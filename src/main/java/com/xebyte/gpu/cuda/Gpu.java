package com.xebyte.gpu.cuda;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;

/**
 * The process-wide CUDA runtime: one context on one device, one loaded module.
 *
 * <p>Initialisation is lazy and never throws to callers: if there is no NVIDIA driver,
 * no device, or the PTX does not JIT, {@link #get()} returns {@code null} and
 * {@link #unavailableReason()} says why, so every op can fall back to its CPU twin
 * and tell the caller what happened instead of failing the request.
 *
 * <p>All device work goes through {@link #run}, which serialises callers (the MCP
 * server is multi-threaded, a CUDA context is not something to race on), makes the
 * context current on the calling thread, and frees every buffer the session
 * allocated when it ends -- including on exceptions.
 *
 * <p>Set {@code GHIDRA_GPU_DISABLE=1} (or {@code -Dghidra.gpu.disable=true}) to force
 * the CPU path everywhere, e.g. to compare results or on a shared GPU.
 */
public final class Gpu {

    public static final String PTX_RESOURCE = "/com/xebyte/gpu/ghidra_gpu.ptx";
    public static final int BLOCK = 256;

    private static final Object INIT_LOCK = new Object();
    private static volatile boolean initialised;
    private static Gpu instance;
    private static String unavailableReason = "not initialised";

    private final CudaDriver cu;
    private final Pointer context;
    private final Map<String, Pointer> functions = new HashMap<>();
    private final Map<String, Object> info = new LinkedHashMap<>();
    private final int smCount;

    /** The runtime, or {@code null} when no usable GPU is present. */
    public static Gpu get() {
        if (!initialised) {
            synchronized (INIT_LOCK) {
                if (!initialised) {
                    try {
                        if (disabledByConfig()) {
                            unavailableReason = "disabled by GHIDRA_GPU_DISABLE / -Dghidra.gpu.disable";
                        } else {
                            instance = new Gpu();
                            unavailableReason = null;
                        }
                    } catch (Throwable t) {
                        instance = null;
                        unavailableReason = t.getClass().getSimpleName() + ": " + t.getMessage();
                    }
                    initialised = true;
                }
            }
        }
        return instance;
    }

    public static boolean isAvailable() {
        return get() != null;
    }

    /** Why {@link #get()} returned null; null when the GPU is available. */
    public static String unavailableReason() {
        get();
        return unavailableReason;
    }

    private static boolean disabledByConfig() {
        String env = System.getenv("GHIDRA_GPU_DISABLE");
        return Boolean.getBoolean("ghidra.gpu.disable")
            || (env != null && !env.isBlank() && !env.equals("0") && !env.equalsIgnoreCase("false"));
    }

    private Gpu() throws IOException {
        cu = CudaDriver.load();
        check(cu.cuInit(0), "cuInit");
        IntByReference count = new IntByReference();
        check(cu.cuDeviceGetCount(count), "cuDeviceGetCount");
        if (count.getValue() < 1) throw new IllegalStateException("no CUDA device");
        int ordinal = Integer.getInteger("ghidra.gpu.device", envInt("GHIDRA_GPU_DEVICE", 0));
        IntByReference dev = new IntByReference();
        check(cu.cuDeviceGet(dev, ordinal), "cuDeviceGet");
        int device = dev.getValue();

        byte[] name = new byte[256];
        check(cu.cuDeviceGetName(name, name.length, device), "cuDeviceGetName");
        IntByReference v = new IntByReference();
        cu.cuDriverGetVersion(v);
        int driverVersion = v.getValue();
        cu.cuDeviceGetAttribute(v, CudaDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        int major = v.getValue();
        cu.cuDeviceGetAttribute(v, CudaDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        int minor = v.getValue();
        cu.cuDeviceGetAttribute(v, CudaDriver.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
        smCount = Math.max(1, v.getValue());
        cu.cuDeviceGetAttribute(v, CudaDriver.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
        int clockKhz = v.getValue();
        LongByReference total = new LongByReference();
        cu.cuDeviceTotalMem_v2(total, device);

        PointerByReference ctx = new PointerByReference();
        check(cu.cuCtxCreate_v2(ctx, 0, device), "cuCtxCreate");
        context = ctx.getValue();

        String ptx = readPtx();
        Memory image = new Memory(ptx.length() + 1L);
        image.setString(0, ptx, StandardCharsets.US_ASCII.name());
        Memory log = new Memory(16384);
        log.clear();
        PointerByReference module = new PointerByReference();
        int rc = cu.cuModuleLoadDataEx(module, image, 2,
            new int[] {CudaDriver.CU_JIT_ERROR_LOG_BUFFER, CudaDriver.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES},
            new Pointer[] {log, new Pointer(16384)});
        if (rc != 0) {
            throw new IllegalStateException("PTX JIT failed (" + errorName(rc) + "): " + log.getString(0));
        }
        Pointer mod = module.getValue();
        for (String k : KERNELS) {
            PointerByReference f = new PointerByReference();
            check(cu.cuModuleGetFunction(f, mod, k), "cuModuleGetFunction(" + k + ")");
            functions.put(k, f.getValue());
        }

        info.put("device", Native.toString(name));
        info.put("device_ordinal", ordinal);
        info.put("compute_capability", major + "." + minor);
        info.put("multiprocessors", smCount);
        info.put("clock_mhz", clockKhz / 1000);
        info.put("total_memory_mb", total.getValue() / (1024 * 1024));
        info.put("driver_cuda_version", (driverVersion / 1000) + "." + (driverVersion % 1000) / 10);
        info.put("kernels", KERNELS.size());
        info.put("ptx", PTX_RESOURCE);
    }

    /** Every kernel the module must export; checked at load so a stale PTX fails loudly. */
    public static final List<String> KERNELS = List.of(
        "pattern_scan", "string_scan", "byte_histogram", "window_entropy", "pointer_scan",
        "fnv1a64_batch", "sha256_batch", "cosine_matrix", "minhash", "minhash_similarity",
        "levenshtein_batch", "bfs_step", "scc_color_propagate", "scc_backward", "scc_trim", "scc_seed_roots",
        "scc_reset_colors", "pagerank_step", "dominators_batch", "substring_search", "ioc_scan",
        "pcode_vm", "token_subset_pairs");

    private static int envInt(String name, int dflt) {
        try {
            String s = System.getenv(name);
            return s == null ? dflt : Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return dflt;
        }
    }

    private static String readPtx() throws IOException {
        try (InputStream in = Gpu.class.getResourceAsStream(PTX_RESOURCE)) {
            if (in == null) throw new IOException("missing resource " + PTX_RESOURCE);
            return new String(in.readAllBytes(), StandardCharsets.US_ASCII);
        }
    }

    public Map<String, Object> info() {
        Map<String, Object> out = new LinkedHashMap<>(info);
        synchronized (this) {
            cu.cuCtxSetCurrent(context);
            LongByReference free = new LongByReference();
            LongByReference total = new LongByReference();
            if (cu.cuMemGetInfo_v2(free, total) == 0) {
                out.put("free_memory_mb", free.getValue() / (1024 * 1024));
            }
        }
        return out;
    }

    public int multiprocessors() {
        return smCount;
    }

    /** Runs {@code body} with exclusive use of the device; all its buffers are freed afterwards. */
    public synchronized <T> T run(Function<Session, T> body) {
        check(cu.cuCtxSetCurrent(context), "cuCtxSetCurrent");
        Session s = new Session();
        try {
            return body.apply(s);
        } catch (GpuException e) {
            if (isSticky(e.code())) {
                // The context is unusable after these; route everything to the CPU from now on.
                synchronized (INIT_LOCK) {
                    instance = null;
                    unavailableReason = "CUDA context lost after " + e.getMessage();
                }
            }
            throw e;
        } finally {
            try {
                s.freeAll();
            } catch (Throwable ignored) {
                // a lost context cannot free; the driver reclaims it with the context
            }
        }
    }

    /** Errors after which CUDA refuses every further call on the context. */
    private static boolean isSticky(int code) {
        return code == 700 || code == 701 || code == 702 || code == 714 || code == 715
            || code == 716 || code == 717 || code == 718 || code == 719;
    }

    void check(int rc, String what) {
        if (rc != 0) throw new GpuException(what + " failed: " + errorName(rc), rc);
    }

    private String errorName(int rc) {
        try {
            PointerByReference p = new PointerByReference();
            if (cu.cuGetErrorName(rc, p) == 0 && p.getValue() != null) {
                return p.getValue().getString(0) + " (" + rc + ")";
            }
        } catch (Throwable ignored) {
            // fall through to the bare code
        }
        return "CUDA error " + rc;
    }

    /** A device allocation; valid until its session ends. */
    public record Buf(long ptr, long bytes) {}

    /** Scoped device work. Not thread-safe; it only exists inside {@link Gpu#run}. */
    public final class Session {
        private final List<Buf> owned = new ArrayList<>();

        public Buf alloc(long bytes) {
            LongByReference p = new LongByReference();
            check(cu.cuMemAlloc_v2(p, Math.max(1, bytes)), "cuMemAlloc(" + bytes + ")");
            Buf b = new Buf(p.getValue(), bytes);
            owned.add(b);
            return b;
        }

        public Buf zeros(long bytes) {
            Buf b = alloc(bytes);
            check(cu.cuMemsetD8_v2(b.ptr, (byte) 0, Math.max(1, bytes)), "cuMemsetD8");
            return b;
        }

        public Buf fill(long bytes, byte value) {
            Buf b = alloc(bytes);
            check(cu.cuMemsetD8_v2(b.ptr, value, Math.max(1, bytes)), "cuMemsetD8");
            return b;
        }

        public Buf put(byte[] a) {
            Buf b = alloc(a.length);
            if (a.length > 0) check(cu.cuMemcpyHtoD_v2(b.ptr, a, a.length), "HtoD");
            return b;
        }

        public Buf put(int[] a) {
            Buf b = alloc(4L * a.length);
            if (a.length > 0) check(cu.cuMemcpyHtoD_v2(b.ptr, a, 4L * a.length), "HtoD");
            return b;
        }

        public Buf put(long[] a) {
            Buf b = alloc(8L * a.length);
            if (a.length > 0) check(cu.cuMemcpyHtoD_v2(b.ptr, a, 8L * a.length), "HtoD");
            return b;
        }

        public Buf put(float[] a) {
            Buf b = alloc(4L * a.length);
            if (a.length > 0) check(cu.cuMemcpyHtoD_v2(b.ptr, a, 4L * a.length), "HtoD");
            return b;
        }

        public Buf put(double[] a) {
            Buf b = alloc(8L * a.length);
            if (a.length > 0) check(cu.cuMemcpyHtoD_v2(b.ptr, a, 8L * a.length), "HtoD");
            return b;
        }

        public void write(Buf dst, int[] a) {
            if (a.length > 0) check(cu.cuMemcpyHtoD_v2(dst.ptr, a, 4L * a.length), "HtoD");
        }

        public void write(Buf dst, double[] a) {
            if (a.length > 0) check(cu.cuMemcpyHtoD_v2(dst.ptr, a, 8L * a.length), "HtoD");
        }

        public byte[] bytes(Buf b, int n) {
            byte[] a = new byte[n];
            if (n > 0) check(cu.cuMemcpyDtoH_v2(a, b.ptr, n), "DtoH");
            return a;
        }

        public int[] ints(Buf b, int n) {
            int[] a = new int[n];
            if (n > 0) check(cu.cuMemcpyDtoH_v2(a, b.ptr, 4L * n), "DtoH");
            return a;
        }

        public long[] longs(Buf b, int n) {
            long[] a = new long[n];
            if (n > 0) check(cu.cuMemcpyDtoH_v2(a, b.ptr, 8L * n), "DtoH");
            return a;
        }

        public float[] floats(Buf b, int n) {
            float[] a = new float[n];
            if (n > 0) check(cu.cuMemcpyDtoH_v2(a, b.ptr, 4L * n), "DtoH");
            return a;
        }

        public double[] doubles(Buf b, int n) {
            double[] a = new double[n];
            if (n > 0) check(cu.cuMemcpyDtoH_v2(a, b.ptr, 8L * n), "DtoH");
            return a;
        }

        public int readInt(Buf b) {
            return ints(b, 1)[0];
        }

        /** Grid size for a grid-stride kernel over {@code work} items. */
        public int gridFor(long work) {
            long blocks = (work + BLOCK - 1) / BLOCK;
            return (int) Math.max(1, Math.min(blocks, (long) smCount * 32));
        }

        /** Launches a grid-stride kernel sized for {@code work} items and waits for it. */
        public void launch(String kernel, long work, Object... args) {
            launchGrid(kernel, gridFor(work), BLOCK, 0, args);
        }

        /**
         * Launches {@code kernel} and synchronises. Arguments: {@link Buf} (device
         * pointer), {@link Long} (64-bit), {@link Integer}, {@link Float}, {@link Double}.
         * The boxed type must match the kernel's parameter width exactly.
         */
        public void launchGrid(String kernel, int grid, int block, int sharedBytes, Object... args) {
            Pointer f = functions.get(kernel);
            if (f == null) throw new GpuException("unknown kernel " + kernel, -1);
            Memory values = new Memory(8L * Math.max(1, args.length));
            Memory params = new Memory((long) Native.POINTER_SIZE * Math.max(1, args.length));
            for (int i = 0; i < args.length; i++) {
                long off = 8L * i;
                Object a = args[i];
                if (a instanceof Buf b) values.setLong(off, b.ptr);
                else if (a instanceof Long l) values.setLong(off, l);
                else if (a instanceof Integer n) values.setInt(off, n);
                else if (a instanceof Float x) values.setFloat(off, x);
                else if (a instanceof Double d) values.setDouble(off, d);
                else throw new IllegalArgumentException("kernel arg " + i + " of " + kernel + ": " + a);
                params.setPointer((long) Native.POINTER_SIZE * i, values.share(off));
            }
            check(cu.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, sharedBytes, null, params, null),
                "cuLaunchKernel(" + kernel + ")");
            check(cu.cuCtxSynchronize(), kernel);
        }

        /** A point to {@link #release} back to: frees everything allocated after it. */
        public int mark() {
            return owned.size();
        }

        public void release(int mark) {
            while (owned.size() > mark) cu.cuMemFree_v2(owned.remove(owned.size() - 1).ptr);
        }

        void freeAll() {
            for (Buf b : owned) cu.cuMemFree_v2(b.ptr);
            owned.clear();
        }
    }

    /** A CUDA driver call returned an error. */
    public static final class GpuException extends RuntimeException {
        private final int code;

        public GpuException(String message, int code) {
            super(message);
            this.code = code;
        }

        public int code() {
            return code;
        }
    }
}
