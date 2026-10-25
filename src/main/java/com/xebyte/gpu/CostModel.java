package com.xebyte.gpu;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Decides {@link Backend#AUTO}: for each op, the work size above which the GPU wins.
 *
 * <p>The defaults are conservative guesses (transfer + launch overhead is ~0.1-1 ms, so
 * small inputs stay on the CPU). {@code gpu_benchmark} replaces them with measured
 * crossovers for this machine and persists them to
 * {@code ~/.ghidra-gpu/calibration.properties} (override with {@code GHIDRA_GPU_CALIBRATION}),
 * so the next Ghidra session starts calibrated. A crossover of {@link Long#MAX_VALUE}
 * means "the GPU never won in the measured range"; the op still runs on the GPU when
 * asked explicitly.
 *
 * <p>"Work" is op-specific and documented beside each default: bytes scanned, pairs
 * compared, lanes emulated, nodes+edges, and so on. Tools report the work they computed
 * so a caller can reason about the choice.
 */
public final class CostModel {

    private static final Map<String, Long> DEFAULTS = new LinkedHashMap<>();
    static {
        DEFAULTS.put(Ops.PATTERN_SCAN, 4L << 20);      // bytes * patterns
        DEFAULTS.put(Ops.STRING_SCAN, 8L << 20);       // bytes
        DEFAULTS.put(Ops.ENTROPY, 16L << 20);          // bytes
        DEFAULTS.put(Ops.HISTOGRAM, 32L << 20);        // bytes
        DEFAULTS.put(Ops.POINTER_SCAN, 4L << 20);      // bytes
        DEFAULTS.put(Ops.FNV_HASH, 64L << 20);         // bytes hashed
        DEFAULTS.put(Ops.SHA256, 4L << 20);            // bytes hashed
        DEFAULTS.put(Ops.COSINE, 2L << 20);            // pairs * dims
        DEFAULTS.put(Ops.MINHASH, 4L << 20);           // tokens * k
        DEFAULTS.put(Ops.MINHASH_SIM, 4L << 20);       // pairs * k
        DEFAULTS.put(Ops.LEVENSHTEIN, 16L << 20);      // sum of lenA*lenB
        DEFAULTS.put(Ops.BFS, 2L << 20);               // nodes + edges
        DEFAULTS.put(Ops.SCC, 2L << 20);               // nodes + edges
        DEFAULTS.put(Ops.PAGERANK, 1L << 20);          // (nodes + edges) * iterations
        DEFAULTS.put(Ops.DOMINATORS, 1L << 20);        // sum of blocks^2 / 32
        DEFAULTS.put(Ops.SUBSTRING, 8L << 20);         // haystack bytes
        DEFAULTS.put(Ops.IOC_SCAN, 4L << 20);          // bytes
        DEFAULTS.put(Ops.PCODE_VM, 256L);              // lanes
    }

    private static final Map<String, Long> calibrated = new ConcurrentHashMap<>();
    /** op -> observed runs {work, cpuMicros, gpuMicros}, newest last, at most {@link #MAX_OBSERVED}. */
    private static final Map<String, List<long[]>> observed = new ConcurrentHashMap<>();
    static final int MAX_OBSERVED = 64;
    private static volatile boolean loaded;

    private CostModel() {}

    /**
     * True when AUTO should run {@code op} on the GPU for this much work.
     *
     * <p>Observed runs win over the crossover: the run nearest in work size (log scale)
     * decides. A single crossover cannot express "GPU wins in the middle, CPU wins on huge
     * graphs" -- measured on an 80k-function binary, PageRank and dominators were 6-20x
     * slower on the GPU far above the synthetic crossover -- but one real backend=both run
     * at that size is enough to make AUTO choose right from then on.
     */
    public static boolean prefersGpu(String op, long work) {
        long[] near = nearestObserved(op, work);
        if (near != null) return near[2] < near[1];
        return work >= crossover(op);
    }

    /** One-line reason for AUTO's choice, naming the run or crossover it followed. */
    public static String explain(String op, long work) {
        long[] near = nearestObserved(op, work);
        if (near != null) {
            return "nearest observed run (work " + near[0] + "): cpu " + near[1] / 1000.0 + " ms, gpu "
                + near[2] / 1000.0 + " ms";
        }
        long c = crossover(op);
        return (work >= c ? "work >= " : "work < ") + "crossover " + c
            + (isCalibrated(op) ? " (measured)" : " (default; run gpu_benchmark or a backend=both call to measure)");
    }

    /** Records one measured CPU-vs-GPU run (backend=both, gpu_benchmark) and persists it. */
    public static void observe(String op, long work, double cpuMs, double gpuMs) {
        if (work <= 0 || cpuMs < 0 || gpuMs < 0) return;
        ensureLoaded();
        List<long[]> l = observed.computeIfAbsent(op, k -> new java.util.concurrent.CopyOnWriteArrayList<>());
        l.removeIf(s -> s[0] == work);
        l.add(new long[] {work, Math.round(cpuMs * 1000), Math.round(gpuMs * 1000)});
        while (l.size() > MAX_OBSERVED) l.remove(0);
        save();
    }

    static long[] nearestObserved(String op, long work) {
        ensureLoaded();
        List<long[]> l = observed.get(op);
        if (l == null || l.isEmpty() || work <= 0) return null;
        long[] best = null;
        double bestD = Double.MAX_VALUE;
        double lw = Math.log(work);
        for (long[] s : l) {
            double d = Math.abs(Math.log(s[0]) - lw);
            if (d < bestD) { bestD = d; best = s; }
        }
        return best;
    }

    public static int observedCount(String op) {
        ensureLoaded();
        List<long[]> l = observed.get(op);
        return l == null ? 0 : l.size();
    }

    public static long crossover(String op) {
        ensureLoaded();
        Long c = calibrated.get(op);
        if (c != null) return c;
        return DEFAULTS.getOrDefault(op, 1L << 20);
    }

    public static boolean isCalibrated(String op) {
        ensureLoaded();
        return calibrated.containsKey(op);
    }

    /** Records a measured crossover and persists the whole table. */
    public static void calibrate(String op, long crossover) {
        ensureLoaded();
        calibrated.put(op, crossover);
        save();
    }

    public static void resetCalibration() {
        calibrated.clear();
        observed.clear();
        try {
            Files.deleteIfExists(file());
        } catch (IOException ignored) {
            // nothing persisted to delete, or not ours to delete
        }
    }

    /** op -> {crossover, source} for every known op, for gpu_info / gpu_advise. */
    public static Map<String, Map<String, Object>> table() {
        ensureLoaded();
        Map<String, Map<String, Object>> out = new TreeMap<>();
        for (String op : DEFAULTS.keySet()) {
            Map<String, Object> row = new LinkedHashMap<>();
            long c = crossover(op);
            row.put("gpu_from_work", c == Long.MAX_VALUE ? "never (in measured range)" : c);
            row.put("source", calibrated.containsKey(op) ? "measured" : "default");
            row.put("work_unit", workUnit(op));
            row.put("observed_runs", observedCount(op));
            out.put(op, row);
        }
        return out;
    }

    public static String workUnit(String op) {
        return switch (op) {
            case Ops.PATTERN_SCAN -> "bytes x patterns";
            case Ops.COSINE -> "pairs x dims";
            case Ops.MINHASH -> "tokens x k";
            case Ops.MINHASH_SIM -> "pairs x k";
            case Ops.LEVENSHTEIN -> "sum(lenA x lenB)";
            case Ops.BFS, Ops.SCC -> "nodes + edges";
            case Ops.PAGERANK -> "(nodes + edges) x iterations";
            case Ops.DOMINATORS -> "sum(blocks^2) / 32";
            case Ops.PCODE_VM -> "lanes";
            case Ops.FNV_HASH, Ops.SHA256 -> "bytes hashed";
            case Ops.SUBSTRING -> "haystack bytes";
            default -> "bytes";
        };
    }

    public static Iterable<String> ops() {
        return DEFAULTS.keySet();
    }

    private static Path file() {
        String prop = System.getProperty("ghidra.gpu.calibration");
        if (prop != null && !prop.isBlank()) return Paths.get(prop);
        String env = System.getenv("GHIDRA_GPU_CALIBRATION");
        if (env != null && !env.isBlank()) return Paths.get(env);
        return Paths.get(System.getProperty("user.home"), ".ghidra-gpu", "calibration.properties");
    }

    private static synchronized void ensureLoaded() {
        if (loaded) return;
        loaded = true;
        Path f = file();
        if (!Files.isRegularFile(f)) return;
        Properties p = new Properties();
        try (InputStream in = Files.newInputStream(f)) {
            p.load(in);
            for (String k : p.stringPropertyNames()) {
                if (k.startsWith("observed.")) {
                    List<long[]> l = new java.util.concurrent.CopyOnWriteArrayList<>();
                    for (String run : p.getProperty(k).split(";")) {
                        String[] f3 = run.trim().split(":");
                        if (f3.length != 3) continue;
                        try {
                            l.add(new long[] {Long.parseLong(f3[0]), Long.parseLong(f3[1]), Long.parseLong(f3[2])});
                        } catch (NumberFormatException ignored) {
                            // skip a malformed run, keep the rest
                        }
                    }
                    if (!l.isEmpty()) observed.put(k.substring("observed.".length()), l);
                    continue;
                }
                try {
                    calibrated.put(k, Long.parseLong(p.getProperty(k).trim()));
                } catch (NumberFormatException ignored) {
                    // a hand-edited line we cannot read is skipped, not fatal
                }
            }
        } catch (IOException ignored) {
            // unreadable calibration means defaults
        }
    }

    private static synchronized void save() {
        Properties p = new Properties();
        calibrated.forEach((k, v) -> p.setProperty(k, Long.toString(v)));
        observed.forEach((k, l) -> {
            StringBuilder sb = new StringBuilder();
            for (long[] r : l) sb.append(sb.length() == 0 ? "" : ";").append(r[0]).append(':').append(r[1]).append(':').append(r[2]);
            p.setProperty("observed." + k, sb.toString());
        });
        Path f = file();
        try {
            Files.createDirectories(f.getParent());
            try (OutputStream out = Files.newOutputStream(f)) {
                p.store(out, "ghidra-gpu AUTO model: crossovers (work units per op) and observed runs "
                    + "(observed.<op>=work:cpu_us:gpu_us;...), written by gpu_benchmark and backend=both calls");
            }
        } catch (IOException ignored) {
            // calibration still applies for this session
        }
    }
}
