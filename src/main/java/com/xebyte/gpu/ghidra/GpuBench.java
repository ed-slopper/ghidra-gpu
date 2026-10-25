package com.xebyte.gpu.ghidra;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

import com.xebyte.gpu.CostModel;
import com.xebyte.gpu.GraphOps;
import com.xebyte.gpu.HashOps;
import com.xebyte.gpu.Ops;
import com.xebyte.gpu.PcodeVm;
import com.xebyte.gpu.ScanOps;
import com.xebyte.gpu.SimilarityOps;
import com.xebyte.gpu.cuda.Gpu;

/**
 * Synthetic workloads for every op, used two ways: {@link #run} times CPU against GPU at
 * doubling sizes and records the crossover in {@link CostModel}; {@link #selftest} runs
 * each op once on both backends and reports agreement. Neither needs a program open.
 */
public final class GpuBench {

    private GpuBench() {}

    /** One sized workload: its work in CostModel units, a CPU run and a GPU run returning comparable values. */
    record Case(long work, Supplier<Object> cpu, Function<Gpu.Session, Object> gpu) {}

    interface Factory {
        Case make(int scale, Random r);
    }

    static final Map<String, Factory> FACTORIES = new LinkedHashMap<>();
    static {
        FACTORIES.put(Ops.PATTERN_SCAN, (k, r) -> {
            byte[] d = bytes(r, (64 << 10) << k);
            List<byte[]> p = List.of(new byte[] {0x55, (byte) 0x8b, (byte) 0xec}, new byte[] {(byte) 0xe8, 0, 0, 0, 0});
            List<byte[]> m = List.of(new byte[] {-1, -1, -1}, new byte[] {-1, 0, 0, 0, 0});
            return new Case((long) d.length * 2, () -> ScanOps.patternCpu(d, p, m, 1, 1000), s -> ScanOps.patternGpu(s, d, p, m, 1, 1000));
        });
        FACTORIES.put(Ops.STRING_SCAN, (k, r) -> {
            byte[] d = textish(r, (64 << 10) << k);
            return new Case(d.length, () -> ScanOps.stringsCpu(d, 5, false, 4096, 1 << 20), s -> ScanOps.stringsGpu(s, d, 5, false, 4096, 1 << 20));
        });
        FACTORIES.put(Ops.ENTROPY, (k, r) -> {
            byte[] d = bytes(r, (64 << 10) << k);
            return new Case(d.length, () -> ScanOps.entropyCpu(d, 4096), s -> ScanOps.entropyGpu(s, d, 4096));
        });
        FACTORIES.put(Ops.HISTOGRAM, (k, r) -> {
            byte[] d = bytes(r, (64 << 10) << k);
            return new Case(d.length, () -> ScanOps.histogramCpu(d), s -> ScanOps.histogramGpu(s, d));
        });
        FACTORIES.put(Ops.POINTER_SCAN, (k, r) -> {
            byte[] d = bytes(r, (64 << 10) << k);
            return new Case(d.length, () -> ScanOps.pointersCpu(d, 4, 4, false, 0x10000000L, 0x20000000L, null, 1000),
                s -> ScanOps.pointersGpu(s, d, 4, 4, false, 0x10000000L, 0x20000000L, null, 1000));
        });
        FACTORIES.put(Ops.IOC_SCAN, (k, r) -> {
            byte[] d = textish(r, (64 << 10) << k);
            return new Case(d.length, () -> ScanOps.iocCpu(d, 512, 1000), s -> ScanOps.iocGpu(s, d, 512, 1000));
        });
        FACTORIES.put(Ops.SUBSTRING, (k, r) -> {
            ScanOps.Packed p = ScanOps.Packed.of(names(r, 2000 << k));
            byte[] n = "file".getBytes();
            return new Case(p.bytes().length, () -> ScanOps.substringCpu(p, n, true), s -> ScanOps.substringGpu(s, p, n, true));
        });
        FACTORIES.put(Ops.FNV_HASH, (k, r) -> {
            ScanOps.Packed p = ScanOps.Packed.of(names(r, 2000 << k));
            return new Case(p.bytes().length, () -> HashOps.fnvCpu(p), s -> HashOps.fnvGpu(s, p));
        });
        FACTORIES.put(Ops.SHA256, (k, r) -> {
            List<String> segs = new ArrayList<>();
            for (int i = 0; i < (200 << k); i++) segs.add(new String(textish(r, 512), StandardCharsets.ISO_8859_1));
            ScanOps.Packed p = ScanOps.Packed.of(segs);
            return new Case(p.bytes().length, () -> HashOps.sha256Cpu(p), s -> HashOps.sha256Gpu(s, p));
        });
        FACTORIES.put(Ops.COSINE, (k, r) -> {
            int n = 64 << (k / 2), m = 64 << ((k + 1) / 2), dim = 64;
            float[] a = floats(r, n * dim), b = floats(r, m * dim);
            return new Case((long) n * m * dim, () -> SimilarityOps.cosineCpu(a, n, b, m, dim), s -> SimilarityOps.cosineGpu(s, a, n, b, m, dim));
        });
        FACTORIES.put(Ops.MINHASH, (k, r) -> {
            SimilarityOps.TokenSets t = tokenSets(r, 100 << k, 100);
            return new Case((long) t.tokens().length * 64, () -> SimilarityOps.minhashCpu(t, 64, 3), s -> SimilarityOps.minhashGpu(s, t, 64, 3));
        });
        FACTORIES.put(Ops.MINHASH_SIM, (k, r) -> {
            int n = 64 << (k / 2), m = 64 << ((k + 1) / 2);
            int[] a = ints(r, n * 64), b = ints(r, m * 64);
            return new Case((long) n * m * 64, () -> SimilarityOps.minhashSimCpu(a, n, b, m, 64), s -> SimilarityOps.minhashSimGpu(s, a, n, b, m, 64));
        });
        FACTORIES.put(Ops.LEVENSHTEIN, (k, r) -> {
            SimilarityOps.TokenSets a = tokenSets(r, 16 << k, 120), b = tokenSets(r, 16 << k, 120);
            long w = 0;
            for (int i = 0; i < a.count(); i++) w += (long) a.len()[i] * b.len()[i];
            return new Case(w, () -> SimilarityOps.levenshteinCpu(a, b), s -> SimilarityOps.levenshteinGpu(s, a, b));
        });
        FACTORIES.put(Ops.BFS, (k, r) -> {
            GraphOps.Csr g = graph(r, 2000 << k, 6000 << k);
            int[] src = {0};
            return new Case((long) g.n() + g.edges(), () -> GraphOps.bfsCpu(g, src, Integer.MAX_VALUE), s -> GraphOps.bfsGpu(s, g, src, Integer.MAX_VALUE));
        });
        FACTORIES.put(Ops.SCC, (k, r) -> {
            GraphOps.Csr g = graph(r, 2000 << k, 4000 << k);
            return new Case((long) g.n() + g.edges(), () -> GraphOps.sccCpu(g), s -> GraphOps.sccGpu(s, g));
        });
        FACTORIES.put(Ops.PAGERANK, (k, r) -> {
            GraphOps.Csr g = graph(r, 2000 << k, 6000 << k);
            return new Case(((long) g.n() + g.edges()) * 20, () -> GraphOps.pagerankCpu(g, 0.85, 20), s -> GraphOps.pagerankGpu(s, g, 0.85, 20));
        });
        FACTORIES.put(Ops.DOMINATORS, (k, r) -> {
            List<int[][]> gs = new ArrayList<>();
            long w = 0;
            for (int i = 0; i < (64 << k); i++) {
                int nb = 2 + r.nextInt(60), m = nb * 2;
                int[] s = new int[m], d = new int[m];
                for (int e = 0; e < m; e++) { s[e] = r.nextInt(nb); d[e] = r.nextInt(nb); }
                gs.add(GraphOps.Cfgs.graph(nb, s, d));
                w += (long) nb * nb / 32 + nb;
            }
            GraphOps.Cfgs c = GraphOps.Cfgs.of(gs);
            return new Case(w, () -> GraphOps.dominatorsCpu(c), s -> GraphOps.dominatorsGpu(s, c));
        });
        FACTORIES.put(Ops.PCODE_VM, (k, r) -> {
            int n = 32 << k;
            Object[] vm = ror13Vm(r, n);
            PcodeVm.Program p = (PcodeVm.Program) vm[0];
            PcodeVm.Memory m = (PcodeVm.Memory) vm[1];
            PcodeVm.Lanes l = (PcodeVm.Lanes) vm[2];
            return new Case(n, () -> PcodeVm.runCpu(p, m, l, 0xDEADBEEFL, 100_000), s -> PcodeVm.runGpu(s, p, m, l, 0xDEADBEEFL, 100_000));
        });
    }

    /** Times each op at {@code scales} doubling sizes and calibrates crossovers. */
    public static Map<String, Object> run(List<String> ops, int scales) {
        Gpu g = Gpu.get();
        Map<String, Object> out = new LinkedHashMap<>();
        Map<String, Object> table = new LinkedHashMap<>();
        Random r = new Random(42);
        for (String op : ops) {
            Factory f = FACTORIES.get(op);
            if (f == null) { table.put(op, "unknown op"); continue; }
            List<Map<String, Object>> rows = new ArrayList<>();
            long[] work = new long[scales];
            boolean[] gpuWins = new boolean[scales];
            try {
                g.run(f.make(0, r).gpu()); // warm-up: first launch pays JIT/module costs
                for (int k = 0; k < scales; k++) {
                    Case c = f.make(k, r);
                    double cpu = best(() -> c.cpu().get());
                    double gpu = best(() -> g.run(c.gpu()));
                    work[k] = c.work();
                    gpuWins[k] = gpu < cpu;
                    CostModel.observe(op, c.work(), cpu, gpu);
                    rows.add(Map.of("work", c.work(), "cpu_ms", round(cpu), "gpu_ms", round(gpu),
                        "speedup", round(cpu / Math.max(1e-6, gpu))));
                }
            } catch (RuntimeException e) {
                table.put(op, "failed: " + e.getMessage());
                continue;
            }
            long crossover = Long.MAX_VALUE;
            for (int k = scales - 1; k >= 0 && gpuWins[k]; k--) crossover = work[k];
            CostModel.calibrate(op, crossover);
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("work_unit", CostModel.workUnit(op));
            m.put("gpu_from_work", crossover == Long.MAX_VALUE ? "never in measured range" : crossover);
            m.put("runs", rows);
            table.put(op, m);
        }
        out.put("device", g.info().get("device"));
        out.put("ops", table);
        out.put("note", "Crossovers saved; backend=auto now uses them. Wall times include host<->device transfers.");
        return out;
    }

    /** One run per op on both backends; reports agreement and timings. */
    public static Map<String, Object> selftest() {
        Gpu g = Gpu.get();
        Map<String, Object> rows = new LinkedHashMap<>();
        Random r = new Random(7);
        int pass = 0, fail = 0;
        for (Map.Entry<String, Factory> e : FACTORIES.entrySet()) {
            Map<String, Object> m = new LinkedHashMap<>();
            try {
                Case c = e.getValue().make(2, r);
                long t0 = System.nanoTime();
                Object cv = c.cpu().get();
                double cpu = (System.nanoTime() - t0) / 1e6;
                long t1 = System.nanoTime();
                Object gv = g.run(c.gpu());
                double gpu = (System.nanoTime() - t1) / 1e6;
                boolean ok = equal(cv, gv);
                m.put("agree", ok);
                m.put("cpu_ms", round(cpu));
                m.put("gpu_ms", round(gpu));
                if (ok) pass++; else fail++;
            } catch (RuntimeException ex) {
                m.put("agree", false);
                m.put("error", ex.getMessage());
                fail++;
            }
            rows.put(e.getKey(), m);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("device", g.info().get("device"));
        out.put("passed", pass);
        out.put("failed", fail);
        out.put("ops", rows);
        return out;
    }

    static boolean equal(Object a, Object b) {
        if (a instanceof ScanOps.Hits x && b instanceof ScanOps.Hits y) return ScanOps.Hits.same(x, y);
        if (a instanceof float[] x && b instanceof float[] y) return ScanOps.sameFloats(x, y, 1e-4f);
        if (a instanceof double[] x && b instanceof double[] y) {
            if (x.length != y.length) return false;
            for (int i = 0; i < x.length; i++) if (Math.abs(x[i] - y[i]) > 1e-12) return false;
            return true;
        }
        if (a instanceof long[] x && b instanceof long[] y) return Arrays.equals(x, y);
        if (a instanceof int[] x && b instanceof int[] y) return Arrays.equals(x, y);
        if (a instanceof byte[] x && b instanceof byte[] y) return Arrays.equals(x, y);
        if (a instanceof GraphOps.DomResult x && b instanceof GraphOps.DomResult y) return GraphOps.DomResult.same(x, y);
        if (a instanceof PcodeVm.Result x && b instanceof PcodeVm.Result y) return PcodeVm.Result.same(x, y);
        return a.equals(b);
    }

    static double best(Runnable r) {
        double best = Double.MAX_VALUE;
        for (int i = 0; i < 3; i++) {
            long t = System.nanoTime();
            r.run();
            best = Math.min(best, (System.nanoTime() - t) / 1e6);
        }
        return best;
    }

    static double round(double v) {
        return Math.round(v * 1000.0) / 1000.0;
    }

    // ------------------------------------------------------------------ generators

    static byte[] bytes(Random r, int n) {
        byte[] d = new byte[n];
        r.nextBytes(d);
        return d;
    }

    static byte[] textish(Random r, int n) {
        byte[] d = new byte[n];
        String[] plant = {"http://example.com/a", "10.1.2.3", "a.b@example.org", "HKEY_CURRENT_USER\\Run", "GetProcAddress"};
        for (int i = 0; i < n; i++) d[i] = r.nextInt(4) == 0 ? 0 : (byte) (0x20 + r.nextInt(95));
        for (int i = 0; i < n / 1024; i++) {
            byte[] s = plant[r.nextInt(plant.length)].getBytes(StandardCharsets.US_ASCII);
            int at = r.nextInt(Math.max(1, n - s.length - 1));
            System.arraycopy(s, 0, d, at, s.length);
        }
        return d;
    }

    static List<String> names(Random r, int n) {
        List<String> l = new ArrayList<>(n);
        for (int i = 0; i < n; i++) l.add((r.nextBoolean() ? "FUN_" : "CreateFile") + Integer.toHexString(r.nextInt()));
        return l;
    }

    static float[] floats(Random r, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = r.nextInt(5);
        return a;
    }

    static int[] ints(Random r, int n) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) a[i] = r.nextInt(8);
        return a;
    }

    static SimilarityOps.TokenSets tokenSets(Random r, int sets, int maxLen) {
        List<int[]> l = new ArrayList<>();
        for (int i = 0; i < sets; i++) {
            int[] t = new int[1 + r.nextInt(maxLen)];
            for (int k = 0; k < t.length; k++) t[k] = r.nextInt(40);
            l.add(t);
        }
        return SimilarityOps.TokenSets.of(l);
    }

    static GraphOps.Csr graph(Random r, int n, int m) {
        int[] s = new int[m], d = new int[m];
        for (int e = 0; e < m; e++) { s[e] = r.nextInt(n); d[e] = r.nextInt(n); }
        return GraphOps.Csr.of(n, s, d);
    }

    /** A hand-lowered ROR13 hash loop and {@code n} lanes of random API-like names. */
    static Object[] ror13Vm(Random r, int n) {
        long scratch = 0x7FFE0000L;
        long[][] ops = {
            // opcode, nIn, outKind,outSize,outVal, in0Kind,in0Size,in0Val, in1Kind,in1Size,in1Val
            {2, 2, 1, 1, 16, 2, 4, 0, 1, 8, 0},
            {11, 2, 1, 1, 17, 1, 1, 16, 2, 1, 0},
            {5, 2, 0, 0, 0, 2, 8, 10, 1, 1, 17},
            {30, 2, 1, 4, 20, 1, 4, 24, 2, 4, 13},
            {29, 2, 1, 4, 28, 1, 4, 24, 2, 4, 19},
            {28, 2, 1, 4, 24, 1, 4, 20, 1, 4, 28},
            {17, 1, 1, 4, 32, 1, 1, 16, 0, 0, 0},
            {19, 2, 1, 4, 24, 1, 4, 24, 1, 4, 32},
            {19, 2, 1, 8, 0, 1, 8, 0, 2, 8, 1},
            {4, 1, 0, 0, 0, 2, 8, 0, 0, 0, 0},
            {10, 1, 0, 0, 0, 2, 8, 0xDEADBEEFL, 0, 0, 0},
        };
        int c = ops.length;
        int[] opc = new int[c], instr = new int[c], nIn = new int[c], kind = new int[4 * c], size = new int[4 * c];
        long[] val = new long[4 * c];
        for (int i = 0; i < c; i++) {
            opc[i] = (int) ops[i][0];
            nIn[i] = (int) ops[i][1];
            instr[i] = i;
            for (int v = 0; v < 3; v++) {
                kind[4 * i + v] = (int) ops[i][2 + 3 * v];
                size[4 * i + v] = (int) ops[i][3 + 3 * v];
                val[4 * i + v] = ops[i][4 + 3 * v];
            }
        }
        int[] first = new int[c];
        long[] addr = new long[c];
        for (int i = 0; i < c; i++) { first[i] = i; addr[i] = 0x401000L + i; }
        PcodeVm.Program p = new PcodeVm.Program(c, opc, instr, nIn, kind, size, val, first, addr, false, 40);
        PcodeVm.Memory m = new PcodeVm.Memory(new long[] {scratch}, new int[] {4096}, new byte[0], new long[0], new int[0], new long[0]);
        byte[] state = new byte[n * 40];
        ByteArrayOutputStream priv = new ByteArrayOutputStream();
        long[] off = new long[n];
        int[] len = new int[n];
        for (int i = 0; i < n; i++) {
            for (int b = 0; b < 8; b++) state[i * 40 + b] = (byte) (scratch >>> (8 * b));
            String s = "Api" + Integer.toHexString(r.nextInt()) + "W\0";
            byte[] rec = PcodeVm.privRecords(new long[] {0}, new byte[][] {s.getBytes(StandardCharsets.US_ASCII)});
            off[i] = priv.size();
            len[i] = rec.length;
            priv.writeBytes(rec);
        }
        return new Object[] {p, m, new PcodeVm.Lanes(n, state, priv.toByteArray(), off, len)};
    }
}
