package com.xebyte.offline;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import com.xebyte.gpu.GraphOps;
import com.xebyte.gpu.GraphOps.Csr;
import com.xebyte.gpu.HashOps;
import com.xebyte.gpu.PcodeVm;
import com.xebyte.gpu.ScanOps;
import com.xebyte.gpu.ScanOps.Packed;
import com.xebyte.gpu.SimilarityOps;
import com.xebyte.gpu.SimilarityOps.TokenSets;
import com.xebyte.gpu.cuda.Gpu;

/**
 * Every GPU op against its CPU twin on randomized inputs, plus CPU-only checks against
 * independent references (so the CPU side is pinned even on a machine with no GPU).
 *
 * <p>The GPU half self-skips without an NVIDIA GPU ({@code assumeTrue}); the skip count is
 * the signal, not the exit status.
 */
public class GpuKernelParityTest {

    private static final Random R = new Random(0x6b1d7a);

    private static Gpu gpu() {
        Gpu g = Gpu.get();
        assumeTrue("no usable GPU: " + Gpu.unavailableReason(), g != null);
        return g;
    }

    /** Binary-ish data with planted strings, patterns, pointers and IOCs. */
    private static byte[] corpus(int n) {
        byte[] d = new byte[n];
        R.nextBytes(d);
        for (int i = 0; i < n / 7; i++) d[R.nextInt(n)] = 0;
        String[] plant = {"http://evil.example.com/x?a=1", "10.0.0.255", "203.0.113.7", "user.name@corp.example.org",
            "HKEY_LOCAL_MACHINE\\SOFTWARE\\Run", "kernel32.dll", "GetProcAddress", "999.1.1.1", "1.2.3.4.5"};
        for (int i = 0; i < n / 512; i++) {
            byte[] s = plant[R.nextInt(plant.length)].getBytes(StandardCharsets.US_ASCII);
            int at = R.nextInt(Math.max(1, n - s.length - 2));
            System.arraycopy(s, 0, d, at, s.length);
            d[at + s.length] = 0;
        }
        for (int i = 0; i < n / 256; i++) {           // UTF-16 strings
            String s = "WideString" + i;
            int at = R.nextInt(Math.max(1, n - 2 * s.length() - 2)) & ~1;
            for (int k = 0; k < s.length(); k++) { d[at + 2 * k] = (byte) s.charAt(k); d[at + 2 * k + 1] = 0; }
        }
        return d;
    }

    // ------------------------------------------------------------------ scans

    @Test
    public void patternScanParity() {
        Gpu g = gpu();
        byte[] d = corpus(1 << 20);
        List<byte[]> pats = List.of(new byte[] {0x55, (byte) 0x8B, (byte) 0xEC}, new byte[] {0, 0},
            "http".getBytes(), new byte[] {(byte) 0xE8, 0, 0, 0, 0});
        List<byte[]> masks = List.of(new byte[] {-1, -1, -1}, new byte[] {-1, -1}, new byte[] {-1, -1, -1, -1},
            new byte[] {-1, 0, 0, 0, 0});
        for (int align : new int[] {1, 4}) {
            for (int max : new int[] {10, 100_000}) {
                ScanOps.Hits c = ScanOps.patternCpu(d, pats, masks, align, max);
                ScanOps.Hits gv = g.run(s -> ScanOps.patternGpu(s, d, pats, masks, align, max));
                assertTrue("align=" + align + " max=" + max, ScanOps.Hits.same(c, gv));
                assertTrue(c.total() > 0);
            }
        }
    }

    @Test
    public void stringScanParity() {
        Gpu g = gpu();
        byte[] d = corpus(1 << 20);
        for (boolean wide : new boolean[] {false, true}) {
            ScanOps.Hits c = ScanOps.stringsCpu(d, 5, wide, 4096, 50_000);
            ScanOps.Hits gv = g.run(s -> ScanOps.stringsGpu(s, d, 5, wide, 4096, 50_000));
            assertTrue("wide=" + wide, ScanOps.Hits.same(c, gv));
            assertTrue(c.total() > 0);
        }
    }

    @Test
    public void entropyAndHistogramParity() {
        Gpu g = gpu();
        byte[] d = corpus((1 << 20) + 333);
        assertArrayEquals(ScanOps.histogramCpu(d), g.run(s -> ScanOps.histogramGpu(s, d)));
        float[] c = ScanOps.entropyCpu(d, 4096);
        float[] gv = g.run(s -> ScanOps.entropyGpu(s, d, 4096));
        assertTrue(ScanOps.sameFloats(c, gv, 1e-5f));
    }

    @Test
    public void pointerScanParity() {
        Gpu g = gpu();
        byte[] d = corpus(1 << 20);
        long[] targets = new long[64];
        for (int i = 0; i < targets.length; i++) {
            int at = R.nextInt(d.length - 8) & ~3;
            targets[i] = ScanOps.pointerValue(ScanOps.pointersCpu(Arrays.copyOfRange(d, at, at + 4), 4, 4, false, 0, -1L, null, 1), 0);
        }
        for (boolean be : new boolean[] {false, true}) {
            ScanOps.Hits c = ScanOps.pointersCpu(d, 4, 4, be, 0x10000000L, 0x70000000L, null, 100_000);
            ScanOps.Hits gv = g.run(s -> ScanOps.pointersGpu(s, d, 4, 4, be, 0x10000000L, 0x70000000L, null, 100_000));
            assertTrue(ScanOps.Hits.same(c, gv));
        }
        ScanOps.Hits c = ScanOps.pointersCpu(d, 4, 4, false, 0, -1L, targets, 100_000);
        ScanOps.Hits gv = g.run(s -> ScanOps.pointersGpu(s, d, 4, 4, false, 0, -1L, targets, 100_000));
        assertTrue(ScanOps.Hits.same(c, gv));
        assertTrue(c.total() >= 64);
        ScanOps.Hits c8 = ScanOps.pointersCpu(d, 8, 8, false, 0, 1L << 40, null, 100_000);
        ScanOps.Hits g8 = g.run(s -> ScanOps.pointersGpu(s, d, 8, 8, false, 0, 1L << 40, null, 100_000));
        assertTrue(ScanOps.Hits.same(c8, g8));
    }

    @Test
    public void iocScanParity() {
        Gpu g = gpu();
        byte[] d = corpus(1 << 20);
        ScanOps.Hits c = ScanOps.iocCpu(d, 512, 100_000);
        ScanOps.Hits gv = g.run(s -> ScanOps.iocGpu(s, d, 512, 100_000));
        assertTrue(ScanOps.Hits.same(c, gv));
        boolean[] seen = new boolean[5];
        for (int t : c.tag()) seen[t] = true;
        assertTrue("url", seen[1]);
        assertTrue("ipv4", seen[2]);
        assertTrue("email", seen[3]);
        assertTrue("registry", seen[4]);
    }

    @Test
    public void iocCpuRejectsMalformedIpv4() {
        byte[] d = "x 999.1.1.1 y 1.2.3.4.5 z 10.20.30.40 w".getBytes(StandardCharsets.US_ASCII);
        ScanOps.Hits h = ScanOps.iocCpu(d, 512, 100);
        assertEquals(1, h.total());
        assertEquals("10.20.30.40", new String(d, (int) h.pos()[0], h.len()[0], StandardCharsets.US_ASCII));
    }

    @Test
    public void substringParity() {
        Gpu g = gpu();
        List<String> names = new ArrayList<>();
        for (int i = 0; i < 50_000; i++) names.add((i % 7 == 0 ? "Create" : "FUN_") + Integer.toHexString(R.nextInt()) + (i % 11 == 0 ? "FileW" : ""));
        Packed p = Packed.of(names);
        for (boolean ci : new boolean[] {false, true}) {
            byte[] needle = "filew".getBytes();
            assertArrayEquals(ScanOps.substringCpu(p, needle, ci), g.run(s -> ScanOps.substringGpu(s, p, needle, ci)));
        }
    }

    // ------------------------------------------------------------------ hashes

    @Test
    public void hashParityAndReference() throws Exception {
        List<String> segs = new ArrayList<>();
        for (int i = 0; i < 2000; i++) {
            char[] c = new char[R.nextInt(300)];
            for (int k = 0; k < c.length; k++) c[k] = (char) ('a' + R.nextInt(26));
            segs.add(new String(c));
        }
        segs.add("");
        segs.add("x".repeat(55));
        segs.add("x".repeat(56));
        segs.add("x".repeat(64));
        Packed p = Packed.of(segs);
        byte[] cpu = HashOps.sha256Cpu(p);
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        assertArrayEquals(md.digest(segs.get(5).getBytes()), Arrays.copyOfRange(cpu, 32 * 5, 32 * 6));
        long[] fnv = HashOps.fnvCpu(Packed.of(List.of("a")));
        assertEquals(0xaf63dc4c8601ec8cL, fnv[0]); // published FNV-1a-64 test vector for "a"
        Gpu g = gpu();
        assertArrayEquals(cpu, g.run(s -> HashOps.sha256Gpu(s, p)));
        assertArrayEquals(HashOps.fnvCpu(p), g.run(s -> HashOps.fnvGpu(s, p)));
    }

    // ------------------------------------------------------------------ similarity

    @Test
    public void similarityParity() {
        Gpu g = gpu();
        int nA = 37, nB = 411, dim = 48;
        float[] A = new float[nA * dim], B = new float[nB * dim];
        for (int i = 0; i < A.length; i++) A[i] = R.nextInt(5);
        for (int i = 0; i < B.length; i++) B[i] = R.nextInt(5);
        Arrays.fill(B, 0, dim, 0f); // a zero vector scores 0, not NaN
        assertArrayEquals(SimilarityOps.cosineCpu(A, nA, B, nB, dim), g.run(s -> SimilarityOps.cosineGpu(s, A, nA, B, nB, dim)), 0f);

        List<int[]> sets = new ArrayList<>();
        for (int i = 0; i < 500; i++) {
            int[] t = new int[R.nextInt(200)];
            for (int k = 0; k < t.length; k++) t[k] = R.nextInt(64);
            sets.add(t);
        }
        TokenSets ts = TokenSets.of(sets);
        int[] sc = SimilarityOps.minhashCpu(ts, 64, 3);
        assertArrayEquals(sc, g.run(s -> SimilarityOps.minhashGpu(s, ts, 64, 3)));
        assertArrayEquals(SimilarityOps.minhashSimCpu(sc, 100, sc, 500, 64),
            g.run(s -> SimilarityOps.minhashSimGpu(s, sc, 100, sc, 500, 64)), 0f);

        TokenSets a = TokenSets.of(sets.subList(0, 250)), b = TokenSets.of(sets.subList(250, 500));
        assertArrayEquals(SimilarityOps.levenshteinCpu(a, b), g.run(s -> SimilarityOps.levenshteinGpu(s, a, b)));
    }

    @Test
    public void levenshteinCpuReference() {
        TokenSets a = TokenSets.of(List.of("kitten".chars().toArray(), "".chars().toArray(), "flaw".chars().toArray()));
        TokenSets b = TokenSets.of(List.of("sitting".chars().toArray(), "abc".chars().toArray(), "lawn".chars().toArray()));
        assertArrayEquals(new int[] {3, 3, 2}, SimilarityOps.levenshteinCpu(a, b));
    }

    // ------------------------------------------------------------------ graphs

    private static Csr randomGraph(int n, int m) {
        int[] src = new int[m], dst = new int[m];
        for (int e = 0; e < m; e++) { src[e] = R.nextInt(n); dst[e] = R.nextInt(n); }
        return Csr.of(n, src, dst);
    }

    @Test
    public void graphParity() {
        Gpu g = gpu();
        for (int[] nm : new int[][] {{1, 0}, {50, 40}, {5000, 12000}, {20000, 20000}}) {
            Csr c = randomGraph(nm[0], nm[1]);
            int[] src = {0};
            assertArrayEquals(GraphOps.bfsCpu(c, src, Integer.MAX_VALUE), g.run(s -> GraphOps.bfsGpu(s, c, src, Integer.MAX_VALUE)));
            assertArrayEquals(GraphOps.bfsCpu(c, src, 3), g.run(s -> GraphOps.bfsGpu(s, c, src, 3)));
            assertArrayEquals(GraphOps.sccCpu(c), g.run(s -> GraphOps.sccGpu(s, c)));
            assertArrayEquals(GraphOps.pagerankCpu(c, 0.85, 20), g.run(s -> GraphOps.pagerankGpu(s, c, 0.85, 20)), 0.0);
        }
    }

    @Test
    public void sccCpuReference() {
        // 0->1->2->0 is one SCC (label 2), 3->4 are singletons, 5 self-loop singleton
        Csr c = Csr.of(6, new int[] {0, 1, 2, 2, 3, 5}, new int[] {1, 2, 0, 3, 4, 5});
        assertArrayEquals(new int[] {2, 2, 2, 3, 4, 5}, GraphOps.sccCpu(c));
    }

    @Test
    public void dominatorsParityAndReference() {
        // Diamond with a loop: 0->1, 0->2, 1->3, 2->3, 3->1, plus unreachable 4->3
        int[][] g0 = GraphOps.Cfgs.graph(5, new int[] {0, 0, 1, 2, 3, 4}, new int[] {1, 2, 3, 3, 1, 3});
        GraphOps.DomResult ref = GraphOps.dominatorsCpu(GraphOps.Cfgs.of(java.util.Collections.singletonList(g0)));
        assertArrayEquals(new int[] {-1, 0, 0, 0, -2}, ref.idom());

        List<int[][]> graphs = new ArrayList<>();
        graphs.add(g0);
        for (int i = 0; i < 300; i++) {
            int nb = 1 + R.nextInt(i % 10 == 0 ? 300 : 40), m = R.nextInt(nb * 3);
            int[] s = new int[m], d = new int[m];
            for (int e = 0; e < m; e++) { s[e] = R.nextInt(nb); d[e] = R.nextInt(nb); }
            graphs.add(GraphOps.Cfgs.graph(nb, s, d));
        }
        GraphOps.Cfgs cfgs = GraphOps.Cfgs.of(graphs);
        GraphOps.DomResult c = GraphOps.dominatorsCpu(cfgs);
        GraphOps.DomResult gv = gpu().run(s -> GraphOps.dominatorsGpu(s, cfgs));
        assertArrayEquals(c.idom(), gv.idom());
    }

    // ------------------------------------------------------------------ P-code VM

    /** Minimal op builder for hand-written VM programs. */
    static final class Ops {
        final List<int[]> meta = new ArrayList<>();   // opcode, nIn
        final List<int[]> kinds = new ArrayList<>(), sizes = new ArrayList<>();
        final List<long[]> vals = new ArrayList<>();

        int add(int opcode, long[] out, long[]... in) {
            int[] k = new int[4], z = new int[4];
            long[] v = new long[4];
            if (out != null) { k[0] = (int) out[0]; z[0] = (int) out[1]; v[0] = out[2]; }
            for (int i = 0; i < in.length; i++) { k[i + 1] = (int) in[i][0]; z[i + 1] = (int) in[i][1]; v[i + 1] = in[i][2]; }
            meta.add(new int[] {opcode, in.length});
            kinds.add(k);
            sizes.add(z);
            vals.add(v);
            return meta.size() - 1;
        }

        PcodeVm.Program build(int stateSize) {
            int n = meta.size();
            int[] opc = new int[n], instr = new int[n], nIn = new int[n], kind = new int[4 * n], size = new int[4 * n];
            long[] val = new long[4 * n];
            for (int i = 0; i < n; i++) {
                opc[i] = meta.get(i)[0];
                nIn[i] = meta.get(i)[1];
                instr[i] = i;
                System.arraycopy(kinds.get(i), 0, kind, 4 * i, 4);
                System.arraycopy(sizes.get(i), 0, size, 4 * i, 4);
                System.arraycopy(vals.get(i), 0, val, 4 * i, 4);
            }
            int[] first = new int[n];
            long[] addr = new long[n];
            for (int i = 0; i < n; i++) { first[i] = i; addr[i] = 0x401000L + 4L * i; }
            return new PcodeVm.Program(n, opc, instr, nIn, kind, size, val, first, addr, false, stateSize);
        }
    }

    static long[] st(int off, int size) { return new long[] {PcodeVm.KIND_STATE, size, off}; }
    static long[] k(long v, int size) { return new long[] {PcodeVm.KIND_CONST, size, v}; }

    static int ror13(String s) {
        int h = 0;
        for (byte b : s.getBytes(StandardCharsets.US_ASCII)) h = Integer.rotateRight(h, 13) + (b & 0xFF);
        return h;
    }

    @Test
    public void pcodeVmRor13ParityAndReference() {
        long sentinel = 0xDEADBEEFL, scratch = 0x7FFE0000L;
        Ops o = new Ops();
        o.add(2, st(16, 1), k(0, 4), st(0, 8));            // 0 LOAD c = [ptr]
        o.add(11, st(17, 1), st(16, 1), k(0, 1));          // 1 z = c == 0
        o.add(5, null, k(10, 8), st(17, 1));               // 2 CBRANCH 10 if z
        o.add(30, st(20, 4), st(24, 4), k(13, 4));         // 3 a = h >> 13
        o.add(29, st(28, 4), st(24, 4), k(19, 4));         // 4 b = h << 19
        o.add(28, st(24, 4), st(20, 4), st(28, 4));        // 5 h = a | b
        o.add(17, st(32, 4), st(16, 1));                   // 6 w = zext(c)
        o.add(19, st(24, 4), st(24, 4), st(32, 4));        // 7 h += w
        o.add(19, st(0, 8), st(0, 8), k(1, 8));            // 8 ptr++
        o.add(4, null, k(0, 8));                           // 9 BRANCH 0
        o.add(10, null, k(sentinel, 8));                   // 10 RETURN
        PcodeVm.Program p = o.build(40);
        PcodeVm.Memory m = new PcodeVm.Memory(new long[] {scratch}, new int[] {4096}, new byte[0],
            new long[0], new int[0], new long[0]);

        List<String> words = new ArrayList<>();
        for (int i = 0; i < 3000; i++) words.add("Api" + Integer.toHexString(R.nextInt()) + "W");
        int n = words.size();
        byte[] stateInit = new byte[n * p.stateSize];
        java.io.ByteArrayOutputStream priv = new java.io.ByteArrayOutputStream();
        long[] off = new long[n];
        int[] len = new int[n];
        for (int i = 0; i < n; i++) {
            for (int b = 0; b < 8; b++) stateInit[i * p.stateSize + b] = (byte) (scratch >>> (8 * b));
            byte[] rec = PcodeVm.privRecords(new long[] {0}, new byte[][] {(words.get(i) + "\0").getBytes(StandardCharsets.US_ASCII)});
            off[i] = priv.size();
            len[i] = rec.length;
            priv.writeBytes(rec);
        }
        PcodeVm.Lanes lanes = new PcodeVm.Lanes(n, stateInit, priv.toByteArray(), off, len);
        PcodeVm.Result c = PcodeVm.runCpu(p, m, lanes, sentinel, 100_000);
        for (int i = 0; i < n; i++) {
            assertEquals(PcodeVm.Status.RETURNED, c.status(i));
            assertEquals(ror13(words.get(i)), (int) c.readState(i, 24, 4, false));
        }
        PcodeVm.Result gv = gpu().run(s -> PcodeVm.runGpu(s, p, m, lanes, sentinel, 100_000));
        assertTrue(PcodeVm.Result.same(c, gv));
        // Same lanes forced through many small chunks must give the identical answer.
        PcodeVm.setLaneBudgetBytes(64L * (p.stateSize + m.privSize) + 1);
        try {
            PcodeVm.Result chunked = gpu().run(s -> PcodeVm.runGpu(s, p, m, lanes, sentinel, 100_000));
            assertTrue(PcodeVm.Result.same(c, chunked));
        } finally {
            PcodeVm.setLaneBudgetBytes(512L << 20);
        }
    }

    /** Random straight-line integer programs over random lane states: every op, every size. */
    @Test
    public void pcodeVmRandomOpParity() {
        int[] binary = {19, 20, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 11, 12, 13, 14, 15, 16, 21, 22, 23};
        int[] unary = {1, 24, 25, 72, 73};
        int stateSize = 64 * 8;
        Ops o = new Ops();
        for (int i = 0; i < 400; i++) {
            int sz = new int[] {1, 2, 4, 8}[R.nextInt(4)];
            int a = 8 * R.nextInt(64), b = 8 * R.nextInt(64), d = 8 * R.nextInt(64);
            int kind = R.nextInt(10);
            if (kind < 6) {
                int opc = binary[R.nextInt(binary.length)];
                int outSz = (opc >= 11 && opc <= 16) || (opc >= 21 && opc <= 23) ? 1 : sz;
                long[] in1 = R.nextBoolean() ? st(b, sz) : k(R.nextInt(70), sz);
                o.add(opc, st(d, outSz), st(a, sz), in1);
            } else if (kind < 8) {
                o.add(unary[R.nextInt(unary.length)], st(d, sz), st(a, sz));
            } else if (kind == 8) {
                int small = new int[] {1, 2, 4}[R.nextInt(3)];
                o.add(R.nextBoolean() ? 17 : 18, st(d, 8), st(a, small));
            } else {
                int lo = new int[] {1, 2, 4}[R.nextInt(3)];
                if (R.nextBoolean()) o.add(62, st(d, 2 * lo), st(a, lo), st(b, lo));
                else o.add(63, st(d, lo), st(a, 8), k(R.nextInt(8 - lo + 1), 4));
            }
        }
        o.add(10, null, k(0x1234L, 8));
        PcodeVm.Program p = o.build(stateSize);
        PcodeVm.Memory m = new PcodeVm.Memory(new long[0], new int[0], new byte[0], new long[0], new int[0], new long[0]);
        int n = 512;
        byte[] init = new byte[n * stateSize];
        R.nextBytes(init);
        for (int i = 0; i < init.length; i += 8) if (R.nextInt(8) == 0) Arrays.fill(init, i, i + 8, (byte) 0);
        PcodeVm.Lanes lanes = new PcodeVm.Lanes(n, init, new byte[0], new long[n], new int[n]);
        PcodeVm.Result c = PcodeVm.runCpu(p, m, lanes, 0x1234L, 10_000);
        PcodeVm.Result gv = gpu().run(s -> PcodeVm.runGpu(s, p, m, lanes, 0x1234L, 10_000));
        assertArrayEquals(c.status(), gv.status());
        assertArrayEquals(c.steps(), gv.steps());
        assertArrayEquals(c.state(), gv.state());
    }
}
