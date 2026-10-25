package com.xebyte.gpu;

import java.util.Arrays;
import java.util.List;

import com.xebyte.gpu.cuda.Gpu;

/**
 * Linear scans over raw bytes: masked patterns, strings, entropy, pointers, IOCs, and
 * substring search over many strings. Each op has a {@code cpu} and a {@code gpu}
 * implementation that return identical results for identical inputs.
 *
 * <p>GPU kernels append hits through an atomic counter, so their raw order is arbitrary.
 * Each GPU op reruns once with an exact-size buffer when the first pass overflowed and
 * sorts, so the GPU returns the same first-{@code max} hits the CPU does and the same
 * {@code total}.
 */
public final class ScanOps {

    private ScanOps() {}

    /** Hard ceiling on hits materialised on the device for a deterministic answer. */
    static final int DEVICE_HIT_CEILING = 16 << 20;

    // ------------------------------------------------------------------ hits

    /** Sorted hits: positions, an int tag per hit (pattern index / length / type), and the untruncated total. */
    public record Hits(long[] pos, int[] tag, int[] len, int total) {
        public static boolean same(Hits a, Hits b) {
            return a.total == b.total && Arrays.equals(a.pos, b.pos) && Arrays.equals(a.tag, b.tag)
                && Arrays.equals(a.len, b.len);
        }

        public int size() {
            return pos.length;
        }
    }

    /** Sort by (pos, tag) and keep the first {@code max}. */
    static Hits finish(long[] pos, int[] tag, int[] len, int n, int total, int max) {
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (x, y) -> pos[x] != pos[y] ? Long.compare(pos[x], pos[y]) : Integer.compare(tag[x], tag[y]));
        int k = Math.min(n, max);
        long[] p = new long[k];
        int[] t = new int[k];
        int[] l = len == null ? null : new int[k];
        for (int i = 0; i < k; i++) {
            p[i] = pos[idx[i]];
            t[i] = tag[idx[i]];
            if (l != null) l[i] = len[idx[i]];
        }
        return new Hits(p, t, l, total);
    }

    // ------------------------------------------------------------------ pattern scan

    /**
     * Every offset where a masked pattern matches. tag = pattern index. {@code alignment}
     * > 1 restricts starts to multiples of it.
     */
    public static Hits patternCpu(byte[] data, List<byte[]> pats, List<byte[]> masks, int alignment, int max) {
        long[] pos = new long[Math.min(max, 1 << 16)];
        int[] tag = new int[pos.length];
        int n = 0, total = 0;
        int step = Math.max(1, alignment);
        for (int i = 0; i < data.length; i += step) {
            for (int p = 0; p < pats.size(); p++) {
                byte[] pp = pats.get(p), mm = masks.get(p);
                if (i + pp.length > data.length) continue;
                boolean ok = true;
                for (int k = 0; k < pp.length; k++) {
                    if ((data[i + k] & mm[k]) != (pp[k] & mm[k])) { ok = false; break; }
                }
                if (!ok) continue;
                total++;
                if (n < max) {
                    if (n == pos.length) { pos = Arrays.copyOf(pos, Math.min(max, n * 2)); tag = Arrays.copyOf(tag, pos.length); }
                    pos[n] = i;
                    tag[n] = p;
                    n++;
                }
            }
        }
        // already in (pos, pattern) order
        return new Hits(Arrays.copyOf(pos, n), Arrays.copyOf(tag, n), null, total);
    }

    public static Hits patternGpu(Gpu.Session s, byte[] data, List<byte[]> pats, List<byte[]> masks, int alignment, int max) {
        int total = 0;
        for (byte[] p : pats) total += p.length;
        byte[] flatP = new byte[total], flatM = new byte[total];
        int[] off = new int[pats.size()], len = new int[pats.size()];
        int o = 0;
        for (int i = 0; i < pats.size(); i++) {
            off[i] = o;
            len[i] = pats.get(i).length;
            System.arraycopy(pats.get(i), 0, flatP, o, len[i]);
            System.arraycopy(masks.get(i), 0, flatM, o, len[i]);
            o += len[i];
        }
        Gpu.Buf d = s.put(data), bp = s.put(flatP), bm = s.put(flatM), bo = s.put(off), bl = s.put(len);
        int cap = Math.max(1024, Math.min(max, DEVICE_HIT_CEILING));
        for (int attempt = 0; ; attempt++) {
            Gpu.Buf hp = s.alloc(8L * cap), ht = s.alloc(4L * cap), cnt = s.zeros(4);
            s.launch("pattern_scan", data.length, d, (long) data.length, bp, bm, bo, bl, pats.size(),
                Math.max(1, alignment), hp, ht, cnt, cap);
            int c = s.readInt(cnt);
            if (c <= cap || attempt > 0 || c > DEVICE_HIT_CEILING) {
                int got = Math.min(c, cap);
                return finish(s.longs(hp, got), s.ints(ht, got), null, got, c, max);
            }
            cap = c;
        }
    }

    // ------------------------------------------------------------------ strings

    static boolean isPrint(int c) {
        return (c >= 0x20 && c <= 0x7E) || c == 0x09;
    }

    /** Printable runs >= minLen; tag = 0 (ascii) or 1 (utf16le), len = characters. */
    public static Hits stringsCpu(byte[] d, int minLen, boolean wide, int maxStrLen, int max) {
        long[] pos = new long[1024];
        int[] len = new int[1024];
        int n = 0, total = 0;
        int N = d.length;
        for (int i = 0; i < N; i++) {
            int L = 0;
            if (!wide) {
                if (!isPrint(d[i] & 0xFF)) continue;
                if (i > 0 && isPrint(d[i - 1] & 0xFF)) continue;
                while (i + L < N && isPrint(d[i + L] & 0xFF) && L < maxStrLen) L++;
            } else {
                if (i + 1 >= N || !isPrint(d[i] & 0xFF) || d[i + 1] != 0) continue;
                if (i >= 2 && isPrint(d[i - 2] & 0xFF) && d[i - 1] == 0) continue;
                while (i + 2L * L + 1 < N && isPrint(d[i + 2 * L] & 0xFF) && d[i + 2 * L + 1] == 0 && L < maxStrLen) L++;
            }
            if (L < minLen) continue;
            total++;
            if (n < max) {
                if (n == pos.length) { pos = Arrays.copyOf(pos, n * 2); len = Arrays.copyOf(len, n * 2); }
                pos[n] = i;
                len[n] = L;
                n++;
            }
        }
        int[] tag = new int[n];
        Arrays.fill(tag, wide ? 1 : 0);
        return new Hits(Arrays.copyOf(pos, n), tag, Arrays.copyOf(len, n), total);
    }

    public static Hits stringsGpu(Gpu.Session s, byte[] d, int minLen, boolean wide, int maxStrLen, int max) {
        Gpu.Buf data = s.put(d);
        int cap = Math.max(1024, Math.min(max, DEVICE_HIT_CEILING));
        for (int attempt = 0; ; attempt++) {
            Gpu.Buf p = s.alloc(8L * cap), l = s.alloc(4L * cap), cnt = s.zeros(4);
            s.launch("string_scan", d.length, data, (long) d.length, minLen, wide ? 1 : 0, p, l, cnt, cap, maxStrLen);
            int c = s.readInt(cnt);
            if (c <= cap || attempt > 0 || c > DEVICE_HIT_CEILING) {
                int got = Math.min(c, cap);
                int[] tag = new int[got];
                Arrays.fill(tag, wide ? 1 : 0);
                return finish(s.longs(p, got), tag, s.ints(l, got), got, c, max);
            }
            cap = c;
        }
    }

    // ------------------------------------------------------------------ entropy / histogram

    public static long[] histogramCpu(byte[] d) {
        long[] h = new long[256];
        for (byte b : d) h[b & 0xFF]++;
        return h;
    }

    public static long[] histogramGpu(Gpu.Session s, byte[] d) {
        Gpu.Buf data = s.put(d), h = s.zeros(256 * 8);
        s.launch("byte_histogram", d.length, data, (long) d.length, h);
        return s.longs(h, 256);
    }

    /** Shannon entropy (bits/byte) of each {@code window}-byte window; the last may be short. */
    public static float[] entropyCpu(byte[] d, int window) {
        int nw = (int) ((d.length + (long) window - 1) / window);
        float[] out = new float[nw];
        int[] h = new int[256];
        double ln2 = Math.log(2.0);
        for (int w = 0; w < nw; w++) {
            Arrays.fill(h, 0);
            int start = w * window, end = (int) Math.min(d.length, (long) start + window);
            for (int i = start; i < end; i++) h[d[i] & 0xFF]++;
            double len = end - start, e = 0;
            for (int k = 0; k < 256; k++) {
                if (h[k] != 0) {
                    double p = h[k] / len;
                    e -= p * (Math.log(p) / ln2);
                }
            }
            out[w] = (float) e;
        }
        return out;
    }

    public static float[] entropyGpu(Gpu.Session s, byte[] d, int window) {
        int nw = (int) ((d.length + (long) window - 1) / window);
        Gpu.Buf data = s.put(d), out = s.alloc(4L * Math.max(1, nw));
        int grid = Math.max(1, Math.min(nw, 65535));
        s.launchGrid("window_entropy", grid, Gpu.BLOCK, 0, data, (long) d.length, window, (long) nw, out);
        return s.floats(out, nw);
    }

    public static boolean sameFloats(float[] a, float[] b, float tol) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > tol) return false;
        return true;
    }

    // ------------------------------------------------------------------ pointer scan

    /** Values at aligned offsets inside [lo, hi), optionally only those in {@code targets} (any order). */
    public static Hits pointersCpu(byte[] d, int ptrSize, int alignment, boolean bigEndian, long lo, long hi,
                                   long[] targets, int max) {
        long[] t = targets == null ? new long[0] : sortedUnsigned(targets);
        long[] pos = new long[1024];
        long[] val = new long[1024];
        int n = 0, total = 0;
        long slots = d.length / alignment;
        for (long sl = 0; sl < slots; sl++) {
            int i = (int) (sl * alignment);
            if (i + ptrSize > d.length) continue;
            long v = read(d, i, ptrSize, bigEndian);
            if (Long.compareUnsigned(v, lo) < 0 || Long.compareUnsigned(v, hi) >= 0) continue;
            if (t.length > 0 && binarySearchUnsigned(t, v) < 0) continue;
            total++;
            if (n < max) {
                if (n == pos.length) { pos = Arrays.copyOf(pos, n * 2); val = Arrays.copyOf(val, n * 2); }
                pos[n] = i;
                val[n] = v;
                n++;
            }
        }
        return pointerHits(Arrays.copyOf(pos, n), Arrays.copyOf(val, n), total);
    }

    public static Hits pointersGpu(Gpu.Session s, byte[] d, int ptrSize, int alignment, boolean bigEndian,
                                   long lo, long hi, long[] targets, int max) {
        long[] t = targets == null ? new long[0] : sortedUnsigned(targets);
        Gpu.Buf data = s.put(d), tb = s.put(t);
        int cap = Math.max(1024, Math.min(max, DEVICE_HIT_CEILING));
        for (int attempt = 0; ; attempt++) {
            Gpu.Buf p = s.alloc(8L * cap), v = s.alloc(8L * cap), cnt = s.zeros(4);
            s.launch("pointer_scan", d.length / Math.max(1, alignment), data, (long) d.length, ptrSize, alignment,
                bigEndian ? 1 : 0, lo, hi, tb, t.length, p, v, cnt, cap);
            int c = s.readInt(cnt);
            if (c <= cap || attempt > 0 || c > DEVICE_HIT_CEILING) {
                int got = Math.min(c, cap);
                long[] pos = s.longs(p, got), val = s.longs(v, got);
                Integer[] idx = new Integer[got];
                for (int i = 0; i < got; i++) idx[i] = i;
                Arrays.sort(idx, (x, y) -> Long.compare(pos[x], pos[y]));
                int k = Math.min(got, max);
                long[] sp = new long[k], sv = new long[k];
                for (int i = 0; i < k; i++) { sp[i] = pos[idx[i]]; sv[i] = val[idx[i]]; }
                return pointerHits(sp, sv, c);
            }
            cap = c;
        }
    }

    /** Pointer hits carry the value split into tag (low 32) and len (high 32). */
    static Hits pointerHits(long[] pos, long[] val, int total) {
        int[] lo = new int[val.length], hi = new int[val.length];
        for (int i = 0; i < val.length; i++) { lo[i] = (int) val[i]; hi[i] = (int) (val[i] >>> 32); }
        return new Hits(pos, lo, hi, total);
    }

    public static long pointerValue(Hits h, int i) {
        return (h.tag()[i] & 0xFFFFFFFFL) | ((long) h.len()[i] << 32);
    }

    static long read(byte[] d, int i, int size, boolean be) {
        long v = 0;
        if (be) for (int k = 0; k < size; k++) v = (v << 8) | (d[i + k] & 0xFF);
        else for (int k = size - 1; k >= 0; k--) v = (v << 8) | (d[i + k] & 0xFF);
        return v;
    }

    static long[] sortedUnsigned(long[] a) {
        long[] t = a.clone();
        for (int i = 0; i < t.length; i++) t[i] ^= Long.MIN_VALUE;
        Arrays.sort(t);
        for (int i = 0; i < t.length; i++) t[i] ^= Long.MIN_VALUE;
        return t;
    }

    static int binarySearchUnsigned(long[] t, long v) {
        int a = 0, b = t.length - 1;
        while (a <= b) {
            int m = (a + b) >>> 1;
            int c = Long.compareUnsigned(t[m], v);
            if (c == 0) return m;
            if (c < 0) a = m + 1; else b = m - 1;
        }
        return -1;
    }

    // ------------------------------------------------------------------ substring search

    /** Packed strings: UTF-8 bytes, an offset and a length per string. */
    public record Packed(byte[] bytes, long[] off, int[] len) {
        public static Packed of(List<String> strings) {
            int total = 0;
            byte[][] enc = new byte[strings.size()][];
            for (int i = 0; i < enc.length; i++) {
                enc[i] = strings.get(i) == null ? new byte[0] : strings.get(i).getBytes(java.nio.charset.StandardCharsets.UTF_8);
                total += enc[i].length;
            }
            byte[] b = new byte[total];
            long[] off = new long[enc.length];
            int[] len = new int[enc.length];
            int o = 0;
            for (int i = 0; i < enc.length; i++) {
                off[i] = o;
                len[i] = enc[i].length;
                System.arraycopy(enc[i], 0, b, o, len[i]);
                o += len[i];
            }
            return new Packed(b, off, len);
        }

        public int count() {
            return len.length;
        }
    }

    static int lower(int c) {
        return c >= 'A' && c <= 'Z' ? c + 32 : c;
    }

    /** First byte index of {@code needle} in each string, or -1. */
    public static int[] substringCpu(Packed hay, byte[] needle, boolean ignoreCase) {
        int[] out = new int[hay.count()];
        for (int s = 0; s < out.length; s++) {
            int base = (int) hay.off()[s], L = hay.len()[s], found = -1;
            for (int i = 0; i + needle.length <= L && found < 0; i++) {
                int k = 0;
                for (; k < needle.length; k++) {
                    int a = hay.bytes()[base + i + k] & 0xFF, b = needle[k] & 0xFF;
                    if (ignoreCase) { a = lower(a); b = lower(b); }
                    if (a != b) break;
                }
                if (k == needle.length) found = i;
            }
            out[s] = found;
        }
        return out;
    }

    public static int[] substringGpu(Gpu.Session s, Packed hay, byte[] needle, boolean ignoreCase) {
        Gpu.Buf h = s.put(hay.bytes()), o = s.put(hay.off()), l = s.put(hay.len()), n = s.put(needle);
        Gpu.Buf out = s.alloc(4L * Math.max(1, hay.count()));
        s.launch("substring_search", hay.count(), h, o, l, hay.count(), n, needle.length, ignoreCase ? 1 : 0, out);
        return s.ints(out, hay.count());
    }

    // ------------------------------------------------------------------ IOC scan

    public static final String[] IOC_TYPES = {"", "url", "ipv4", "email", "registry"};

    static boolean isAlnum(int c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }

    static boolean isDigit(int c) {
        return c >= '0' && c <= '9';
    }

    static boolean isUrlChar(int c) {
        return isAlnum(c) || "-._~:/?#[]@!$&'()*+,;=%".indexOf(c) >= 0;
    }

    static boolean isLocalChar(int c) {
        return isAlnum(c) || c == '.' || c == '_' || c == '-' || c == '+';
    }

    static boolean isDomainChar(int c) {
        return isAlnum(c) || c == '.' || c == '-';
    }

    static int at(byte[] d, long i) {
        return d[(int) i] & 0xFF;
    }

    static boolean prefixAt(byte[] d, int i, String lit, boolean ci) {
        for (int k = 0; k < lit.length(); k++) {
            if (i + k >= d.length) return false;
            int a = d[i + k] & 0xFF, b = lit.charAt(k);
            if (ci) { a = lower(a); b = lower(b); }
            if (a != b) return false;
        }
        return true;
    }

    static int matchIpv4(byte[] d, int i) {
        int p = i, n = d.length;
        for (int part = 0; part < 4; part++) {
            int digits = 0, val = 0;
            while (p < n && isDigit(d[p]) && digits < 4) { val = val * 10 + (d[p] - '0'); p++; digits++; }
            if (digits == 0 || digits > 3 || val > 255) return 0;
            if (part < 3) { if (p >= n || d[p] != '.') return 0; p++; }
        }
        if (p < n && (isDigit(d[p]) || (d[p] == '.' && p + 1 < n && isDigit(d[p + 1])))) return 0;
        return p - i;
    }

    /** URL / IPv4 / e-mail / registry candidates; tag = type (see {@link #IOC_TYPES}). */
    public static Hits iocCpu(byte[] d, int maxLen, int max) {
        long[] pos = new long[256];
        int[] tag = new int[256], len = new int[256];
        int n = 0, total = 0, N = d.length;
        for (int i = 0; i < N; i++) {
            int type = 0, L = 0;
            long start = i;
            int prev = i > 0 ? d[i - 1] & 0xFF : 0;
            int cur = d[i] & 0xFF;
            int scheme = 0;
            if (!isAlnum(prev)) {
                if (prefixAt(d, i, "https://", true)) scheme = 8;
                else if (prefixAt(d, i, "http://", true)) scheme = 7;
                else if (prefixAt(d, i, "ftp://", true)) scheme = 6;
            }
            if (scheme != 0) {
                int q = 0;
                while (i + q < N && isUrlChar(d[i + q] & 0xFF) && q < maxLen) q++;
                if (q > scheme + 3) { type = 1; L = q; }
            } else if (isDigit(cur) && !isDigit(prev) && prev != '.') {
                int q = matchIpv4(d, i);
                if (q != 0) { type = 2; L = q; }
            } else if (cur == '@' && i > 0 && isLocalChar(prev)) {
                int s = i;
                while (s > 0 && isLocalChar(d[s - 1] & 0xFF) && i - s < 64) s--;
                int dl = 0;
                boolean dot = false;
                while (i + 1 + dl < N && isDomainChar(d[i + 1 + dl] & 0xFF) && dl < 253) {
                    if (d[i + 1 + dl] == '.') dot = true;
                    dl++;
                }
                while (dl > 0 && (d[i + dl] == '.' || d[i + dl] == '-')) dl--;
                if (dot && dl >= 3 && isAlnum(d[s] & 0xFF) && isAlnum(d[i + dl] & 0xFF)) {
                    boolean tailDot = false;
                    for (int q = 1; q <= dl; q++) if (d[i + q] == '.') tailDot = true;
                    if (tailDot) { type = 3; start = s; L = (i - s) + 1 + dl; }
                }
            } else if (!isAlnum(prev) && (prefixAt(d, i, "HKEY_", false) || prefixAt(d, i, "HKLM\\", false)
                || prefixAt(d, i, "HKCU\\", false) || prefixAt(d, i, "SOFTWARE\\", true)
                || prefixAt(d, i, "SYSTEM\\CurrentControlSet", true))) {
                int q = 0;
                while (i + q < N && isPrint(d[i + q] & 0xFF) && d[i + q] != '"' && q < maxLen) q++;
                if (q >= 8) { type = 4; L = q; }
            }
            if (type == 0) continue;
            total++;
            if (n == pos.length) { pos = Arrays.copyOf(pos, n * 2); tag = Arrays.copyOf(tag, n * 2); len = Arrays.copyOf(len, n * 2); }
            pos[n] = start;
            tag[n] = type;
            len[n] = L;
            n++;
        }
        return finish(pos, tag, len, n, total, max);
    }

    public static Hits iocGpu(Gpu.Session s, byte[] d, int maxLen, int max) {
        Gpu.Buf data = s.put(d);
        int cap = Math.max(1024, Math.min(Math.max(max, 1 << 16), DEVICE_HIT_CEILING));
        for (int attempt = 0; ; attempt++) {
            Gpu.Buf p = s.alloc(8L * cap), l = s.alloc(4L * cap), t = s.alloc(4L * cap), cnt = s.zeros(4);
            s.launch("ioc_scan", d.length, data, (long) d.length, p, l, t, cnt, cap, maxLen);
            int c = s.readInt(cnt);
            if (c <= cap || attempt > 0 || c > DEVICE_HIT_CEILING) {
                int got = Math.min(c, cap);
                return finish(s.longs(p, got), s.ints(t, got), s.ints(l, got), got, c, max);
            }
            cap = c;
        }
    }
}
