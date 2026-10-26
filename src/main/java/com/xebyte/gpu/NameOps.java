package com.xebyte.gpu;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import com.xebyte.gpu.cuda.Gpu;

/**
 * Token-subset name collisions, batched: the check rename_function runs against every
 * function name for one new name, done for many candidate names at once (or every name
 * against every name, as a whole-program naming audit).
 *
 * <p>A name is encoded as its sorted, de-duplicated token hashes plus hashes of its module
 * prefix and of the whole name. Candidate a collides with existing b when prefixes match,
 * the names differ, the token-set sizes differ, and one set contains the other -- the rule
 * of {@code NamingConventions.findTokenSubsetCollision}, whose first-in-order answer is
 * kept by reporting the smallest colliding index.
 */
public final class NameOps {

    private NameOps() {}

    /** Encoded names. tokens are unsigned-sorted per name. */
    public record Encoded(int[] tokens, long[] off, int[] len, int[] prefix, int[] name) {
        public int count() {
            return len.length;
        }
    }

    static int fnv32(String s) {
        int h = 0x811c9dc5;
        for (byte b : s.getBytes(StandardCharsets.UTF_8)) {
            h ^= b & 0xFF;
            h *= 0x01000193;
        }
        return h;
    }

    /**
     * Encodes names with the caller's tokenizer and prefix extractor (the server passes
     * NamingConventions' own, so the rule cannot drift from rename_function's).
     */
    public static Encoded encode(List<String> names, Function<String, List<String>> tokenizer,
                                 Function<String, String> prefixOf) {
        int n = names.size();
        int[][] sets = new int[n][];
        int total = 0;
        int[] prefix = new int[n], name = new int[n];
        for (int i = 0; i < n; i++) {
            String s = names.get(i) == null ? "" : names.get(i);
            List<String> toks = s.isEmpty() ? List.of() : tokenizer.apply(s);
            long[] h = toks.stream().mapToLong(t -> fnv32(t) & 0xFFFFFFFFL).distinct().sorted().toArray();
            sets[i] = new int[h.length];
            for (int k = 0; k < h.length; k++) sets[i][k] = (int) h[k];
            total += h.length;
            String p = s.isEmpty() ? null : prefixOf.apply(s);
            prefix[i] = p == null ? 0 : fnv32(p) | 1;   // 0 is reserved for "no prefix"
            name[i] = fnv32(s);
        }
        int[] tokens = new int[total];
        long[] off = new long[n];
        int[] len = new int[n];
        int o = 0;
        for (int i = 0; i < n; i++) {
            off[i] = o;
            len[i] = sets[i].length;
            System.arraycopy(sets[i], 0, tokens, o, len[i]);
            o += len[i];
        }
        return new Encoded(tokens, off, len, prefix, name);
    }

    static boolean subset(int[] t, int sOff, int ns, int[] u, int bOff, int nb) {
        int i = 0, j = 0;
        while (i < ns && j < nb) {
            int c = Integer.compareUnsigned(t[sOff + i], u[bOff + j]);
            if (c == 0) { i++; j++; }
            else if (c > 0) j++;
            else return false;
        }
        return i == ns;
    }

    /** First colliding index in B for each name in A, or -1. */
    public static int[] collisionsCpu(Encoded a, Encoded b) {
        int[] out = new int[a.count()];
        Arrays.fill(out, -1);
        for (int i = 0; i < a.count(); i++) {
            int la = a.len()[i];
            if (la == 0) continue;
            for (int j = 0; j < b.count(); j++) {
                int lb = b.len()[j];
                if (lb == 0 || la == lb || a.prefix()[i] != b.prefix()[j] || a.name()[i] == b.name()[j]) continue;
                boolean hit = la < lb
                    ? subset(a.tokens(), (int) a.off()[i], la, b.tokens(), (int) b.off()[j], lb)
                    : subset(b.tokens(), (int) b.off()[j], lb, a.tokens(), (int) a.off()[i], la);
                if (hit) { out[i] = j; break; }
            }
        }
        return out;
    }

    public static int[] collisionsGpu(Gpu.Session s, Encoded a, Encoded b) {
        int[] init = new int[Math.max(1, a.count())];
        Arrays.fill(init, Integer.MAX_VALUE);
        Gpu.Buf ta = s.put(a.tokens().length == 0 ? new int[1] : a.tokens()), oa = s.put(a.off()), la = s.put(a.len());
        Gpu.Buf pa = s.put(a.prefix()), na = s.put(a.name());
        Gpu.Buf tb = s.put(b.tokens().length == 0 ? new int[1] : b.tokens()), ob = s.put(b.off()), lb = s.put(b.len());
        Gpu.Buf pb = s.put(b.prefix()), nb = s.put(b.name());
        Gpu.Buf out = s.put(init);
        s.launch("token_subset_pairs", (long) a.count() * b.count(), ta, oa, la, pa, na, a.count(),
            tb, ob, lb, pb, nb, b.count(), out);
        int[] r = s.ints(out, a.count());
        for (int i = 0; i < r.length; i++) if (r[i] == Integer.MAX_VALUE) r[i] = -1;
        return r;
    }
}
