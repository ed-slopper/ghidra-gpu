package com.xebyte.gpu;

import com.xebyte.gpu.cuda.Gpu;

/**
 * Pairwise similarity: cosine over dense feature vectors, MinHash signatures and their
 * Jaccard estimate, and batched Levenshtein distance over token sequences.
 */
public final class SimilarityOps {

    private SimilarityOps() {}

    /** out[a*nB+b] = cosine(A[a], B[b]); rows of length {@code dim}. */
    public static float[] cosineCpu(float[] A, int nA, float[] B, int nB, int dim) {
        float[] out = new float[nA * nB];
        for (int a = 0; a < nA; a++) {
            for (int b = 0; b < nB; b++) {
                double dot = 0, nx = 0, ny = 0;
                for (int k = 0; k < dim; k++) {
                    double u = A[a * dim + k], v = B[b * dim + k];
                    dot += u * v;
                    nx += u * u;
                    ny += v * v;
                }
                out[a * nB + b] = (nx == 0 || ny == 0) ? 0f : (float) (dot / (Math.sqrt(nx) * Math.sqrt(ny)));
            }
        }
        return out;
    }

    public static float[] cosineGpu(Gpu.Session s, float[] A, int nA, float[] B, int nB, int dim) {
        Gpu.Buf a = s.put(A), b = s.put(B), out = s.alloc(4L * Math.max(1, (long) nA * nB));
        s.launch("cosine_matrix", (long) nA * nB, a, nA, b, nB, dim, out);
        return s.floats(out, nA * nB);
    }

    static int mix32(int h) {
        h ^= h >>> 16;
        h *= 0x85ebca6b;
        h ^= h >>> 13;
        h *= 0xc2b2ae35;
        h ^= h >>> 16;
        return h;
    }

    /** Token sets packed as (tokens, off, len). */
    public record TokenSets(int[] tokens, long[] off, int[] len) {
        public int count() {
            return len.length;
        }

        public static TokenSets of(java.util.List<int[]> sets) {
            int total = 0;
            for (int[] s : sets) total += s.length;
            int[] t = new int[total];
            long[] off = new long[sets.size()];
            int[] len = new int[sets.size()];
            int o = 0;
            for (int i = 0; i < sets.size(); i++) {
                off[i] = o;
                len[i] = sets.get(i).length;
                System.arraycopy(sets.get(i), 0, t, o, len[i]);
                o += len[i];
            }
            return new TokenSets(t, off, len);
        }
    }

    /** sig[s*k+j]: min over shingles of mix32(fnv32(shingle) ^ seed_j). Unsigned min. */
    public static int[] minhashCpu(TokenSets ts, int k, int shingle) {
        int[] sig = new int[ts.count() * k];
        for (int s = 0; s < ts.count(); s++) {
            int base = (int) ts.off()[s], L = ts.len()[s];
            int nSh = L >= shingle ? L - shingle + 1 : (L > 0 ? 1 : 0);
            int w = L >= shingle ? shingle : L;
            int[] sh = new int[nSh];
            for (int i = 0; i < nSh; i++) {
                int h = 0x811c9dc5;
                for (int q = 0; q < w; q++) {
                    h ^= ts.tokens()[base + i + q];
                    h *= 0x01000193;
                }
                sh[i] = h;
            }
            for (int j = 0; j < k; j++) {
                int seed = mix32(0x9e3779b9 * (j + 1));
                int best = -1; // 0xffffffff
                for (int h : sh) {
                    int v = mix32(h ^ seed);
                    if (Integer.compareUnsigned(v, best) < 0) best = v;
                }
                sig[s * k + j] = best;
            }
        }
        return sig;
    }

    public static int[] minhashGpu(Gpu.Session s, TokenSets ts, int k, int shingle) {
        Gpu.Buf t = s.put(ts.tokens()), o = s.put(ts.off()), l = s.put(ts.len());
        Gpu.Buf sig = s.alloc(4L * Math.max(1, (long) ts.count() * k));
        s.launch("minhash", (long) ts.count() * k, t, o, l, ts.count(), k, shingle, sig);
        return s.ints(sig, ts.count() * k);
    }

    public static float[] minhashSimCpu(int[] sigA, int nA, int[] sigB, int nB, int k) {
        float[] out = new float[nA * nB];
        for (int a = 0; a < nA; a++) {
            for (int b = 0; b < nB; b++) {
                int eq = 0;
                for (int j = 0; j < k; j++) if (sigA[a * k + j] == sigB[b * k + j]) eq++;
                out[a * nB + b] = (float) eq / (float) k;
            }
        }
        return out;
    }

    public static float[] minhashSimGpu(Gpu.Session s, int[] sigA, int nA, int[] sigB, int nB, int k) {
        Gpu.Buf a = s.put(sigA), b = s.put(sigB), out = s.alloc(4L * Math.max(1, (long) nA * nB));
        s.launch("minhash_similarity", (long) nA * nB, a, nA, b, nB, k, out);
        return s.floats(out, nA * nB);
    }

    /** Levenshtein distance of pair p = (A[p], B[p]). */
    public static int[] levenshteinCpu(TokenSets A, TokenSets B) {
        int n = A.count();
        int[] out = new int[n];
        for (int p = 0; p < n; p++) {
            int ao = (int) A.off()[p], bo = (int) B.off()[p], la = A.len()[p], lb = B.len()[p];
            int[] prev = new int[lb + 1], cur = new int[lb + 1];
            for (int j = 0; j <= lb; j++) prev[j] = j;
            for (int i = 1; i <= la; i++) {
                cur[0] = i;
                for (int j = 1; j <= lb; j++) {
                    int c = prev[j - 1] + (A.tokens()[ao + i - 1] != B.tokens()[bo + j - 1] ? 1 : 0);
                    cur[j] = Math.min(c, Math.min(prev[j] + 1, cur[j - 1] + 1));
                }
                int[] t = prev; prev = cur; cur = t;
            }
            out[p] = prev[lb];
        }
        return out;
    }

    public static int[] levenshteinGpu(Gpu.Session s, TokenSets A, TokenSets B) {
        int n = A.count();
        int stride = 1;
        for (int l : B.len()) stride = Math.max(stride, l + 1);
        Gpu.Buf ta = s.put(A.tokens()), oa = s.put(A.off()), la = s.put(A.len());
        Gpu.Buf tb = s.put(B.tokens()), ob = s.put(B.off()), lb = s.put(B.len());
        Gpu.Buf scratch = s.alloc(8L * stride * Math.max(1, n));
        Gpu.Buf out = s.alloc(4L * Math.max(1, n));
        s.launch("levenshtein_batch", n, ta, oa, la, tb, ob, lb, n, scratch, stride, out);
        return s.ints(out, n);
    }
}
