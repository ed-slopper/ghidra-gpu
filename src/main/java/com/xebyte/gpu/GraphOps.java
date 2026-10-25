package com.xebyte.gpu;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.List;

import com.xebyte.gpu.cuda.Gpu;

/**
 * Graph algorithms over CSR graphs: BFS depths, strongly connected components, PageRank,
 * and batched dominator trees for many CFGs at once.
 *
 * <p>Canonical outputs keep CPU and GPU comparable: an SCC is labelled by the largest node
 * id it contains (what the GPU colouring algorithm yields naturally), BFS reports depths
 * (not a parent tree, which depends on visit order), and dominators report the immediate
 * dominator, which is unique.
 */
public final class GraphOps {

    private GraphOps() {}

    /** Directed graph in compressed-sparse-row form. */
    public record Csr(int n, int[] rowPtr, int[] col) {

        public static Csr of(int n, int[] src, int[] dst) {
            int[] rowPtr = new int[n + 1];
            for (int s : src) rowPtr[s + 1]++;
            for (int i = 0; i < n; i++) rowPtr[i + 1] += rowPtr[i];
            int[] col = new int[src.length];
            int[] fill = Arrays.copyOf(rowPtr, n);
            for (int e = 0; e < src.length; e++) col[fill[src[e]]++] = dst[e];
            // stable successor order makes PageRank summation order identical on both sides
            for (int u = 0; u < n; u++) Arrays.sort(col, rowPtr[u], rowPtr[u + 1]);
            return new Csr(n, rowPtr, col);
        }

        public Csr reverse() {
            int[] src = new int[col.length], dst = new int[col.length];
            int e = 0;
            for (int u = 0; u < n; u++) {
                for (int k = rowPtr[u]; k < rowPtr[u + 1]; k++) { src[e] = col[k]; dst[e] = u; e++; }
            }
            return of(n, src, dst);
        }

        public int edges() {
            return col.length;
        }

        public int outDegree(int u) {
            return rowPtr[u + 1] - rowPtr[u];
        }
    }

    // ------------------------------------------------------------------ BFS

    /** Depth of each node from the nearest source (-1: unreachable within maxDepth). */
    public static int[] bfsCpu(Csr g, int[] sources, int maxDepth) {
        int[] depth = new int[g.n()];
        Arrays.fill(depth, -1);
        ArrayDeque<Integer> q = new ArrayDeque<>();
        for (int s : sources) if (depth[s] < 0) { depth[s] = 0; q.add(s); }
        while (!q.isEmpty()) {
            int u = q.poll();
            if (depth[u] >= maxDepth) continue;
            for (int k = g.rowPtr()[u]; k < g.rowPtr()[u + 1]; k++) {
                int v = g.col()[k];
                if (depth[v] < 0) { depth[v] = depth[u] + 1; q.add(v); }
            }
        }
        return depth;
    }

    public static int[] bfsGpu(Gpu.Session s, Csr g, int[] sources, int maxDepth) {
        int[] init = new int[g.n()];
        Arrays.fill(init, -1);
        for (int src : sources) init[src] = 0;
        Gpu.Buf rp = s.put(g.rowPtr()), col = s.put(g.col()), depth = s.put(init), changed = s.alloc(4);
        for (int level = 0; level < maxDepth; level++) {
            s.write(changed, new int[] {0});
            s.launch("bfs_step", g.n(), rp, col, g.n(), depth, level, maxDepth, changed);
            if (s.readInt(changed) == 0) break;
        }
        return s.ints(depth, g.n());
    }

    // ------------------------------------------------------------------ SCC

    /** Component label per node: the largest node id in its SCC. Iterative Tarjan. */
    public static int[] sccCpu(Csr g) {
        int n = g.n();
        int[] index = new int[n], low = new int[n], comp = new int[n];
        Arrays.fill(index, -1);
        boolean[] onStack = new boolean[n];
        int[] stack = new int[n];
        int sp = 0, counter = 0;
        int[] callNode = new int[n], callEdge = new int[n];
        for (int root = 0; root < n; root++) {
            if (index[root] >= 0) continue;
            int depth = 0;
            callNode[0] = root;
            callEdge[0] = g.rowPtr()[root];
            index[root] = low[root] = counter++;
            stack[sp++] = root;
            onStack[root] = true;
            while (depth >= 0) {
                int u = callNode[depth];
                if (callEdge[depth] < g.rowPtr()[u + 1]) {
                    int v = g.col()[callEdge[depth]++];
                    if (index[v] < 0) {
                        index[v] = low[v] = counter++;
                        stack[sp++] = v;
                        onStack[v] = true;
                        depth++;
                        callNode[depth] = v;
                        callEdge[depth] = g.rowPtr()[v];
                    } else if (onStack[v]) {
                        low[u] = Math.min(low[u], index[v]);
                    }
                } else {
                    if (low[u] == index[u]) {
                        int top = sp, max = -1;
                        int w;
                        do { w = stack[--top]; max = Math.max(max, w); } while (w != u);
                        do { w = stack[--sp]; onStack[w] = false; comp[w] = max; } while (w != u);
                    }
                    depth--;
                    if (depth >= 0) {
                        int p = callNode[depth];
                        low[p] = Math.min(low[p], low[u]);
                    }
                }
            }
        }
        return comp;
    }

    public static int[] sccGpu(Gpu.Session s, Csr g) {
        Csr r = g.reverse();
        int n = g.n();
        int[] minus = new int[n];
        Arrays.fill(minus, -1);
        Gpu.Buf rp = s.put(g.rowPtr()), col = s.put(g.col()), rrp = s.put(r.rowPtr()), rcol = s.put(r.col());
        Gpu.Buf scc = s.put(minus), color = s.alloc(4L * Math.max(1, n)), changed = s.alloc(4);
        while (true) {
            do {
                s.write(changed, new int[] {0});
                s.launch("scc_trim", n, rp, col, rrp, rcol, n, scc, changed);
            } while (s.readInt(changed) != 0);
            int[] cur = s.ints(scc, n);
            boolean left = false;
            for (int c : cur) if (c < 0) { left = true; break; }
            if (!left) return cur;
            s.launch("scc_reset_colors", n, n, scc, color);
            do {
                s.write(changed, new int[] {0});
                s.launch("scc_color_propagate", n, rp, col, n, scc, color, changed);
            } while (s.readInt(changed) != 0);
            s.launch("scc_seed_roots", n, n, scc, color);
            do {
                s.write(changed, new int[] {0});
                s.launch("scc_backward", n, rrp, rcol, n, scc, color, changed);
            } while (s.readInt(changed) != 0);
        }
    }

    // ------------------------------------------------------------------ PageRank

    public static double[] pagerankCpu(Csr g, double damping, int iterations) {
        Csr r = g.reverse();
        int n = g.n();
        double[] rank = new double[n], next = new double[n];
        Arrays.fill(rank, 1.0 / n);
        for (int it = 0; it < iterations; it++) {
            double share = danglingShare(g, rank);
            for (int v = 0; v < n; v++) {
                double sum = 0;
                for (int k = r.rowPtr()[v]; k < r.rowPtr()[v + 1]; k++) {
                    int u = r.col()[k];
                    sum += rank[u] / g.outDegree(u);
                }
                next[v] = (1.0 - damping) / n + damping * (sum + share);
            }
            double[] t = rank; rank = next; next = t;
        }
        return rank;
    }

    public static double[] pagerankGpu(Gpu.Session s, Csr g, double damping, int iterations) {
        Csr r = g.reverse();
        int n = g.n();
        int[] outDeg = new int[n];
        for (int u = 0; u < n; u++) outDeg[u] = g.outDegree(u);
        double[] rank = new double[n];
        Arrays.fill(rank, 1.0 / n);
        Gpu.Buf rrp = s.put(r.rowPtr()), rcol = s.put(r.col()), deg = s.put(outDeg);
        Gpu.Buf a = s.put(rank), b = s.alloc(8L * Math.max(1, n));
        for (int it = 0; it < iterations; it++) {
            double share = danglingShare(g, rank);
            s.launch("pagerank_step", n, rrp, rcol, deg, n, a, b, damping, share);
            Gpu.Buf t = a; a = b; b = t;
            rank = s.doubles(a, n);
        }
        return rank;
    }

    static double danglingShare(Csr g, double[] rank) {
        double d = 0;
        for (int u = 0; u < g.n(); u++) if (g.outDegree(u) == 0) d += rank[u];
        return d / g.n();
    }

    // ------------------------------------------------------------------ dominators

    /**
     * Many CFGs, concatenated. Graph {@code i} owns global nodes
     * {@code [nodeOff[i], nodeOff[i+1])}; its local block 0 is the entry. Predecessor lists
     * are in global CSR {@code predRowPtr/predCol} with LOCAL indices.
     */
    public record Cfgs(int[] nodeOff, int[] predRowPtr, int[] predCol) {
        public int graphs() {
            return nodeOff.length - 1;
        }

        public int nodes() {
            return nodeOff[nodeOff.length - 1];
        }

        /** Build from per-graph local successor edge lists. */
        public static Cfgs of(List<int[][]> graphs) {
            int[] nodeOff = new int[graphs.size() + 1];
            int edges = 0;
            for (int g = 0; g < graphs.size(); g++) {
                nodeOff[g + 1] = nodeOff[g] + graphs.get(g)[0][0];
                edges += graphs.get(g)[1].length;
            }
            int total = nodeOff[graphs.size()];
            int[] cnt = new int[total + 1];
            for (int g = 0; g < graphs.size(); g++) {
                int[] dst = graphs.get(g)[2];
                for (int d : dst) cnt[nodeOff[g] + d + 1]++;
            }
            for (int i = 0; i < total; i++) cnt[i + 1] += cnt[i];
            int[] predCol = new int[edges];
            int[] fill = Arrays.copyOf(cnt, total);
            for (int g = 0; g < graphs.size(); g++) {
                int[] src = graphs.get(g)[1], dst = graphs.get(g)[2];
                for (int e = 0; e < src.length; e++) predCol[fill[nodeOff[g] + dst[e]]++] = src[e];
            }
            return new Cfgs(nodeOff, cnt, predCol);
        }

        /** Encodes one CFG for {@link #of}: {{nodes}, src[], dst[]}. */
        public static int[][] graph(int nodes, int[] src, int[] dst) {
            return new int[][] {{nodes}, src, dst};
        }
    }

    /** Immediate dominators (local index; -1 entry, -2 unreachable) per global node. */
    public record DomResult(int[] idom) {
        public static boolean same(DomResult a, DomResult b) {
            return Arrays.equals(a.idom, b.idom);
        }
    }

    public static DomResult dominatorsCpu(Cfgs c) {
        int[] idom = new int[c.nodes()];
        for (int g = 0; g < c.graphs(); g++) {
            int base = c.nodeOff()[g], nb = c.nodeOff()[g + 1] - base;
            if (nb == 0) continue;
            boolean[] reach = new boolean[nb];
            reach[0] = true;
            boolean changed = true;
            while (changed) {
                changed = false;
                for (int v = 0; v < nb; v++) {
                    if (reach[v]) continue;
                    for (int e = c.predRowPtr()[base + v]; e < c.predRowPtr()[base + v + 1]; e++) {
                        if (reach[c.predCol()[e]]) { reach[v] = true; changed = true; break; }
                    }
                }
            }
            int words = (nb + 31) / 32;
            int[][] dom = new int[nb][words];
            for (int v = 1; v < nb; v++) {
                Arrays.fill(dom[v], -1);
                if ((nb & 31) != 0) dom[v][words - 1] = (1 << (nb & 31)) - 1;
            }
            dom[0][0] = 1;
            changed = true;
            while (changed) {
                changed = false;
                for (int v = 1; v < nb; v++) {
                    if (!reach[v]) continue;
                    for (int w = 0; w < words; w++) {
                        int acc = -1;
                        boolean any = false;
                        for (int e = c.predRowPtr()[base + v]; e < c.predRowPtr()[base + v + 1]; e++) {
                            int p = c.predCol()[e];
                            if (!reach[p]) continue;
                            acc &= dom[p][w];
                            any = true;
                        }
                        if (!any) acc = 0;
                        if ((v >> 5) == w) acc |= 1 << (v & 31);
                        if (acc != dom[v][w]) { dom[v][w] = acc; changed = true; }
                    }
                }
            }
            for (int v = 0; v < nb; v++) {
                if (!reach[v]) { idom[base + v] = -2; continue; }
                if (v == 0) { idom[base] = -1; continue; }
                int best = -1, bestCount = -1;
                for (int d = 0; d < nb; d++) {
                    if (d == v || (dom[v][d >> 5] & (1 << (d & 31))) == 0) continue;
                    int cnt = 0;
                    for (int w = 0; w < words; w++) cnt += Integer.bitCount(dom[d][w]);
                    if (cnt > bestCount) { bestCount = cnt; best = d; }
                }
                idom[base + v] = best;
            }
        }
        return new DomResult(idom);
    }

    public static DomResult dominatorsGpu(Gpu.Session s, Cfgs c) {
        int ng = c.graphs();
        long[] domOff = new long[Math.max(1, ng)];
        long total = 0;
        for (int g = 0; g < ng; g++) {
            int nb = c.nodeOff()[g + 1] - c.nodeOff()[g];
            domOff[g] = total;
            total += (long) nb * ((nb + 31) / 32);
        }
        int[] empty = new int[Math.max(1, c.nodes())];
        Gpu.Buf no = s.put(c.nodeOff()), dummy = s.put(new int[1]);
        Gpu.Buf prp = s.put(c.predRowPtr()), pc = s.put(c.predCol().length == 0 ? new int[1] : c.predCol());
        Gpu.Buf bits = s.alloc(4L * Math.max(1, total)), off = s.put(domOff);
        Gpu.Buf idom = s.put(empty), reach = s.put(empty);
        if (ng > 0) {
            s.launchGrid("dominators_batch", ng, Gpu.BLOCK, 0, no, dummy, prp, pc, ng, bits, off, idom, reach);
        }
        return new DomResult(s.ints(idom, c.nodes()));
    }
}
