// ghidra-gpu CUDA kernels.
//
// Compiled to PTX by tools/gpu/build_ptx.py (NVRTC, no host compiler needed) and
// loaded at runtime through the CUDA driver API (com.xebyte.gpu.cuda.CudaDriver),
// so a user needs only the NVIDIA driver, not the CUDA toolkit.
//
// Every kernel here has a CPU twin in com.xebyte.gpu.ops that produces the same
// answer from the same inputs; GpuKernelParityTest holds the two together. Keep
// that invariant when editing: output order the CPU cannot reproduce (atomic
// append order) is sorted on the host before it reaches a caller.
//
// Conventions: all kernels are extern "C" (stable names for cuModuleGetFunction),
// sizes are long long where a buffer may exceed 2 GiB, and appends go through an
// atomic counter that is allowed to run past the cap -- the host reports the
// overflow as "truncated" instead of guessing.

typedef unsigned char      u8;
typedef unsigned int       u32;
typedef unsigned long long u64;
typedef long long          i64;

#define GRID_STRIDE(i, n) for (i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (i64)blockDim.x * gridDim.x)

// ---------------------------------------------------------------------------
// 1. Masked multi-pattern byte scan (search_byte_patterns, crypto constants,
//    anti-analysis signatures, prologue scan)
// ---------------------------------------------------------------------------
extern "C" __global__ void pattern_scan(const u8* data, i64 n,
        const u8* pats, const u8* masks, const int* patOff, const int* patLen, int nPats,
        int alignment, i64* hitPos, int* hitPat, int* hitCount, int maxHits) {
    GRID_STRIDE(i, n) {
        if (alignment > 1 && (i % alignment) != 0) continue;
        for (int p = 0; p < nPats; p++) {
            int len = patLen[p];
            if (i + len > n) continue;
            const u8* pp = pats + patOff[p];
            const u8* mm = masks + patOff[p];
            bool ok = true;
            for (int k = 0; k < len; k++) {
                if ((data[i + k] & mm[k]) != (pp[k] & mm[k])) { ok = false; break; }
            }
            if (ok) {
                int slot = atomicAdd(hitCount, 1);
                if (slot < maxHits) { hitPos[slot] = i; hitPat[slot] = p; }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. String extraction (list_strings / search_strings / inspect_memory_content)
//    A run starts where the previous unit is not printable. ASCII: printable is
//    0x20..0x7E plus \t. UTF-16LE: a printable ASCII low byte with a zero high byte,
//    at either byte alignment.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool isPrint(u8 c) { return (c >= 0x20 && c <= 0x7E) || c == 0x09; }

extern "C" __global__ void string_scan(const u8* data, i64 n, int minLen, int wide,
        i64* outPos, int* outLen, int* outCount, int maxOut, int maxStrLen) {
    GRID_STRIDE(i, n) {
        if (!wide) {
            if (!isPrint(data[i])) continue;
            if (i > 0 && isPrint(data[i - 1])) continue;
            int len = 0;
            while (i + len < n && isPrint(data[i + len]) && len < maxStrLen) len++;
            if (len >= minLen) {
                int s = atomicAdd(outCount, 1);
                if (s < maxOut) { outPos[s] = i; outLen[s] = len; }
            }
        } else {
            if (i + 1 >= n || !isPrint(data[i]) || data[i + 1] != 0) continue;
            if (i >= 2 && isPrint(data[i - 2]) && data[i - 1] == 0) continue;
            int len = 0;
            while (i + 2 * len + 1 < n && isPrint(data[i + 2 * len]) && data[i + 2 * len + 1] == 0
                   && len < maxStrLen) len++;
            if (len >= minLen) {
                int s = atomicAdd(outCount, 1);
                if (s < maxOut) { outPos[s] = i; outLen[s] = len; }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Byte histogram + windowed Shannon entropy (analyze_data_region,
//    inspect_memory_content, find_code_gaps content typing)
// ---------------------------------------------------------------------------
extern "C" __global__ void byte_histogram(const u8* data, i64 n, u64* hist) {
    __shared__ u32 local[256];
    for (int k = threadIdx.x; k < 256; k += blockDim.x) local[k] = 0;
    __syncthreads();
    GRID_STRIDE(i, n) atomicAdd(&local[data[i]], 1u);
    __syncthreads();
    for (int k = threadIdx.x; k < 256; k += blockDim.x)
        if (local[k]) atomicAdd(&hist[k], (u64)local[k]);
}

// One block per window. Entropy in bits per byte (0..8), computed in double so
// the CPU twin matches to the last ulp after both round to float.
extern "C" __global__ void window_entropy(const u8* data, i64 n, int window, i64 nWindows, float* out) {
    __shared__ u32 h[256];
    for (i64 w = blockIdx.x; w < nWindows; w += gridDim.x) {
        for (int k = threadIdx.x; k < 256; k += blockDim.x) h[k] = 0;
        __syncthreads();
        i64 start = w * (i64)window;
        i64 end = start + window; if (end > n) end = n;
        for (i64 i = start + threadIdx.x; i < end; i += blockDim.x) atomicAdd(&h[data[i]], 1u);
        __syncthreads();
        if (threadIdx.x == 0) {
            double len = (double)(end - start), e = 0.0;
            for (int k = 0; k < 256; k++) if (h[k]) { double p = h[k] / len; e -= p * log2(p); }
            out[w] = (float)e;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// 4. Pointer scan (raw data xrefs, list_data_items_by_xrefs, vtable hunting).
//    Reads a ptrSize-byte value at every alignment-th offset and keeps it if it is
//    in [lo, hi) and, when targets are given, equal to one of them (sorted array,
//    binary search).
// ---------------------------------------------------------------------------
__device__ __forceinline__ u64 readUnsigned(const u8* p, int size, int bigEndian) {
    u64 v = 0;
    if (bigEndian) { for (int k = 0; k < size; k++) v = (v << 8) | p[k]; }
    else           { for (int k = size - 1; k >= 0; k--) v = (v << 8) | p[k]; }
    return v;
}

extern "C" __global__ void pointer_scan(const u8* data, i64 n, int ptrSize, int alignment, int bigEndian,
        u64 lo, u64 hi, const u64* targets, int nTargets,
        i64* outPos, u64* outVal, int* outCount, int maxOut) {
    i64 slots = n / alignment;
    GRID_STRIDE(s, slots) {
        i64 i = s * alignment;
        if (i + ptrSize > n) continue;
        u64 v = readUnsigned(data + i, ptrSize, bigEndian);
        if (v < lo || v >= hi) continue;
        if (nTargets > 0) {
            int a = 0, b = nTargets - 1; bool found = false;
            while (a <= b) { int m = (a + b) >> 1; u64 t = targets[m];
                if (t == v) { found = true; break; } if (t < v) a = m + 1; else b = m - 1; }
            if (!found) continue;
        }
        int slot = atomicAdd(outCount, 1);
        if (slot < maxOut) { outPos[slot] = i; outVal[slot] = v; }
    }
}

// ---------------------------------------------------------------------------
// 5. Batched hashing: FNV-1a 64 and SHA-256, one thread per segment
//    (get_function_hash over normalized instruction text, compare_programs)
// ---------------------------------------------------------------------------
extern "C" __global__ void fnv1a64_batch(const u8* data, const i64* off, const int* len, int nSeg, u64* out) {
    GRID_STRIDE(s, nSeg) {
        u64 h = 0xcbf29ce484222325ULL;
        const u8* p = data + off[s];
        for (int k = 0; k < len[s]; k++) { h ^= p[k]; h *= 0x100000001b3ULL; }
        out[s] = h;
    }
}

__constant__ u32 K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ u32 rotr(u32 x, int r) { return (x >> r) | (x << (32 - r)); }

__device__ void sha256_block(u32* st, const u8* blk) {
    u32 w[64];
    for (int t = 0; t < 16; t++)
        w[t] = ((u32)blk[4*t] << 24) | ((u32)blk[4*t+1] << 16) | ((u32)blk[4*t+2] << 8) | blk[4*t+3];
    for (int t = 16; t < 64; t++) {
        u32 s0 = rotr(w[t-15], 7) ^ rotr(w[t-15], 18) ^ (w[t-15] >> 3);
        u32 s1 = rotr(w[t-2], 17) ^ rotr(w[t-2], 19) ^ (w[t-2] >> 10);
        w[t] = w[t-16] + s0 + w[t-7] + s1;
    }
    u32 a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for (int t = 0; t < 64; t++) {
        u32 S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        u32 ch = (e & f) ^ (~e & g);
        u32 t1 = h + S1 + ch + K256[t] + w[t];
        u32 S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        u32 mj = (a & b) ^ (a & c) ^ (b & c);
        u32 t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

extern "C" __global__ void sha256_batch(const u8* data, const i64* off, const int* len, int nSeg, u8* out) {
    GRID_STRIDE(s, nSeg) {
        u32 st[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
        const u8* p = data + off[s];
        int L = len[s];
        int full = L / 64;
        for (int b = 0; b < full; b++) sha256_block(st, p + 64 * b);
        u8 tail[128];
        int rem = L - full * 64;
        for (int k = 0; k < rem; k++) tail[k] = p[full * 64 + k];
        tail[rem] = 0x80;
        int tailLen = (rem + 9 <= 64) ? 64 : 128;
        for (int k = rem + 1; k < tailLen; k++) tail[k] = 0;
        u64 bits = (u64)L * 8;
        for (int k = 0; k < 8; k++) tail[tailLen - 1 - k] = (u8)(bits >> (8 * k));
        sha256_block(st, tail);
        if (tailLen == 128) sha256_block(st, tail + 64);
        for (int k = 0; k < 8; k++) {
            out[32*s + 4*k]   = (u8)(st[k] >> 24); out[32*s + 4*k+1] = (u8)(st[k] >> 16);
            out[32*s + 4*k+2] = (u8)(st[k] >> 8);  out[32*s + 4*k+3] = (u8)st[k];
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Similarity: cosine over feature vectors, MinHash, MinHash Jaccard estimate
//    (find_similar_functions, find_similar_functions_fuzzy, bulk_fuzzy_match)
// ---------------------------------------------------------------------------
// out[a*nB+b] = cos(A[a], B[b]); zero vectors score 0. Accumulated in double so
// the CPU twin agrees exactly after the final rounding to float.
extern "C" __global__ void cosine_matrix(const float* A, int nA, const float* B, int nB, int dim, float* out) {
    i64 total = (i64)nA * nB;
    GRID_STRIDE(idx, total) {
        int a = (int)(idx / nB), b = (int)(idx % nB);
        const float* x = A + (i64)a * dim; const float* y = B + (i64)b * dim;
        double dot = 0, nx = 0, ny = 0;
        for (int k = 0; k < dim; k++) { double u = x[k], v = y[k]; dot += u*v; nx += u*u; ny += v*v; }
        out[idx] = (nx == 0 || ny == 0) ? 0.0f : (float)(dot / (sqrt(nx) * sqrt(ny)));
    }
}

__device__ __forceinline__ u32 mix32(u32 h) {
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h;
}

// sig[s*k + j] = min over shingles of mix32(shingleHash ^ seed_j). A shingle is
// `shingle` consecutive tokens; sets shorter than one shingle hash the whole set.
extern "C" __global__ void minhash(const u32* tokens, const i64* off, const int* len, int nSets,
        int k, int shingle, u32* sig) {
    i64 total = (i64)nSets * k;
    GRID_STRIDE(idx, total) {
        int s = (int)(idx / k), j = (int)(idx % k);
        u32 seed = mix32(0x9e3779b9u * (u32)(j + 1));
        const u32* t = tokens + off[s];
        int L = len[s];
        int nSh = L >= shingle ? L - shingle + 1 : (L > 0 ? 1 : 0);
        int w = L >= shingle ? shingle : L;
        u32 best = 0xffffffffu;
        for (int i = 0; i < nSh; i++) {
            u32 h = 0x811c9dc5u;
            for (int q = 0; q < w; q++) { h ^= t[i + q]; h *= 0x01000193u; }
            u32 v = mix32(h ^ seed);
            if (v < best) best = v;
        }
        sig[idx] = best;
    }
}

extern "C" __global__ void minhash_similarity(const u32* sigA, int nA, const u32* sigB, int nB, int k, float* out) {
    i64 total = (i64)nA * nB;
    GRID_STRIDE(idx, total) {
        int a = (int)(idx / nB), b = (int)(idx % nB);
        int eq = 0;
        for (int j = 0; j < k; j++) eq += sigA[(i64)a*k + j] == sigB[(i64)b*k + j];
        out[idx] = (float)eq / (float)k;
    }
}

// ---------------------------------------------------------------------------
// 7. Batched Levenshtein over token sequences (diff_functions, cross-program
//    matching). One thread per pair, two DP rows in caller-provided scratch.
// ---------------------------------------------------------------------------
extern "C" __global__ void levenshtein_batch(const u32* tokA, const i64* offA, const int* lenA,
        const u32* tokB, const i64* offB, const int* lenB, int nPairs, int* rowScratch, int rowStride, int* out) {
    GRID_STRIDE(p, nPairs) {
        const u32* a = tokA + offA[p]; const u32* b = tokB + offB[p];
        int la = lenA[p], lb = lenB[p];
        int* prev = rowScratch + (i64)p * 2 * rowStride;
        int* cur = prev + rowStride;
        for (int j = 0; j <= lb; j++) prev[j] = j;
        for (int i = 1; i <= la; i++) {
            cur[0] = i;
            for (int j = 1; j <= lb; j++) {
                int c = prev[j - 1] + (a[i - 1] != b[j - 1]);
                int d = prev[j] + 1, e = cur[j - 1] + 1;
                cur[j] = c < d ? (c < e ? c : e) : (d < e ? d : e);
            }
            int* t = prev; prev = cur; cur = t;
        }
        out[p] = prev[lb];
    }
}

// ---------------------------------------------------------------------------
// 8. Graphs in CSR form (call graph, CFG reachability, def-use slices)
// ---------------------------------------------------------------------------
// Level-synchronous BFS step: every node at `level` relaxes its successors.
// Successor ordering does not matter: depth is the only thing written.
extern "C" __global__ void bfs_step(const int* rowPtr, const int* col, int n, int* depth, int level, int maxDepth, int* changed) {
    GRID_STRIDE(u, n) {
        if (depth[u] != level || level >= maxDepth) continue;
        for (int e = rowPtr[u]; e < rowPtr[u + 1]; e++) {
            int v = col[e];
            if (atomicCAS(&depth[v], -1, level + 1) == -1) *changed = 1;
        }
    }
}

// SCC by forward colouring / backward closure. color[] holds the max node id
// that reaches each unfinished node; scc[] < 0 means unfinished.
extern "C" __global__ void scc_color_propagate(const int* rowPtr, const int* col, int n, const int* scc, int* color, int* changed) {
    GRID_STRIDE(u, n) {
        if (scc[u] >= 0) continue;
        int cu = color[u];
        for (int e = rowPtr[u]; e < rowPtr[u + 1]; e++) {
            int v = col[e];
            if (scc[v] >= 0) continue;
            if (atomicMax(&color[v], cu) < cu) *changed = 1;
        }
    }
}

// Backward closure over the reverse graph, restricted to one colour class.
extern "C" __global__ void scc_backward(const int* rRowPtr, const int* rCol, int n, int* scc, const int* color, int* changed) {
    GRID_STRIDE(u, n) {
        if (scc[u] < 0) continue;
        int root = scc[u];
        for (int e = rRowPtr[u]; e < rRowPtr[u + 1]; e++) {
            int v = rCol[e];
            if (scc[v] < 0 && color[v] == root) {
                if (atomicCAS(&scc[v], -1, root) == -1) *changed = 1;
            }
        }
    }
}

// Trimming: an unfinished node with no unfinished predecessor or no unfinished
// successor is its own SCC. Repeated until stable it peels every acyclic part of
// the graph, which colouring alone would take one round per node to finish.
extern "C" __global__ void scc_trim(const int* rowPtr, const int* col, const int* rRowPtr, const int* rCol,
        int n, int* scc, int* changed) {
    GRID_STRIDE(u, n) {
        if (scc[u] >= 0) continue;
        bool hasOut = false, hasIn = false;
        for (int e = rowPtr[u]; e < rowPtr[u + 1] && !hasOut; e++) { int v = col[e]; if (v != u && scc[v] < 0) hasOut = true; }
        for (int e = rRowPtr[u]; e < rRowPtr[u + 1] && !hasIn; e++) { int v = rCol[e]; if (v != u && scc[v] < 0) hasIn = true; }
        if (!hasOut || !hasIn) { scc[u] = (int)u; *changed = 1; }
    }
}

extern "C" __global__ void scc_seed_roots(int n, int* scc, const int* color) {
    GRID_STRIDE(u, n) if (scc[u] < 0 && color[u] == u) scc[u] = (int)u;
}

extern "C" __global__ void scc_reset_colors(int n, const int* scc, int* color) {
    GRID_STRIDE(u, n) if (scc[u] < 0) color[u] = (int)u;
}

// PageRank pull step over the reverse graph. Dangling mass is passed in.
extern "C" __global__ void pagerank_step(const int* rRowPtr, const int* rCol, const int* outDeg, int n,
        const double* rankIn, double* rankOut, double damping, double danglingShare) {
    GRID_STRIDE(v, n) {
        double s = 0;
        for (int e = rRowPtr[v]; e < rRowPtr[v + 1]; e++) { int u = rCol[e]; s += rankIn[u] / outDeg[u]; }
        rankOut[v] = (1.0 - damping) / n + damping * (s + danglingShare);
    }
}

// ---------------------------------------------------------------------------
// 9. Batched dominators: one block per CFG (analyze_control_flow, natural loops,
//    find_dead_code). Iterative bitset algorithm; entry is local block 0.
//    Unreachable blocks keep the all-ones set and are reported with idom = -2.
// ---------------------------------------------------------------------------
extern "C" __global__ void dominators_batch(const int* graphNodeOff, const int* graphNodes,
        const int* predRowPtr, const int* predCol, int nGraphs, u32* domBits, const i64* domOff,
        int* idom, int* reachable) {
    int g = blockIdx.x;
    if (g >= nGraphs) return;
    int base = graphNodeOff[g];
    int nb = graphNodeOff[g + 1] - base;
    int words = (nb + 31) / 32;
    u32* dom = domBits + domOff[g];
    for (int v = threadIdx.x; v < nb; v += blockDim.x) {
        for (int w = 0; w < words; w++) dom[(i64)v * words + w] = v == 0 ? 0u : 0xffffffffu;
        if (v == 0) dom[0] = 1u;
        int tail = nb & 31;
        if (tail && v != 0) dom[(i64)v * words + words - 1] = (1u << tail) - 1;
    }
    __syncthreads();
    // Reachability from entry first (predecessor sets ignore unreachable preds).
    for (int v = threadIdx.x; v < nb; v += blockDim.x) reachable[base + v] = v == 0;
    __syncthreads();
    int changed = 1;
    while (changed) {
        int local = 0;
        for (int v = threadIdx.x; v < nb; v += blockDim.x) {
            if (reachable[base + v]) continue;
            for (int e = predRowPtr[base + v]; e < predRowPtr[base + v + 1]; e++)
                if (reachable[base + predCol[e]]) { reachable[base + v] = 1; local = 1; break; }
        }
        changed = __syncthreads_or(local);
    }
    changed = 1;
    while (changed) {
        int local = 0;
        for (int v = 1 + threadIdx.x; v < nb; v += blockDim.x) {
            if (!reachable[base + v]) continue;
            for (int w = 0; w < words; w++) {
                u32 acc = 0xffffffffu; bool any = false;
                for (int e = predRowPtr[base + v]; e < predRowPtr[base + v + 1]; e++) {
                    int p = predCol[e];
                    if (!reachable[base + p]) continue;
                    acc &= dom[(i64)p * words + w]; any = true;
                }
                if (!any) acc = 0;
                if ((v >> 5) == w) acc |= 1u << (v & 31);
                if (acc != dom[(i64)v * words + w]) { dom[(i64)v * words + w] = acc; local = 1; }
            }
        }
        changed = __syncthreads_or(local);
    }
    // idom(v): the strict dominator whose own dominator set is largest.
    for (int v = threadIdx.x; v < nb; v += blockDim.x) {
        if (!reachable[base + v]) { idom[base + v] = -2; continue; }
        if (v == 0) { idom[base] = -1; continue; }
        int best = -1, bestCount = -1;
        for (int d = 0; d < nb; d++) {
            if (d == v) continue;
            if (!(dom[(i64)v * words + (d >> 5)] & (1u << (d & 31)))) continue;
            int c = 0;
            for (int w = 0; w < words; w++) c += __popc(dom[(i64)d * words + w]);
            if (c > bestCount) { bestCount = c; best = d; }
        }
        idom[base + v] = best;
    }
}

// ---------------------------------------------------------------------------
// 10. Substring search over many strings (find_functions, search_strings,
//     find_data_types, find_undocumented_by_string). ASCII case folding.
// ---------------------------------------------------------------------------
__device__ __forceinline__ u8 lower(u8 c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

extern "C" __global__ void substring_search(const u8* hay, const i64* off, const int* len, int nStr,
        const u8* needle, int nlen, int ignoreCase, int* firstIndex) {
    GRID_STRIDE(s, nStr) {
        const u8* h = hay + off[s];
        int L = len[s], found = -1;
        for (int i = 0; i + nlen <= L && found < 0; i++) {
            int k = 0;
            for (; k < nlen; k++) {
                u8 a = h[i + k], b = needle[k];
                if (ignoreCase) { a = lower(a); b = lower(b); }
                if (a != b) break;
            }
            if (k == nlen) found = i;
        }
        firstIndex[s] = found;
    }
}

// ---------------------------------------------------------------------------
// 11. IOC candidate scan (extract_iocs_with_context): URLs, IPv4 literals,
//     e-mail addresses and registry paths, found in raw bytes.
//     type: 1=url 2=ipv4 3=email 4=registry
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool isAlnum(u8 c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
__device__ __forceinline__ bool isDigit(u8 c) { return c >= '0' && c <= '9'; }
__device__ __forceinline__ bool isUrlChar(u8 c) {
    return isAlnum(c) || c == '-' || c == '.' || c == '_' || c == '~' || c == ':' || c == '/' || c == '?' ||
           c == '#' || c == '[' || c == ']' || c == '@' || c == '!' || c == '$' || c == '&' || c == '\'' ||
           c == '(' || c == ')' || c == '*' || c == '+' || c == ',' || c == ';' || c == '=' || c == '%';
}
__device__ __forceinline__ bool isLocalChar(u8 c) { return isAlnum(c) || c == '.' || c == '_' || c == '-' || c == '+'; }
__device__ __forceinline__ bool isDomainChar(u8 c) { return isAlnum(c) || c == '.' || c == '-'; }

__device__ bool prefixAt(const u8* d, i64 n, i64 i, const char* lit, int ci) {
    for (int k = 0; lit[k]; k++) {
        if (i + k >= n) return false;
        u8 a = d[i + k]; u8 b = (u8)lit[k];
        if (ci) { a = lower(a); b = lower(b); }
        if (a != b) return false;
    }
    return true;
}

__device__ int matchIpv4(const u8* d, i64 n, i64 i) {
    i64 p = i;
    for (int part = 0; part < 4; part++) {
        int digits = 0, val = 0;
        while (p < n && isDigit(d[p]) && digits < 4) { val = val * 10 + (d[p] - '0'); p++; digits++; }
        if (digits == 0 || digits > 3 || val > 255) return 0;
        if (part < 3) { if (p >= n || d[p] != '.') return 0; p++; }
    }
    if (p < n && (isDigit(d[p]) || (d[p] == '.' && p + 1 < n && isDigit(d[p + 1])))) return 0;
    return (int)(p - i);
}

extern "C" __global__ void ioc_scan(const u8* d, i64 n, i64* outPos, int* outLen, int* outType, int* outCount, int maxOut, int maxLen) {
    GRID_STRIDE(i, n) {
        int type = 0, len = 0;
        u8 prev = i > 0 ? d[i - 1] : 0;
        int scheme = 0;
        if (!isAlnum(prev)) {
            if (prefixAt(d, n, i, "https://", 1)) scheme = 8;
            else if (prefixAt(d, n, i, "http://", 1)) scheme = 7;
            else if (prefixAt(d, n, i, "ftp://", 1)) scheme = 6;
        }
        if (scheme) {
            int L = 0; while (i + L < n && isUrlChar(d[i + L]) && L < maxLen) L++;
            if (L > scheme + 3) { type = 1; len = L; }
        } else if (isDigit(d[i]) && !isDigit(prev) && prev != '.') {
            int L = matchIpv4(d, n, i);
            if (L) { type = 2; len = L; }
        } else if (d[i] == '@' && i > 0 && isLocalChar(prev)) {
            i64 s = i; while (s > 0 && isLocalChar(d[s - 1]) && i - s < 64) s--;
            int dl = 0; bool dot = false;
            while (i + 1 + dl < n && isDomainChar(d[i + 1 + dl]) && dl < 253) { if (d[i + 1 + dl] == '.') dot = true; dl++; }
            while (dl > 0 && (d[i + dl] == '.' || d[i + dl] == '-')) dl--;
            if (dot && dl >= 3 && isAlnum(d[s]) && isAlnum(d[i + dl])) {
                bool tailDot = false; for (int q = 1; q <= dl; q++) if (d[i + q] == '.') tailDot = true;
                if (tailDot) { int sl = atomicAdd(outCount, 1);
                    if (sl < maxOut) { outPos[sl] = s; outLen[sl] = (int)(i - s) + 1 + dl; outType[sl] = 3; } }
            }
        } else if (!isAlnum(prev) && (prefixAt(d, n, i, "HKEY_", 0) || prefixAt(d, n, i, "HKLM\\", 0) ||
                   prefixAt(d, n, i, "HKCU\\", 0) || prefixAt(d, n, i, "SOFTWARE\\", 1) || prefixAt(d, n, i, "SYSTEM\\CurrentControlSet", 1))) {
            int L = 0; while (i + L < n && isPrint(d[i + L]) && d[i + L] != '"' && L < maxLen) L++;
            if (L >= 8) { type = 4; len = L; }
        }
        if (type) {
            int s = atomicAdd(outCount, 1);
            if (s < maxOut) { outPos[s] = i; outLen[s] = len; outType[s] = type; }
        }
    }
}

// ---------------------------------------------------------------------------
// 12. Batched raw-P-code virtual machine (emulate_hash_batch, emulate_function
//     with many inputs). One thread == one lane == one independent execution.
//
// The host lowers a function's raw P-code (Instruction.getPcode) into PcOp
// records. Varnodes are pre-resolved: register/unique varnodes become byte
// offsets into the lane's private state block, constants carry their value, RAM
// varnodes stay addresses. Memory a lane can touch:
//   * private windows (stack, scratch): read/write, per lane
//   * the program image: read-only, shared, looked up through segment table
// Anything else faults the lane rather than inventing a value.
// ---------------------------------------------------------------------------
enum { VN_NONE = 0, VN_STATE = 1, VN_CONST = 2, VN_RAM = 3 };

struct PcVarnode { int kind; int size; u64 val; };          // val = state offset, constant, or address
struct PcOp { int opcode; int instrIndex; PcVarnode out; PcVarnode in[3]; int nIn; int pad; };

// Raw P-code opcodes (ghidra.program.model.pcode.PcodeOp constants).
enum {
    OP_COPY=1, OP_LOAD=2, OP_STORE=3, OP_BRANCH=4, OP_CBRANCH=5, OP_BRANCHIND=6, OP_CALL=7, OP_CALLIND=8,
    OP_CALLOTHER=9, OP_RETURN=10, OP_INT_EQUAL=11, OP_INT_NOTEQUAL=12, OP_INT_SLESS=13, OP_INT_SLESSEQUAL=14,
    OP_INT_LESS=15, OP_INT_LESSEQUAL=16, OP_INT_ZEXT=17, OP_INT_SEXT=18, OP_INT_ADD=19, OP_INT_SUB=20,
    OP_INT_CARRY=21, OP_INT_SCARRY=22, OP_INT_SBORROW=23, OP_INT_2COMP=24, OP_INT_NEGATE=25, OP_INT_XOR=26,
    OP_INT_AND=27, OP_INT_OR=28, OP_INT_LEFT=29, OP_INT_RIGHT=30, OP_INT_SRIGHT=31, OP_INT_MULT=32,
    OP_INT_DIV=33, OP_INT_SDIV=34, OP_INT_REM=35, OP_INT_SREM=36, OP_BOOL_NEGATE=37, OP_BOOL_XOR=38,
    OP_BOOL_AND=39, OP_BOOL_OR=40, OP_PIECE=62, OP_SUBPIECE=63, OP_POPCOUNT=72, OP_LZCOUNT=73
};

// Lane status codes, mirrored in PcodeVm.Status.
enum { ST_RUNNING=0, ST_RETURNED=1, ST_STEP_LIMIT=2, ST_BAD_READ=3, ST_BAD_WRITE=4, ST_UNSUPPORTED=5,
       ST_DIV_ZERO=6, ST_BAD_BRANCH=7, ST_CALL=8 };

struct VmMem {
    u8* lanePriv; i64 privSize;                 // this lane's private bytes (windows packed)
    const u64* winBase; const int* winSize; const i64* winOff; int nWin;
    const u8* image; const u64* segBase; const int* segSize; const i64* segOff; int nSeg;
    int bigEndian;
};

__device__ __forceinline__ u64 maskOf(int size) { return size >= 8 ? ~0ULL : ((1ULL << (8 * size)) - 1); }

__device__ __forceinline__ u64 sext(u64 v, int size) {
    if (size >= 8) return v;
    int sh = 64 - 8 * size; return (u64)(((i64)(v << sh)) >> sh);
}

__device__ bool memPtr(const VmMem& m, u64 addr, int size, bool write, u8** p) {
    for (int w = 0; w < m.nWin; w++) {
        if (addr >= m.winBase[w] && addr + size <= m.winBase[w] + (u64)m.winSize[w]) {
            *p = m.lanePriv + m.winOff[w] + (addr - m.winBase[w]); return true;
        }
    }
    if (write) return false;
    // Segments are sorted by base: binary search the last base <= addr.
    int a = 0, b = m.nSeg - 1, hit = -1;
    while (a <= b) { int mid = (a + b) >> 1; if (m.segBase[mid] <= addr) { hit = mid; a = mid + 1; } else b = mid - 1; }
    if (hit >= 0 && addr + size <= m.segBase[hit] + (u64)m.segSize[hit]) {
        *p = (u8*)(m.image + m.segOff[hit] + (addr - m.segBase[hit])); return true;
    }
    return false;
}

__device__ u64 loadBytes(const u8* p, int size, int be) {
    u64 v = 0;
    if (be) for (int k = 0; k < size; k++) v = (v << 8) | p[k];
    else    for (int k = size - 1; k >= 0; k--) v = (v << 8) | p[k];
    return v;
}
__device__ void storeBytes(u8* p, int size, int be, u64 v) {
    if (be) for (int k = size - 1; k >= 0; k--) { p[k] = (u8)v; v >>= 8; }
    else    for (int k = 0; k < size; k++) { p[k] = (u8)v; v >>= 8; }
}

extern "C" __global__ void pcode_vm(const PcOp* ops, int nOps, const int* instrFirstOp, const u64* instrAddr, int nInstr,
        u8* laneState, int stateSize, const u8* stateInit,
        u8* lanePrivAll, i64 privSize, const u8* privInit, const i64* privInitLaneOff, const int* privInitLaneLen,
        const u64* winBase, const int* winSize, const i64* winOff, int nWin,
        const u8* image, const u64* segBase, const int* segSize, const i64* segOff, int nSeg,
        int bigEndian, u64 returnSentinel, int maxSteps, int nLanes,
        int* status, int* steps, int* faultOp, u64* faultAddr) {
    GRID_STRIDE(lane, nLanes) {
        u8* st = laneState + lane * (i64)stateSize;
        for (int k = 0; k < stateSize; k++) st[k] = stateInit[lane * (i64)stateSize + k];
        VmMem m;
        m.lanePriv = lanePrivAll + lane * privSize; m.privSize = privSize;
        m.winBase = winBase; m.winSize = winSize; m.winOff = winOff; m.nWin = nWin;
        m.image = image; m.segBase = segBase; m.segSize = segSize; m.segOff = segOff; m.nSeg = nSeg;
        m.bigEndian = bigEndian;
        for (i64 k = 0; k < privSize; k++) m.lanePriv[k] = 0;
        // Per-lane overlay of initial private bytes: (window-relative offset is pre-baked by host)
        {
            const u8* src = privInit + privInitLaneOff[lane];
            int L = privInitLaneLen[lane];
            // Host packs records as [u32 privOffset][u32 len][bytes...]
            int q = 0;
            while (q + 8 <= L) {
                u32 po = src[q] | (src[q+1] << 8) | (src[q+2] << 16) | ((u32)src[q+3] << 24);
                u32 ln = src[q+4] | (src[q+5] << 8) | (src[q+6] << 16) | ((u32)src[q+7] << 24);
                q += 8;
                for (u32 k = 0; k < ln && po + k < privSize; k++) m.lanePriv[po + k] = src[q + k];
                q += ln;
            }
        }
        int pc = 0, n = 0, code = ST_RUNNING;
        u64 fa = 0;
        while (code == ST_RUNNING) {
            if (pc < 0 || pc >= nOps) { code = ST_BAD_BRANCH; break; }
            if (n >= maxSteps) { code = ST_STEP_LIMIT; break; }
            n++;
            const PcOp& op = ops[pc];
            u64 in[3] = {0, 0, 0};
            bool bad = false;
            for (int k = 0; k < op.nIn && k < 3; k++) {
                const PcVarnode& v = op.in[k];
                if (v.kind == VN_CONST) in[k] = v.val & maskOf(v.size);
                else if (v.kind == VN_STATE) in[k] = loadBytes(st + v.val, v.size, bigEndian);
                else if (v.kind == VN_RAM) {
                    u8* p; if (!memPtr(m, v.val, v.size, false, &p)) { code = ST_BAD_READ; fa = v.val; bad = true; break; }
                    in[k] = loadBytes(p, v.size, bigEndian);
                }
            }
            if (bad) break;
            int osz = op.out.size;
            u64 r = 0; bool write = op.out.kind != VN_NONE; int next = pc + 1;
            int s0 = op.in[0].size, s1 = op.in[1].size;
            switch (op.opcode) {
            case OP_COPY: r = in[0]; break;
            case OP_LOAD: {
                u64 addr = in[1]; u8* p;
                if (!memPtr(m, addr, osz, false, &p)) { code = ST_BAD_READ; fa = addr; write = false; break; }
                r = loadBytes(p, osz, bigEndian); break; }
            case OP_STORE: {
                u64 addr = in[1]; u8* p; int sz = op.in[2].size;
                if (!memPtr(m, addr, sz, true, &p)) { code = ST_BAD_WRITE; fa = addr; break; }
                storeBytes(p, sz, bigEndian, in[2]); write = false; break; }
            case OP_BRANCH: case OP_CBRANCH: {
                if (op.opcode == OP_CBRANCH && in[1] == 0) { write = false; break; }
                // Host resolved the target into in[0].val as an absolute op index,
                // or -1 when it left the function.
                long long t = (long long)op.in[0].val;
                if (t < 0) { code = ST_BAD_BRANCH; fa = 0; }
                else next = (int)t;
                write = false; break; }
            case OP_BRANCHIND: case OP_RETURN: {
                u64 target = in[0];
                write = false;
                if (target == returnSentinel) { code = ST_RETURNED; break; }
                int a = 0, b = nInstr - 1, hit = -1;
                while (a <= b) { int mid = (a + b) >> 1; if (instrAddr[mid] == target) { hit = mid; break; }
                    if (instrAddr[mid] < target) a = mid + 1; else b = mid - 1; }
                if (hit < 0) { code = ST_BAD_BRANCH; fa = target; } else next = instrFirstOp[hit];
                break; }
            case OP_CALL: case OP_CALLIND: case OP_CALLOTHER:
                code = ST_CALL; fa = in[0]; write = false; break;
            case OP_INT_EQUAL: r = in[0] == in[1]; break;
            case OP_INT_NOTEQUAL: r = in[0] != in[1]; break;
            case OP_INT_LESS: r = in[0] < in[1]; break;
            case OP_INT_LESSEQUAL: r = in[0] <= in[1]; break;
            case OP_INT_SLESS: r = (i64)sext(in[0], s0) < (i64)sext(in[1], s1); break;
            case OP_INT_SLESSEQUAL: r = (i64)sext(in[0], s0) <= (i64)sext(in[1], s1); break;
            case OP_INT_ZEXT: r = in[0]; break;
            case OP_INT_SEXT: r = sext(in[0], s0); break;
            case OP_INT_ADD: r = in[0] + in[1]; break;
            case OP_INT_SUB: r = in[0] - in[1]; break;
            case OP_INT_CARRY: r = ((in[0] + in[1]) & maskOf(s0)) < in[0]; break;
            case OP_INT_SCARRY: { i64 a = (i64)sext(in[0], s0), b = (i64)sext(in[1], s0);
                i64 c = (i64)sext((in[0] + in[1]) & maskOf(s0), s0);
                r = ((a >= 0) == (b >= 0)) && ((c >= 0) != (a >= 0)); break; }
            case OP_INT_SBORROW: { i64 a = (i64)sext(in[0], s0), b = (i64)sext(in[1], s0);
                i64 c = (i64)sext((in[0] - in[1]) & maskOf(s0), s0);
                r = ((a >= 0) != (b >= 0)) && ((c >= 0) != (a >= 0)); break; }
            case OP_INT_2COMP: r = (u64)(-(i64)in[0]); break;
            case OP_INT_NEGATE: r = ~in[0]; break;
            case OP_INT_XOR: r = in[0] ^ in[1]; break;
            case OP_INT_AND: r = in[0] & in[1]; break;
            case OP_INT_OR: r = in[0] | in[1]; break;
            case OP_INT_LEFT: r = in[1] >= 64 ? 0 : in[0] << in[1]; break;
            case OP_INT_RIGHT: r = in[1] >= 64 ? 0 : in[0] >> in[1]; break;
            case OP_INT_SRIGHT: { i64 a = (i64)sext(in[0], s0); r = (u64)(in[1] >= 64 ? (a < 0 ? -1 : 0) : (a >> in[1])); break; }
            case OP_INT_MULT: r = in[0] * in[1]; break;
            case OP_INT_DIV: if (in[1] == 0) { code = ST_DIV_ZERO; write = false; } else r = in[0] / in[1]; break;
            case OP_INT_REM: if (in[1] == 0) { code = ST_DIV_ZERO; write = false; } else r = in[0] % in[1]; break;
            case OP_INT_SDIV: if (in[1] == 0) { code = ST_DIV_ZERO; write = false; }
                else r = (u64)((i64)sext(in[0], s0) / (i64)sext(in[1], s1)); break;
            case OP_INT_SREM: if (in[1] == 0) { code = ST_DIV_ZERO; write = false; }
                else r = (u64)((i64)sext(in[0], s0) % (i64)sext(in[1], s1)); break;
            case OP_BOOL_NEGATE: r = in[0] ^ 1; break;
            case OP_BOOL_XOR: r = in[0] ^ in[1]; break;
            case OP_BOOL_AND: r = in[0] & in[1]; break;
            case OP_BOOL_OR: r = in[0] | in[1]; break;
            case OP_PIECE: r = (s1 >= 8 ? 0 : (in[0] << (8 * s1))) | in[1]; break;
            case OP_SUBPIECE: r = in[1] >= 8 ? 0 : in[0] >> (8 * in[1]); break;
            case OP_POPCOUNT: r = __popcll(in[0]); break;
            case OP_LZCOUNT: { u64 x = in[0]; int c = __clzll(x) - (64 - 8 * s0); r = x == 0 ? 8 * s0 : c; break; }
            default: code = ST_UNSUPPORTED; fa = op.opcode; write = false; break;
            }
            if (code != ST_RUNNING) break;
            if (write) {
                r &= maskOf(osz);
                if (op.out.kind == VN_STATE) storeBytes(st + op.out.val, osz, bigEndian, r);
                else if (op.out.kind == VN_RAM) {
                    u8* p; if (!memPtr(m, op.out.val, osz, true, &p)) { code = ST_BAD_WRITE; fa = op.out.val; break; }
                    storeBytes(p, osz, bigEndian, r);
                }
            }
            pc = next;
        }
        status[lane] = code; steps[lane] = n; faultOp[lane] = pc; faultAddr[lane] = fa;
    }
}

// ---------------------------------------------------------------------------
// 13. Token-subset name collisions (rename_function's naming gate, batched).
//     Each name is a sorted, de-duplicated array of token hashes plus a module-
//     prefix hash and a full-name hash. Candidate a collides with existing b when
//     the prefixes match, the names differ, the set sizes differ, and one token set
//     contains the other. out[a] = the smallest colliding b (atomicMin), matching
//     NamingConventions.findTokenSubsetCollision's first-in-order answer.
// ---------------------------------------------------------------------------
__device__ bool sortedSubset(const u32* small, int ns, const u32* big, int nb) {
    int i = 0, j = 0;
    while (i < ns && j < nb) {
        if (small[i] == big[j]) { i++; j++; }
        else if (small[i] > big[j]) j++;
        else return false;
    }
    return i == ns;
}

extern "C" __global__ void token_subset_pairs(const u32* tokA, const i64* offA, const int* lenA, const u32* prefA, const u32* nameA, int nA,
        const u32* tokB, const i64* offB, const int* lenB, const u32* prefB, const u32* nameB, int nB, int* out) {
    i64 total = (i64)nA * nB;
    GRID_STRIDE(idx, total) {
        int a = (int)(idx / nB), b = (int)(idx % nB);
        int la = lenA[a], lb = lenB[b];
        if (la == 0 || lb == 0 || la == lb) continue;
        if (prefA[a] != prefB[b] || nameA[a] == nameB[b]) continue;
        const u32* ta = tokA + offA[a]; const u32* tb = tokB + offB[b];
        bool hit = la < lb ? sortedSubset(ta, la, tb, lb) : sortedSubset(tb, lb, ta, la);
        if (hit) atomicMin(&out[a], b);
    }
}

