package com.xebyte.gpu;

/** Op names: the keys of {@link CostModel}, the {@code op} in every execution report. */
public final class Ops {
    private Ops() {}

    public static final String PATTERN_SCAN = "pattern_scan";
    public static final String STRING_SCAN = "string_scan";
    public static final String ENTROPY = "window_entropy";
    public static final String HISTOGRAM = "byte_histogram";
    public static final String POINTER_SCAN = "pointer_scan";
    public static final String FNV_HASH = "fnv1a64";
    public static final String SHA256 = "sha256";
    public static final String COSINE = "cosine_similarity";
    public static final String MINHASH = "minhash";
    public static final String MINHASH_SIM = "minhash_similarity";
    public static final String LEVENSHTEIN = "levenshtein";
    public static final String BFS = "bfs";
    public static final String SCC = "scc";
    public static final String PAGERANK = "pagerank";
    public static final String DOMINATORS = "dominators";
    public static final String SUBSTRING = "substring_search";
    public static final String IOC_SCAN = "ioc_scan";
    public static final String PCODE_VM = "pcode_vm";
}
