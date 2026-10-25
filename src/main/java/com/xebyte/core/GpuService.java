package com.xebyte.core;

import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import com.xebyte.gpu.Backend;
import com.xebyte.gpu.CostModel;
import com.xebyte.gpu.Dispatch;
import com.xebyte.gpu.GraphOps;
import com.xebyte.gpu.HashOps;
import com.xebyte.gpu.Ops;
import com.xebyte.gpu.PcodeVm;
import com.xebyte.gpu.ScanOps;
import com.xebyte.gpu.SimilarityOps;
import com.xebyte.gpu.cuda.Gpu;
import com.xebyte.gpu.ghidra.GpuBench;
import com.xebyte.gpu.ghidra.PcodeLowering;
import com.xebyte.gpu.ghidra.ProgramData;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.program.model.address.Address;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.Program;
import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.PcodeOpAST;
import ghidra.program.model.pcode.Varnode;
import ghidra.program.model.pcode.VarnodeAST;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.lang.PrototypeModel;
import ghidra.util.task.TaskMonitor;

/**
 * GPU-accelerated (CUDA) twins of every compute-bearing ghidra-mcp tool.
 *
 * <p>Each tool takes {@code backend=cpu|gpu|auto|both} and returns an {@code execution}
 * block saying which backend ran, how long the op and the program extraction took, and
 * (for {@code both}) whether CPU and GPU agreed. That is the contract that lets an agent
 * choose per task: small inputs are usually faster on the CPU (no transfer, no launch),
 * big scans, all-pairs similarity and batched emulation are usually faster on the GPU,
 * and {@code gpu_benchmark} measures where the line is on this machine so {@code auto}
 * uses real numbers. {@code gpu_advise} answers "which backend for this?" without running.
 *
 * <p>What stays on the CPU regardless: reading the program database (it lives in the
 * JVM) and Ghidra's decompiler itself (native C++, single function per call). Tools that
 * consume decompiler output report decompile time apart from the op time.
 *
 * <p>The upstream tool each one accelerates is named in its description; the full
 * mapping of all upstream tools is in docs/GPU.md.
 */
@McpToolGroup(value = "gpu", description = "CUDA GPU twins of compute-heavy analysis tools; every tool takes backend=cpu|gpu|auto|both")
public class GpuService {

    private static final String P_PROGRAM = "Target program name (omit to use the active program — always specify "
        + "when multiple programs are open)";
    private static final String P_BACKEND = "cpu | gpu | auto (default: cost model, calibrated by gpu_benchmark) | "
        + "both (run both, return GPU result, report agreement and both timings)";
    private static final long SCRATCH_BASE = 0x7FFE0000L;
    private static final long STACK_TOP = 0x7FFF0000L;
    private static final long RETURN_SENTINEL = 0xDEADBEEFL;

    private final ProgramProvider programProvider;
    private final ThreadingStrategy threadingStrategy;

    public GpuService(ProgramProvider programProvider, ThreadingStrategy threadingStrategy) {
        this.programProvider = programProvider;
        this.threadingStrategy = threadingStrategy;
    }

    // =====================================================================
    // Runtime, calibration, advice
    // =====================================================================

    @McpTool(path = "/gpu_info", description = "GPU status: device, driver, memory, kernels, whether CUDA is usable "
        + "(and why not), the AUTO cost model per op, and which upstream tools each GPU tool accelerates.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuInfo() {
        Map<String, Object> out = new LinkedHashMap<>();
        Gpu g = Gpu.get();
        out.put("gpu_available", g != null);
        if (g != null) out.put("device", g.info());
        else out.put("unavailable_reason", Gpu.unavailableReason());
        out.put("cost_model", CostModel.table());
        out.put("tools", TOOL_MAP);
        out.put("note", "Every gpu_* tool accepts backend=cpu|gpu|auto|both. Run gpu_benchmark once per machine "
            + "to replace default crossovers with measured ones.");
        return Response.ok(out);
    }

    /** gpu tool -> upstream tools it accelerates. */
    static final Map<String, List<String>> TOOL_MAP = new LinkedHashMap<>();
    static {
        TOOL_MAP.put("gpu_search_byte_patterns", List.of("search_byte_patterns"));
        TOOL_MAP.put("gpu_list_strings", List.of("list_strings", "search_strings", "find_undocumented_by_string"));
        TOOL_MAP.put("gpu_entropy_map", List.of("analyze_data_region", "inspect_memory_content"));
        TOOL_MAP.put("gpu_pointer_scan", List.of("get_xrefs_to (raw data pointers)", "list_data_items_by_xrefs"));
        TOOL_MAP.put("gpu_detect_crypto_constants", List.of("detect_crypto_constants"));
        TOOL_MAP.put("gpu_find_anti_analysis", List.of("find_anti_analysis_techniques", "detect_malware_behaviors"));
        TOOL_MAP.put("gpu_extract_iocs", List.of("extract_iocs_with_context"));
        TOOL_MAP.put("gpu_find_functions", List.of("find_functions"));
        TOOL_MAP.put("gpu_search_instructions", List.of("search_instructions"));
        TOOL_MAP.put("gpu_function_hashes", List.of("get_function_hash", "compare_programs_documentation"));
        TOOL_MAP.put("gpu_find_similar_functions", List.of("find_similar_functions", "find_similar_functions_fuzzy", "bulk_fuzzy_match"));
        TOOL_MAP.put("gpu_diff_functions", List.of("diff_functions"));
        TOOL_MAP.put("gpu_call_graph_analysis", List.of("analyze_call_graph", "get_full_call_graph", "get_function_call_graph", "analyze_api_call_chains"));
        TOOL_MAP.put("gpu_control_flow_analysis", List.of("analyze_control_flow", "find_dead_code"));
        TOOL_MAP.put("gpu_find_function_starts", List.of("find_code_gaps", "find_next_undefined_function"));
        TOOL_MAP.put("gpu_dataflow_slice", List.of("analyze_dataflow"));
        // New capability rather than a twin: batch decompile (CPU) + analysis of the results (CPU or GPU).
        TOOL_MAP.put("gpu_decompile_batch", List.of());
        TOOL_MAP.put("gpu_analysis_passes", List.of("run_analysis (data-parallel passes)", "reanalyze (data-parallel passes)"));
        TOOL_MAP.put("gpu_emulate_hash_batch", List.of("emulate_hash_batch"));
        TOOL_MAP.put("gpu_emulate_function_batch", List.of("emulate_function"));
    }

    @McpTool(path = "/gpu_benchmark", method = "POST", dryRun = false,
        description = "Measure CPU vs GPU for each op on synthetic inputs of growing size, record the crossover "
            + "work size where the GPU starts winning, and persist it so backend=auto uses measured numbers "
            + "(~/.ghidra-gpu/calibration.properties). Returns the timing table.",
        category = "gpu", access = ToolAccess.WRITE)
    public Response gpuBenchmark(
            @Param(value = "ops", defaultValue = "", source = ParamSource.BODY,
                   description = "Comma-separated op names (see gpu_info cost_model); empty = all") String ops,
            @Param(value = "max_scale", defaultValue = "6", source = ParamSource.BODY,
                   description = "Number of doubling steps per op (1-10); larger measures bigger inputs") int maxScale,
            @Param(value = "reset", defaultValue = "false", source = ParamSource.BODY,
                   description = "Discard previous calibration first") boolean reset) {
        if (Gpu.get() == null) return Response.err("GPU unavailable: " + Gpu.unavailableReason());
        if (reset) CostModel.resetCalibration();
        List<String> wanted = new ArrayList<>();
        if (ops == null || ops.isBlank()) CostModel.ops().forEach(wanted::add);
        else for (String s : ops.split(",")) if (!s.isBlank()) wanted.add(s.trim());
        return Response.ok(GpuBench.run(wanted, Math.max(1, Math.min(10, maxScale))));
    }

    @McpTool(path = "/gpu_advise", description = "Recommend cpu or gpu for an op (or gpu_* tool) at a given work size, "
        + "without running it. Explains the crossover and whether it was measured.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuAdvise(
            @Param(value = "op", description = "Op name (pattern_scan, pcode_vm, ...) or gpu_* tool name") String op,
            @Param(value = "work", defaultValue = "0", description = "Work size in the op's unit (see gpu_info); 0 = just show the crossover") long work) {
        String key = TOOL_TO_OP.getOrDefault(op, op);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("op", key);
        out.put("work_unit", CostModel.workUnit(key));
        long c = CostModel.crossover(key);
        out.put("gpu_from_work", c == Long.MAX_VALUE ? "never in measured range" : c);
        out.put("calibrated", CostModel.isCalibrated(key));
        out.put("gpu_available", Gpu.isAvailable());
        if (work > 0) {
            boolean gpu = Gpu.isAvailable() && CostModel.prefersGpu(key, work);
            out.put("recommendation", gpu ? "gpu" : "cpu");
            out.put("reason", !Gpu.isAvailable() ? "no GPU: " + Gpu.unavailableReason()
                : (gpu ? "work " + work + " >= crossover " + c : "work " + work + " < crossover " + c
                    + " (transfer + launch overhead dominates)"));
        }
        return Response.ok(out);
    }

    static final Map<String, String> TOOL_TO_OP = Map.ofEntries(
        Map.entry("gpu_search_byte_patterns", Ops.PATTERN_SCAN), Map.entry("gpu_list_strings", Ops.STRING_SCAN),
        Map.entry("gpu_entropy_map", Ops.ENTROPY), Map.entry("gpu_pointer_scan", Ops.POINTER_SCAN),
        Map.entry("gpu_detect_crypto_constants", Ops.PATTERN_SCAN), Map.entry("gpu_find_anti_analysis", Ops.PATTERN_SCAN),
        Map.entry("gpu_extract_iocs", Ops.IOC_SCAN), Map.entry("gpu_find_functions", Ops.SUBSTRING),
        Map.entry("gpu_search_instructions", Ops.SUBSTRING), Map.entry("gpu_function_hashes", Ops.SHA256),
        Map.entry("gpu_find_similar_functions", Ops.MINHASH_SIM), Map.entry("gpu_diff_functions", Ops.LEVENSHTEIN),
        Map.entry("gpu_call_graph_analysis", Ops.SCC), Map.entry("gpu_control_flow_analysis", Ops.DOMINATORS),
        Map.entry("gpu_find_function_starts", Ops.PATTERN_SCAN), Map.entry("gpu_dataflow_slice", Ops.BFS),
        Map.entry("gpu_decompile_batch", Ops.DOMINATORS),
        Map.entry("gpu_analysis_passes", Ops.POINTER_SCAN),
        Map.entry("gpu_emulate_hash_batch", Ops.PCODE_VM), Map.entry("gpu_emulate_function_batch", Ops.PCODE_VM));

    @McpTool(path = "/gpu_selftest", description = "Run every GPU kernel against its CPU twin on generated inputs "
        + "and report per-op agreement and timing. Use to confirm results are trustworthy on this GPU/driver.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuSelftest() {
        if (Gpu.get() == null) return Response.err("GPU unavailable: " + Gpu.unavailableReason());
        return Response.ok(GpuBench.selftest());
    }

    // =====================================================================
    // Memory scans
    // =====================================================================

    @McpTool(path = "/gpu_search_byte_patterns", description = "GPU twin of search_byte_patterns: search all "
        + "initialized memory for one or more hex patterns at once. '??' (or a mask) wildcards bytes.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuSearchBytePatterns(
            @Param(value = "patterns", description = "Hex patterns separated by ';' or newlines, e.g. '55 8B EC;E8 ?? ?? ?? ??'") String patterns,
            @Param(value = "exec_only", defaultValue = "false", description = "Only executable blocks") boolean execOnly,
            @Param(value = "block", defaultValue = "", description = "Only this memory block (name)") String block,
            @Param(value = "alignment", defaultValue = "1", description = "Only match at offsets that are multiples of this") int alignment,
            @Param(value = "limit", defaultValue = "1000", description = "Max hits returned (total is always counted)") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        try {
            List<byte[]> pats = new ArrayList<>(), masks = new ArrayList<>();
            List<String> labels = new ArrayList<>();
            for (String raw : patterns.split("[;\\n]")) {
                if (raw.isBlank()) continue;
                byte[][] pm = parsePattern(raw);
                pats.add(pm[0]);
                masks.add(pm[1]);
                labels.add(raw.trim());
            }
            if (pats.isEmpty()) return Response.err("no patterns given");
            return Response.ok(scanPatterns(pe.program(), pats, masks, labels, null, execOnly, block, alignment, limit,
                Backend.parse(backend)));
        } catch (IllegalArgumentException e) {
            return Response.err(e.getMessage());
        }
    }

    /** Runs a pattern scan over regions; shared by the pattern, crypto, anti-analysis and prologue tools. */
    private Map<String, Object> scanPatterns(Program p, List<byte[]> pats, List<byte[]> masks, List<String> labels,
                                             List<String> categories, boolean execOnly, String block, int alignment,
                                             int limit, Backend backend) {
        long t0 = System.nanoTime();
        ProgramData.Regions rs = ProgramData.regions(p, execOnly, block, maxBytes());
        double extractMs = ms(t0);
        long patBytes = 0;
        for (byte[] b : pats) patBytes += b.length;
        long work = rs.bytes() * Math.max(1, pats.size());
        int max = Math.max(1, limit);
        Dispatch.Outcome<List<ScanOps.Hits>> o = Dispatch.run(Ops.PATTERN_SCAN, backend, work,
            () -> { List<ScanOps.Hits> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.patternCpu(r.bytes(), pats, masks, alignment, max)); return l; },
            s -> { List<ScanOps.Hits> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.patternGpu(s, r.bytes(), pats, masks, alignment, max)); return l; },
            GpuService::sameHitLists);
        List<Map<String, Object>> hits = new ArrayList<>();
        int total = 0;
        int[] perPattern = new int[pats.size()];
        for (int ri = 0; ri < rs.regions().size(); ri++) {
            ProgramData.Region r = rs.regions().get(ri);
            ScanOps.Hits h = o.value().get(ri);
            total += h.total();
            for (int i = 0; i < h.size(); i++) {
                perPattern[h.tag()[i]]++;
                if (hits.size() >= max) continue;
                Address a = r.addressAt(h.pos()[i]);
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("address", a.toString());
                m.put("pattern", labels.get(h.tag()[i]));
                if (categories != null) m.put("category", categories.get(h.tag()[i]));
                m.put("block", r.name());
                Function f = p.getFunctionManager().getFunctionContaining(a);
                if (f != null) m.put("function", f.getName());
                hits.add(m);
            }
        }
        Map<String, Object> counts = new LinkedHashMap<>();
        for (int i = 0; i < pats.size(); i++) if (perPattern[i] > 0) counts.put(labels.get(i), perPattern[i]);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("total_hits", total);
        out.put("returned", hits.size());
        out.put("hits_per_pattern_returned", counts);
        out.put("hits", hits);
        out.put("scanned", rs.describe());
        out.put("execution", exec(o, extractMs));
        return out;
    }

    static byte[][] parsePattern(String raw) {
        String[] toks = raw.trim().replace(",", " ").split("\\s+");
        if (toks.length == 1 && toks[0].length() > 2) {
            String s = toks[0].replace("0x", "");
            toks = new String[s.length() / 2];
            for (int i = 0; i < toks.length; i++) toks[i] = s.substring(2 * i, 2 * i + 2);
        }
        byte[] p = new byte[toks.length], m = new byte[toks.length];
        for (int i = 0; i < toks.length; i++) {
            String t = toks[i];
            if (t.equals("??") || t.equals("?") || t.equals("**")) { p[i] = 0; m[i] = 0; continue; }
            if (t.length() != 2) throw new IllegalArgumentException("bad pattern byte '" + t + "' in '" + raw + "'");
            int hi = nib(t.charAt(0)), lo = nib(t.charAt(1));
            p[i] = (byte) (((hi < 0 ? 0 : hi) << 4) | (lo < 0 ? 0 : lo));
            m[i] = (byte) (((hi < 0 ? 0 : 0xF) << 4) | (lo < 0 ? 0 : 0xF));
        }
        return new byte[][] {p, m};
    }

    private static int nib(char c) {
        if (c == '?') return -1;
        int v = Character.digit(c, 16);
        if (v < 0) throw new IllegalArgumentException("bad hex digit '" + c + "'");
        return v;
    }

    @McpTool(path = "/gpu_list_strings", description = "GPU twin of list_strings / search_strings / "
        + "find_undocumented_by_string: extract ASCII and UTF-16LE strings from raw memory (defined or not), "
        + "optionally filtered by substring, with the containing function and referencing functions.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuListStrings(
            @Param(value = "min_length", defaultValue = "5", description = "Minimum characters") int minLength,
            @Param(value = "encoding", defaultValue = "both", description = "ascii | utf16 | both") String encoding,
            @Param(value = "filter", defaultValue = "", description = "Keep strings containing this (case-insensitive)") String filter,
            @Param(value = "include_xrefs", defaultValue = "false", description = "Add referencing functions per returned string") boolean includeXrefs,
            @Param(value = "offset", defaultValue = "0", description = "Pagination offset") int offset,
            @Param(value = "limit", defaultValue = "200", description = "Max strings returned") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        long t0 = System.nanoTime();
        ProgramData.Regions rs = ProgramData.regions(p, false, null, maxBytes());
        double extractMs = ms(t0);
        boolean doAscii = !encoding.equalsIgnoreCase("utf16"), doWide = !encoding.equalsIgnoreCase("ascii");
        int cap = 4_000_000;
        int minLen = Math.max(1, minLength);
        Dispatch.Outcome<List<ScanOps.Hits>> o = Dispatch.run(Ops.STRING_SCAN, be, rs.bytes() * ((doAscii ? 1 : 0) + (doWide ? 1 : 0)),
            () -> { List<ScanOps.Hits> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) {
                l.add(doAscii ? ScanOps.stringsCpu(r.bytes(), minLen, false, 4096, cap) : null);
                l.add(doWide ? ScanOps.stringsCpu(r.bytes(), minLen, true, 4096, cap) : null); } return l; },
            s -> { List<ScanOps.Hits> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) {
                l.add(doAscii ? ScanOps.stringsGpu(s, r.bytes(), minLen, false, 4096, cap) : null);
                l.add(doWide ? ScanOps.stringsGpu(s, r.bytes(), minLen, true, 4096, cap) : null); } return l; },
            GpuService::sameHitLists);

        // Materialise text, then (optionally) filter with a second op: substring search.
        List<String> texts = new ArrayList<>();
        List<Address> addrs = new ArrayList<>();
        List<String> enc = new ArrayList<>();
        for (int ri = 0; ri < rs.regions().size(); ri++) {
            ProgramData.Region r = rs.regions().get(ri);
            for (int w = 0; w < 2; w++) {
                ScanOps.Hits h = o.value().get(2 * ri + w);
                if (h == null) continue;
                for (int i = 0; i < h.size(); i++) {
                    int pos = (int) h.pos()[i], len = h.len()[i];
                    String t = w == 0 ? new String(r.bytes(), pos, len, StandardCharsets.ISO_8859_1)
                        : new String(r.bytes(), pos, 2 * len, StandardCharsets.UTF_16LE);
                    texts.add(t);
                    addrs.add(r.addressAt(pos));
                    enc.add(w == 0 ? "ascii" : "utf16");
                }
            }
        }
        Map<String, Object> filterExec = null;
        List<Integer> keep = new ArrayList<>();
        if (filter != null && !filter.isEmpty()) {
            ScanOps.Packed packed = ScanOps.Packed.of(texts);
            byte[] needle = filter.getBytes(StandardCharsets.UTF_8);
            Dispatch.Outcome<int[]> f = Dispatch.run(Ops.SUBSTRING, be, packed.bytes().length,
                () -> ScanOps.substringCpu(packed, needle, true), s -> ScanOps.substringGpu(s, packed, needle, true), Arrays::equals);
            for (int i = 0; i < texts.size(); i++) if (f.value()[i] >= 0) keep.add(i);
            filterExec = f.meta();
        } else {
            for (int i = 0; i < texts.size(); i++) keep.add(i);
        }
        keep.sort((x, y) -> addrs.get(x).compareTo(addrs.get(y)));
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int k = Math.max(0, offset); k < keep.size() && rows.size() < Math.max(1, limit); k++) {
            int i = keep.get(k);
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("address", addrs.get(i).toString());
            m.put("encoding", enc.get(i));
            m.put("length", texts.get(i).length());
            m.put("string", texts.get(i).length() > 512 ? texts.get(i).substring(0, 512) + "..." : texts.get(i));
            Data d = p.getListing().getDefinedDataAt(addrs.get(i));
            m.put("defined", d != null);
            if (includeXrefs) {
                Set<String> refs = new java.util.TreeSet<>();
                for (var ref : p.getReferenceManager().getReferencesTo(addrs.get(i))) {
                    Function fn = p.getFunctionManager().getFunctionContaining(ref.getFromAddress());
                    refs.add(fn != null ? fn.getName() + "@" + fn.getEntryPoint() : ref.getFromAddress().toString());
                    if (refs.size() >= 32) break;
                }
                m.put("referenced_from", new ArrayList<>(refs));
            }
            rows.add(m);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("total_strings_found", texts.size());
        out.put("matching", keep.size());
        out.put("offset", offset);
        out.put("returned", rows.size());
        out.put("strings", rows);
        out.put("scanned", rs.describe());
        Map<String, Object> ex = exec(o, extractMs);
        if (filterExec != null) ex.put("filter", filterExec);
        out.put("execution", ex);
        return Response.ok(out);
    }

    @McpTool(path = "/gpu_entropy_map", description = "GPU twin of analyze_data_region / inspect_memory_content at "
        + "program scale: windowed Shannon entropy over every block plus byte histograms; flags high-entropy "
        + "(packed/encrypted/compressed) and zero/low-entropy ranges.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuEntropyMap(
            @Param(value = "window", defaultValue = "4096", description = "Window size in bytes") int window,
            @Param(value = "high_threshold", defaultValue = "7.2", description = "Bits/byte at or above which a window is 'high'") double high,
            @Param(value = "block", defaultValue = "", description = "Only this memory block") String block,
            @Param(value = "max_ranges", defaultValue = "100", description = "Max high-entropy ranges returned") int maxRanges,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        int win = Math.max(64, window);
        long t0 = System.nanoTime();
        ProgramData.Regions rs = ProgramData.regions(pe.program(), false, block, maxBytes());
        double extractMs = ms(t0);
        Dispatch.Outcome<List<float[]>> o = Dispatch.run(Ops.ENTROPY, be, rs.bytes(),
            () -> { List<float[]> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.entropyCpu(r.bytes(), win)); return l; },
            s -> { List<float[]> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.entropyGpu(s, r.bytes(), win)); return l; },
            (a, b) -> { for (int i = 0; i < a.size(); i++) if (!ScanOps.sameFloats(a.get(i), b.get(i), 1e-4f)) return false; return true; });
        Dispatch.Outcome<List<long[]>> h = Dispatch.run(Ops.HISTOGRAM, be, rs.bytes(),
            () -> { List<long[]> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.histogramCpu(r.bytes())); return l; },
            s -> { List<long[]> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.histogramGpu(s, r.bytes())); return l; },
            (a, b) -> { for (int i = 0; i < a.size(); i++) if (!Arrays.equals(a.get(i), b.get(i))) return false; return true; });
        List<Map<String, Object>> blocks = new ArrayList<>();
        List<Map<String, Object>> ranges = new ArrayList<>();
        for (int ri = 0; ri < rs.regions().size(); ri++) {
            ProgramData.Region r = rs.regions().get(ri);
            float[] e = o.value().get(ri);
            long[] hist = h.value().get(ri);
            double sum = 0, mx = 0;
            int highCount = 0;
            for (float v : e) { sum += v; mx = Math.max(mx, v); if (v >= high) highCount++; }
            long zero = hist[0];
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("block", r.name());
            m.put("start", r.block().getStart().toString());
            m.put("size", r.bytes().length);
            m.put("executable", r.block().isExecute());
            m.put("mean_entropy", round3(e.length == 0 ? 0 : sum / e.length));
            m.put("max_entropy", round3(mx));
            m.put("whole_block_entropy", round3(entropyOf(hist, r.bytes().length)));
            m.put("high_entropy_windows", highCount);
            m.put("zero_byte_ratio", round3(r.bytes().length == 0 ? 0 : (double) zero / r.bytes().length));
            m.put("classification", classify(entropyOf(hist, r.bytes().length), (double) zero / Math.max(1, r.bytes().length)));
            blocks.add(m);
            int i = 0;
            while (i < e.length && ranges.size() < maxRanges) {
                if (e[i] < high) { i++; continue; }
                int j = i;
                double s = 0;
                while (j < e.length && e[j] >= high) s += e[j++];
                Map<String, Object> rr = new LinkedHashMap<>();
                rr.put("start", r.addressAt((long) i * win).toString());
                rr.put("end", r.addressAt(Math.min((long) j * win, r.bytes().length) - 1).toString());
                rr.put("bytes", Math.min((long) j * win, r.bytes().length) - (long) i * win);
                rr.put("mean_entropy", round3(s / (j - i)));
                rr.put("block", r.name());
                ranges.add(rr);
                i = j;
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("window", win);
        out.put("blocks", blocks);
        out.put("high_entropy_ranges", ranges);
        Map<String, Object> ex = exec(o, extractMs);
        ex.put("histogram", h.meta());
        out.put("execution", ex);
        return Response.ok(out);
    }

    static double entropyOf(long[] hist, long n) {
        if (n == 0) return 0;
        double e = 0;
        for (long c : hist) if (c > 0) { double q = (double) c / n; e -= q * Math.log(q) / Math.log(2); }
        return e;
    }

    static String classify(double entropy, double zeroRatio) {
        if (zeroRatio > 0.9) return "mostly zero / padding";
        if (entropy >= 7.5) return "encrypted or compressed";
        if (entropy >= 6.5) return "code or packed data";
        if (entropy >= 4.0) return "code / structured data";
        return "text or sparse data";
    }

    @McpTool(path = "/gpu_pointer_scan", description = "GPU twin of get_xrefs_to for raw data: find every aligned "
        + "pointer-sized value in memory that points at the given target addresses (or anywhere in the program), "
        + "including references Ghidra has not created. Finds vtables, jump tables, callback arrays, string tables.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuPointerScan(
            @Param(value = "targets", defaultValue = "", description = "Comma-separated target addresses; empty = any value inside a memory block") String targets,
            @Param(value = "pointer_size", defaultValue = "0", description = "4 or 8; 0 = the program's pointer size") int pointerSize,
            @Param(value = "alignment", defaultValue = "0", description = "Scan stride; 0 = pointer size") int alignment,
            @Param(value = "block", defaultValue = "", description = "Only scan this memory block") String block,
            @Param(value = "limit", defaultValue = "1000", description = "Max hits returned") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        int ps = pointerSize == 4 || pointerSize == 8 ? pointerSize : p.getDefaultPointerSize();
        int al = alignment > 0 ? alignment : ps;
        long[] t = null;
        long lo = 0, hi = -1L;
        if (targets != null && !targets.isBlank()) {
            List<Long> l = new ArrayList<>();
            for (String s : targets.split(",")) {
                if (s.isBlank()) continue;
                Address a = ServiceUtils.parseAddress(p, s.trim());
                if (a == null) return Response.err("bad target address '" + s + "': " + ServiceUtils.getLastParseError());
                l.add(a.getOffset());
            }
            t = l.stream().mapToLong(Long::longValue).toArray();
        } else {
            lo = p.getMinAddress().getOffset();
            hi = p.getMaxAddress().getOffset() + 1;
        }
        long[] tt = t;
        long flo = lo, fhi = hi;
        boolean bigEndian = p.getLanguage().isBigEndian();
        long t0 = System.nanoTime();
        ProgramData.Regions rs = ProgramData.regions(p, false, block, maxBytes());
        double extractMs = ms(t0);
        int max = Math.max(1, limit);
        Dispatch.Outcome<List<ScanOps.Hits>> o = Dispatch.run(Ops.POINTER_SCAN, be, rs.bytes(),
            () -> { List<ScanOps.Hits> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.pointersCpu(r.bytes(), ps, al, bigEndian, flo, fhi, tt, max)); return l; },
            s -> { List<ScanOps.Hits> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.pointersGpu(s, r.bytes(), ps, al, bigEndian, flo, fhi, tt, max)); return l; },
            GpuService::sameHitLists);
        List<Map<String, Object>> hits = new ArrayList<>();
        int total = 0;
        Map<String, Integer> perTarget = new TreeMap<>();
        for (int ri = 0; ri < rs.regions().size(); ri++) {
            ProgramData.Region r = rs.regions().get(ri);
            ScanOps.Hits h = o.value().get(ri);
            total += h.total();
            for (int i = 0; i < h.size(); i++) {
                long v = ScanOps.pointerValue(h, i);
                Address target = p.getAddressFactory().getDefaultAddressSpace().getAddress(v);
                if (tt == null && p.getMemory().getBlock(target) == null) { total--; continue; }
                perTarget.merge(target.toString(), 1, Integer::sum);
                if (hits.size() >= max) continue;
                Address at = r.addressAt(h.pos()[i]);
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("address", at.toString());
                m.put("points_to", target.toString());
                Symbol sym = p.getSymbolTable().getPrimarySymbol(target);
                if (sym != null) m.put("target_symbol", sym.getName());
                m.put("existing_reference", p.getReferenceManager().getReference(at, target, 0) != null
                    || p.getReferenceManager().getReferencesFrom(at).length > 0);
                m.put("block", r.name());
                hits.add(m);
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("pointer_size", ps);
        out.put("alignment", al);
        out.put("total_hits", total);
        out.put("returned", hits.size());
        if (tt != null) out.put("hits_per_target_returned", perTarget);
        out.put("hits", hits);
        out.put("scanned", rs.describe());
        out.put("execution", exec(o, extractMs));
        return Response.ok(out);
    }

    // =====================================================================
    // Signatures: crypto constants, anti-analysis, IOCs, function starts
    // =====================================================================

    /** {name, category, hex}. Little- and big-endian forms of word constants are both listed. */
    static final String[][] CRYPTO_SIGS = {
        {"AES S-box", "AES", "63 7c 77 7b f2 6b 6f c5 30 01 67 2b fe d7 ab 76"},
        {"AES inverse S-box", "AES", "52 09 6a d5 30 36 a5 38 bf 40 a3 9e 81 f3 d7 fb"},
        {"AES Te0 table (LE)", "AES", "a5 63 63 c6 84 7c 7c f8"},
        {"AES Te0 table (BE)", "AES", "c6 63 63 a5 f8 7c 7c 84"},
        {"SHA-256 K (LE)", "SHA-256", "98 2f 8a 42 91 44 37 71 cf fb c0 b5"},
        {"SHA-256 K (BE)", "SHA-256", "42 8a 2f 98 71 37 44 91 b5 c0 fb cf"},
        {"SHA-256 initial H (LE)", "SHA-256", "67 e6 09 6a 85 ae 67 bb"},
        {"SHA-512 K (LE)", "SHA-512", "22 ae 28 d7 98 2f 8a 42"},
        {"SHA-512 K (BE)", "SHA-512", "42 8a 2f 98 d7 28 ae 22"},
        {"MD5/SHA-1 initial state (LE)", "MD5/SHA-1", "01 23 45 67 89 ab cd ef fe dc ba 98 76 54 32 10"},
        {"SHA-1 H4 (LE)", "SHA-1", "f0 e1 d2 c3"},
        {"MD5 T table (LE)", "MD5", "78 a4 6a d7 56 b7 c7 e8"},
        {"CRC-32 table (LE)", "CRC-32", "00 00 00 00 96 30 07 77 2c 61 0e ee"},
        {"CRC-32 reversed polynomial (LE)", "CRC-32", "20 83 b8 ed"},
        {"Blowfish P-array (LE)", "Blowfish", "88 6a 3f 24 d3 08 a3 85"},
        {"Blowfish P-array (BE)", "Blowfish", "24 3f 6a 88 85 a3 08 d3"},
        {"ChaCha/Salsa sigma 'expand 32-byte k'", "ChaCha20/Salsa20", "65 78 70 61 6e 64 20 33 32 2d 62 79 74 65 20 6b"},
        {"ChaCha/Salsa tau 'expand 16-byte k'", "ChaCha20/Salsa20", "65 78 70 61 6e 64 20 31 36 2d 62 79 74 65 20 6b"},
        {"TEA/XTEA delta (LE)", "TEA/XTEA", "b9 79 37 9e"},
        {"TEA/XTEA delta sum*32 (LE)", "TEA/XTEA", "20 37 ef c6"},
        {"RC5/RC6 P32 (LE)", "RC5/RC6", "63 51 e1 b7"},
        {"Murmur3 c1 (LE)", "MurmurHash3", "51 2d 9e cc"},
        {"Base64 alphabet", "Base64", "41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f 50 51 52 53 54 55 56 57 58 59 5a 61 62 63 64"},
        {"DES initial permutation table", "DES", "3a 32 2a 22 1a 12 0a 02 3c 34 2c 24"},
        {"Twofish q0 table", "Twofish", "a9 67 b3 e8 04 fd a3 76"},
        {"Camellia sigma1 (BE)", "Camellia", "a0 9e 66 7f 3b cc 90 8b"},
        {"Whirlpool S-box", "Whirlpool", "18 23 c6 e8 87 b8 01 4f"},
        {"RIPEMD-160 K (LE)", "RIPEMD-160", "99 79 82 5a a1 eb d9 6e"},
        {"Serpent golden ratio (LE)", "Serpent", "b9 79 37 9e"},
    };

    @McpTool(path = "/gpu_detect_crypto_constants", description = "GPU twin of detect_crypto_constants (upstream is "
        + "a placeholder; this one works): scan all memory for AES, SHA-1/2, MD5, CRC-32, Blowfish, ChaCha/Salsa, "
        + "TEA, RC5/6, DES, Twofish, Camellia, Whirlpool, RIPEMD, Base64 and other constant tables.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuDetectCryptoConstants(
            @Param(value = "limit", defaultValue = "500", description = "Max hits returned") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        try {
            return Response.ok(signatureScan(pe.program(), CRYPTO_SIGS, false, limit, Backend.parse(backend)));
        } catch (IllegalArgumentException e) {
            return Response.err(e.getMessage());
        }
    }

    static final String[][] ANTI_ANALYSIS_SIGS = {
        {"rdtsc (timing check)", "timing", "0f 31"},
        {"rdtscp (timing check)", "timing", "0f 01 f9"},
        {"cpuid (VM / hypervisor detection)", "vm_detection", "0f a2"},
        {"int 2Dh (debugger trap)", "debugger_detection", "cd 2d"},
        {"int 3 + ret pattern", "debugger_detection", "cc c3"},
        {"PEB access mov eax,fs:[30h]", "debugger_detection", "64 a1 30 00 00 00"},
        {"PEB access mov ecx,fs:[30h]", "debugger_detection", "64 8b 0d 30 00 00 00"},
        {"PEB access mov edx,fs:[30h]", "debugger_detection", "64 8b 15 30 00 00 00"},
        {"PEB access mov rax,gs:[60h]", "debugger_detection", "65 48 8b 04 25 60 00 00 00"},
        {"VMware backdoor magic 'VMXh'", "vm_detection", "58 68 4d 56"},
        {"VMware backdoor port 'VX'", "vm_detection", "66 ba 58 56"},
        {"sidt (red pill)", "vm_detection", "0f 01 0d"},
        {"sgdt", "vm_detection", "0f 01 05"},
        {"sldt", "vm_detection", "0f 00 05"},
        {"str", "vm_detection", "0f 00 0d"},
        {"IsDebuggerPresent", "debugger_detection", hexOf("IsDebuggerPresent")},
        {"CheckRemoteDebuggerPresent", "debugger_detection", hexOf("CheckRemoteDebuggerPresent")},
        {"NtQueryInformationProcess", "debugger_detection", hexOf("NtQueryInformationProcess")},
        {"NtSetInformationThread", "debugger_detection", hexOf("NtSetInformationThread")},
        {"OutputDebugString", "debugger_detection", hexOf("OutputDebugString")},
        {"GetTickCount", "timing", hexOf("GetTickCount")},
        {"QueryPerformanceCounter", "timing", hexOf("QueryPerformanceCounter")},
        {"VBox", "vm_detection", hexOf("VBox")},
        {"VMware", "vm_detection", hexOf("VMware")},
        {"vmtoolsd", "vm_detection", hexOf("vmtoolsd")},
        {"SbieDll.dll (Sandboxie)", "sandbox_detection", hexOf("SbieDll")},
        {"wine_get_unix_file_name", "sandbox_detection", hexOf("wine_get_unix_file_name")},
        {"ollydbg", "analysis_tool_detection", hexOf("ollydbg")},
        {"x64dbg", "analysis_tool_detection", hexOf("x64dbg")},
        {"wireshark", "analysis_tool_detection", hexOf("wireshark")},
        {"procmon", "analysis_tool_detection", hexOf("procmon")},
        {"VirtualProtect", "code_injection", hexOf("VirtualProtect")},
        {"WriteProcessMemory", "code_injection", hexOf("WriteProcessMemory")},
        {"CreateRemoteThread", "code_injection", hexOf("CreateRemoteThread")},
        {"SetWindowsHookEx", "hooking", hexOf("SetWindowsHookEx")},
    };

    static String hexOf(String s) {
        StringBuilder sb = new StringBuilder();
        for (byte b : s.getBytes(StandardCharsets.US_ASCII)) sb.append(String.format("%02x ", b));
        return sb.toString().trim();
    }

    @McpTool(path = "/gpu_find_anti_analysis", description = "GPU twin of find_anti_analysis_techniques / "
        + "detect_malware_behaviors: one pass over all memory for anti-debug, anti-VM, timing and sandbox "
        + "instruction encodings and API/tool name strings, grouped by category with containing functions.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuFindAntiAnalysis(
            @Param(value = "limit", defaultValue = "500", description = "Max hits returned") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        try {
            return Response.ok(signatureScan(pe.program(), ANTI_ANALYSIS_SIGS, false, limit, Backend.parse(backend)));
        } catch (IllegalArgumentException e) {
            return Response.err(e.getMessage());
        }
    }

    private Map<String, Object> signatureScan(Program p, String[][] sigs, boolean execOnly, int limit, Backend be) {
        List<byte[]> pats = new ArrayList<>(), masks = new ArrayList<>();
        List<String> labels = new ArrayList<>(), cats = new ArrayList<>();
        for (String[] s : sigs) {
            byte[][] pm = parsePattern(s[2]);
            pats.add(pm[0]);
            masks.add(pm[1]);
            labels.add(s[0]);
            cats.add(s[1]);
        }
        Map<String, Object> out = scanPatterns(p, pats, masks, labels, cats, execOnly, null, 1, limit, be);
        Map<String, Integer> byCat = new TreeMap<>();
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> hits = (List<Map<String, Object>>) out.get("hits");
        for (Map<String, Object> h : hits) byCat.merge(String.valueOf(h.get("category")), 1, Integer::sum);
        out.put("returned_by_category", byCat);
        return out;
    }

    @McpTool(path = "/gpu_extract_iocs", description = "GPU twin of extract_iocs_with_context: scan raw memory "
        + "(not only defined strings) for URLs, IPv4 literals, e-mail addresses and registry paths, with the "
        + "containing/referencing function for context.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuExtractIocs(
            @Param(value = "types", defaultValue = "", description = "Comma-separated subset of url,ipv4,email,registry; empty = all") String types,
            @Param(value = "limit", defaultValue = "500", description = "Max IOCs returned") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        Set<String> want = new HashSet<>();
        if (types != null && !types.isBlank()) for (String t : types.split(",")) want.add(t.trim().toLowerCase(Locale.ROOT));
        long t0 = System.nanoTime();
        ProgramData.Regions rs = ProgramData.regions(p, false, null, maxBytes());
        double extractMs = ms(t0);
        int cap = 1 << 20;
        Dispatch.Outcome<List<ScanOps.Hits>> o = Dispatch.run(Ops.IOC_SCAN, be, rs.bytes(),
            () -> { List<ScanOps.Hits> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.iocCpu(r.bytes(), 1024, cap)); return l; },
            s -> { List<ScanOps.Hits> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.iocGpu(s, r.bytes(), 1024, cap)); return l; },
            GpuService::sameHitLists);
        Map<String, List<Map<String, Object>>> byType = new LinkedHashMap<>();
        Map<String, Set<String>> unique = new LinkedHashMap<>();
        int returned = 0;
        for (int ri = 0; ri < rs.regions().size(); ri++) {
            ProgramData.Region r = rs.regions().get(ri);
            ScanOps.Hits h = o.value().get(ri);
            for (int i = 0; i < h.size(); i++) {
                String type = ScanOps.IOC_TYPES[h.tag()[i]];
                if (!want.isEmpty() && !want.contains(type)) continue;
                String value = new String(r.bytes(), (int) h.pos()[i], h.len()[i], StandardCharsets.ISO_8859_1);
                unique.computeIfAbsent(type, k -> new java.util.TreeSet<>()).add(value);
                if (returned >= Math.max(1, limit)) continue;
                Address a = r.addressAt(h.pos()[i]);
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("value", value);
                m.put("address", a.toString());
                Function f = p.getFunctionManager().getFunctionContaining(a);
                if (f != null) m.put("in_function", f.getName());
                List<String> refs = new ArrayList<>();
                for (var ref : p.getReferenceManager().getReferencesTo(a)) {
                    Function rf = p.getFunctionManager().getFunctionContaining(ref.getFromAddress());
                    refs.add(rf != null ? rf.getName() : ref.getFromAddress().toString());
                    if (refs.size() >= 8) break;
                }
                if (!refs.isEmpty()) m.put("referenced_from", refs);
                byType.computeIfAbsent(type, k -> new ArrayList<>()).add(m);
                returned++;
            }
        }
        Map<String, Object> counts = new LinkedHashMap<>();
        unique.forEach((k, v) -> counts.put(k, v.size()));
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("unique_counts", counts);
        out.put("returned", returned);
        out.put("iocs", byType);
        out.put("scanned", rs.describe());
        out.put("execution", exec(o, extractMs));
        return Response.ok(out);
    }

    /** {name, hex pattern, processor substring, alignment}. */
    static final String[][] PROLOGUES = {
        {"x86 push ebp; mov ebp,esp", "55 8b ec", "x86", "1"},
        {"x86 push ebp; mov ebp,esp (gas)", "55 89 e5", "x86", "1"},
        {"x86 hotpatch mov edi,edi; push ebp", "8b ff 55 8b ec", "x86", "1"},
        {"x64 push rbp; mov rbp,rsp", "55 48 89 e5", "x86", "1"},
        {"x64 sub rsp,imm8", "48 83 ec ??", "x86", "1"},
        {"x64 mov [rsp+8],rbx", "48 89 5c 24 ??", "x86", "1"},
        {"x64 push rbx; sub rsp", "40 53 48 83 ec", "x86", "1"},
        {"x64 endbr64", "f3 0f 1e fa", "x86", "1"},
        {"ARM Thumb push {..,lr}", "?? b5", "ARM", "2"},
        {"ARM push {..,lr}", "?? ?? 2d e9", "ARM", "4"},
        {"AArch64 stp x29,x30,[sp,#-n]!", "fd 7b ?? a9", "AARCH64", "4"},
        {"AArch64 paciasp", "3f 23 03 d5", "AARCH64", "4"},
        {"MIPS addiu sp,sp,-n (LE)", "?? ff bd 27", "MIPS", "4"},
        {"MIPS addiu sp,sp,-n (BE)", "27 bd ff ??", "MIPS", "4"},
        {"PowerPC stwu r1,-n(r1)", "94 21 ff ??", "PowerPC", "4"},
    };

    @McpTool(path = "/gpu_find_function_starts", description = "GPU twin of find_code_gaps / "
        + "find_next_undefined_function: scan executable memory for this architecture's function prologues "
        + "and report the ones not inside any function or defined data -- likely missed functions.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuFindFunctionStarts(
            @Param(value = "limit", defaultValue = "500", description = "Max candidates returned") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        String proc = p.getLanguage().getProcessor().toString();
        List<byte[]> pats = new ArrayList<>(), masks = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        int align = 1;
        for (String[] s : PROLOGUES) {
            if (!proc.toUpperCase(Locale.ROOT).contains(s[2].toUpperCase(Locale.ROOT))
                || (s[2].equals("ARM") && proc.toUpperCase(Locale.ROOT).contains("AARCH64"))) continue;
            byte[][] pm = parsePattern(s[1]);
            pats.add(pm[0]);
            masks.add(pm[1]);
            labels.add(s[0]);
            align = Math.max(align, Integer.parseInt(s[3]));
        }
        if (pats.isEmpty()) return Response.err("no prologue signatures for processor " + proc);
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        Map<String, Object> scan = scanPatterns(p, pats, masks, labels, null, true, null,
            proc.toUpperCase(Locale.ROOT).contains("ARM") ? 2 : (align > 1 ? align : 1), 1_000_000, be);
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> hits = (List<Map<String, Object>>) scan.get("hits");
        List<Map<String, Object>> cands = new ArrayList<>();
        int outside = 0;
        for (Map<String, Object> h : hits) {
            if (h.containsKey("function")) continue;
            Address a = p.getAddressFactory().getAddress((String) h.get("address"));
            if (a == null) continue;
            if (p.getListing().getDefinedDataContaining(a) != null) continue;
            Instruction ins = p.getListing().getInstructionContaining(a);
            if (ins != null && !ins.getAddress().equals(a)) continue; // mid-instruction false positive
            outside++;
            if (cands.size() < Math.max(1, limit)) {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("address", a.toString());
                m.put("pattern", h.get("pattern"));
                m.put("disassembled", ins != null);
                cands.add(m);
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("processor", proc);
        out.put("prologue_hits_total", scan.get("total_hits"));
        out.put("candidates_outside_functions", outside);
        out.put("candidates", cands);
        out.put("note", "Create functions at good candidates with create_function; hits are heuristics.");
        out.put("execution", scan.get("execution"));
        return Response.ok(out);
    }

    // =====================================================================
    // Auto-analysis passes
    // =====================================================================

    @McpTool(path = "/gpu_analysis_passes", method = "POST", description = "GPU twin of the data-parallel passes "
        + "of run_analysis / reanalyze: (1) scalar-operand references -- every instruction scalar that is an "
        + "address inside a memory block and has no reference yet (the Scalar Operand References analyzer); "
        + "(2) address tables -- runs of consecutive pointers into the program (Create Address Tables). "
        + "Function starts and strings, the other data-parallel passes, are gpu_find_function_starts and "
        + "gpu_list_strings. apply=true creates the scalar references (source ANALYSIS) in one transaction. "
        + "The decompiler-driven analyzers (Parameter ID, Switch Analysis) stay on the CPU.",
        category = "gpu", access = ToolAccess.WRITE)
    public Response gpuAnalysisPasses(
            @Param(value = "min_table_entries", source = ParamSource.BODY, defaultValue = "4",
                   description = "Minimum consecutive pointers reported as an address table") int minTable,
            @Param(value = "apply", source = ParamSource.BODY, defaultValue = "false",
                   description = "Create the found scalar-operand references") boolean apply,
            @Param(value = "include_existing", source = ParamSource.BODY, defaultValue = "false",
                   description = "Also check operands that already have a reference, and report how many of those "
                               + "the pass finds with the same target (validation against Ghidra's own analysis)") boolean includeExisting,
            @Param(value = "limit", source = ParamSource.BODY, defaultValue = "200", description = "Max rows listed per pass") int limit,
            @Param(value = "backend", source = ParamSource.BODY, defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        boolean bigE = p.getLanguage().isBigEndian();
        long lo = p.getMinAddress().getOffset(), hi = p.getMaxAddress().getOffset() + 1;
        var mem = p.getMemory();
        var space = p.getAddressFactory().getDefaultAddressSpace();

        // Pass 1: scalar operands. Each candidate scalar becomes 8 bytes of a flat array so the
        // pointer-scan kernel range-checks all of them in one launch.
        long t0 = System.nanoTime();
        List<Address> fromAddr = new ArrayList<>();
        List<Integer> opIndex = new ArrayList<>();
        List<Address> existingTo = new ArrayList<>();
        java.io.ByteArrayOutputStream packed = new java.io.ByteArrayOutputStream();
        var insIt = p.getListing().getInstructions(true);
        while (insIt.hasNext()) {
            Instruction ins = insIt.next();
            for (int op = 0; op < ins.getNumOperands(); op++) {
                var opRefs = ins.getOperandReferences(op);
                if (opRefs.length > 0 && !includeExisting) continue;
                for (Object o : ins.getOpObjects(op)) {
                    if (!(o instanceof ghidra.program.model.scalar.Scalar sc)) continue;
                    long v = sc.getUnsignedValue();
                    for (int k = 0; k < 8; k++) packed.write((int) (v >>> (8 * k)));
                    fromAddr.add(ins.getAddress());
                    opIndex.add(op);
                    existingTo.add(opRefs.length > 0 ? opRefs[0].getToAddress() : null);
                }
            }
        }
        byte[] scalars = packed.toByteArray();
        ProgramData.Regions rs = ProgramData.regions(p, false, null, maxBytes());
        double extractMs = ms(t0);
        int cap = Math.max(1, scalars.length / 8);
        Dispatch.Outcome<ScanOps.Hits> sc = Dispatch.run(Ops.POINTER_SCAN, be, scalars.length,
            () -> ScanOps.pointersCpu(scalars, 8, 8, false, lo, hi, null, cap),
            s -> ScanOps.pointersGpu(s, scalars, 8, 8, false, lo, hi, null, cap), ScanOps.Hits::same);
        List<Map<String, Object>> refs = new ArrayList<>();
        List<Object[]> toCreate = new ArrayList<>();
        int matchedExisting = 0, existingTotal = 0;
        for (Address e : existingTo) if (e != null && e.getAddressSpace().equals(space)) existingTotal++;
        for (int i = 0; i < sc.value().size(); i++) {
            int idx = (int) (sc.value().pos()[i] / 8);
            Address to = space.getAddress(ScanOps.pointerValue(sc.value(), i));
            if (!mem.contains(to)) continue;
            if (existingTo.get(idx) != null) {
                if (existingTo.get(idx).equals(to)) matchedExisting++;
                continue;
            }
            toCreate.add(new Object[] {fromAddr.get(idx), to, opIndex.get(idx)});
            if (refs.size() < Math.max(1, limit)) {
                Symbol sym = p.getSymbolTable().getPrimarySymbol(to);
                refs.add(JsonHelper.mapOf("from", fromAddr.get(idx).toString(), "operand", opIndex.get(idx),
                    "to", to.toString(), "to_symbol", sym == null ? null : sym.getName()));
            }
        }

        // Pass 2: address tables -- runs of aligned pointers into memory, from one pointer scan.
        int ps = p.getDefaultPointerSize();
        Dispatch.Outcome<List<ScanOps.Hits>> pt = Dispatch.run(Ops.POINTER_SCAN, be, rs.bytes(),
            () -> { List<ScanOps.Hits> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.pointersCpu(r.bytes(), ps, ps, bigE, lo, hi, null, 1 << 22)); return l; },
            s -> { List<ScanOps.Hits> l = new ArrayList<>(); for (ProgramData.Region r : rs.regions()) l.add(ScanOps.pointersGpu(s, r.bytes(), ps, ps, bigE, lo, hi, null, 1 << 22)); return l; },
            GpuService::sameHitLists);
        List<Map<String, Object>> tables = new ArrayList<>();
        int tableCount = 0;
        for (int ri = 0; ri < rs.regions().size(); ri++) {
            ProgramData.Region r = rs.regions().get(ri);
            ScanOps.Hits h = pt.value().get(ri);
            int i = 0;
            while (i < h.size()) {
                int j = i;
                while (j < h.size() && mem.contains(space.getAddress(ScanOps.pointerValue(h, j)))
                       && (j == i || h.pos()[j] == h.pos()[j - 1] + ps)) j++;
                int run = j - i;
                if (run >= Math.max(2, minTable)) {
                    tableCount++;
                    if (tables.size() < Math.max(1, limit)) {
                        Address start = r.addressAt(h.pos()[i]);
                        tables.add(JsonHelper.mapOf("address", start.toString(), "entries", run, "block", r.name(),
                            "first_target", space.getAddress(ScanOps.pointerValue(h, i)).toString(),
                            "defined", p.getListing().getDefinedDataAt(start) != null));
                    }
                }
                i = Math.max(j, i + 1);
            }
        }

        int created = 0;
        if (apply && !toCreate.isEmpty()) {
            try {
                Response w = threadingStrategy.executeWrite(p, "GPU scalar operand references", () -> {
                    var rm = p.getReferenceManager();
                    int n = 0;
                    for (Object[] c : toCreate) {
                        if (rm.addMemoryReference((Address) c[0], (Address) c[1], ghidra.program.model.symbol.RefType.DATA,
                            ghidra.program.model.symbol.SourceType.ANALYSIS, (Integer) c[2]) != null) n++;
                    }
                    return Response.ok(n);
                });
                if (w instanceof Response.Ok ok && ok.data() instanceof Integer n) created = n;
            } catch (Exception e) {
                return Response.err("applying references failed: " + e.getMessage());
            }
        }
        Map<String, Object> ex = new LinkedHashMap<>();
        ex.put("extract_ms", extractMs);
        ex.put("scalar_references", sc.meta());
        ex.put("address_tables", pt.meta());
        if (includeExisting) {
            ex.put("validation", JsonHelper.mapOf("existing_memory_references_on_scalar_operands", existingTotal,
                "found_by_this_pass_with_same_target", matchedExisting));
        }
        return Response.ok(JsonHelper.mapOf("scalars_checked", scalars.length / 8, "scalar_references_found", toCreate.size(),
            "scalar_references_created", created, "scalar_references", refs, "address_tables_found", tableCount,
            "address_tables", tables, "execution", ex));
    }

    // =====================================================================
    // Names and instructions
    // =====================================================================

    @McpTool(path = "/gpu_find_functions", description = "GPU twin of find_functions: substring search over every "
        + "function name (and optionally every symbol name) at once.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuFindFunctions(
            @Param(value = "query", description = "Substring to find") String query,
            @Param(value = "case_sensitive", defaultValue = "false", description = "Match case") boolean caseSensitive,
            @Param(value = "all_symbols", defaultValue = "false", description = "Search every symbol, not just functions") boolean allSymbols,
            @Param(value = "limit", defaultValue = "200", description = "Max matches returned") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        if (query == null || query.isEmpty()) return Response.err("query is required");
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        long t0 = System.nanoTime();
        List<String> names = new ArrayList<>();
        List<String> where = new ArrayList<>();
        List<String> kinds = new ArrayList<>();
        if (allSymbols) {
            for (Symbol s : p.getSymbolTable().getAllSymbols(true)) {
                names.add(s.getName(true));
                where.add(s.getAddress().toString());
                kinds.add(s.getSymbolType().toString());
            }
        } else {
            for (Function f : ProgramData.functions(p)) {
                names.add(f.getName(true));
                where.add(f.getEntryPoint().toString());
                kinds.add("Function");
            }
        }
        ScanOps.Packed packed = ScanOps.Packed.of(names);
        double extractMs = ms(t0);
        byte[] needle = query.getBytes(StandardCharsets.UTF_8);
        Dispatch.Outcome<int[]> o = Dispatch.run(Ops.SUBSTRING, be, packed.bytes().length,
            () -> ScanOps.substringCpu(packed, needle, !caseSensitive),
            s -> ScanOps.substringGpu(s, packed, needle, !caseSensitive), Arrays::equals);
        List<Map<String, Object>> rows = new ArrayList<>();
        int total = 0;
        for (int i = 0; i < names.size(); i++) {
            if (o.value()[i] < 0) continue;
            total++;
            if (rows.size() < Math.max(1, limit)) rows.add(JsonHelper.mapOf("name", names.get(i), "address", where.get(i), "type", kinds.get(i)));
        }
        return Response.ok(JsonHelper.mapOf("searched", names.size(), "total_matches", total, "returned", rows.size(),
            "matches", rows, "execution", exec(o, extractMs)));
    }

    @McpTool(path = "/gpu_search_instructions", description = "GPU twin of search_instructions: substring search "
        + "over the text of every instruction (e.g. 'ecx+0xd0', 'call dword ptr', 'xor eax,eax'), optional mnemonic filter.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuSearchInstructions(
            @Param(value = "query", description = "Substring of the instruction text as Ghidra prints it") String query,
            @Param(value = "mnemonic", defaultValue = "", description = "Only instructions with this mnemonic") String mnemonic,
            @Param(value = "max_instructions", defaultValue = "5000000", description = "Cap on instructions considered") int maxInstructions,
            @Param(value = "limit", defaultValue = "200", description = "Max matches returned") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        if (query == null || query.isEmpty()) return Response.err("query is required");
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        long t0 = System.nanoTime();
        ProgramData.InstructionText it = ProgramData.instructionText(p, null, Math.max(1, maxInstructions));
        double extractMs = ms(t0);
        byte[] needle = query.getBytes(StandardCharsets.UTF_8);
        Dispatch.Outcome<int[]> o = Dispatch.run(Ops.SUBSTRING, be, it.packed().bytes().length,
            () -> ScanOps.substringCpu(it.packed(), needle, true),
            s -> ScanOps.substringGpu(s, it.packed(), needle, true), Arrays::equals);
        List<Map<String, Object>> rows = new ArrayList<>();
        int total = 0;
        String mn = mnemonic == null ? "" : mnemonic.trim().toUpperCase(Locale.ROOT);
        for (int i = 0; i < it.text().size(); i++) {
            if (o.value()[i] < 0) continue;
            String text = it.text().get(i);
            if (!mn.isEmpty() && !text.toUpperCase(Locale.ROOT).startsWith(mn + " ") && !text.equalsIgnoreCase(mn)) continue;
            total++;
            if (rows.size() >= Math.max(1, limit)) continue;
            Address a = it.addresses().get(i);
            Function f = p.getFunctionManager().getFunctionContaining(a);
            rows.add(JsonHelper.mapOf("address", a.toString(), "instruction", text, "function", f == null ? null : f.getName()));
        }
        return Response.ok(JsonHelper.mapOf("instructions_searched", it.text().size(), "total_matches", total,
            "returned", rows.size(), "matches", rows, "execution", exec(o, extractMs)));
    }

    // =====================================================================
    // Hashing, similarity, diffing
    // =====================================================================

    @McpTool(path = "/gpu_function_hashes", description = "GPU twin of get_function_hash at program scale: "
        + "SHA-256 of every function's normalized instructions (kind=normalized, byte-identical to "
        + "get_function_hash) or of its opcode/operand-type stream (kind=opcode, byte-identical to the "
        + "opcode_hash archive_ingest_program computes), plus groups of identical functions.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuFunctionHashes(
            @Param(value = "filter", defaultValue = "", description = "Only functions whose name contains this") String filter,
            @Param(value = "kind", defaultValue = "normalized", description = "normalized (get_function_hash) | opcode (archive opcode_hash)") String kind,
            @Param(value = "limit", defaultValue = "500", description = "Max hashes listed (duplicate groups are always complete)") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        boolean opcode = "opcode".equalsIgnoreCase(kind);
        if (!opcode && !"normalized".equalsIgnoreCase(kind)) return Response.err("kind must be normalized or opcode");
        long t0 = System.nanoTime();
        List<Function> fns = new ArrayList<>();
        List<byte[]> texts = new ArrayList<>();
        for (Function f : ProgramData.functions(p)) {
            if (f.isThunk() || f.isExternal()) continue;
            if (filter != null && !filter.isEmpty() && !f.getName().toLowerCase(Locale.ROOT).contains(filter.toLowerCase(Locale.ROOT))) continue;
            fns.add(f);
            texts.add(opcode ? ProgramData.opcodeText(p, f) : ProgramData.normalizedText(p, f));
        }
        ScanOps.Packed packed = packBytes(texts);
        double extractMs = ms(t0);
        Dispatch.Outcome<byte[]> o = Dispatch.run(Ops.SHA256, be, packed.bytes().length,
            () -> HashOps.sha256Cpu(packed), s -> HashOps.sha256Gpu(s, packed), Arrays::equals);
        Map<String, List<String>> groups = new LinkedHashMap<>();
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int i = 0; i < fns.size(); i++) {
            String h = HashOps.hex(o.value(), i);
            groups.computeIfAbsent(h, k -> new ArrayList<>()).add(fns.get(i).getName() + "@" + fns.get(i).getEntryPoint());
            if (rows.size() < Math.max(1, limit)) {
                rows.add(JsonHelper.mapOf("function", fns.get(i).getName(), "address", fns.get(i).getEntryPoint().toString(), "hash", h));
            }
        }
        List<Map<String, Object>> dups = new ArrayList<>();
        groups.forEach((h, l) -> { if (l.size() > 1) dups.add(JsonHelper.mapOf("hash", h, "count", l.size(), "functions", l)); });
        dups.sort((a, b) -> Integer.compare((int) b.get("count"), (int) a.get("count")));
        return Response.ok(JsonHelper.mapOf("kind", opcode ? "opcode" : "normalized", "functions_hashed", fns.size(), "unique_hashes", groups.size(),
            "duplicate_groups", dups, "hashes", rows, "execution", exec(o, extractMs)));
    }

    static ScanOps.Packed packBytes(List<byte[]> segs) {
        int total = 0;
        for (byte[] b : segs) total += b.length;
        byte[] all = new byte[total];
        long[] off = new long[segs.size()];
        int[] len = new int[segs.size()];
        int o = 0;
        for (int i = 0; i < segs.size(); i++) {
            off[i] = o;
            len[i] = segs.get(i).length;
            System.arraycopy(segs.get(i), 0, all, o, len[i]);
            o += len[i];
        }
        return new ScanOps.Packed(all, off, len);
    }

    @McpTool(path = "/gpu_find_similar_functions", description = "GPU twin of find_similar_functions / "
        + "find_similar_functions_fuzzy / bulk_fuzzy_match: compare one function (or every function) against all "
        + "functions of this or another program. method=minhash (Jaccard over mnemonic 3-grams) or cosine "
        + "(mnemonic histograms). All-pairs on the GPU is where the speedup is largest.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuFindSimilarFunctions(
            @Param(value = "function", defaultValue = "", paramType = Param.FUNCTION_REF,
                   description = "Query function (name or address); empty = all-vs-all") String function,
            @Param(value = "program_b", defaultValue = "", description = "Compare against this program instead of the same one") String programB,
            @Param(value = "method", defaultValue = "minhash", description = "minhash | cosine") String method,
            @Param(value = "threshold", defaultValue = "0.7", description = "Minimum similarity reported") double threshold,
            @Param(value = "top_k", defaultValue = "10", description = "Matches per query function") int topK,
            @Param(value = "min_instructions", defaultValue = "8", description = "Ignore functions shorter than this") int minInstructions,
            @Param(value = "limit", defaultValue = "200", description = "Max query functions reported in all-vs-all mode") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program pa = pe.program();
        Program pb = pa;
        if (programB != null && !programB.isBlank()) {
            ServiceUtils.ProgramOrError pe2 = ServiceUtils.getProgramOrError(programProvider, programB);
            if (pe2.hasError()) return pe2.error();
            pb = pe2.program();
        }
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        long t0 = System.nanoTime();
        Map<String, Integer> vocab = new HashMap<>();
        List<Function> fb = new ArrayList<>();
        List<int[]> tb = new ArrayList<>();
        for (Function f : ProgramData.functions(pb)) {
            if (f.isThunk() || f.isExternal()) continue;
            int[] t = ProgramData.mnemonicTokens(pb, f, vocab);
            if (t.length < minInstructions) continue;
            fb.add(f);
            tb.add(t);
        }
        List<Function> fa = new ArrayList<>();
        List<int[]> ta = new ArrayList<>();
        if (function != null && !function.isBlank()) {
            ServiceUtils.FunctionOrError fe = ServiceUtils.getFunctionOrError(pa, function);
            if (fe.hasError()) return fe.error();
            fa.add(fe.function());
            ta.add(ProgramData.mnemonicTokens(pa, fe.function(), vocab));
        } else if (pa == pb) {
            fa = fb;
            ta = tb;
        } else {
            for (Function f : ProgramData.functions(pa)) {
                if (f.isThunk() || f.isExternal()) continue;
                int[] t = ProgramData.mnemonicTokens(pa, f, vocab);
                if (t.length < minInstructions) continue;
                fa.add(f);
                ta.add(t);
            }
        }
        if (fa.isEmpty() || fb.isEmpty()) return Response.err("no functions with >= " + minInstructions + " instructions");
        double extractMs = ms(t0);
        int nA = fa.size(), nB = fb.size();
        long pairs = (long) nA * nB;
        if (pairs > 400_000_000L) return Response.err("too many pairs (" + pairs + "); pass function= or raise min_instructions");
        boolean cosine = method.equalsIgnoreCase("cosine");
        Dispatch.Outcome<float[]> o;
        Map<String, Object> sigExec = null;
        if (cosine) {
            int dim = Math.min(vocab.size(), 512);
            float[] A = histo(ta, dim), B = ta == tb ? null : histo(tb, dim);
            float[] BB = B == null ? A : B;
            o = Dispatch.run(Ops.COSINE, be, pairs * dim,
                () -> SimilarityOps.cosineCpu(A, nA, BB, nB, dim), s -> SimilarityOps.cosineGpu(s, A, nA, BB, nB, dim),
                (x, y) -> ScanOps.sameFloats(x, y, 1e-6f));
        } else {
            int k = 64;
            SimilarityOps.TokenSets sa = SimilarityOps.TokenSets.of(ta), sb = ta == tb ? sa : SimilarityOps.TokenSets.of(tb);
            long tok = sa.tokens().length + (sa == sb ? 0 : sb.tokens().length);
            Dispatch.Outcome<int[][]> sig = Dispatch.run(Ops.MINHASH, be, tok * k,
                () -> new int[][] {SimilarityOps.minhashCpu(sa, k, 3), sa == sb ? null : SimilarityOps.minhashCpu(sb, k, 3)},
                s -> new int[][] {SimilarityOps.minhashGpu(s, sa, k, 3), sa == sb ? null : SimilarityOps.minhashGpu(s, sb, k, 3)},
                (x, y) -> Arrays.equals(x[0], y[0]) && Arrays.equals(x[1], y[1]));
            int[] sA = sig.value()[0], sB = sig.value()[1] == null ? sA : sig.value()[1];
            sigExec = sig.meta();
            o = Dispatch.run(Ops.MINHASH_SIM, be, pairs * k,
                () -> SimilarityOps.minhashSimCpu(sA, nA, sB, nB, k), s -> SimilarityOps.minhashSimGpu(s, sA, nA, sB, nB, k),
                Arrays::equals);
        }
        float[] sim = o.value();
        List<Map<String, Object>> results = new ArrayList<>();
        int pairsAbove = 0;
        boolean same = pa == pb;
        for (int a = 0; a < nA && results.size() < Math.max(1, limit); a++) {
            List<int[]> best = new ArrayList<>();
            for (int b = 0; b < nB; b++) {
                if (same && fa.get(a).equals(fb.get(b))) continue;
                if (sim[a * nB + b] >= threshold) best.add(new int[] {b});
            }
            pairsAbove += best.size();
            if (best.isEmpty()) continue;
            int aa = a;
            best.sort((x, y) -> Float.compare(sim[aa * nB + y[0]], sim[aa * nB + x[0]]));
            List<Map<String, Object>> m = new ArrayList<>();
            for (int i = 0; i < Math.min(topK, best.size()); i++) {
                int b = best.get(i)[0];
                m.add(JsonHelper.mapOf("function", fb.get(b).getName(), "address", fb.get(b).getEntryPoint().toString(),
                    "similarity", round3(sim[a * nB + b]), "instructions", tb.get(b).length));
            }
            results.add(JsonHelper.mapOf("query", fa.get(a).getName(), "address", fa.get(a).getEntryPoint().toString(),
                "instructions", ta.get(a).length, "matches", m));
        }
        Map<String, Object> ex = exec(o, extractMs);
        if (sigExec != null) ex.put("signatures", sigExec);
        return Response.ok(JsonHelper.mapOf("method", cosine ? "cosine" : "minhash", "queries", nA, "candidates", nB,
            "pairs_compared", pairs, "queries_with_matches_reported", results.size(), "results", results,
            "execution", ex));
    }

    static float[] histo(List<int[]> sets, int dim) {
        float[] out = new float[sets.size() * dim];
        for (int i = 0; i < sets.size(); i++) for (int t : sets.get(i)) if (t < dim) out[i * dim + t]++;
        return out;
    }

    @McpTool(path = "/gpu_diff_functions", description = "GPU twin of diff_functions, batched: edit distance between "
        + "the mnemonic sequences of many function pairs at once (same or different programs). "
        + "similarity = 1 - distance / max(len).",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuDiffFunctions(
            @Param(value = "pairs", description = "Pairs 'funcA:funcB' separated by commas (names or addresses)") String pairs,
            @Param(value = "program_b", defaultValue = "", description = "Program holding the second function of each pair (default: same)") String programB,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program pa = pe.program(), pb = pa;
        if (programB != null && !programB.isBlank()) {
            ServiceUtils.ProgramOrError pe2 = ServiceUtils.getProgramOrError(programProvider, programB);
            if (pe2.hasError()) return pe2.error();
            pb = pe2.program();
        }
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        long t0 = System.nanoTime();
        Map<String, Integer> vocab = new HashMap<>();
        List<int[]> la = new ArrayList<>(), lb = new ArrayList<>();
        List<String[]> names = new ArrayList<>();
        for (String pair : pairs.split(",")) {
            if (pair.isBlank()) continue;
            int c = pair.lastIndexOf(':');
            if (c <= 0) return Response.err("pair '" + pair + "' must be funcA:funcB");
            String a = pair.substring(0, c).trim(), b = pair.substring(c + 1).trim();
            // allow 'space:hex' addresses on the left by preferring the split that resolves
            ServiceUtils.FunctionOrError fa = ServiceUtils.getFunctionOrError(pa, a);
            ServiceUtils.FunctionOrError fb = ServiceUtils.getFunctionOrError(pb, b);
            if (fa.hasError()) return fa.error();
            if (fb.hasError()) return fb.error();
            la.add(ProgramData.mnemonicTokens(pa, fa.function(), vocab));
            lb.add(ProgramData.mnemonicTokens(pb, fb.function(), vocab));
            names.add(new String[] {fa.function().getName(), fb.function().getName()});
        }
        if (la.isEmpty()) return Response.err("no pairs given");
        SimilarityOps.TokenSets A = SimilarityOps.TokenSets.of(la), B = SimilarityOps.TokenSets.of(lb);
        double extractMs = ms(t0);
        long work = 0;
        for (int i = 0; i < la.size(); i++) work += (long) la.get(i).length * lb.get(i).length;
        Dispatch.Outcome<int[]> o = Dispatch.run(Ops.LEVENSHTEIN, be, work,
            () -> SimilarityOps.levenshteinCpu(A, B), s -> SimilarityOps.levenshteinGpu(s, A, B), Arrays::equals);
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int i = 0; i < la.size(); i++) {
            int mx = Math.max(la.get(i).length, lb.get(i).length);
            rows.add(JsonHelper.mapOf("function_a", names.get(i)[0], "function_b", names.get(i)[1],
                "instructions_a", la.get(i).length, "instructions_b", lb.get(i).length, "edit_distance", o.value()[i],
                "similarity", round3(mx == 0 ? 1.0 : 1.0 - (double) o.value()[i] / mx)));
        }
        return Response.ok(JsonHelper.mapOf("pairs", rows, "execution", exec(o, extractMs)));
    }

    // =====================================================================
    // Graphs
    // =====================================================================

    @McpTool(path = "/gpu_call_graph_analysis", description = "GPU twin of analyze_call_graph / get_full_call_graph / "
        + "get_function_call_graph: whole-program call graph with recursion groups (SCCs), PageRank importance, "
        + "functions unreachable from entry points, and -- given a function -- its callees/callers by depth.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuCallGraphAnalysis(
            @Param(value = "function", defaultValue = "", paramType = Param.FUNCTION_REF,
                   description = "Optional focus function for callee/caller depth maps") String function,
            @Param(value = "depth", defaultValue = "3", description = "Max depth for the callee/caller maps") int depth,
            @Param(value = "top_n", defaultValue = "25", description = "How many most-important functions to list") int topN,
            @Param(value = "pagerank_iterations", defaultValue = "30", description = "PageRank iterations") int iterations,
            @Param(value = "limit", defaultValue = "200", description = "Max rows in each list") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        long t0 = System.nanoTime();
        ProgramData.CallGraph cg;
        try {
            cg = ProgramData.callGraph(p, TaskMonitor.DUMMY);
        } catch (Exception e) {
            return Response.err("call graph extraction failed: " + e.getMessage());
        }
        double extractMs = ms(t0);
        GraphOps.Csr g = cg.csr();
        long work = (long) g.n() + g.edges();
        int lim = Math.max(1, limit);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("functions", g.n());
        out.put("call_edges", g.edges());

        Dispatch.Outcome<int[]> scc = Dispatch.run(Ops.SCC, be, work, () -> GraphOps.sccCpu(g), s -> GraphOps.sccGpu(s, g), Arrays::equals);
        Map<Integer, List<Integer>> comps = new HashMap<>();
        for (int u = 0; u < g.n(); u++) comps.computeIfAbsent(scc.value()[u], k -> new ArrayList<>()).add(u);
        List<Map<String, Object>> recursion = new ArrayList<>();
        int groups = 0;
        for (List<Integer> c : comps.values()) {
            boolean selfLoop = false;
            if (c.size() == 1) {
                int u = c.get(0);
                for (int k = g.rowPtr()[u]; k < g.rowPtr()[u + 1]; k++) if (g.col()[k] == u) selfLoop = true;
            }
            if (c.size() < 2 && !selfLoop) continue;
            groups++;
            if (recursion.size() >= lim) continue;
            List<String> names = new ArrayList<>();
            for (int u : c) if (names.size() < 50) names.add(cg.functions().get(u).getName());
            recursion.add(JsonHelper.mapOf("size", c.size(), "kind", c.size() == 1 ? "direct" : "mutual", "functions", names));
        }
        recursion.sort((a, b) -> Integer.compare((int) b.get("size"), (int) a.get("size")));
        out.put("recursion_groups", groups);
        out.put("recursive", recursion);

        int iters = Math.max(1, Math.min(200, iterations));
        Dispatch.Outcome<double[]> pr = Dispatch.run(Ops.PAGERANK, be, work * iters,
            () -> GraphOps.pagerankCpu(g, 0.85, iters), s -> GraphOps.pagerankGpu(s, g, 0.85, iters),
            (a, b) -> { for (int i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 1e-12) return false; return true; });
        Integer[] order = new Integer[g.n()];
        for (int i = 0; i < order.length; i++) order[i] = i;
        double[] rank = pr.value();
        Arrays.sort(order, (x, y) -> Double.compare(rank[y], rank[x]));
        List<Map<String, Object>> top = new ArrayList<>();
        for (int i = 0; i < Math.min(Math.max(1, topN), order.length); i++) {
            Function f = cg.functions().get(order[i]);
            top.add(JsonHelper.mapOf("function", f.getName(), "address", f.getEntryPoint().toString(),
                "pagerank", rank[order[i]], "external", f.isExternal()));
        }
        out.put("most_called_into", top);

        // entry points: exported/entry symbols that are functions, plus functions with no callers
        List<Integer> roots = new ArrayList<>();
        var it = p.getSymbolTable().getExternalEntryPointIterator();
        while (it.hasNext()) {
            Function f = p.getFunctionManager().getFunctionAt(it.next());
            Integer i = f == null ? null : cg.index().get(f);
            if (i != null) roots.add(i);
        }
        Dispatch.Outcome<int[]> reach = null;
        if (!roots.isEmpty()) {
            int[] src = roots.stream().mapToInt(Integer::intValue).toArray();
            reach = Dispatch.run(Ops.BFS, be, work, () -> GraphOps.bfsCpu(g, src, Integer.MAX_VALUE),
                s -> GraphOps.bfsGpu(s, g, src, Integer.MAX_VALUE), Arrays::equals);
            List<String> dead = new ArrayList<>();
            int deadCount = 0;
            for (int u = 0; u < g.n(); u++) {
                Function f = cg.functions().get(u);
                if (f.isExternal() || reach.value()[u] >= 0) continue;
                deadCount++;
                if (dead.size() < lim) dead.add(f.getName() + "@" + f.getEntryPoint());
            }
            out.put("entry_points", roots.size());
            out.put("unreachable_from_entry_points", deadCount);
            out.put("unreachable", dead);
            out.put("unreachable_note", "Unreachable by direct calls only: callbacks, vtables and indirect calls are not edges.");
        }

        Dispatch.Outcome<int[]> fwd = null, bwd = null;
        if (function != null && !function.isBlank()) {
            ServiceUtils.FunctionOrError fe = ServiceUtils.getFunctionOrError(p, function);
            if (fe.hasError()) return fe.error();
            Integer fi = cg.index().get(fe.function());
            if (fi != null) {
                int[] src = {fi};
                int d = Math.max(1, depth);
                GraphOps.Csr rev = g.reverse();
                fwd = Dispatch.run(Ops.BFS, be, work, () -> GraphOps.bfsCpu(g, src, d), s -> GraphOps.bfsGpu(s, g, src, d), Arrays::equals);
                bwd = Dispatch.run(Ops.BFS, be, work, () -> GraphOps.bfsCpu(rev, src, d), s -> GraphOps.bfsGpu(s, rev, src, d), Arrays::equals);
                out.put("focus", fe.function().getName());
                out.put("callees_by_depth", byDepth(fwd.value(), cg, lim));
                out.put("callers_by_depth", byDepth(bwd.value(), cg, lim));
            }
        }
        Map<String, Object> ex = new LinkedHashMap<>();
        ex.put("extract_ms", extractMs);
        ex.put("scc", scc.meta());
        ex.put("pagerank", pr.meta());
        if (reach != null) ex.put("reachability", reach.meta());
        if (fwd != null) { ex.put("callees", fwd.meta()); ex.put("callers", bwd.meta()); }
        out.put("execution", ex);
        return Response.ok(out);
    }

    static Map<String, List<String>> byDepth(int[] depth, ProgramData.CallGraph cg, int lim) {
        Map<String, List<String>> m = new TreeMap<>();
        int n = 0;
        for (int u = 0; u < depth.length; u++) {
            if (depth[u] <= 0 || n >= lim) continue;
            m.computeIfAbsent(String.valueOf(depth[u]), k -> new ArrayList<>()).add(cg.functions().get(u).getName());
            n++;
        }
        return m;
    }

    @McpTool(path = "/gpu_control_flow_analysis", description = "GPU twin of analyze_control_flow / find_dead_code, "
        + "batched over every function at once: basic blocks, edges, cyclomatic complexity, dominator-based "
        + "natural loops (back edges) and unreachable blocks. Sorted by complexity.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuControlFlowAnalysis(
            @Param(value = "function", defaultValue = "", paramType = Param.FUNCTION_REF,
                   description = "One function (name or address); empty = all functions") String function,
            @Param(value = "max_functions", defaultValue = "100000", description = "Cap on functions analysed") int maxFunctions,
            @Param(value = "limit", defaultValue = "100", description = "Max functions listed") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        List<Function> fns = new ArrayList<>();
        if (function != null && !function.isBlank()) {
            ServiceUtils.FunctionOrError fe = ServiceUtils.getFunctionOrError(p, function);
            if (fe.hasError()) return fe.error();
            fns.add(fe.function());
        } else {
            for (Function f : ProgramData.functions(p)) {
                if (f.isThunk() || f.isExternal()) continue;
                if (fns.size() >= maxFunctions) break;
                fns.add(f);
            }
        }
        long t0 = System.nanoTime();
        ProgramData.CfgBatch cb;
        try {
            cb = ProgramData.cfgs(p, fns, TaskMonitor.DUMMY);
        } catch (Exception e) {
            return Response.err("CFG extraction failed: " + e.getMessage());
        }
        double extractMs = ms(t0);
        GraphOps.Cfgs c = cb.cfgs();
        long work = 0;
        for (int gi = 0; gi < c.graphs(); gi++) { long nb = c.nodeOff()[gi + 1] - c.nodeOff()[gi]; work += nb * nb / 32 + nb; }
        Dispatch.Outcome<GraphOps.DomResult> o = Dispatch.run(Ops.DOMINATORS, be, work,
            () -> GraphOps.dominatorsCpu(c), s -> GraphOps.dominatorsGpu(s, c), GraphOps.DomResult::same);
        int[] idom = o.value().idom();
        List<Map<String, Object>> rows = new ArrayList<>();
        long totalUnreachable = 0, totalLoops = 0;
        for (int gi = 0; gi < c.graphs(); gi++) {
            int base = c.nodeOff()[gi], nb = c.nodeOff()[gi + 1] - base;
            int edges = cb.edges()[gi];
            int loops = 0;
            List<String> loopHeads = new ArrayList<>();
            List<String> unreachable = new ArrayList<>();
            // successor edges are the predecessor CSR inverted: edge p->v for p in pred(v)
            for (int v = 0; v < nb; v++) {
                if (idom[base + v] == -2) {
                    unreachable.add(cb.blocks().get(gi).get(v).getFirstStartAddress().toString());
                    continue;
                }
                for (int e = c.predRowPtr()[base + v]; e < c.predRowPtr()[base + v + 1]; e++) {
                    int u = c.predCol()[e];
                    if (idom[base + u] == -2) continue;
                    if (dominates(idom, base, v, u)) {
                        loops++;
                        if (loopHeads.size() < 16) {
                            String h = cb.blocks().get(gi).get(v).getFirstStartAddress().toString();
                            if (!loopHeads.contains(h)) loopHeads.add(h);
                        }
                    }
                }
            }
            totalUnreachable += unreachable.size();
            totalLoops += loops;
            Function f = cb.functions().get(gi);
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("function", f.getName());
            m.put("address", f.getEntryPoint().toString());
            m.put("blocks", nb);
            m.put("edges", edges);
            m.put("cyclomatic_complexity", nb == 0 ? 0 : edges - nb + 2);
            m.put("back_edges", loops);
            m.put("loop_headers", loopHeads);
            m.put("unreachable_blocks", unreachable);
            rows.add(m);
        }
        rows.sort((a, b) -> Integer.compare((int) b.get("cyclomatic_complexity"), (int) a.get("cyclomatic_complexity")));
        return Response.ok(JsonHelper.mapOf("functions_analyzed", rows.size(), "total_basic_blocks", c.nodes(),
            "total_back_edges", totalLoops, "total_unreachable_blocks", totalUnreachable,
            "functions", rows.subList(0, Math.min(rows.size(), Math.max(1, limit))), "execution", exec(o, extractMs)));
    }

    /** Does local block d dominate local block v (walking v's idom chain)? */
    static boolean dominates(int[] idom, int base, int d, int v) {
        int x = v, guard = 0;
        while (x >= 0 && guard++ < 1_000_000) {
            if (x == d) return true;
            x = idom[base + x];
        }
        return false;
    }

    @McpTool(path = "/gpu_dataflow_slice", description = "GPU twin of analyze_dataflow: decompile the function "
        + "(Ghidra's decompiler, CPU), build its whole def-use graph, and slice forward (consumers) or backward "
        + "(producers) from every P-code value at an address, optionally only those touching a named variable/register.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuDataflowSlice(
            @Param(value = "address", paramType = "address", description = "Instruction address to slice from") String address,
            @Param(value = "variable", defaultValue = "", description = "Only seed varnodes whose high variable or register has this name") String variable,
            @Param(value = "direction", defaultValue = "forward", description = "forward | backward") String direction,
            @Param(value = "max_steps", defaultValue = "64", description = "Max def-use depth") int maxSteps,
            @Param(value = "limit", defaultValue = "300", description = "Max ops listed") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        Address addr = ServiceUtils.parseAddress(p, address);
        if (addr == null) return Response.err(ServiceUtils.getLastParseError());
        Function fn = p.getFunctionManager().getFunctionContaining(addr);
        if (fn == null) return Response.err("no function contains " + address);
        boolean forward = !direction.equalsIgnoreCase("backward");
        long t0 = System.nanoTime();
        HighFunction hf;
        DecompInterface ifc = new DecompInterface();
        try {
            ifc.openProgram(p);
            DecompileResults res = ifc.decompileFunction(fn, 60, TaskMonitor.DUMMY);
            hf = res.getHighFunction();
            if (hf == null) return Response.err("decompile failed: " + res.getErrorMessage());
        } finally {
            ifc.dispose();
        }
        double decompileMs = ms(t0);
        long t1 = System.nanoTime();
        // Nodes: ops then varnodes. Edges: varnode -> consuming op, op -> output varnode.
        List<PcodeOpAST> ops = new ArrayList<>();
        hf.getPcodeOps().forEachRemaining(ops::add);
        Map<Integer, Integer> opIndex = new HashMap<>();
        Map<Integer, Integer> vnIndex = new HashMap<>();
        List<Varnode> vns = new ArrayList<>();
        for (int i = 0; i < ops.size(); i++) opIndex.put(ops.get(i).getSeqnum().getTime(), i);
        List<Integer> src = new ArrayList<>(), dst = new ArrayList<>();
        int nOps = ops.size();
        java.util.function.Function<Varnode, Integer> vid = v -> vnIndex.computeIfAbsent(((VarnodeAST) v).getUniqueId(),
            k -> { vns.add(v); return nOps + vns.size() - 1; });
        for (int i = 0; i < nOps; i++) {
            PcodeOpAST op = ops.get(i);
            for (Varnode in : op.getInputs()) {
                int v = vid.apply(in);
                if (forward) { src.add(v); dst.add(i); } else { src.add(i); dst.add(v); }
            }
            if (op.getOutput() != null) {
                int v = vid.apply(op.getOutput());
                if (forward) { src.add(i); dst.add(v); } else { src.add(v); dst.add(i); }
            }
        }
        int n = nOps + vns.size();
        int[] s = src.stream().mapToInt(Integer::intValue).toArray(), d = dst.stream().mapToInt(Integer::intValue).toArray();
        GraphOps.Csr g = GraphOps.Csr.of(n, s, d);
        List<Integer> seeds = new ArrayList<>();
        String var = variable == null ? "" : variable.trim();
        for (int i = 0; i < nOps; i++) {
            PcodeOpAST op = ops.get(i);
            if (!op.getSeqnum().getTarget().equals(addr)) continue;
            List<Varnode> cand = new ArrayList<>();
            if (forward) { if (op.getOutput() != null) cand.add(op.getOutput()); }
            else cand.addAll(Arrays.asList(op.getInputs()));
            boolean any = false;
            for (Varnode v : cand) {
                if (!var.isEmpty() && !varMatches(p, v, var)) continue;
                seeds.add(vnIndex.get(((VarnodeAST) v).getUniqueId()));
                any = true;
            }
            if (var.isEmpty() && !any) seeds.add(i);
        }
        if (seeds.isEmpty()) return Response.err("no P-code at " + addr + (var.isEmpty() ? "" : " touching '" + var + "'"));
        double buildMs = ms(t1);
        int[] seedArr = seeds.stream().mapToInt(Integer::intValue).toArray();
        int steps = Math.max(1, maxSteps) * 2; // op and varnode levels alternate
        Dispatch.Outcome<int[]> o = Dispatch.run(Ops.BFS, be, (long) n + g.edges(),
            () -> GraphOps.bfsCpu(g, seedArr, steps), x -> GraphOps.bfsGpu(x, g, seedArr, steps), Arrays::equals);
        List<Map<String, Object>> reached = new ArrayList<>();
        int total = 0;
        Set<String> addrs = new java.util.TreeSet<>();
        Integer[] order = new Integer[nOps];
        for (int i = 0; i < nOps; i++) order[i] = i;
        int[] depthArr = o.value();
        Arrays.sort(order, (x, y) -> Integer.compare(depthArr[x] < 0 ? Integer.MAX_VALUE : depthArr[x], depthArr[y] < 0 ? Integer.MAX_VALUE : depthArr[y]));
        for (int i : order) {
            if (depthArr[i] < 0) continue;
            total++;
            PcodeOpAST op = ops.get(i);
            addrs.add(op.getSeqnum().getTarget().toString());
            if (reached.size() < Math.max(1, limit)) {
                reached.add(JsonHelper.mapOf("address", op.getSeqnum().getTarget().toString(), "step", (depthArr[i] + 1) / 2,
                    "op", op.toString()));
            }
        }
        Map<String, Object> ex = exec(o, buildMs);
        ex.put("decompile_ms", decompileMs);
        ex.put("decompile_backend", "cpu (Ghidra decompiler, native)");
        return Response.ok(JsonHelper.mapOf("function", fn.getName(), "direction", forward ? "forward" : "backward",
            "seeds", seeds.size(), "graph_nodes", n, "graph_edges", g.edges(), "ops_reached", total,
            "addresses_reached", new ArrayList<>(addrs), "ops", reached, "execution", ex));
    }

    /** One decompiled function reduced to arrays: high-level CFG and def-use graph. */
    private record Decompiled(Function fn, double ms, String error, int blocks, int[] cfgSrc, int[] cfgDst,
                              int nodes, int ops, int[] duSrc, int[] duDst, int[] paramSeeds, int[] returnSeeds) {}

    @McpTool(path = "/gpu_decompile_batch", description = "Decompile many functions and analyse the decompiler's "
        + "output, with the analysis on the backend of your choice. Decompilation itself is Ghidra's native "
        + "decompiler on a CPU thread pool (it cannot run on a GPU); the analysis of every result -- dominators and "
        + "loops of the high-level CFG, ops influenced by the parameters, ops contributing to the return value -- "
        + "runs batched on CPU or GPU per backend=. Returns structure, not C text: use get_functions/force_decompile "
        + "for pseudocode.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuDecompileBatch(
            @Param(value = "functions", defaultValue = "", description = "Comma-separated names/addresses; empty = all non-thunk functions up to max_functions") String functions,
            @Param(value = "max_functions", defaultValue = "200", description = "Cap when functions is empty") int maxFunctions,
            @Param(value = "threads", defaultValue = "0", description = "Decompiler threads; 0 = available processors") int threads,
            @Param(value = "timeout_secs", defaultValue = "60", description = "Per-function decompile timeout") int timeoutSecs,
            @Param(value = "limit", defaultValue = "100", description = "Max functions listed") int limit,
            @Param(value = "backend", defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        List<Function> fns = new ArrayList<>();
        if (functions != null && !functions.isBlank()) {
            for (String ref : functions.split(",")) {
                if (ref.isBlank()) continue;
                ServiceUtils.FunctionOrError fe = ServiceUtils.getFunctionOrError(p, ref.trim());
                if (fe.hasError()) return fe.error();
                fns.add(fe.function());
            }
        } else {
            for (Function f : ProgramData.functions(p)) {
                if (f.isThunk() || f.isExternal()) continue;
                if (fns.size() >= Math.max(1, maxFunctions)) break;
                fns.add(f);
            }
        }
        if (fns.isEmpty()) return Response.err("no functions to decompile");

        // Stage 1: native decompiler, CPU, one DecompInterface per worker thread.
        int nThreads = threads > 0 ? threads : Math.max(1, Runtime.getRuntime().availableProcessors());
        nThreads = Math.min(nThreads, fns.size());
        long t0 = System.nanoTime();
        Decompiled[] out = new Decompiled[fns.size()];
        java.util.concurrent.atomic.AtomicInteger next = new java.util.concurrent.atomic.AtomicInteger();
        List<Thread> workers = new ArrayList<>();
        for (int w = 0; w < nThreads; w++) {
            Thread th = new Thread(() -> {
                DecompInterface ifc = new DecompInterface();
                try {
                    ifc.openProgram(p);
                    int i;
                    while ((i = next.getAndIncrement()) < fns.size()) out[i] = reduce(ifc, fns.get(i), timeoutSecs);
                } finally {
                    ifc.dispose();
                }
            }, "gpu-decompile-" + w);
            th.setDaemon(true);
            workers.add(th);
            th.start();
        }
        for (Thread th : workers) {
            try {
                th.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return Response.err("interrupted");
            }
        }
        double decompileMs = ms(t0);

        // Stage 2: concatenate every function's graphs and analyse them in one batch.
        long t1 = System.nanoTime();
        List<int[][]> cfgs = new ArrayList<>();
        List<Integer> src = new ArrayList<>(), dst = new ArrayList<>(), pSeeds = new ArrayList<>(), rSeeds = new ArrayList<>();
        int[] nodeBase = new int[fns.size() + 1];
        int base = 0;
        for (int i = 0; i < out.length; i++) {
            Decompiled d = out[i];
            nodeBase[i] = base;
            if (d.error == null) {
                cfgs.add(GraphOps.Cfgs.graph(d.blocks, d.cfgSrc, d.cfgDst));
                for (int k = 0; k < d.duSrc.length; k++) { src.add(base + d.duSrc[k]); dst.add(base + d.duDst[k]); }
                for (int s : d.paramSeeds) pSeeds.add(base + s);
                for (int s : d.returnSeeds) rSeeds.add(base + s);
                base += d.nodes;
            } else {
                cfgs.add(GraphOps.Cfgs.graph(0, new int[0], new int[0]));
            }
        }
        nodeBase[fns.size()] = base;
        GraphOps.Cfgs c = GraphOps.Cfgs.of(cfgs);
        int n = base;
        GraphOps.Csr du = GraphOps.Csr.of(n, src.stream().mapToInt(Integer::intValue).toArray(), dst.stream().mapToInt(Integer::intValue).toArray());
        GraphOps.Csr duRev = du.reverse();
        int[] ps = pSeeds.stream().mapToInt(Integer::intValue).toArray(), rs = rSeeds.stream().mapToInt(Integer::intValue).toArray();
        double buildMs = ms(t1);
        long domWork = 0;
        for (int gi = 0; gi < c.graphs(); gi++) { long nb = c.nodeOff()[gi + 1] - c.nodeOff()[gi]; domWork += nb * nb / 32 + nb; }
        Dispatch.Outcome<GraphOps.DomResult> dom = Dispatch.run(Ops.DOMINATORS, be, domWork,
            () -> GraphOps.dominatorsCpu(c), s -> GraphOps.dominatorsGpu(s, c), GraphOps.DomResult::same);
        Dispatch.Outcome<int[]> fromParams = Dispatch.run(Ops.BFS, be, (long) n + du.edges(),
            () -> GraphOps.bfsCpu(du, ps, Integer.MAX_VALUE), s -> GraphOps.bfsGpu(s, du, ps, Integer.MAX_VALUE), Arrays::equals);
        Dispatch.Outcome<int[]> toReturn = Dispatch.run(Ops.BFS, be, (long) n + du.edges(),
            () -> GraphOps.bfsCpu(duRev, rs, Integer.MAX_VALUE), s -> GraphOps.bfsGpu(s, duRev, rs, Integer.MAX_VALUE), Arrays::equals);

        List<Map<String, Object>> rows = new ArrayList<>();
        int failed = 0;
        int[] idom = dom.value().idom();
        for (int i = 0; i < out.length; i++) {
            Decompiled d = out[i];
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("function", d.fn.getName());
            m.put("address", d.fn.getEntryPoint().toString());
            m.put("decompile_ms", d.ms);
            if (d.error != null) { failed++; m.put("error", d.error); rows.add(m); continue; }
            int gb = c.nodeOff()[i], nb = d.blocks, loops = 0, unreachable = 0;
            for (int v = 0; v < nb; v++) {
                if (idom[gb + v] == -2) { unreachable++; continue; }
                for (int e = c.predRowPtr()[gb + v]; e < c.predRowPtr()[gb + v + 1]; e++) {
                    int u = c.predCol()[e];
                    if (idom[gb + u] != -2 && dominates(idom, gb, v, u)) loops++;
                }
            }
            int paramOps = 0, retOps = 0;
            for (int k = nodeBase[i]; k < nodeBase[i] + d.ops; k++) {
                if (fromParams.value()[k] >= 0) paramOps++;
                if (toReturn.value()[k] >= 0) retOps++;
            }
            m.put("high_blocks", nb);
            m.put("high_edges", d.cfgSrc.length);
            m.put("structured_complexity", nb == 0 ? 0 : d.cfgSrc.length - nb + 2);
            m.put("loops", loops);
            m.put("unreachable_blocks", unreachable);
            m.put("pcode_ops", d.ops);
            m.put("ops_influenced_by_params", paramOps);
            m.put("ops_feeding_return", retOps);
            rows.add(m);
        }
        Map<String, Object> ex = new LinkedHashMap<>();
        ex.put("decompile_ms", decompileMs);
        ex.put("decompile_backend", "cpu (Ghidra native decompiler, " + nThreads + " threads)");
        ex.put("graph_build_ms", buildMs);
        ex.put("dominators", dom.meta());
        ex.put("param_influence", fromParams.meta());
        ex.put("return_slice", toReturn.meta());
        return Response.ok(JsonHelper.mapOf("functions_decompiled", fns.size() - failed, "failed", failed,
            "total_high_blocks", c.nodes(), "total_dataflow_nodes", n, "functions", rows.subList(0, Math.min(rows.size(), Math.max(1, limit))),
            "execution", ex));
    }

    /** Decompiles one function and flattens its HighFunction into the arrays the batch needs. */
    private static Decompiled reduce(DecompInterface ifc, Function fn, int timeoutSecs) {
        long t0 = System.nanoTime();
        DecompileResults res;
        try {
            res = ifc.decompileFunction(fn, Math.max(1, timeoutSecs), TaskMonitor.DUMMY);
        } catch (Exception e) {
            return new Decompiled(fn, ms(t0), e.getMessage(), 0, null, null, 0, 0, null, null, null, null);
        }
        HighFunction hf = res == null ? null : res.getHighFunction();
        double took = ms(t0);
        if (hf == null) {
            return new Decompiled(fn, took, res == null ? "no result" : String.valueOf(res.getErrorMessage()),
                0, null, null, 0, 0, null, null, null, null);
        }
        List<ghidra.program.model.pcode.PcodeBlockBasic> blocks = hf.getBasicBlocks();
        int entry = 0;
        for (int i = 0; i < blocks.size(); i++) {
            if (blocks.get(i).getStart() != null && blocks.get(i).getStart().equals(fn.getEntryPoint())) { entry = i; break; }
        }
        // local index with the entry block first
        int[] local = new int[blocks.size()];
        for (int i = 0, k = 1; i < blocks.size(); i++) local[i] = i == entry ? 0 : k++;
        Map<ghidra.program.model.pcode.PcodeBlock, Integer> idx = new HashMap<>();
        for (int i = 0; i < blocks.size(); i++) idx.put(blocks.get(i), local[i]);
        List<Integer> cs = new ArrayList<>(), cd = new ArrayList<>();
        for (int i = 0; i < blocks.size(); i++) {
            var b = blocks.get(i);
            for (int o = 0; o < b.getOutSize(); o++) {
                Integer j = idx.get(b.getOut(o));
                if (j != null) { cs.add(local[i]); cd.add(j); }
            }
        }
        List<PcodeOpAST> ops = new ArrayList<>();
        hf.getPcodeOps().forEachRemaining(ops::add);
        Map<Integer, Integer> vn = new HashMap<>();
        int nOps = ops.size();
        int[] counter = {nOps};
        java.util.function.Function<Varnode, Integer> vid = v -> vn.computeIfAbsent(((VarnodeAST) v).getUniqueId(), k -> counter[0]++);
        List<Integer> s = new ArrayList<>(), d = new ArrayList<>(), ret = new ArrayList<>();
        for (int i = 0; i < nOps; i++) {
            PcodeOpAST op = ops.get(i);
            for (Varnode in : op.getInputs()) { s.add(vid.apply(in)); d.add(i); }
            if (op.getOutput() != null) { s.add(i); d.add(vid.apply(op.getOutput())); }
            if (op.getOpcode() == ghidra.program.model.pcode.PcodeOp.RETURN) ret.add(i);
        }
        List<Integer> params = new ArrayList<>();
        var lsm = hf.getLocalSymbolMap();
        for (int pi = 0; pi < lsm.getNumParams(); pi++) {
            var sym = lsm.getParamSymbol(pi);
            var hv = sym == null ? null : sym.getHighVariable();
            if (hv == null) continue;
            for (Varnode v : hv.getInstances()) {
                Integer id = vn.get(((VarnodeAST) v).getUniqueId());
                if (id != null) params.add(id);
            }
        }
        return new Decompiled(fn, took, null, blocks.size(), toArr(cs), toArr(cd), counter[0], nOps,
            toArr(s), toArr(d), toArr(params), toArr(ret));
    }

    private static int[] toArr(List<Integer> l) {
        return l.stream().mapToInt(Integer::intValue).toArray();
    }

    static boolean varMatches(Program p, Varnode v, String name) {
        if (v.getHigh() != null && name.equalsIgnoreCase(v.getHigh().getName())) return true;
        if (v.isRegister()) {
            Register r = p.getRegister(v.getAddress(), v.getSize());
            return r != null && r.getName().equalsIgnoreCase(name);
        }
        return false;
    }

    // =====================================================================
    // Batched emulation
    // =====================================================================

    @McpTool(path = "/gpu_emulate_hash_batch", method = "POST", description = "GPU twin of emulate_hash_batch: run a "
        + "hash function on every candidate string in parallel (one GPU thread per candidate) on a P-code VM "
        + "lowered from the function, and return the candidates producing target_hash. Resolves ROR13/CRC32/"
        + "djb2/FNV/custom API hashing for 10^5+ candidates in one call. Works for any processor Ghidra decodes.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuEmulateHashBatch(
            @Param(value = "hash_function_address", paramType = "address", source = ParamSource.BODY,
                   description = "Entry of the hash function") String hashFuncAddr,
            @Param(value = "string_register", source = ParamSource.BODY, defaultValue = "",
                   description = "Register receiving the string pointer (e.g. ECX, RCX, R0); empty = pass it on the stack as the first argument") String stringRegister,
            @Param(value = "result_register", source = ParamSource.BODY, defaultValue = "",
                   description = "Register holding the hash on return; empty = the calling convention's return register") String resultRegister,
            @Param(value = "target_hash", source = ParamSource.BODY, defaultValue = "",
                   description = "Hash to match (hex or decimal); empty = return every candidate's hash") String targetHashStr,
            @Param(value = "candidates", source = ParamSource.BODY, fieldsJson = true,
                   description = "JSON array of candidate strings") String candidatesJson,
            @Param(value = "initial_registers", source = ParamSource.BODY, fieldsJson = true, defaultValue = "",
                   description = "JSON object of extra register values set in every lane") String initialRegistersJson,
            @Param(value = "wide_string", source = ParamSource.BODY, defaultValue = "false",
                   description = "Write candidates as UTF-16LE") boolean wideString,
            @Param(value = "hash_bits", source = ParamSource.BODY, defaultValue = "32",
                   description = "Compare the low N bits of the result (upstream compares 32)") int hashBits,
            @Param(value = "max_steps", source = ParamSource.BODY, defaultValue = "100000",
                   description = "P-code ops per lane before it is stopped") int maxSteps,
            @Param(value = "backend", source = ParamSource.BODY, defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        List<String> cands = new ArrayList<>();
        try {
            Object parsed = JsonHelper.parseJson("{\"c\":" + candidatesJson + "}").get("c");
            if (parsed instanceof List<?> l) for (Object x : l) cands.add(String.valueOf(x));
        } catch (Exception e) {
            return Response.err("Invalid candidates JSON: " + e.getMessage());
        }
        if (cands.isEmpty()) return Response.err("No candidates provided");
        if (cands.size() > 5_000_000) return Response.err("Too many candidates (max 5,000,000)");
        Long target = null;
        if (targetHashStr != null && !targetHashStr.isBlank()) {
            try { target = parseLong(targetHashStr); } catch (Exception e) { return Response.err("Invalid target_hash: " + targetHashStr); }
        }
        long mask = hashBits >= 64 ? -1L : (1L << Math.max(1, hashBits)) - 1;
        List<Map<String, Long>> laneRegs = new ArrayList<>();
        Map<String, Long> extra;
        try { extra = parseRegs(initialRegistersJson); } catch (Exception e) { return Response.err("Invalid initial_registers: " + e.getMessage()); }
        List<byte[]> strings = new ArrayList<>();
        for (String c : cands) {
            strings.add((c + "\0").getBytes(wideString ? StandardCharsets.UTF_16LE : StandardCharsets.ISO_8859_1));
            laneRegs.add(extra);
        }
        EmuSetup setup;
        try {
            int longest = 0;
            for (byte[] b : strings) longest = Math.max(longest, b.length);
            setup = setupEmulation(p, hashFuncAddr, stringRegister, resultRegister, extra.keySet(), List.of(), longest);
        } catch (IllegalArgumentException e) {
            return Response.err(e.getMessage());
        }
        EmuRun run = runLanes(setup, laneRegs, strings, null, stringRegister, maxSteps, be);
        Map<String, Integer> statusCounts = new TreeMap<>();
        List<Map<String, Object>> matches = new ArrayList<>();
        List<Map<String, Object>> all = new ArrayList<>();
        for (int i = 0; i < cands.size(); i++) {
            PcodeVm.Status st = run.result.status(i);
            statusCounts.merge(st.label(), 1, Integer::sum);
            if (st != PcodeVm.Status.RETURNED) continue;
            long h = run.result.readState(i, setup.result[0], setup.result[1], p.getLanguage().isBigEndian()) & mask;
            if (target != null && h == (target & mask)) {
                matches.add(JsonHelper.mapOf("api_name", cands.get(i), "computed_hash", "0x" + Long.toHexString(h), "index", i));
            } else if (target == null && all.size() < 10_000) {
                all.add(JsonHelper.mapOf("candidate", cands.get(i), "hash", "0x" + Long.toHexString(h)));
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("function", setup.fn.getName());
        if (target != null) out.put("target_hash", "0x" + Long.toHexString(target & mask));
        out.put("tested", cands.size());
        out.put("lane_status", statusCounts);
        if (target != null) {
            out.put("matches", matches);
            out.put("resolved", !matches.isEmpty());
            if (!matches.isEmpty()) out.put("best_match", matches.get(0).get("api_name"));
        } else {
            out.put("hashes", all);
        }
        out.put("lowering", setup.describe());
        if (statusCounts.getOrDefault("returned", 0) < cands.size()) out.put("fault_example", run.faultExample(setup));
        out.put("execution", exec(run.outcome, run.setupMs));
        return Response.ok(out);
    }

    @McpTool(path = "/gpu_emulate_function_batch", method = "POST", description = "GPU twin of emulate_function, "
        + "batched: run one function once per input set (one GPU thread each) and return the requested output "
        + "registers per run. For fuzzing branches, tabulating a decoder, or brute-forcing a check routine.",
        category = "gpu", access = ToolAccess.READ_ONLY)
    public Response gpuEmulateFunctionBatch(
            @Param(value = "function", paramType = Param.FUNCTION_REF, source = ParamSource.BODY,
                   description = "Function to run (name or address)") String function,
            @Param(value = "inputs", source = ParamSource.BODY, fieldsJson = true,
                   description = "JSON array, one object per run: {\"ECX\": \"0x10\", \"buffer\": \"hex bytes placed in scratch memory\", "
                       + "\"args\": [\"buffer\", 5]} -- registers by name, an optional scratch buffer, and optional stack "
                       + "arguments in order (\"buffer\" = the buffer's address)") String inputsJson,
            @Param(value = "output_registers", source = ParamSource.BODY, defaultValue = "",
                   description = "Comma-separated registers to read back; empty = the return register") String outputRegisters,
            @Param(value = "buffer_register", source = ParamSource.BODY, defaultValue = "",
                   description = "Register that receives the scratch-buffer address when an input has 'buffer'") String bufferRegister,
            @Param(value = "max_steps", source = ParamSource.BODY, defaultValue = "100000", description = "P-code ops per run") int maxSteps,
            @Param(value = "limit", source = ParamSource.BODY, defaultValue = "1000", description = "Max runs listed") int limit,
            @Param(value = "backend", source = ParamSource.BODY, defaultValue = "auto", description = P_BACKEND) String backend,
            @Param(value = "program", defaultValue = "", description = P_PROGRAM) String programName) {
        ServiceUtils.ProgramOrError pe = ServiceUtils.getProgramOrError(programProvider, programName);
        if (pe.hasError()) return pe.error();
        Program p = pe.program();
        Backend be;
        try { be = Backend.parse(backend); } catch (IllegalArgumentException e) { return Response.err(e.getMessage()); }
        List<Map<String, Long>> lanes = new ArrayList<>();
        List<byte[]> buffers = new ArrayList<>();
        List<long[]> stackArgs = new ArrayList<>();
        Set<String> regNames = new java.util.LinkedHashSet<>();
        try {
            Object parsed = JsonHelper.parseJson("{\"c\":" + inputsJson + "}").get("c");
            if (!(parsed instanceof List<?> l)) return Response.err("inputs must be a JSON array of objects");
            for (Object x : l) {
                if (!(x instanceof Map<?, ?> m)) return Response.err("each input must be an object");
                Map<String, Long> regs = new LinkedHashMap<>();
                byte[] buf = null;
                long[] args = new long[0];
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    String k = String.valueOf(e.getKey());
                    if (k.equalsIgnoreCase("buffer")) { buf = parsePattern(String.valueOf(e.getValue()))[0]; continue; }
                    if (k.equalsIgnoreCase("args")) {
                        if (!(e.getValue() instanceof List<?> al)) return Response.err("args must be an array");
                        args = new long[al.size()];
                        for (int i = 0; i < args.length; i++) {
                            String v = String.valueOf(al.get(i));
                            args[i] = v.equalsIgnoreCase("buffer") ? SCRATCH_BASE : parseLong(v);
                        }
                        continue;
                    }
                    regs.put(k, parseLong(String.valueOf(e.getValue())));
                    regNames.add(k);
                }
                lanes.add(regs);
                buffers.add(buf == null ? new byte[0] : buf);
                stackArgs.add(args);
            }
        } catch (Exception e) {
            return Response.err("Invalid inputs: " + e.getMessage());
        }
        if (lanes.isEmpty()) return Response.err("no inputs");
        List<String> outs = new ArrayList<>();
        if (outputRegisters != null) for (String s : outputRegisters.split(",")) if (!s.isBlank()) outs.add(s.trim());
        EmuSetup setup;
        try {
            int longest = 0;
            for (byte[] b : buffers) longest = Math.max(longest, b.length);
            setup = setupEmulation(p, function, bufferRegister, outs.isEmpty() ? "" : outs.get(0), regNames, outs, longest);
        } catch (IllegalArgumentException e) {
            return Response.err(e.getMessage());
        }
        if (outs.isEmpty()) outs.add(setup.resultName);
        EmuRun run = runLanes(setup, lanes, buffers, stackArgs, bufferRegister, maxSteps, be);
        boolean bigE = p.getLanguage().isBigEndian();
        List<Map<String, Object>> rows = new ArrayList<>();
        Map<String, Integer> statusCounts = new TreeMap<>();
        for (int i = 0; i < lanes.size(); i++) {
            PcodeVm.Status st = run.result.status(i);
            statusCounts.merge(st.label(), 1, Integer::sum);
            if (rows.size() >= Math.max(1, limit)) continue;
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("run", i);
            m.put("status", st.label());
            m.put("steps", run.result.steps()[i]);
            Map<String, String> vals = new LinkedHashMap<>();
            for (String r : outs) {
                int[] slot = setup.slots.get(r.toUpperCase(Locale.ROOT));
                if (slot != null) vals.put(r, "0x" + Long.toHexString(run.result.readState(i, slot[0], slot[1], bigE)));
            }
            m.put("outputs", vals);
            rows.add(m);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("function", setup.fn.getName());
        out.put("runs", lanes.size());
        out.put("lane_status", statusCounts);
        out.put("results", rows);
        out.put("lowering", setup.describe());
        if (statusCounts.getOrDefault("returned", 0) < lanes.size()) out.put("fault_example", run.faultExample(setup));
        out.put("execution", exec(run.outcome, run.setupMs));
        return Response.ok(out);
    }

    /** Lowered function plus the register layout and memory windows lanes share. */
    static final class EmuSetup {
        Function fn;
        PcodeLowering.Lowered lowered;
        PcodeVm.Memory memory;
        Map<String, int[]> slots = new HashMap<>();
        int[] sp, result;
        String resultName;
        boolean returnOnStack;
        int returnOffset;
        int[] returnReg;
        int ptrSize;
        double lowerMs;
        int stackWindow, scratchWindow;

        Map<String, Object> describe() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("instructions", lowered.instructions);
            m.put("pcode_ops", lowered.program.nOps);
            m.put("lane_state_bytes", lowered.program.stateSize);
            m.put("unsupported_ops", lowered.unsupportedOps);
            if (!lowered.unsupportedKinds.isEmpty()) m.put("unsupported_kinds", lowered.unsupportedKinds);
            m.put("result_register", resultName);
            m.put("lower_ms", lowerMs);
            return m;
        }
    }

    /** Stack window: this much below the initial SP (locals) and above it (return address, args). */
    static final int STACK_BELOW = 0x4000, STACK_ABOVE = 0x1000;

    EmuSetup setupEmulation(Program p, String fnRef, String ptrRegister, String resultRegister,
                            Iterable<String> extraRegs, List<String> outRegs, int scratchBytes) {
        EmuSetup e = new EmuSetup();
        long t0 = System.nanoTime();
        ServiceUtils.FunctionOrError fe = ServiceUtils.getFunctionOrError(p, fnRef);
        if (fe.hasError()) {
            Address a = ServiceUtils.parseAddress(p, fnRef);
            Function f = a == null ? null : p.getFunctionManager().getFunctionAt(a);
            if (f == null) throw new IllegalArgumentException(fe.message());
            e.fn = f;
        } else {
            e.fn = fe.function();
        }
        Register sp = p.getCompilerSpec().getStackPointer();
        PrototypeModel cc = e.fn.getCallingConvention() != null ? e.fn.getCallingConvention()
            : p.getCompilerSpec().getDefaultCallingConvention();
        List<Register> regs = new ArrayList<>();
        regs.add(sp);
        Register result = null;
        if (resultRegister != null && !resultRegister.isBlank()) {
            result = p.getRegister(resultRegister);
            if (result == null) throw new IllegalArgumentException("unknown register " + resultRegister);
        } else if (cc != null) {
            var ret = cc.getReturnLocation(ghidra.program.model.data.IntegerDataType.dataType, p);
            if (ret != null && ret.isRegisterStorage()) result = ret.getRegister();
        }
        if (result == null) throw new IllegalArgumentException("cannot determine result register; pass result_register");
        regs.add(result);
        e.resultName = result.getName();
        if (ptrRegister != null && !ptrRegister.isBlank()) {
            Register r = p.getRegister(ptrRegister);
            if (r == null) throw new IllegalArgumentException("unknown register " + ptrRegister);
            regs.add(r);
        }
        for (String name : extraRegs) {
            Register r = p.getRegister(name);
            if (r == null) throw new IllegalArgumentException("unknown register " + name);
            regs.add(r);
        }
        for (String name : outRegs) {
            Register r = p.getRegister(name);
            if (r == null) throw new IllegalArgumentException("unknown register " + name);
            regs.add(r);
        }
        // Return address: a register (LR / RA) or a stack slot (x86).
        Varnode[] retAddr = cc == null ? null : cc.getReturnAddress();
        Register retReg = null;
        if (retAddr != null && retAddr.length == 1 && retAddr[0].isRegister()) {
            retReg = p.getRegister(retAddr[0].getAddress(), retAddr[0].getSize());
            if (retReg != null) regs.add(retReg);
        }
        e.ptrSize = p.getDefaultPointerSize();
        e.lowered = PcodeLowering.lower(p, e.fn, regs);
        for (Register r : regs) e.slots.put(r.getName().toUpperCase(Locale.ROOT), e.lowered.slot(r));
        e.sp = e.lowered.slot(sp);
        e.result = e.lowered.slot(result);
        if (retReg != null) {
            e.returnReg = e.lowered.slot(retReg);
        } else {
            e.returnOnStack = true;
            e.returnOffset = retAddr != null && retAddr.length == 1 && retAddr[0].getAddress().isStackAddress()
                ? (int) retAddr[0].getOffset() : 0;
        }
        // Windows are per lane on the device, so they are sized to the inputs, not generously.
        e.stackWindow = STACK_BELOW + STACK_ABOVE;
        e.scratchWindow = Math.max(256, (scratchBytes + 15) & ~15);
        e.memory = PcodeLowering.memory(p, new long[] {STACK_TOP - STACK_BELOW, SCRATCH_BASE},
            new int[] {e.stackWindow, e.scratchWindow}, maxBytes());
        e.lowerMs = ms(t0);
        return e;
    }

    record EmuRun(PcodeVm.Result result, Dispatch.Outcome<PcodeVm.Result> outcome, double setupMs) {
        Map<String, Object> faultExample(EmuSetup s) {
            for (int i = 0; i < result.status().length; i++) {
                PcodeVm.Status st = result.status(i);
                if (st == PcodeVm.Status.RETURNED) continue;
                int op = result.faultOp()[i];
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("lane", i);
                m.put("status", st.label());
                if (op >= 0 && op < s.lowered.program.nOps) {
                    int ins = s.lowered.program.instr[op];
                    m.put("at_instruction", "0x" + Long.toHexString(s.lowered.program.instrAddr[ins]));
                }
                m.put("detail", "0x" + Long.toHexString(result.faultAddr()[i]));
                m.put("hint", switch (st) {
                    case CALL -> "the function calls another function; the VM runs leaf code only";
                    case BAD_READ -> "read outside the image, stack and scratch windows (uninitialized global or pointer arg?)";
                    case BAD_WRITE -> "write outside the stack/scratch windows";
                    case UNSUPPORTED -> "floating point, >8-byte values or an unsupported P-code op";
                    case STEP_LIMIT -> "raise max_steps";
                    case BAD_BRANCH -> "jumped outside the function (tail call?) or to an unknown address";
                    default -> "";
                });
                return m;
            }
            return Map.of();
        }
    }

    /**
     * Builds per-lane state and private memory, then runs the VM. {@code stackArgs} (nullable,
     * per lane) are written in order where the calling convention expects stack arguments:
     * just above the return address when that lives on the stack, else from SP upward.
     */
    EmuRun runLanes(EmuSetup e, List<Map<String, Long>> laneRegs, List<byte[]> scratch, List<long[]> stackArgs,
                    String ptrRegister, int maxSteps, Backend be) {
        long t0 = System.nanoTime();
        PcodeVm.Program prog = e.lowered.program;
        int n = laneRegs.size();
        boolean bigE = prog.bigEndian;
        byte[] state = new byte[n * prog.stateSize];
        java.io.ByteArrayOutputStream priv = new java.io.ByteArrayOutputStream();
        long[] off = new long[n];
        int[] len = new int[n];
        long spVal = STACK_TOP;
        long stackPrivBase = e.memory.privOffset(spVal, e.ptrSize);
        int[] ptrSlot = ptrRegister == null || ptrRegister.isBlank() ? null : e.slots.get(ptrRegister.toUpperCase(Locale.ROOT));
        long scratchPriv = e.memory.privOffset(SCRATCH_BASE, 1);
        for (int i = 0; i < n; i++) {
            int base = i * prog.stateSize;
            for (Map.Entry<String, Long> r : laneRegs.get(i).entrySet()) {
                int[] s = e.slots.get(r.getKey().toUpperCase(Locale.ROOT));
                if (s != null) PcodeVmAccess.store(state, base + s[0], s[1], bigE, r.getValue());
            }
            PcodeVmAccess.store(state, base + e.sp[0], e.sp[1], bigE, spVal);
            if (e.returnReg != null) PcodeVmAccess.store(state, base + e.returnReg[0], e.returnReg[1], bigE, RETURN_SENTINEL);
            List<Long> pos = new ArrayList<>();
            List<byte[]> payload = new ArrayList<>();
            if (e.returnOnStack) {
                pos.add(stackPrivBase + e.returnOffset);
                payload.add(PcodeVmAccess.bytes(RETURN_SENTINEL, e.ptrSize, bigE));
            }
            long[] args = stackArgs == null ? new long[0] : stackArgs.get(i);
            long argBase = e.returnOnStack ? stackPrivBase + e.returnOffset + e.ptrSize : stackPrivBase;
            for (int k = 0; k < args.length; k++) {
                pos.add(argBase + (long) k * e.ptrSize);
                payload.add(PcodeVmAccess.bytes(args[k], e.ptrSize, bigE));
            }
            byte[] buf = scratch.get(i);
            if (buf.length > 0) {
                pos.add(scratchPriv);
                payload.add(buf.length > e.scratchWindow ? Arrays.copyOf(buf, e.scratchWindow) : buf);
                if (ptrSlot != null) PcodeVmAccess.store(state, base + ptrSlot[0], ptrSlot[1], bigE, SCRATCH_BASE);
                else if (e.returnOnStack && args.length == 0) {
                    // stack argument: first parameter right above the return address
                    pos.add(stackPrivBase + e.returnOffset + e.ptrSize);
                    payload.add(PcodeVmAccess.bytes(SCRATCH_BASE, e.ptrSize, bigE));
                }
            }
            byte[] rec = PcodeVm.privRecords(pos.stream().mapToLong(Long::longValue).toArray(), payload.toArray(new byte[0][]));
            off[i] = priv.size();
            len[i] = rec.length;
            priv.writeBytes(rec);
        }
        PcodeVm.Lanes lanes = new PcodeVm.Lanes(n, state, priv.toByteArray(), off, len);
        double setupMs = e.lowerMs + ms(t0);
        int steps = Math.max(1, Math.min(10_000_000, maxSteps));
        Dispatch.Outcome<PcodeVm.Result> o = Dispatch.run(Ops.PCODE_VM, be, n,
            () -> PcodeVm.runCpu(prog, e.memory, lanes, RETURN_SENTINEL, steps),
            s -> PcodeVm.runGpu(s, prog, e.memory, lanes, RETURN_SENTINEL, steps), PcodeVm.Result::same);
        return new EmuRun(o.value(), o, setupMs);
    }

    /** Small byte helpers, kept out of the public VM API. */
    static final class PcodeVmAccess {
        static void store(byte[] a, int off, int size, boolean be, long v) {
            if (be) for (int k = size - 1; k >= 0; k--) { a[off + k] = (byte) v; v >>>= 8; }
            else for (int k = 0; k < size; k++) { a[off + k] = (byte) v; v >>>= 8; }
        }

        static byte[] bytes(long v, int size, boolean be) {
            byte[] b = new byte[size];
            store(b, 0, size, be, v);
            return b;
        }
    }

    static Map<String, Long> parseRegs(String json) {
        Map<String, Long> m = new LinkedHashMap<>();
        if (json == null || json.isBlank()) return m;
        for (Map.Entry<String, Object> e : JsonHelper.parseJson(json).entrySet()) m.put(e.getKey(), parseLong(String.valueOf(e.getValue())));
        return m;
    }

    static long parseLong(String s) {
        s = s.trim();
        if (s.endsWith(".0")) s = s.substring(0, s.length() - 2);
        if (s.startsWith("0x") || s.startsWith("0X")) return new BigInteger(s.substring(2), 16).longValue();
        if (s.startsWith("-")) return Long.parseLong(s);
        return new BigInteger(s).longValue();
    }

    // =====================================================================
    // helpers
    // =====================================================================

    static long maxBytes() {
        String env = System.getenv("GHIDRA_GPU_MAX_SCAN_BYTES");
        try {
            if (env != null && !env.isBlank()) return Long.parseLong(env.trim());
        } catch (NumberFormatException ignored) {
            // default below
        }
        return 1L << 30;
    }

    static Map<String, Object> exec(Dispatch.Outcome<?> o, double extractMs) {
        Map<String, Object> m = new LinkedHashMap<>(o.meta());
        m.put("extract_ms", extractMs);
        return m;
    }

    static boolean sameHitLists(List<ScanOps.Hits> a, List<ScanOps.Hits> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) {
            ScanOps.Hits x = a.get(i), y = b.get(i);
            if (x == null || y == null) { if (x != y) return false; continue; }
            if (!ScanOps.Hits.same(x, y)) return false;
        }
        return true;
    }

    static double ms(long t0) {
        return Math.round((System.nanoTime() - t0) / 1e4) / 100.0;
    }

    static double round3(double v) {
        return Math.round(v * 1000.0) / 1000.0;
    }
}
