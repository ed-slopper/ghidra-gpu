package com.xebyte.gpu.ghidra;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.xebyte.gpu.GraphOps;
import com.xebyte.gpu.ScanOps;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressSetView;
import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;
import ghidra.program.model.block.CodeBlockReference;
import ghidra.program.model.block.CodeBlockReferenceIterator;
import ghidra.program.model.lang.OperandType;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.listing.Program;
import ghidra.program.model.mem.MemoryAccessException;
import ghidra.program.model.mem.MemoryBlock;
import ghidra.program.model.scalar.Scalar;
import ghidra.program.model.symbol.RefType;
import ghidra.program.model.symbol.Reference;
import ghidra.util.exception.CancelledException;
import ghidra.util.task.TaskMonitor;

/**
 * Pulls flat arrays out of a Ghidra {@link Program} for the GPU ops: memory bytes,
 * the call graph, per-function CFGs, and normalized instruction streams.
 *
 * <p>Extraction runs on the CPU (the program database lives in the JVM) and is shared by
 * both backends, so a CPU/GPU timing comparison measures the op, not the extraction.
 * Tools report extraction time separately.
 */
public final class ProgramData {

    private ProgramData() {}

    // ------------------------------------------------------------------ memory

    /** One initialized memory block's bytes. */
    public record Region(MemoryBlock block, byte[] bytes) {
        public Address addressAt(long offset) {
            return block.getStart().add(offset);
        }

        public String name() {
            return block.getName();
        }
    }

    /**
     * Initialized, loaded blocks (optionally executable only, optionally one block by
     * name). Stops adding blocks once {@code maxBytes} is reached and reports that through
     * {@link Regions#truncated}.
     */
    public record Regions(List<Region> regions, long bytes, boolean truncated, List<String> skipped) {
        public Map<String, Object> describe() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("blocks", regions.size());
            m.put("bytes", bytes);
            if (truncated) m.put("truncated_at_bytes", bytes);
            if (!skipped.isEmpty()) m.put("skipped_blocks", skipped);
            return m;
        }
    }

    public static Regions regions(Program p, boolean execOnly, String blockName, long maxBytes) {
        List<Region> out = new ArrayList<>();
        List<String> skipped = new ArrayList<>();
        long total = 0;
        boolean truncated = false;
        for (MemoryBlock b : p.getMemory().getBlocks()) {
            if (!b.isInitialized() || !b.isLoaded()) continue;
            if (execOnly && !b.isExecute()) continue;
            if (blockName != null && !blockName.isBlank() && !b.getName().equalsIgnoreCase(blockName)) continue;
            long size = b.getSize();
            if (size > Integer.MAX_VALUE - 16) { skipped.add(b.getName() + " (larger than 2 GiB)"); continue; }
            if (total + size > maxBytes) { truncated = true; skipped.add(b.getName() + " (over max_bytes)"); continue; }
            byte[] bytes = new byte[(int) size];
            try {
                int got = b.getBytes(b.getStart(), bytes);
                if (got < bytes.length) bytes = java.util.Arrays.copyOf(bytes, Math.max(0, got));
            } catch (MemoryAccessException e) {
                skipped.add(b.getName() + " (" + e.getMessage() + ")");
                continue;
            }
            out.add(new Region(b, bytes));
            total += bytes.length;
        }
        return new Regions(out, total, truncated, skipped);
    }

    // ------------------------------------------------------------------ functions

    public static List<Function> functions(Program p) {
        List<Function> out = new ArrayList<>();
        FunctionIterator it = p.getFunctionManager().getFunctions(true);
        while (it.hasNext()) out.add(it.next());
        return out;
    }

    // ------------------------------------------------------------------ call graph

    /** Internal + external functions as nodes; an edge per distinct caller->callee pair. */
    public record CallGraph(List<Function> functions, Map<Function, Integer> index, GraphOps.Csr csr) {}

    public static CallGraph callGraph(Program p, TaskMonitor monitor) throws CancelledException {
        List<Function> fns = functions(p);
        Map<Function, Integer> index = new HashMap<>();
        for (Function f : fns) index.put(f, index.size());
        FunctionIterator ext = p.getFunctionManager().getExternalFunctions();
        while (ext.hasNext()) {
            Function f = ext.next();
            index.put(f, index.size());
            fns.add(f);
        }
        List<Integer> src = new ArrayList<>(), dst = new ArrayList<>();
        int internal = fns.size();
        for (int i = 0; i < internal; i++) {
            Function f = fns.get(i);
            if (f.isExternal()) continue;
            monitor.checkCancelled();
            for (Function callee : f.getCalledFunctions(monitor)) {
                Integer j = index.get(callee);
                if (j == null) { j = index.size(); index.put(callee, j); fns.add(callee); }
                src.add(i);
                dst.add(j);
            }
        }
        return new CallGraph(fns, index, GraphOps.Csr.of(fns.size(), toInts(src), toInts(dst)));
    }

    // ------------------------------------------------------------------ CFGs

    /** Basic blocks of many functions; block 0 of each is the one holding the entry point. */
    public record CfgBatch(List<Function> functions, List<List<CodeBlock>> blocks, int[] edges, GraphOps.Cfgs cfgs) {}

    public static CfgBatch cfgs(Program p, List<Function> fns, TaskMonitor monitor) throws CancelledException {
        BasicBlockModel model = new BasicBlockModel(p);
        List<int[][]> graphs = new ArrayList<>();
        List<List<CodeBlock>> allBlocks = new ArrayList<>();
        int[] edgeCounts = new int[fns.size()];
        for (int fi = 0; fi < fns.size(); fi++) {
            Function f = fns.get(fi);
            monitor.checkCancelled();
            List<CodeBlock> blocks = new ArrayList<>();
            CodeBlockIterator it = model.getCodeBlocksContaining(f.getBody(), monitor);
            while (it.hasNext()) blocks.add(it.next());
            // entry block first
            for (int i = 0; i < blocks.size(); i++) {
                if (blocks.get(i).contains(f.getEntryPoint())) {
                    CodeBlock e = blocks.remove(i);
                    blocks.add(0, e);
                    break;
                }
            }
            Map<Address, Integer> byStart = new HashMap<>();
            for (int i = 0; i < blocks.size(); i++) byStart.put(blocks.get(i).getFirstStartAddress(), i);
            List<Integer> s = new ArrayList<>(), d = new ArrayList<>();
            for (int i = 0; i < blocks.size(); i++) {
                CodeBlockReferenceIterator refs = blocks.get(i).getDestinations(monitor);
                while (refs.hasNext()) {
                    CodeBlockReference r = refs.next();
                    if (r.getFlowType().isCall()) continue;
                    Integer j = byStart.get(r.getDestinationAddress());
                    if (j != null) { s.add(i); d.add(j); }
                }
            }
            edgeCounts[fi] = s.size();
            graphs.add(GraphOps.Cfgs.graph(blocks.size(), toInts(s), toInts(d)));
            allBlocks.add(blocks);
        }
        return new CfgBatch(fns, allBlocks, edgeCounts, GraphOps.Cfgs.of(graphs));
    }

    // ------------------------------------------------------------------ instruction streams

    public static List<Instruction> instructions(Program p, Function f) {
        List<Instruction> out = new ArrayList<>();
        InstructionIterator it = p.getListing().getInstructions(f.getBody(), true);
        while (it.hasNext()) out.add(it.next());
        return out;
    }

    /**
     * {@code fn} over {@code items}, results in input order. Sequential by default:
     * extraction is program-database reads, and Ghidra's database lock serialises them --
     * measured on kernel32, 16 threads made hash-text extraction 2.2x SLOWER (1,659 vs
     * 762 ms) and token extraction 1.7x slower (614 vs 354 ms) through lock contention.
     * -Dghidra.gpu.parallelExtract=true opts in, for Ghidra builds where that changes.
     */
    public static <T, R> List<R> parallelMap(List<T> items, java.util.function.Function<T, R> fn) {
        if (items.size() < 64 || !Boolean.getBoolean("ghidra.gpu.parallelExtract")) {
            List<R> out = new ArrayList<>(items.size());
            for (T t : items) out.add(fn.apply(t));
            return out;
        }
        java.util.concurrent.ForkJoinPool pool = new java.util.concurrent.ForkJoinPool(
            Math.max(1, Runtime.getRuntime().availableProcessors()));
        try {
            return pool.submit(() -> items.parallelStream().map(fn).toList()).get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted", e);
        } catch (java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException(e.getCause());
        } finally {
            pool.shutdown();
        }
    }

    /** Mnemonic strings of a function, in address order (intern them to tokens afterwards). */
    public static String[] mnemonics(Program p, Function f) {
        List<Instruction> ins = instructions(p, f);
        String[] m = new String[ins.size()];
        for (int i = 0; i < m.length; i++) m[i] = ins.get(i).getMnemonicString();
        return m;
    }

    /** Interns mnemonic strings into {@code vocab}; sequential, so ids are deterministic. */
    public static int[] intern(String[] mnemonics, Map<String, Integer> vocab) {
        int[] t = new int[mnemonics.length];
        for (int i = 0; i < t.length; i++) t[i] = vocab.computeIfAbsent(mnemonics[i], k -> vocab.size());
        return t;
    }

    /** Mnemonic token ids for a function, interning into {@code vocab}. */
    public static int[] mnemonicTokens(Program p, Function f, Map<String, Integer> vocab) {
        List<Instruction> ins = instructions(p, f);
        int[] t = new int[ins.size()];
        for (int i = 0; i < t.length; i++) {
            t[i] = vocab.computeIfAbsent(ins.get(i).getMnemonicString(), k -> vocab.size());
        }
        return t;
    }

    /**
     * The normalized text {@code get_function_hash} hashes, reproduced exactly from
     * {@code DocumentationHashService.computeNormalizedFunctionHash} so that SHA-256 of it
     * equals the upstream hash (the GPU tool's hashes are interchangeable with it).
     */
    public static byte[] normalizedText(Program program, Function func) {
        StringBuilder normalized = new StringBuilder();
        Listing listing = program.getListing();
        AddressSetView functionBody = func.getBody();
        InstructionIterator instructions = listing.getInstructions(functionBody, true);
        Address funcStart = func.getEntryPoint();
        while (instructions.hasNext()) {
            Instruction instr = instructions.next();
            normalized.append(instr.getMnemonicString()).append(" ");
            int numOperands = instr.getNumOperands();
            for (int i = 0; i < numOperands; i++) {
                int opType = instr.getOperandType(i);
                boolean isAddressRef = (opType & OperandType.ADDRESS) != 0
                    || (opType & OperandType.CODE) != 0 || (opType & OperandType.DATA) != 0;
                if (isAddressRef) {
                    Reference[] refs = instr.getOperandReferences(i);
                    if (refs.length > 0) {
                        Address targetAddr = refs[0].getToAddress();
                        if (functionBody.contains(targetAddr)) {
                            normalized.append("REL+").append(targetAddr.subtract(funcStart));
                        } else {
                            RefType refType = refs[0].getReferenceType();
                            if (refType.isCall()) normalized.append("CALL_EXT");
                            else if (refType.isData()) normalized.append("DATA_EXT");
                            else normalized.append("EXT_REF");
                        }
                    } else {
                        normalized.append("ADDR");
                    }
                } else if ((opType & OperandType.REGISTER) != 0) {
                    normalized.append(instr.getDefaultOperandRepresentation(i));
                } else if ((opType & OperandType.SCALAR) != 0) {
                    Object[] opObjects = instr.getOpObjects(i);
                    if (opObjects.length > 0 && opObjects[0] instanceof Scalar scalar) {
                        long value = scalar.getValue();
                        normalized.append(Math.abs(value) < 0x10000 ? "IMM:" + value : "IMM_LARGE");
                    } else {
                        normalized.append(instr.getDefaultOperandRepresentation(i));
                    }
                } else {
                    normalized.append(instr.getDefaultOperandRepresentation(i));
                }
                if (i < numOperands - 1) normalized.append(",");
            }
            normalized.append(";");
        }
        return normalized.toString().getBytes(StandardCharsets.UTF_8);
    }

    /**
     * The text {@code archive_ingest_program} hashes per function, reproduced exactly from
     * {@code DocumentationHashService.computeOpcodeHash}: mnemonic, then {@code |operandType}
     * per operand, then {@code ;}.
     */
    public static byte[] opcodeText(Program program, Function fn) {
        StringBuilder sb = new StringBuilder();
        InstructionIterator it = program.getListing().getInstructions(fn.getBody(), true);
        while (it.hasNext()) {
            Instruction inst = it.next();
            sb.append(inst.getMnemonicString());
            int n = inst.getNumOperands();
            for (int i = 0; i < n; i++) sb.append('|').append(inst.getOperandType(i));
            sb.append(';');
        }
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    /** Each instruction rendered as "MNEMONIC operands" (what search_instructions matches). */
    public record InstructionText(List<Address> addresses, List<String> text, ScanOps.Packed packed) {}

    public static InstructionText instructionText(Program p, AddressSetView where, int max) {
        List<Address> addrs = new ArrayList<>();
        List<String> text = new ArrayList<>();
        InstructionIterator it = where == null ? p.getListing().getInstructions(true) : p.getListing().getInstructions(where, true);
        while (it.hasNext() && addrs.size() < max) {
            Instruction ins = it.next();
            addrs.add(ins.getAddress());
            text.add(ins.toString());
        }
        return new InstructionText(addrs, text, ScanOps.Packed.of(text));
    }

    static int[] toInts(List<Integer> l) {
        int[] a = new int[l.size()];
        for (int i = 0; i < a.length; i++) a[i] = l.get(i);
        return a;
    }
}
