package com.xebyte.gpu.ghidra;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import com.xebyte.gpu.PcodeVm;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressSpace;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.Program;
import ghidra.program.model.mem.MemoryAccessException;
import ghidra.program.model.mem.MemoryBlock;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.Varnode;

/**
 * Lowers a function's raw P-code ({@link Instruction#getPcode()}) into a
 * {@link PcodeVm.Program} both VM backends execute.
 *
 * <ul>
 *   <li>Register and unique varnodes are packed into a compact per-lane state block:
 *       touched byte ranges are merged into intervals and laid out back to back, so
 *       aliasing registers (EAX/AX/AL) still share bytes exactly as in Ghidra's register
 *       space, and lane state is a few hundred bytes instead of the whole register file.</li>
 *   <li>Branch targets are resolved to absolute op indexes: relative (constant-space)
 *       targets within the instruction, and address targets through the function's
 *       instruction map. A target outside the function becomes -1 and stops the lane.</li>
 *   <li>Fall-through is made explicit with a BRANCH whenever the next instruction in
 *       address order is not the fall-through (gaps, delay slots).</li>
 *   <li>Ops the VM cannot run (varnodes wider than 8 bytes, floating point, other
 *       spaces) are emitted as {@link PcodeVm#OP_UNSUPPORTED}; they only matter if a
 *       lane reaches them, and then it stops with status {@code unsupported}.</li>
 * </ul>
 */
public final class PcodeLowering {

    private PcodeLowering() {}

    public static final class Lowered {
        public final PcodeVm.Program program;
        public final int instructions;
        public final int unsupportedOps;
        public final List<String> unsupportedKinds;
        private final Map<String, int[]> registerSlots;
        private final Program ghidraProgram;

        Lowered(PcodeVm.Program program, int instructions, int unsupportedOps, List<String> unsupportedKinds,
                Map<String, int[]> registerSlots, Program ghidraProgram) {
            this.program = program;
            this.instructions = instructions;
            this.unsupportedOps = unsupportedOps;
            this.unsupportedKinds = unsupportedKinds;
            this.registerSlots = registerSlots;
            this.ghidraProgram = ghidraProgram;
        }

        /** {stateOffset, size} of a register that was requested at lowering time. */
        public int[] slot(Register r) {
            int[] s = registerSlots.get(r.getName().toUpperCase());
            if (s == null) throw new IllegalArgumentException("register " + r.getName() + " was not mapped");
            return s;
        }

        public Program ghidraProgram() {
            return ghidraProgram;
        }
    }

    /** Merged byte intervals of one space -> compact state offsets. */
    private static final class Packer {
        final TreeMap<Long, Long> raw = new TreeMap<>(); // start -> end (exclusive)
        final TreeMap<Long, long[]> merged = new TreeMap<>(); // start -> {end, base}

        void add(long off, int size) {
            long end = off + size;
            Long prev = raw.get(off);
            if (prev == null || prev < end) raw.put(off, end);
        }

        int layout(int base) {
            long curS = Long.MIN_VALUE, curE = Long.MIN_VALUE;
            for (Map.Entry<Long, Long> e : raw.entrySet()) {
                if (curE != Long.MIN_VALUE && e.getKey() <= curE) {
                    curE = Math.max(curE, e.getValue());
                } else {
                    if (curE != Long.MIN_VALUE) { merged.put(curS, new long[] {curE, base}); base += align8(curE - curS); }
                    curS = e.getKey();
                    curE = e.getValue();
                }
            }
            if (curE != Long.MIN_VALUE) { merged.put(curS, new long[] {curE, base}); base += align8(curE - curS); }
            return base;
        }

        long offset(long off) {
            Map.Entry<Long, long[]> e = merged.floorEntry(off);
            return e.getValue()[1] + (off - e.getKey());
        }

        static int align8(long n) {
            return (int) ((n + 7) & ~7L);
        }
    }

    public static Lowered lower(Program program, Function fn, Collection<Register> registers) {
        List<Instruction> ins = ProgramData.instructions(program, fn);
        int n = ins.size();
        List<PcodeOp[]> pcode = new ArrayList<>(n);
        Map<Address, Integer> instrIndex = new HashMap<>();
        Packer regs = new Packer(), uniq = new Packer();
        for (int i = 0; i < n; i++) {
            Instruction in = ins.get(i);
            instrIndex.put(in.getAddress(), i);
            PcodeOp[] ops = in.getPcode();
            pcode.add(ops);
            for (PcodeOp op : ops) {
                collect(op.getOutput(), regs, uniq);
                for (Varnode v : op.getInputs()) collect(v, regs, uniq);
            }
        }
        for (Register r : registers) regs.add(r.getOffset(), r.getMinimumByteSize());
        int regEnd = regs.layout(0);
        int stateSize = uniq.layout(regEnd);

        // pass 1: op counts per instruction, with explicit fall-through branches
        int[] firstOp = new int[n];
        boolean[] needsFall = new boolean[n];
        int count = 0;
        for (int i = 0; i < n; i++) {
            firstOp[i] = count;
            count += pcode.get(i).length;
            Address ft = ins.get(i).getFallThrough();
            if (ft != null && (i + 1 >= n || !ins.get(i + 1).getAddress().equals(ft))) {
                needsFall[i] = true;
                count++;
            }
        }

        int[] opcode = new int[count], instr = new int[count], nIn = new int[count], kind = new int[4 * count], size = new int[4 * count];
        long[] val = new long[4 * count];
        int unsupported = 0;
        List<String> unsupportedKinds = new ArrayList<>();
        int o = 0;
        for (int i = 0; i < n; i++) {
            PcodeOp[] ops = pcode.get(i);
            for (int k = 0; k < ops.length; k++, o++) {
                PcodeOp op = ops[k];
                instr[o] = i;
                int code = op.getOpcode();
                String why = supportReason(op);
                Varnode[] inputs = op.getInputs();
                nIn[o] = Math.min(3, inputs.length);
                if (why != null) {
                    opcode[o] = PcodeVm.OP_UNSUPPORTED;
                    unsupported++;
                    if (unsupportedKinds.size() < 16 && !unsupportedKinds.contains(why)) unsupportedKinds.add(why);
                    continue;
                }
                opcode[o] = code;
                encode(op.getOutput(), 4 * o, kind, size, val, regs, uniq);
                for (int j = 0; j < nIn[o]; j++) encode(inputs[j], 4 * o + 1 + j, kind, size, val, regs, uniq);
                if (code == PcodeOp.BRANCH || code == PcodeOp.CBRANCH) {
                    Varnode t = inputs[0];
                    long target;
                    if (t.isConstant()) {
                        target = firstOp[i] + k + (int) t.getOffset();
                    } else {
                        Integer ti = instrIndex.get(t.getAddress());
                        target = ti == null ? -1 : firstOp[ti];
                    }
                    kind[4 * o + 1] = PcodeVm.KIND_CONST;
                    size[4 * o + 1] = 8;
                    val[4 * o + 1] = target;
                } else if (code == PcodeOp.CALL) {
                    kind[4 * o + 1] = PcodeVm.KIND_CONST;
                    size[4 * o + 1] = 8;
                    val[4 * o + 1] = inputs[0].getOffset();
                }
            }
            if (needsFall[i]) {
                Address ft = ins.get(i).getFallThrough();
                Integer ti = instrIndex.get(ft);
                instr[o] = i;
                opcode[o] = PcodeOp.BRANCH;
                nIn[o] = 1;
                kind[4 * o + 1] = PcodeVm.KIND_CONST;
                size[4 * o + 1] = 8;
                val[4 * o + 1] = ti == null ? -1 : firstOp[ti];
                o++;
            }
        }

        long[] addrs = new long[n];
        for (int i = 0; i < n; i++) addrs[i] = ins.get(i).getAddress().getOffset();
        boolean be = program.getLanguage().isBigEndian();
        PcodeVm.Program vm = new PcodeVm.Program(count, opcode, instr, nIn, kind, size, val, firstOp, addrs, be, stateSize);

        Map<String, int[]> slots = new HashMap<>();
        for (Register r : registers) {
            slots.put(r.getName().toUpperCase(), new int[] {(int) regs.offset(r.getOffset()), r.getMinimumByteSize()});
        }
        return new Lowered(vm, n, unsupported, unsupportedKinds, slots, program);
    }

    private static void collect(Varnode v, Packer regs, Packer uniq) {
        if (v == null || v.getSize() > 8) return;
        if (v.isRegister()) regs.add(v.getOffset(), v.getSize());
        else if (v.isUnique()) uniq.add(v.getOffset(), v.getSize());
    }

    private static void encode(Varnode v, int slot, int[] kind, int[] size, long[] val, Packer regs, Packer uniq) {
        if (v == null) { kind[slot] = PcodeVm.KIND_NONE; return; }
        size[slot] = v.getSize();
        if (v.isConstant()) { kind[slot] = PcodeVm.KIND_CONST; val[slot] = v.getOffset(); }
        else if (v.isRegister()) { kind[slot] = PcodeVm.KIND_STATE; val[slot] = regs.offset(v.getOffset()); }
        else if (v.isUnique()) { kind[slot] = PcodeVm.KIND_STATE; val[slot] = uniq.offset(v.getOffset()); }
        else { kind[slot] = PcodeVm.KIND_RAM; val[slot] = v.getOffset(); }
    }

    /** Null when the VM can run {@code op}; otherwise why not. */
    static String supportReason(PcodeOp op) {
        int c = op.getOpcode();
        boolean known = (c >= PcodeOp.COPY && c <= PcodeOp.BOOL_OR) || c == PcodeOp.PIECE || c == PcodeOp.SUBPIECE
            || c == PcodeOp.POPCOUNT || c == PcodeOp.LZCOUNT;
        if (!known) return "opcode " + op.getMnemonic();
        if (op.getOutput() != null && op.getOutput().getSize() > 8) return op.getMnemonic() + " wider than 8 bytes";
        Varnode[] in = op.getInputs();
        for (int j = 0; j < in.length; j++) {
            Varnode v = in[j];
            if (v.getSize() > 8 && !(j == 0 && (c == PcodeOp.BRANCH || c == PcodeOp.CBRANCH || c == PcodeOp.CALL))) {
                return op.getMnemonic() + " wider than 8 bytes";
            }
            AddressSpace sp = v.getAddress().getAddressSpace();
            if (!v.isConstant() && !v.isRegister() && !v.isUnique() && !sp.isLoadedMemorySpace()) {
                return op.getMnemonic() + " on space " + sp.getName();
            }
        }
        return null;
    }

    /** Read-only image of every initialized loaded block, capped at {@code maxBytes}. */
    public static PcodeVm.Memory memory(Program program, long[] winBase, int[] winSize, long maxBytes) {
        List<MemoryBlock> blocks = new ArrayList<>();
        long total = 0;
        for (MemoryBlock b : program.getMemory().getBlocks()) {
            if (!b.isInitialized() || !b.isLoaded() || b.isOverlay()) continue;
            if (total + b.getSize() > maxBytes || b.getSize() > Integer.MAX_VALUE) continue;
            blocks.add(b);
            total += b.getSize();
        }
        blocks.sort((x, y) -> Long.compareUnsigned(x.getStart().getOffset(), y.getStart().getOffset()));
        byte[] image = new byte[(int) total];
        long[] base = new long[blocks.size()], off = new long[blocks.size()];
        int[] size = new int[blocks.size()];
        long o = 0;
        for (int i = 0; i < blocks.size(); i++) {
            MemoryBlock b = blocks.get(i);
            base[i] = b.getStart().getOffset();
            size[i] = (int) b.getSize();
            off[i] = o;
            try {
                b.getBytes(b.getStart(), image, (int) o, size[i]);
            } catch (MemoryAccessException e) {
                size[i] = 0; // unreadable block: present in the table, but no address resolves into it
            }
            o += b.getSize();
        }
        return new PcodeVm.Memory(winBase, winSize, image, base, size, off);
    }
}
