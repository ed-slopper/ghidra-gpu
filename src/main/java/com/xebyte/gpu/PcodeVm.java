package com.xebyte.gpu;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import com.xebyte.gpu.cuda.Gpu;

/**
 * A batched raw-P-code interpreter: one compiled function, many independent lanes.
 *
 * <p>The GPU kernel ({@code pcode_vm} in ghidra_gpu.cu) and {@link #runCpu} implement the
 * same machine; {@link com.xebyte.gpu.ghidra.PcodeLowering} builds the {@link Program} from
 * Ghidra's {@code Instruction.getPcode()}. Varnodes are pre-resolved by the lowering:
 * register and unique varnodes become byte offsets into a lane's private state block,
 * constants carry their value, RAM varnodes keep their address, and branch targets are
 * absolute op indexes (stored as CONST, -1 when the target is outside the function).
 *
 * <p>Memory a lane can address: its private read/write windows (stack, scratch) and the
 * shared read-only program image. Anything else faults the lane with a status instead
 * of inventing a value, so a wrong answer is never silently produced.
 *
 * <p>Supported: integer, boolean, compare, shift, carry/borrow, PIECE/SUBPIECE, POPCOUNT,
 * LZCOUNT, COPY/LOAD/STORE, BRANCH/CBRANCH/BRANCHIND/RETURN; varnodes up to 8 bytes.
 * Calls and floating point stop the lane ({@link Status#CALL}, {@link Status#UNSUPPORTED}).
 */
public final class PcodeVm {

    private PcodeVm() {}

    public static final int KIND_NONE = 0, KIND_STATE = 1, KIND_CONST = 2, KIND_RAM = 3;
    /** Size of one packed op record; must equal sizeof(PcOp) in ghidra_gpu.cu. */
    public static final int OP_BYTES = 80;
    /** Opcode the lowering emits for a raw op the VM cannot execute. */
    public static final int OP_UNSUPPORTED = 999;

    public enum Status {
        RUNNING, RETURNED, STEP_LIMIT, BAD_READ, BAD_WRITE, UNSUPPORTED, DIV_ZERO, BAD_BRANCH, CALL;

        public String label() {
            return name().toLowerCase();
        }
    }

    /** The lowered function. Slot 0 of each varnode array is the output, 1..3 the inputs. */
    public static final class Program {
        public final int nOps;
        public final int[] opcode, instr, nIn;
        public final int[] kind, size;   // [nOps*4]
        public final long[] val;         // [nOps*4]
        public final int[] instrFirstOp;
        public final long[] instrAddr;   // sorted unsigned
        public final boolean bigEndian;
        public final int stateSize;

        public Program(int nOps, int[] opcode, int[] instr, int[] nIn, int[] kind, int[] size, long[] val,
                       int[] instrFirstOp, long[] instrAddr, boolean bigEndian, int stateSize) {
            this.nOps = nOps;
            this.opcode = opcode;
            this.instr = instr;
            this.nIn = nIn;
            this.kind = kind;
            this.size = size;
            this.val = val;
            this.instrFirstOp = instrFirstOp;
            this.instrAddr = instrAddr;
            this.bigEndian = bigEndian;
            this.stateSize = Math.max(8, stateSize);
        }

        /** Packs ops into the device layout: struct PcOp, little-endian, 80 bytes each. */
        public byte[] pack() {
            ByteBuffer b = ByteBuffer.allocate(Math.max(1, nOps) * OP_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < nOps; i++) {
                b.putInt(opcode[i]).putInt(instr[i]);
                for (int v = 0; v < 4; v++) b.putInt(kind[4 * i + v]).putInt(size[4 * i + v]).putLong(val[4 * i + v]);
                b.putInt(nIn[i]).putInt(0);
            }
            return b.array();
        }
    }

    /** Read/write windows private to each lane, plus the read-only image segments. */
    public static final class Memory {
        public final long[] winBase;
        public final int[] winSize;
        public final long[] winOff;
        public final long privSize;
        public final byte[] image;
        public final long[] segBase;   // sorted unsigned
        public final int[] segSize;
        public final long[] segOff;

        public Memory(long[] winBase, int[] winSize, byte[] image, long[] segBase, int[] segSize, long[] segOff) {
            this.winBase = winBase;
            this.winSize = winSize;
            this.winOff = new long[winBase.length];
            long o = 0;
            for (int i = 0; i < winBase.length; i++) { winOff[i] = o; o += winSize[i]; }
            this.privSize = Math.max(8, o);
            this.image = image;
            this.segBase = segBase;
            this.segSize = segSize;
            this.segOff = segOff;
        }

        /** Offset into a lane's private block of {@code addr}, or -1 when outside every window. */
        public long privOffset(long addr, int len) {
            for (int w = 0; w < winBase.length; w++) {
                if (within(addr, len, winBase[w], winSize[w])) return winOff[w] + (addr - winBase[w]);
            }
            return -1;
        }
    }

    static boolean within(long addr, int len, long base, long size) {
        return Long.compareUnsigned(addr, base) >= 0
            && Long.compareUnsigned(addr + len, base + size) <= 0
            && Long.compareUnsigned(addr + len, addr) >= 0;
    }

    /**
     * Per-lane inputs. {@code stateInit} is nLanes*stateSize bytes. {@code privInit} packs,
     * per lane, records of [u32 privOffset][u32 len][bytes] (little-endian headers).
     */
    public record Lanes(int count, byte[] stateInit, byte[] privInit, long[] privInitOff, int[] privInitLen) {}

    public record Result(int[] status, int[] steps, int[] faultOp, long[] faultAddr, byte[] state, int stateSize) {

        public Status status(int lane) {
            return Status.values()[status[lane]];
        }

        public long readState(int lane, int offset, int size, boolean bigEndian) {
            return load(state, (long) lane * stateSize + offset, size, bigEndian);
        }

        public static boolean same(Result a, Result b) {
            return Arrays.equals(a.status, b.status) && Arrays.equals(a.steps, b.steps)
                && Arrays.equals(a.faultOp, b.faultOp) && Arrays.equals(a.faultAddr, b.faultAddr)
                && Arrays.equals(a.state, b.state);
        }
    }

    // ------------------------------------------------------------------ helpers shared by both

    static long maskOf(int size) {
        return size >= 8 ? -1L : (1L << (8 * size)) - 1;
    }

    static long sext(long v, int size) {
        if (size >= 8) return v;
        int sh = 64 - 8 * size;
        return (v << sh) >> sh;
    }

    static long load(byte[] a, long off, int size, boolean be) {
        long v = 0;
        int o = (int) off;
        if (be) for (int k = 0; k < size; k++) v = (v << 8) | (a[o + k] & 0xFF);
        else for (int k = size - 1; k >= 0; k--) v = (v << 8) | (a[o + k] & 0xFF);
        return v;
    }

    static void store(byte[] a, long off, int size, boolean be, long v) {
        int o = (int) off;
        if (be) for (int k = size - 1; k >= 0; k--) { a[o + k] = (byte) v; v >>>= 8; }
        else for (int k = 0; k < size; k++) { a[o + k] = (byte) v; v >>>= 8; }
    }

    /** Builds a lane's private-init record list. */
    public static byte[] privRecords(long[] privOffsets, byte[][] payloads) {
        int total = 0;
        for (byte[] p : payloads) total += 8 + p.length;
        ByteBuffer b = ByteBuffer.allocate(total).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < payloads.length; i++) {
            b.putInt((int) privOffsets[i]).putInt(payloads[i].length).put(payloads[i]);
        }
        return b.array();
    }

    // ------------------------------------------------------------------ CPU interpreter

    public static Result runCpu(Program p, Memory m, Lanes lanes, long returnSentinel, int maxSteps) {
        int n = lanes.count();
        int[] status = new int[n], steps = new int[n], faultOp = new int[n];
        long[] faultAddr = new long[n];
        byte[] state = Arrays.copyOf(lanes.stateInit(), n * p.stateSize);
        byte[] priv = new byte[(int) m.privSize];
        boolean be = p.bigEndian;
        for (int lane = 0; lane < n; lane++) {
            Arrays.fill(priv, (byte) 0);
            {
                int q = (int) lanes.privInitOff()[lane], end = q + lanes.privInitLen()[lane];
                while (q + 8 <= end) {
                    long po = load(lanes.privInit(), q, 4, false) & 0xFFFFFFFFL;
                    long ln = load(lanes.privInit(), q + 4, 4, false) & 0xFFFFFFFFL;
                    q += 8;
                    for (long k = 0; k < ln && po + k < m.privSize; k++) priv[(int) (po + k)] = lanes.privInit()[(int) (q + k)];
                    q += (int) ln;
                }
            }
            long st = (long) lane * p.stateSize;
            int pc = 0, count = 0;
            Status code = Status.RUNNING;
            long fa = 0;
            long[] in = new long[3];
            while (code == Status.RUNNING) {
                if (pc < 0 || pc >= p.nOps) { code = Status.BAD_BRANCH; break; }
                if (count >= maxSteps) { code = Status.STEP_LIMIT; break; }
                count++;
                int b4 = 4 * pc;
                in[0] = in[1] = in[2] = 0;
                boolean bad = false;
                for (int k = 0; k < p.nIn[pc] && k < 3; k++) {
                    int vk = p.kind[b4 + 1 + k], vs = p.size[b4 + 1 + k];
                    long vv = p.val[b4 + 1 + k];
                    if (vk == KIND_CONST) in[k] = vv & maskOf(vs);
                    else if (vk == KIND_STATE) in[k] = load(state, st + vv, vs, be);
                    else if (vk == KIND_RAM) {
                        if (!readable(m, vv, vs)) { code = Status.BAD_READ; fa = vv; bad = true; break; }
                        in[k] = readMem(m, priv, vv, vs, be);
                    }
                }
                if (bad) break;
                int osz = p.size[b4];
                int s0 = p.size[b4 + 1], s1 = p.size[b4 + 2];
                long r = 0;
                boolean write = p.kind[b4] != KIND_NONE;
                int next = pc + 1;
                switch (p.opcode[pc]) {
                    case 1 -> r = in[0];                                             // COPY
                    case 2 -> {                                                      // LOAD
                        long addr = in[1];
                        if (!readable(m, addr, osz)) { code = Status.BAD_READ; fa = addr; write = false; }
                        else r = readMem(m, priv, addr, osz, be);
                    }
                    case 3 -> {                                                      // STORE
                        long addr = in[1];
                        int sz = p.size[b4 + 3];
                        long po = m.privOffset(addr, sz);
                        if (po < 0) { code = Status.BAD_WRITE; fa = addr; }
                        else store(priv, po, sz, be, in[2]);
                        write = false;
                    }
                    case 4, 5 -> {                                                   // BRANCH, CBRANCH
                        write = false;
                        if (p.opcode[pc] == 5 && in[1] == 0) break;
                        long t = p.val[b4 + 1];
                        if (t < 0) { code = Status.BAD_BRANCH; fa = 0; }
                        else next = (int) t;
                    }
                    case 6, 10 -> {                                                  // BRANCHIND, RETURN
                        write = false;
                        long target = in[0];
                        if (target == returnSentinel) { code = Status.RETURNED; break; }
                        int hit = binarySearchUnsigned(p.instrAddr, target);
                        if (hit < 0) { code = Status.BAD_BRANCH; fa = target; }
                        else next = p.instrFirstOp[hit];
                    }
                    case 7, 8, 9 -> { code = Status.CALL; fa = in[0]; write = false; } // CALL, CALLIND, CALLOTHER
                    case 11 -> r = in[0] == in[1] ? 1 : 0;
                    case 12 -> r = in[0] != in[1] ? 1 : 0;
                    case 13 -> r = sext(in[0], s0) < sext(in[1], s1) ? 1 : 0;
                    case 14 -> r = sext(in[0], s0) <= sext(in[1], s1) ? 1 : 0;
                    case 15 -> r = Long.compareUnsigned(in[0], in[1]) < 0 ? 1 : 0;
                    case 16 -> r = Long.compareUnsigned(in[0], in[1]) <= 0 ? 1 : 0;
                    case 17 -> r = in[0];                                            // ZEXT
                    case 18 -> r = sext(in[0], s0);                                  // SEXT
                    case 19 -> r = in[0] + in[1];
                    case 20 -> r = in[0] - in[1];
                    case 21 -> r = Long.compareUnsigned((in[0] + in[1]) & maskOf(s0), in[0]) < 0 ? 1 : 0;
                    case 22 -> {
                        long a = sext(in[0], s0), b = sext(in[1], s0), c = sext((in[0] + in[1]) & maskOf(s0), s0);
                        r = ((a >= 0) == (b >= 0)) && ((c >= 0) != (a >= 0)) ? 1 : 0;
                    }
                    case 23 -> {
                        long a = sext(in[0], s0), b = sext(in[1], s0), c = sext((in[0] - in[1]) & maskOf(s0), s0);
                        r = ((a >= 0) != (b >= 0)) && ((c >= 0) != (a >= 0)) ? 1 : 0;
                    }
                    case 24 -> r = -in[0];
                    case 25 -> r = ~in[0];
                    case 26 -> r = in[0] ^ in[1];
                    case 27 -> r = in[0] & in[1];
                    case 28 -> r = in[0] | in[1];
                    case 29 -> r = Long.compareUnsigned(in[1], 64) >= 0 ? 0 : in[0] << in[1];
                    case 30 -> r = Long.compareUnsigned(in[1], 64) >= 0 ? 0 : in[0] >>> in[1];
                    case 31 -> {
                        long a = sext(in[0], s0);
                        r = Long.compareUnsigned(in[1], 64) >= 0 ? (a < 0 ? -1 : 0) : a >> in[1];
                    }
                    case 32 -> r = in[0] * in[1];
                    case 33 -> { if (in[1] == 0) { code = Status.DIV_ZERO; write = false; } else r = Long.divideUnsigned(in[0], in[1]); }
                    case 35 -> { if (in[1] == 0) { code = Status.DIV_ZERO; write = false; } else r = Long.remainderUnsigned(in[0], in[1]); }
                    case 34 -> { if (in[1] == 0) { code = Status.DIV_ZERO; write = false; } else r = sext(in[0], s0) / sext(in[1], s1); }
                    case 36 -> { if (in[1] == 0) { code = Status.DIV_ZERO; write = false; } else r = sext(in[0], s0) % sext(in[1], s1); }
                    case 37 -> r = in[0] ^ 1;
                    case 38 -> r = in[0] ^ in[1];
                    case 39 -> r = in[0] & in[1];
                    case 40 -> r = in[0] | in[1];
                    case 62 -> r = (s1 >= 8 ? 0 : in[0] << (8 * s1)) | in[1];       // PIECE
                    case 63 -> r = Long.compareUnsigned(in[1], 8) >= 0 ? 0 : in[0] >>> (8 * in[1]); // SUBPIECE
                    case 72 -> r = Long.bitCount(in[0]);
                    case 73 -> r = in[0] == 0 ? 8L * s0 : Long.numberOfLeadingZeros(in[0]) - (64 - 8 * s0);
                    default -> { code = Status.UNSUPPORTED; fa = p.opcode[pc]; write = false; }
                }
                if (code != Status.RUNNING) break;
                if (write) {
                    r &= maskOf(osz);
                    int ok = p.kind[b4];
                    if (ok == KIND_STATE) store(state, st + p.val[b4], osz, be, r);
                    else if (ok == KIND_RAM) {
                        long po = m.privOffset(p.val[b4], osz);
                        if (po < 0) { code = Status.BAD_WRITE; fa = p.val[b4]; break; }
                        store(priv, po, osz, be, r);
                    }
                }
                pc = next;
            }
            status[lane] = code.ordinal();
            steps[lane] = count;
            faultOp[lane] = pc;
            faultAddr[lane] = fa;
        }
        return new Result(status, steps, faultOp, faultAddr, state, p.stateSize);
    }

    static boolean readable(Memory m, long addr, int size) {
        if (m.privOffset(addr, size) >= 0) return true;
        return segment(m, addr, size) >= 0;
    }

    static int segment(Memory m, long addr, int size) {
        int a = 0, b = m.segBase.length - 1, hit = -1;
        while (a <= b) {
            int mid = (a + b) >>> 1;
            if (Long.compareUnsigned(m.segBase[mid], addr) <= 0) { hit = mid; a = mid + 1; } else b = mid - 1;
        }
        if (hit >= 0 && within(addr, size, m.segBase[hit], m.segSize[hit])) return hit;
        return -1;
    }

    /** Value at {@code addr}; callers check {@link #readable} first. */
    static long readMem(Memory m, byte[] priv, long addr, int size, boolean be) {
        long po = m.privOffset(addr, size);
        if (po >= 0) return load(priv, po, size, be);
        int s = segment(m, addr, size);
        return load(m.image, m.segOff[s] + (addr - m.segBase[s]), size, be);
    }

    static int binarySearchUnsigned(long[] t, long v) {
        return ScanOps.binarySearchUnsigned(t, v);
    }

    // ------------------------------------------------------------------ GPU

    public static Result runGpu(Gpu.Session s, Program p, Memory m, Lanes lanes, long returnSentinel, int maxSteps) {
        int n = lanes.count();
        Gpu.Buf ops = s.put(p.pack());
        Gpu.Buf firstOp = s.put(nonEmpty(p.instrFirstOp)), addrs = s.put(nonEmpty(p.instrAddr));
        Gpu.Buf state = s.alloc((long) n * p.stateSize), stateInit = s.put(lanes.stateInit());
        Gpu.Buf priv = s.alloc((long) n * m.privSize);
        Gpu.Buf privInit = s.put(lanes.privInit().length == 0 ? new byte[1] : lanes.privInit());
        Gpu.Buf privOff = s.put(lanes.privInitOff()), privLen = s.put(lanes.privInitLen());
        Gpu.Buf wb = s.put(nonEmpty(m.winBase)), ws = s.put(nonEmpty(m.winSize)), wo = s.put(nonEmpty(m.winOff));
        Gpu.Buf img = s.put(m.image.length == 0 ? new byte[1] : m.image);
        Gpu.Buf sb = s.put(nonEmpty(m.segBase)), ss = s.put(nonEmpty(m.segSize)), so = s.put(nonEmpty(m.segOff));
        Gpu.Buf st = s.alloc(4L * n), steps = s.alloc(4L * n), fop = s.alloc(4L * n), fad = s.alloc(8L * n);
        // One lane per thread with a small block: lanes diverge heavily, and a short block
        // keeps the per-thread local arrays (in[], VmMem) resident.
        int block = 64;
        int grid = Math.max(1, Math.min((n + block - 1) / block, 65535));
        s.launchGrid("pcode_vm", grid, block, 0, ops, p.nOps, firstOp, addrs, p.instrAddr.length,
            state, p.stateSize, stateInit, priv, m.privSize, privInit, privOff, privLen,
            wb, ws, wo, m.winBase.length, img, sb, ss, so, m.segBase.length,
            p.bigEndian ? 1 : 0, returnSentinel, maxSteps, n, st, steps, fop, fad);
        return new Result(s.ints(st, n), s.ints(steps, n), s.ints(fop, n), s.longs(fad, n),
            s.bytes(state, n * p.stateSize), p.stateSize);
    }

    private static int[] nonEmpty(int[] a) {
        return a.length == 0 ? new int[1] : a;
    }

    private static long[] nonEmpty(long[] a) {
        return a.length == 0 ? new long[1] : a;
    }
}
