package com.xebyte.gpu;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

import com.xebyte.gpu.ScanOps.Packed;
import com.xebyte.gpu.cuda.Gpu;

/** Batched FNV-1a-64 and SHA-256 over many byte segments (one hash per segment). */
public final class HashOps {

    private HashOps() {}

    public static long[] fnvCpu(Packed p) {
        long[] out = new long[p.count()];
        byte[] b = p.bytes();
        for (int s = 0; s < out.length; s++) {
            long h = 0xcbf29ce484222325L;
            int base = (int) p.off()[s];
            for (int k = 0; k < p.len()[s]; k++) {
                h ^= b[base + k] & 0xFF;
                h *= 0x100000001b3L;
            }
            out[s] = h;
        }
        return out;
    }

    public static long[] fnvGpu(Gpu.Session s, Packed p) {
        Gpu.Buf d = s.put(p.bytes()), o = s.put(p.off()), l = s.put(p.len()), out = s.alloc(8L * Math.max(1, p.count()));
        s.launch("fnv1a64_batch", p.count(), d, o, l, p.count(), out);
        return s.longs(out, p.count());
    }

    /** 32 bytes per segment, concatenated. */
    public static byte[] sha256Cpu(Packed p) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] out = new byte[32 * p.count()];
            for (int s = 0; s < p.count(); s++) {
                md.reset();
                md.update(p.bytes(), (int) p.off()[s], p.len()[s]);
                System.arraycopy(md.digest(), 0, out, 32 * s, 32);
            }
            return out;
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    public static byte[] sha256Gpu(Gpu.Session s, Packed p) {
        Gpu.Buf d = s.put(p.bytes()), o = s.put(p.off()), l = s.put(p.len()), out = s.alloc(32L * Math.max(1, p.count()));
        s.launch("sha256_batch", p.count(), d, o, l, p.count(), out);
        return s.bytes(out, 32 * p.count());
    }

    public static String hex(byte[] all, int index) {
        StringBuilder sb = new StringBuilder(64);
        for (int k = 0; k < 32; k++) sb.append(String.format("%02x", all[32 * index + k] & 0xFF));
        return sb.toString();
    }

    public static boolean same(byte[] a, byte[] b) {
        return Arrays.equals(a, b);
    }
}
