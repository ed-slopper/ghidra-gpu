package com.xebyte.gpu;

import java.util.Locale;

/**
 * Where an op runs. Every GPU tool takes {@code backend=} with one of these.
 *
 * <ul>
 *   <li>{@link #CPU}: the Java twin, always available.</li>
 *   <li>{@link #GPU}: the CUDA kernel; falls back to CPU (and says so) when no GPU is usable.</li>
 *   <li>{@link #AUTO}: {@link CostModel} picks, from the op and its work size, using
 *       calibrated crossover points when {@code gpu_benchmark} has measured them.</li>
 *   <li>{@link #BOTH}: runs both, returns the GPU answer, and reports whether they agree
 *       plus both timings -- the way to check a result and learn the crossover at once.</li>
 * </ul>
 */
public enum Backend {
    CPU, GPU, AUTO, BOTH;

    public static Backend parse(String s) {
        if (s == null || s.isBlank()) return AUTO;
        try {
            return valueOf(s.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("backend must be one of cpu, gpu, auto, both (got '" + s + "')");
        }
    }
}
