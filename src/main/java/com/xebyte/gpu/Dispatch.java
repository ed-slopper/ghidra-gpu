package com.xebyte.gpu;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.function.Supplier;

import com.xebyte.gpu.cuda.Gpu;

/**
 * Runs one op on the backend the caller asked for and reports what actually happened.
 *
 * <p>The report ({@link Outcome#meta}) is part of every GPU tool's response under
 * {@code "execution"}, so an agent sees which backend ran, how long it took, the work
 * size AUTO judged on, and -- for {@link Backend#BOTH} -- whether CPU and GPU agreed.
 * That is what lets it choose deliberately next time instead of guessing.
 */
public final class Dispatch {

    private Dispatch() {}

    public record Outcome<T>(T value, Map<String, Object> meta) {}

    public static <T> Outcome<T> run(String op, Backend requested, long work,
                                     Supplier<T> cpu, Function<Gpu.Session, T> gpu) {
        return run(op, requested, work, cpu, gpu, null);
    }

    /**
     * @param same equality used by {@link Backend#BOTH}; null means {@link Object#equals}
     */
    public static <T> Outcome<T> run(String op, Backend requested, long work,
                                     Supplier<T> cpu, Function<Gpu.Session, T> gpu,
                                     BiPredicate<T, T> same) {
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("op", op);
        meta.put("requested", requested.name().toLowerCase());
        meta.put("work", work);
        meta.put("work_unit", CostModel.workUnit(op));

        Gpu g = Gpu.get();
        Backend chosen = requested;
        if (requested == Backend.AUTO) {
            boolean wantGpu = CostModel.prefersGpu(op, work);
            chosen = wantGpu && g != null ? Backend.GPU : Backend.CPU;
            meta.put("auto_reason", g == null ? "no GPU: " + Gpu.unavailableReason()
                : (wantGpu ? "work >= " : "work < ") + "crossover " + CostModel.crossover(op)
                    + (CostModel.isCalibrated(op) ? " (measured)" : " (default; run gpu_benchmark to measure)"));
        }
        if ((chosen == Backend.GPU || chosen == Backend.BOTH) && g == null) {
            meta.put("fallback", "GPU unavailable (" + Gpu.unavailableReason() + "); ran on CPU");
            chosen = Backend.CPU;
        }

        T value;
        if (chosen == Backend.CPU) {
            long t0 = System.nanoTime();
            value = cpu.get();
            meta.put("cpu_ms", ms(t0));
        } else if (chosen == Backend.GPU) {
            long t0 = System.nanoTime();
            try {
                value = g.run(gpu);
                meta.put("gpu_ms", ms(t0));
            } catch (RuntimeException e) {
                meta.put("fallback", "GPU failed (" + e.getMessage() + "); ran on CPU");
                long t1 = System.nanoTime();
                value = cpu.get();
                meta.put("cpu_ms", ms(t1));
                chosen = Backend.CPU;
            }
        } else {
            long t0 = System.nanoTime();
            T c = cpu.get();
            double cpuMs = ms(t0);
            long t1 = System.nanoTime();
            T gv;
            try {
                gv = g.run(gpu);
            } catch (RuntimeException e) {
                meta.put("cpu_ms", cpuMs);
                meta.put("fallback", "GPU failed (" + e.getMessage() + "); returned the CPU result");
                meta.put("backend", "cpu");
                return new Outcome<>(c, meta);
            }
            double gpuMs = ms(t1);
            boolean agree = same != null ? same.test(c, gv) : c.equals(gv);
            meta.put("cpu_ms", cpuMs);
            meta.put("gpu_ms", gpuMs);
            meta.put("agree", agree);
            meta.put("faster", gpuMs < cpuMs ? "gpu" : "cpu");
            meta.put("speedup", gpuMs > 0 ? Math.round(cpuMs / gpuMs * 100.0) / 100.0 : null);
            meta.put("backend", "both");
            return new Outcome<>(gv, meta);
        }
        meta.put("backend", chosen.name().toLowerCase());
        return new Outcome<>(value, meta);
    }

    static double ms(long t0) {
        return Math.round((System.nanoTime() - t0) / 1e4) / 100.0;
    }
}
