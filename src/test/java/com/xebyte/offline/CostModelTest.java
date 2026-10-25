package com.xebyte.offline;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.BeforeClass;
import org.junit.Test;

import com.xebyte.gpu.CostModel;
import com.xebyte.gpu.Ops;

/**
 * AUTO follows the observed run nearest in work size, which is what a single crossover
 * could not express: measured on an 80k-function binary, PageRank was faster on the GPU
 * at small sizes and 20x slower at full size.
 */
public class CostModelTest {

    private static Path file;

    @BeforeClass
    public static void isolate() throws Exception {
        file = Files.createTempFile("ghidra-gpu-calibration", ".properties");
        Files.delete(file);
        System.setProperty("ghidra.gpu.calibration", file.toString());
    }

    @Test
    public void autoFollowsTheNearestObservedRunInBothDirections() {
        CostModel.resetCalibration();
        CostModel.calibrate(Ops.PAGERANK, 1_000_000L);              // synthetic crossover: GPU from 1M
        assertTrue(CostModel.prefersGpu(Ops.PAGERANK, 13_000_000L)); // with nothing observed, crossover rules

        CostModel.observe(Ops.PAGERANK, 2_000_000L, 10.0, 5.0);     // GPU faster at 2M
        CostModel.observe(Ops.PAGERANK, 13_000_000L, 77.0, 1597.0); // CPU 20x faster at 13M (real binary)
        assertTrue(CostModel.prefersGpu(Ops.PAGERANK, 2_500_000L));
        assertFalse(CostModel.prefersGpu(Ops.PAGERANK, 13_000_000L));
        assertFalse(CostModel.prefersGpu(Ops.PAGERANK, 50_000_000L)); // beyond the largest run: its verdict holds
        assertTrue(CostModel.explain(Ops.PAGERANK, 13_000_000L).contains("nearest observed run"));
    }

    @Test
    public void observationsPersistAndAreCapped() throws Exception {
        CostModel.resetCalibration();
        for (int i = 1; i <= 100; i++) CostModel.observe(Ops.SCC, i * 1000L, 1.0, 2.0);
        assertEquals(64, CostModel.observedCount(Ops.SCC));
        String saved = Files.readString(file);
        assertTrue(saved.contains("observed.scc="));
    }
}
