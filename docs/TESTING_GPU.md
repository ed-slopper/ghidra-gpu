# Testing ghidra-gpu from a fresh environment

A checklist for an independent tester (person or agent) to verify the GPU work without
relying on anything the author's machine left behind.

## Prerequisites

- Ghidra 12.1.4, JDK 21+ (`-PjavaToolchain=25` if only JDK 25 is installed), Python 3.11+, `uv`.
- An NVIDIA GPU with its driver (no CUDA toolkit needed). Without one, every GPU test
  self-skips and every tool runs its CPU twin -- check skip counts, not exit codes.

## 1. Build and unit tiers

```text
git clone https://github.com/ed-slopper/ghidra-gpu && cd ghidra-gpu
./gradlew test buildExtension -PGHIDRA_INSTALL_DIR=<ghidra> [-PjavaToolchain=25]
uv run pytest tests/unit/ --no-cov
```

Expect 903 Java tests (0 failed, 0 skipped on a GPU machine) and ~1221 Python tests passing.
`GpuKernelParityTest` is the CPU-vs-GPU bit-identity check for all 22 kernels.

## 2. Kernels are current

```text
pip install nvidia-cuda-nvrtc-cu12
python -m tools.gpu.build_ptx --check     # exit 0 = committed PTX matches the CUDA source
```

## 3. Live tools

Install `build/distributions/GhidraMCP-7.0.0.zip` (or run the headless server), open any
analysed binary, then:

- `gpu_info` -- device, kernels, cost model.
- `gpu_selftest` -- every op on both backends; expect `failed: 0`.
- Call each `gpu_*` tool with `backend=both`; every `execution.agree` (including nested
  ones) must be `true`.
- `gpu_function_hashes` must equal `get_function_hash` for the same function.
- `gpu_benchmark` then `gpu_advise` -- auto now uses measured crossovers.

Through the MCP bridge: `connect_instance(<project>)`, `load_tool_group("gpu")`, and the
`gpu_*` tools appear.

## 4. Coverage claims

- `python -m tools.gpu.coverage` -- every upstream tool is ported or has a reason
  (`tests/unit/test_gpu_coverage.py` enforces it).
- `docs/GPU_COVERAGE.md` lists the 176 non-ported tools with a reason and a live timing
  each; `tools/gpu/measure_cpu_only.py` and `tools/gpu/measure_live.py` reproduce them.
  The reasons are the author's; an independent review of whether any of those tools does
  program-wide work worth porting is exactly what a tester can add.

## Report

Note the GPU, driver, OS, Ghidra version, the test counts and skip counts, any tool whose
`agree` was false, and any non-ported tool you believe does port-worthy work (with the code
path).
