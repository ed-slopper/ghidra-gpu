# ghidra-gpu: CUDA acceleration for ghidra-mcp

ghidra-gpu is ghidra-mcp 7.0.0 plus a `gpu` tool group: CUDA twins of every
compute-bearing tool in the catalog. Each twin runs on **either** backend and reports
what it did, so an AI agent can decide per task whether the CPU or the GPU should do the
work, and verify that both give the same answer.

- 22 `gpu_*` MCP tools: 18 analysis tools plus `gpu_info`, `gpu_benchmark`, `gpu_advise`, `gpu_selftest`
- 31 of the 209 upstream tools have a GPU twin; the other 178 have nothing a GPU could compute
  (database edits, single-record lookups, I/O, UI, the native decompiler). The full,
  generated table with a reason per tool is [GPU_COVERAGE.md](GPU_COVERAGE.md).
- 22 CUDA kernels in [`src/main/cuda/ghidra_gpu.cu`](../src/main/cuda/ghidra_gpu.cu), each with a Java twin
  that returns bit-identical results ([`GpuKernelParityTest`](../src/test/java/com/xebyte/offline/GpuKernelParityTest.java)).

## Requirements

- An NVIDIA GPU (Maxwell / compute 5.2 or newer) and its normal display driver.
- **No CUDA toolkit.** The kernels ship as PTX inside the extension jar and are JIT-compiled
  by the driver (`nvcuda.dll` / `libcuda.so`) on first use, through JNA, which Ghidra
  already bundles.
- Without a usable GPU every tool still works: it runs the CPU twin and says why in
  `execution.fallback`.

Install exactly like ghidra-mcp (`./gradlew buildExtension` then install the zip, or
`./gradlew deploy`); see the main README.

## How an agent chooses

Every `gpu_*` analysis tool takes `backend`:

| backend | what runs |
| --- | --- |
| `auto` (default) | the cost model picks: GPU when the op's work size is at or above its crossover |
| `cpu` | the Java twin |
| `gpu` | the CUDA kernel (falls back to CPU with a note if no GPU) |
| `both` | both; returns the GPU result plus `agree`, both timings and the speedup |

and returns an `execution` block, for example:

```json
"execution": {"op": "minhash_similarity", "requested": "both", "work": 192657408,
  "work_unit": "pairs x k", "cpu_ms": 335.43, "gpu_ms": 14.06, "agree": true,
  "faster": "gpu", "speedup": 23.86, "backend": "both", "extract_ms": 337.73}
```

`extract_ms` is the time spent reading the program database (always CPU), reported apart
from the op so the comparison is honest.

- **`gpu_benchmark`** measures CPU vs GPU for every op at doubling sizes on *this*
  machine and stores each crossover in `~/.ghidra-gpu/calibration.properties`
  (`GHIDRA_GPU_CALIBRATION` overrides the path). `auto` uses the measured numbers from
  then on, across Ghidra restarts. Run it once per machine.
- **`gpu_advise(op|tool, work)`** answers "cpu or gpu?" without running anything.
- **`gpu_info`** shows the device, the cost model per op (measured or default) and the
  upstream tools each GPU tool replaces.
- **`gpu_selftest`** runs every kernel against its CPU twin on generated inputs.

Through the MCP bridge the group is lazy-loaded like any other:
`list_tool_groups` → `load_tool_group("gpu")`.

## The tools

| GPU tool | Upstream equivalent(s) | Op(s) |
| --- | --- | --- |
| `gpu_search_byte_patterns` | search_byte_patterns | many masked patterns, one pass |
| `gpu_list_strings` | list_strings, search_strings, find_undocumented_by_string | ASCII + UTF-16LE extraction from raw memory, substring filter |
| `gpu_entropy_map` | analyze_data_region, inspect_memory_content | windowed entropy + byte histograms, packed/encrypted ranges |
| `gpu_pointer_scan` | get_xrefs_to (raw), list_data_items_by_xrefs | every aligned pointer into targets or the image, incl. refs Ghidra never made |
| `gpu_detect_crypto_constants` | detect_crypto_constants (a placeholder upstream) | AES, SHA-1/2, MD5, CRC-32, Blowfish, ChaCha/Salsa, TEA, RC5/6, DES, Twofish, Camellia, Whirlpool, RIPEMD, Base64 |
| `gpu_find_anti_analysis` | find_anti_analysis_techniques, detect_malware_behaviors | anti-debug/VM/timing/sandbox encodings and API names |
| `gpu_extract_iocs` | extract_iocs_with_context | URLs, IPv4, e-mail, registry paths in raw memory |
| `gpu_find_function_starts` | find_code_gaps, find_next_undefined_function | per-architecture prologue scan outside known functions |
| `gpu_find_functions` | find_functions | substring over all function (or symbol) names |
| `gpu_search_instructions` | search_instructions | substring over every instruction's text |
| `gpu_function_hashes` | get_function_hash, compare_programs_documentation | SHA-256, byte-identical to `get_function_hash`, plus duplicate groups |
| `gpu_find_similar_functions` | find_similar_functions(_fuzzy), bulk_fuzzy_match | MinHash Jaccard or cosine, one-vs-all or all-vs-all, cross-program |
| `gpu_diff_functions` | diff_functions | batched Levenshtein over mnemonic sequences |
| `gpu_call_graph_analysis` | analyze_call_graph, get_full_call_graph, get_function_call_graph, analyze_api_call_chains | SCC recursion groups, PageRank, reachability, caller/callee depth maps |
| `gpu_control_flow_analysis` | analyze_control_flow, find_dead_code | batched dominator trees for every function, loops, unreachable blocks |
| `gpu_dataflow_slice` | analyze_dataflow | decompile (CPU), then forward/backward slice over the def-use graph |
| `gpu_emulate_hash_batch` | emulate_hash_batch | one GPU thread per candidate on a P-code VM |
| `gpu_emulate_function_batch` | emulate_function | one GPU thread per input set; registers, scratch buffer, stack args |

## Measured on an RTX 3060 (driver 581.80)

Real binary: Windows `kernel32.dll` (x64; 4301 functions, 25,513 basic blocks, 128k
instructions), `backend=both`, every op agreeing CPU vs GPU:

| Tool / op | CPU ms | GPU ms | |
| --- | ---: | ---: | --- |
| all-vs-all MinHash similarity (3.0M pairs) | 335 | 14 | 24x |
| all-vs-all cosine similarity (3.0M pairs) | 344 | 41 | 8.4x |
| anti-analysis signature scan | 98 | 3.9 | 25x |
| crypto constant scan | 82 | 3.9 | 21x |
| byte pattern scan | 51 | 12 | 4.1x |
| IOC scan | 45 | 12 | 3.7x |
| dominators, 1735 functions | 36 | 10 | 3.5x |
| call-graph SCC (4301 nodes) | 0.9 | 4.1 | CPU wins |

Batched emulation of real x86 code lowered from Ghidra P-code (benchmark fixture):

| Run | CPU ms | GPU ms | |
| --- | ---: | ---: | --- |
| `calc_crc16(data, len)`, 5000 random buffers, all 5000 CRCs correct | 3939 | 173 | 23x |
| `compute_str_len` as a hash, 20000 candidates | 467 | 65 | 7x |

`gpu_benchmark` crossovers on the same card: the GPU wins from ~65 KB for IOC scans,
~100 KB for SHA-256, ~0.5 MB for pattern scans, ~1000 lanes for emulation and a few
thousand blocks for dominators. Plain memory-bound passes (string extraction, histogram,
entropy, pointer scan, FNV) did not beat the CPU up to 4 MB because the PCIe copy
dominates, and small graphs stay on the CPU; `auto` routes them accordingly. That is the
point of keeping both backends.

## Architecture

```text
MCP client -> bridge -> McpHttpServer -> GpuService (@McpToolGroup "gpu")
                                          |  ProgramData / PcodeLowering   (Ghidra DB -> flat arrays, CPU)
                                          |  Dispatch + CostModel          (backend choice, timing, fallback)
                                          |-- ops: ScanOps HashOps SimilarityOps GraphOps PcodeVm  (Java twins)
                                          '-- cuda.Gpu -> JNA -> nvcuda / libcuda -> ghidra_gpu.ptx kernels
```

- `com.xebyte.gpu.cuda` — driver-API binding (`CudaDriver`) and the runtime (`Gpu`): one
  context, serialised sessions that free their buffers, sticky-error detection that
  routes everything to the CPU after a lost context.
- `com.xebyte.gpu` — the ops, each a `cpu(...)`/`gpu(session, ...)` pair over arrays.
  GPU appends are sorted and re-run once at exact size on overflow, so truncated results
  and `total` match the CPU.
- `com.xebyte.gpu.ghidra` — extraction from a `Program`, the P-code lowering, and
  `GpuBench` (benchmark + self-test workloads).
- `com.xebyte.core.GpuService` — the MCP tools; registered in `CoreServices`, so served by
  both the GUI plugin and the headless server.

### The P-code VM

`PcodeLowering` turns a function's raw P-code (`Instruction.getPcode()`) into a compact
program: register and unique varnodes are packed into a small per-lane state block
(aliasing registers keep sharing bytes), branches become absolute op indexes, and
fall-through is made explicit. Lanes get private stack and scratch windows and read the
program image read-only; the return address (stack slot or link register, from the
calling convention) is a sentinel that ends the lane. Big batches are split to fit a
512 MB device budget.

Limits, reported per lane in `lane_status` and `fault_example` rather than guessed:
calls out of the function stop the lane (`call`), floating point and values wider
than 8 bytes are `unsupported`, memory outside the windows and image is `bad_read` /
`bad_write`.

## What stays on the CPU

- Reading the program database — it lives in the JVM; every tool reports `extract_ms`.
- Ghidra's decompiler — native C++, one function per call. `gpu_dataflow_slice`
  decompiles on the CPU and runs the graph walk on the GPU, reporting both times.
- The 178 catalog tools listed in [GPU_COVERAGE.md](GPU_COVERAGE.md).

## Configuration

| Variable | Effect |
| --- | --- |
| `GHIDRA_GPU_DISABLE=1` / `-Dghidra.gpu.disable=true` | never touch the GPU (every tool runs its CPU twin) |
| `GHIDRA_GPU_DEVICE=<n>` / `-Dghidra.gpu.device=<n>` | which CUDA device to use (default 0) |
| `GHIDRA_GPU_CALIBRATION=<path>` | where `gpu_benchmark` stores crossovers |
| `GHIDRA_GPU_MAX_SCAN_BYTES=<n>` | cap on memory a scan reads (default 1 GiB) |

## Development

```text
pip install nvidia-cuda-nvrtc-cu12          # NVRTC only; no host compiler needed
python -m tools.gpu.build_ptx               # src/main/cuda/*.cu -> src/main/resources/com/xebyte/gpu/*.ptx
python -m tools.gpu.build_ptx --check       # fail if the committed PTX is stale
./gradlew test --tests 'com.xebyte.offline.GpuKernelParityTest' -PGHIDRA_INSTALL_DIR=...
python -m tools.gpu.coverage --write        # regenerate docs/GPU_COVERAGE.md
```

Adding a kernel: write it in `ghidra_gpu.cu`, add its name to `Gpu.KERNELS` (a stale PTX
then fails at load, not at first call), give it a Java twin, a parity test, a
`CostModel` default and a `GpuBench` workload, then expose it from `GpuService` and map
the upstream tool in `TOOL_MAP`.

`GpuKernelParityTest` self-skips without a GPU (`assumeTrue`); check the skip count.
The CPU-reference halves of that class run everywhere.
