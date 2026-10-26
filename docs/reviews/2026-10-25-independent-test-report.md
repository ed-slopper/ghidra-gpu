# ghidra-gpu: independent test report (2026-10-25)

Repo HEAD 186d21bf; `build/libs/GhidraMCP-7.0.0.jar` built 13:44.

## Setup

- Headless `GhidraMCPHeadlessServer` on 127.0.0.1:8089, `-Xmx16g`, run on a copy of the Townscaper project.
- Program: `GameAssembly.dll` (x86:LE:32): 80,818 functions, 24 MB, 899,827 basic blocks, 352,818 call edges (about 19x kernel32).
- Machine: RTX 3060 12 GB (driver CUDA 13.0), 16 CPU threads, 32 GB RAM, Windows 11.
- The calibration was already measured on this machine (`~/.ghidra-gpu`).
- `gpu_selftest`: 18 passed, 0 failed.
- Every `backend=both` call below agreed (`agree=true`).

## GPU wins (CPU ms / GPU ms)

| Tool | CPU | GPU | Speed-up |
|---|---:|---:|---:|
| `gpu_detect_crypto_constants` (702M bytes x patterns) | 4,381 | 44 | 100x |
| `gpu_find_anti_analysis` | 5,554 | 96 | 58x |
| `gpu_find_function_starts` | 991 | 51 | 19x |
| `gpu_extract_iocs` | 635 | 53 | 12x |
| `gpu_find_functions` (substring search) | 36 | 6 | 6x |
| `gpu_function_hashes` (sha256, 81 MB hashed) | 263 | 82 | 3.2x |
| `gpu_list_strings` | 287 | 173 | 1.7x |
| All-vs-all MinHash, `min_instructions=120` (~10.7k functions, ~114M pairs) | 19,929 | 874 | 23x |
| All-vs-all cosine, same set | 52,883 | 5,754 | 9.2x |
| Emulating zlib `crc32_little` (0x1014c900), 5,000 lanes x 256 B | 1,738 | 95 | 18x |
| Same, 20,000 lanes | 4,580 | 327 | 14x |

All 25,000 emulated CRCs matched Python's `zlib.crc32`.

## CPU wins

| Tool / op | CPU | GPU |
|---|---:|---:|
| Call graph SCC | 14 | 1,417 |
| PageRank (work 13.0M) | 77 | 1,597 |
| Reachability (bfs) | 0.8 | 7.9 |
| Dominators (work 2.58M, 900k blocks) | 339 | 2,096 |
| `gpu_pointer_scan` | 60 | 483 |
| `gpu_entropy_map` | 42 | 126 |

`Char.GetHashCode` (0x1050f050, 5 instructions) over 20,000 lanes was a tie: 119 against 133 ms. The results I spot-checked were correct.

## Decompile (`gpu_decompile_batch`, backend=both, 16 threads)

- **2,000 functions:** decompile 11,231 ms. Dominators 0.8 / 1.9 ms, param influence 16 / 8 ms, return slice 47 / 7 ms.
- **20,000 functions** (1 failed): 124 s wall, of which decompile 118,275 ms (about 95%) and graph build 358 ms. Dominators 15 / 11 ms, param influence 58 / 101 ms, return slice 22 / 104 ms.
- **500 functions on the CPU backend:** 4,686 ms on 1 thread against 1,792 ms on 16. The native decompiler only scales about 2.6x across 16 threads, and that is the bottleneck for decompile-heavy work.

## Issues

1. **`auto` picks the slower backend at this size.**
   - PageRank: crossover 2,560,000, work 13.0M, so `auto` runs the GPU, which is 20x slower.
   - Dominators: crossover 4,352, work 2.58M, so `auto` picks the GPU, which is 6x slower.
   - `gpu_benchmark` only measures small synthetic inputs, and the GPU's advantage doesn't hold as real graphs grow. The crossover needs an upper bound, or a measurement at real sizes.
2. **Reading the program (`extract_ms`) takes most of the time in many tools.**
   - `gpu_function_hashes`: 45,479 ms to extract against 82 ms of hashing.
   - `gpu_control_flow_analysis`: 41,454 ms. `gpu_call_graph_analysis`: 11,316 ms. Similarity: 13–14 s.
   - Speeding up extraction may pay off more than faster kernels.
3. **All-vs-all similarity refuses a binary this size.** It refused 5,299,257,616 pairs at the default `min_instructions=8`, and still refused 623,201,296 at `min_instructions=60`. A tiled or streamed top-k all-vs-all would let it cover a whole large binary.
4. **The bridge connected to the wrong server.** It was attached to http://127.0.0.1:8090 ("gpu_e2e", active-tcp fallback), but port 8090 belongs to a Windows process, WsToastNotification. Every tool returned HTTP 501.
5. **The bridge's tool list is short of the server.** Its static `gpu` group lists 22 tools; the server's `/mcp/schema` has 24. `gpu_decompile_batch` and `gpu_analysis_passes` are missing from the bridge.

## Checklist coverage

- `gpu_function_hashes` matched `get_function_hash` on the 5 functions compared (4,932 duplicate groups across the program).
- **Not run:**
  - `gradlew test` and pytest;
  - `build_ptx --check`;
  - `gpu_benchmark`;
  - `gpu_analysis_passes`, `gpu_diff_functions`, `gpu_emulate_hash_batch`, `gpu_dataflow_slice`;
  - `gpu_search_byte_patterns` and `gpu_search_instructions` on their own.

The raw replies (`out_*.json`) and the runner (`run.py`) are next to this file. The test server is stopped.
