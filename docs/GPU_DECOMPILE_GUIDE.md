# CPU or GPU? A guide for agents doing decompilation and RE

Every `gpu_*` tool runs on either backend (`backend=cpu|gpu|auto|both`). This guide says
which tasks belong on which side, from measurements on an RTX 3060 against kernel32.dll
(4,301 functions) and an 80,818-function binary (Townscaper `GameAssembly.dll`). The rules
of thumb are below; `auto` learns the exact line for your machine from any `backend=both`
call (see [GPU.md](GPU.md)).

## The short version

- **Decompiling itself is always CPU.** Ghidra's decompiler is native C++, one function per
  call; it cannot run on a GPU. Use `gpu_decompile_batch` to run it on all cores; it scales
  only ~2.6x across 16 threads, and on large batches decompiling is ~95% of the time.
- **Send to the GPU** work that touches *every byte, every function or every pair of
  functions*: whole-binary signature scans, all-vs-all similarity, batched emulation,
  bulk hashing, whole-program name audits.
- **Keep on the CPU** anything about *one* function or address, graph algorithms on the
  call graph or on big CFGs, and simple single-pass memory scans (entropy, pointer scan),
  where copying the data to the GPU costs more than the scan.
- **Reading the program is CPU-bound and serial.** Ghidra's database lock serialises reads;
  `execution.extract_ms` shows it. On very large binaries it can dominate (45 s of
  extraction against 82 ms of GPU hashing). The first call pays it; later calls on the
  unchanged program reuse the cached arrays (`execution.extract_cache: hit`), so run several
  GPU tools back to back before editing.
- **Batch-decompile with ~8 threads, not more.** Scaling flattens at 4-8 workers (300
  functions: 12.6 s on 1 thread, 3.2 s on 8, 3.8 s on 16); that is the default.

## Task table

| Task in a decompilation workflow | Tool | Use | Measured (CPU vs GPU) |
| --- | --- | --- | --- |
| Decompile one function | `force_decompile`, `get_functions` | CPU (only option) | ~350 ms per call on kernel32 |
| C for up to 20 functions | `get_functions functions=a,b,...` | CPU, parallel workers | 20 bundles: 5.8 s on 1 worker, 0.95 s on 8 |
| Decompile many functions + analyse results | `gpu_decompile_batch` | CPU decompile; analysis: `auto` | 2,000 fns: 11.2 s decompile; dominators 0.8/1.9 ms, return slice 47/7 ms |
| Find crypto routines (AES/SHA/CRC/TEA... tables) | `gpu_detect_crypto_constants` | **GPU** | 4,381 / 44 ms (100x) on 24 MB |
| Find anti-debug / anti-VM code | `gpu_find_anti_analysis` | **GPU** | 5,554 / 96 ms (58x) |
| Find missed functions (prologues) | `gpu_find_function_starts` | **GPU** | 991 / 51 ms (19x) |
| Pull URLs, IPs, e-mails, registry paths | `gpu_extract_iocs` | **GPU** | 635 / 53 ms (12x) |
| Find all copies / near-copies of a function across a binary or two | `gpu_find_similar_functions` (all-vs-all) | **GPU** | MinHash 114M pairs: 19,929 / 874 ms (23x); cosine 9x |
| Find the twin of one function | `gpu_find_similar_functions function=` | `auto` (small: CPU) | one-vs-1.7k is sub-ms either way |
| Name an unknown function by its hash | `gpu_function_hashes` | GPU for the hashing; extraction dominates | 82 MB hashed: 263 / 82 ms; extraction 45 s |
| Resolve API-hash imports (ROR13, CRC32, djb2...) | `gpu_emulate_hash_batch` | **GPU** from ~1k candidates | real CRC routine x 5,000: 3,939 / 173 ms (23x) |
| Run a decoder / check routine on many inputs | `gpu_emulate_function_batch` | **GPU** from ~1k inputs | zlib crc32 x 20,000: 4,580 / 327 ms (14x) |
| Audit names after bulk renaming | `gpu_name_collisions` (no names = audit) | **GPU** for whole-program audits | all names x all names |
| Search names / types / class members | `gpu_find_functions scope=` | `auto` (usually CPU: small) | 36 / 6 ms on 80k names |
| Search instruction text (`[ecx+0xd0]`) | `gpu_search_instructions` | `auto` | 128k instructions: ~6 / 3 ms |
| Call graph: recursion groups, PageRank, reachability | `gpu_call_graph_analysis` | **CPU** at any real size | 80k fns: SCC 14 / 1,417 ms, PageRank 77 / 1,597 ms |
| CFG dominators / loops for every function | `gpu_control_flow_analysis` | CPU for big CFG sets; GPU for many small CFGs | 900k blocks: 339 / 2,096 ms; kernel32 25k blocks: 36 / 10 ms |
| Dataflow slice inside one function | `gpu_dataflow_slice` | CPU (one function is tiny) | decompile dominates |
| Entropy map (find packed/encrypted data) | `gpu_entropy_map` | **CPU** | 42 / 126 ms on 24 MB |
| Raw pointers into a target (vtables, jump tables) | `gpu_pointer_scan` | **CPU** | 60 / 483 ms on 24 MB |
| Missing scalar references, address tables | `gpu_analysis_passes` | `auto` | small on kernel32 (ms) |

## Workflow recipes

- **Triage a new binary:** `gpu_detect_crypto_constants`, `gpu_find_anti_analysis`,
  `gpu_extract_iocs`, `gpu_find_function_starts` with `backend=gpu` -- one pass each over
  all of memory. Then `gpu_entropy_map` on the CPU.
- **Map a binary against a known one** (a previous version, a library): `gpu_function_hashes`
  on both for exact matches, then `gpu_find_similar_functions program_b=... ` all-vs-all on
  the GPU for near matches, then `gpu_diff_functions` on the candidate pairs.
- **Resolve hashed imports:** find the hash routine, then `gpu_emulate_hash_batch` with the
  full export-name list of the likely DLLs (10^4-10^5 candidates) on the GPU.
- **Decompile a whole module for documentation:** `gpu_decompile_batch` (CPU decompile, all
  cores); use its structure and dataflow columns to pick what to read first; decompile the
  chosen functions with `get_functions` for C.
- **After renaming many functions:** `gpu_name_collisions` (audit mode) on the GPU to catch
  near-duplicate names the per-call gate would only catch one at a time.

## Letting `auto` decide

- Call a tool once with `backend=both` at the size you care about. It returns both timings
  and `agree`, and records the run; `auto` then follows the nearest measured size.
  `execution.auto_reason` says which run it followed.
- `gpu_advise(op, work)` answers without running anything.
- `gpu_benchmark` measures synthetic sizes; real runs refine it (synthetic crossovers did not
  hold for graph algorithms on an 80k-function binary).
- If `agree` is ever false, report it: CPU and GPU are built to return identical results.
