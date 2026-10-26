# Review of docs/GPU_COVERAGE.md (repo at 0bfdb397)

I read the "Code review" and "Not ported" sections against the source, and against my GameAssembly.dll test run (80,818 functions; REPORT.md is next to this file). Most verdicts hold: the rows are single-record edits, lookups, lifecycle actions or I/O, and none of those would gain from a GPU.

Four things are worth acting on. None of them is a GPU kernel. They are CPU parallelism or caching that the GPU layer already has the machinery for.

## 1. Bulk decompiles that run serially, while gpu_decompile_batch already has the worker pool

- **`get_functions`** in bulk mode (`functions=a,b,c...`), in `FunctionBundleService`:
  - It decompiles each requested function one after another, through `functionService.decompileFunctionNoRetry` (around line 159).
  - The doc's reason, "bundles decompiler output", is right that the cost is the decompiler. But a bulk request is the same many-function decompile `gpu_decompile_batch` runs on N threads.
  - Routing bulk mode through that pool would cut its wall time by about the 2.6x I measured.
- **`analyze_struct_field_usage`** in `DataTypeService`, around lines 2540-2570:
  - It decompiles up to 100 referencing functions in a serial loop over one `DecompInterface`, then regex-scans each function's C text.
  - The doc measured 1,457 ms on kernel32. With 100 functions that is about 100 serial decompiles, so it is the same kind of win.
  - The text scan afterwards is a substring search, which the existing substring kernel could take for big batches.

## 2. gpu_decompile_batch asks for C text it never uses

- **What it does:** `reduce()` only reads `getHighFunction()` (basic blocks and the def-use graph), but each worker's `DecompInterface` is opened with the defaults, so the decompiler also prints C for every function.
- **Fix:** `ifc.toggleCCode(false)`, keeping the syntax tree on. The printing is a real part of each call, so this is a cheap speed-up before anything else.
- **Thread scaling:** 500 functions took 4,686 ms on 1 thread and 1,792 ms on 16 (2.6x). Each `DecompInterface` has its own native process, so the likely limit is the decompiler's callbacks into the shared Program database.
- **Worth measuring:** the speed-up at 4, 8 and 16 threads, to find where it flattens. More threads than that only add processes.

## 3. Extraction is the cost the doc doesn't count

- **The measurements:** on 80k functions, `extract_ms` was the large majority of every program-wide GPU tool.

  | Tool | Extract | Op |
  |---|---|---|
  | `gpu_function_hashes` | 45.5 s | 0.08–0.26 s |
  | `gpu_control_flow_analysis` | 41.5 s | |
  | `gpu_call_graph_analysis` | 11.3 s | |
  | Similarity | 13–14 s | |

- **Why it matters:** each call re-reads the same instructions, blocks and call edges from the database.
- **Suggested fix:** cache the extracted `ProgramData` arrays per program, keyed by `Program.getModificationNumber()`. The second and later tool calls would then be nearly free, which matters more than any kernel speed-up for an agent making several calls in a row.
- **Wording:** this isn't a "non-ported tool", but GPU_COVERAGE.md's claim that "nothing was lost by porting" reads differently once extraction is 99% of the wall time.

## 4. Smaller notes

- **`list_globals`** (ListingService, around line 690): one `getReferenceCountTo` per global symbol, over every global. It's an indexed database read per symbol, so "indexed symbol read" is fair; no GPU win. On a big binary its cost is the per-symbol reference count, so batching the counts (one reference-manager pass) would help more than a GPU.
- **`set_image_base` (26 s), `run_ghidra_script`, `archive_project`, `export_program`:** I agree, these are I/O and compile time, not work a GPU could do.
- **`get_entry_points`:** a symbol walk with a stored flag. Agree.
- **The decompiler-bound rows** (`force_decompile`, `get_function_pcode`, `set_variable_type`, `set_variables`, `get_function_documentation`, `analyze_function_complete`, `analyze_for_documentation`): each is one function per call, so the reasons are right. Only the bulk callers in section 1 should change.

## Verdict on the doc

The classification is accurate as a GPU question: nothing listed as non-ported has data-parallel arithmetic a kernel would win on. Its blind spot is that program-wide time in these tools goes to serial decompiles and repeated database extraction, and the GPU work leaves both in place.
