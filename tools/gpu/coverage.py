"""GPU coverage of the upstream tool catalog: every tool is either ported or explained.

Reads tests/endpoints.json (the authoritative catalog) and GpuService.java's TOOL_MAP
(gpu tool -> upstream tools it accelerates). Every non-gpu endpoint must end up in
exactly one row: ported (named GPU twin) or cpu-only with the reason it has no
computation a GPU could take over. tests/unit/test_gpu_coverage.py fails when a new
upstream tool arrives unclassified, so the table cannot silently go stale.

Usage:
    python -m tools.gpu.coverage            # print summary
    python -m tools.gpu.coverage --write    # regenerate docs/GPU_COVERAGE.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "tests" / "endpoints.json"
SERVICE = REPO / "src" / "main" / "java" / "com" / "xebyte" / "core" / "GpuService.java"
OUT = REPO / "docs" / "GPU_COVERAGE.md"
TIMINGS = REPO / "docs" / "gpu_cpu_only_timings.json"
# A timed cpu-only tool slower than this must be explained by the decompiler.
SLOW_MS = 100.0

# Why a category has nothing to offload. Applied to every tool in the category that
# is not ported; EXPLICIT below overrides per tool.
CATEGORY_REASON = {
    "comment": "database write/read of one comment; no bulk computation",
    "symbol": "database write/read of one symbol; no bulk computation",
    "datatype": "data type manager edits and single-type lookups (DB transactions in the JVM)",
    "program": "program lifecycle, options, memory-map edits or I/O; no computation",
    "project": "project/file lifecycle and I/O",
    "server": "Ghidra Server networking and version control",
    "headless": "headless project lifecycle",
    "debugger": "live debugger control (TraceRmi); latency-bound, not compute-bound",
    "utility": "server identity / UI navigation",
    "system": "prompt policy configuration",
    "getter": "reads UI state",
    "function": "function database edits or single-function reads",
    "xref": "single-address reference lookups/edits (indexed DB reads)",
    "listing": "indexed listing reads; no scan to offload",
    "documentation": "documentation records (DB read/write, archive I/O)",
    "analysis": "Ghidra-internal analysis control or single-item reads",
    "malware": "",
    "emulation": "",
}

EXPLICIT = {
    "/force_decompile": "Ghidra's decompiler is native C++ run one function per process call; it cannot run on a GPU. "
                        "gpu_decompile_batch decompiles many functions on a CPU thread pool and runs the analysis of "
                        "the results (dominators, loops, dataflow slices) on the backend the caller picks.",
    "/get_functions": "bundles decompiler output (native decompiler, see force_decompile) with indexed reads",
    "/analyze_function_complete": "composite of decompile + indexed reads; its structural part is gpu_control_flow_analysis",
    "/analyze_for_documentation": "composite of decompile + indexed reads",
    "/get_function_documentation": "documentation record assembled from decompiler output (native, CPU)",
    "/disassemble_function": "Ghidra's SLEIGH disassembler (CPU); output is a listing read",
    "/disassemble_bytes": "Ghidra's SLEIGH disassembler (CPU) on caller-supplied bytes",
    "/run_analysis": "Ghidra's auto-analyzers; measured on kernel32: of 22 s, 14.3 s is the native decompiler "
                     "(Parameter ID 8.4 s, Switch Analysis 5.9 s), 3.6 s the x86 constant-propagation analyzer (a per-function "
                     "symbolic interpreter writing references into the DB); the rest is many sub-second DB passes. "
                     "Its function-start search is the part gpu_find_function_starts runs on the GPU",
    "/reanalyze": "re-runs Ghidra's auto-analyzers; same profile as run_analysis (mostly the native decompiler)",
    "/analysis_status": "status read",
    "/read_memory": "returns bytes; nothing to compute",
    "/convert_number": "a single number conversion",
    "/get_function_pcode": "dumps P-code produced by the decompiler (CPU)",
    "/get_language_metadata": "language description read",
    "/merge_program_documentation": "documentation DB writes",
    "/apply_documentation": "documentation writes in one transaction; for a single function it also scores "
                            "completeness by default, which decompiles it (native decompiler, CPU)",
    "/get_entry_points": "symbol table read",
    "/get_function_count": "counter read",
    "/list_globals": "indexed symbol read",
    "/list_shadowed_globals": "indexed symbol read",
    "/list_program_items": "indexed listing read",
    "/list_calling_conventions": "compiler-spec read",
    "/get_external_location": "single external-location read",
    "/run_ghidra_script": "compiles (javac + OSGi bundle) and runs an arbitrary user script; compile dominates",
    "/run_script_inline": "compiles (javac + OSGi bundle) and runs user-supplied script code; compile dominates",
    "/analyze_function_completeness": "documentation-quality scoring driven by the decompiler (native, CPU) and naming rules",
    "/batch_string_anchor_report": "joins documentation state to string references (indexed DB reads); raw string "
                                   "extraction itself is gpu_list_strings",
    "/archive_ingest_function": "HTTP upload to an external archive service",
    "/archive_ingest_program": "walks every function building upload payloads (indexed DB reads) for an external "
                               "archive service; its one compute step, an opcode SHA-256 per function, is "
                               "gpu_function_hashes(kind=opcode), byte-identical",
    "/find_data_types": "data type manager name lookup (indexed, small)",
    "/detect_array_bounds": "inspects one address's references and neighbours",
    "/get_assembly_context": "reads a few instructions around one address",
    "/analyze_struct_field_usage": "decompiles each function referencing the struct (native decompiler, CPU) to map "
                                   "field offsets; the per-function decompile is the cost",
    "/get_ui_cursor": "reads the CodeBrowser cursor (GUI state)",
    "/set_image_base": "rebasing rewrites every address-keyed record in the program database (DB I/O, single "
                       "transaction); no arithmetic a GPU could take over",
    "/batch_set_comments": "a handful of comment writes followed by a deliberate 500 ms settle wait "
                           "(Thread.sleep in CommentService), not computation",
    "/set_variable_type": "retypes a variable at the decompiler layer: decompiles the function (native decompiler, CPU)",
    "/debugger/remove_breakpoint": "one Trace RMI round trip to the debugger agent process (dbgeng); latency, not compute",
    "/debugger/launch": "spawns the target and the Python debugger agent, then waits for its Trace RMI connection",
    "/debugger/static_to_dynamic": "waits (up to 10 s) for the trace's static module mapping; a lookup, not compute",
    "/exit_ghidra": "shuts Ghidra down",
    "/set_variables": "retypes/renames variables in one transaction; decompiles the function for fresh SSA variables (native decompiler, CPU)",
    "/export_program": "serialises the program database into a packed .gzf file (DB + file I/O)",
    "/import_program": "unpacks a .gzf into the project (file I/O + DB)",
    "/import_file": "Ghidra's loader parsing the binary format (PE/ELF/...) into a new program database",
    "/archive_project": "packs the whole project directory into a .gar archive (file I/O + zip)",
    "/restore_project": "unpacks a .gar archive into a new project directory (file I/O + unzip)",
    "/checkin_program": "uploads the changed database buffers to the Ghidra Server (network I/O)",
    "/emulate_function": "single run of Ghidra's EmulatorHelper; batched runs are gpu_emulate_function_batch",
}


# Hand review (2026-10-25) of every place a non-GPU tool -- or a helper it calls -- iterates a
# whole-program collection (all functions, instructions, symbols, defined data, data types,
# references, comments, bookmarks). Each was read in context. "ported" verdicts must name a
# GPU tool that TOOL_MAP maps the path to (tests/unit/test_gpu_coverage.py checks it).
REVIEW = {
    "/rename_function": "ported: token-subset collision check against every function name -> gpu_name_collisions "
                        "(batched; CPU twin matches NamingConventions.findTokenSubsetCollision exactly)",
    "/analyze_function_completeness": "ported: its name-collision check over all function names -> gpu_name_collisions; "
                                      "the rest is per-function decompiler-driven scoring",
    "/find_data_types": "ported: substring over every data type name and path -> gpu_find_functions scope=data_types",
    "/list_class_members": "ported: namespace / implicit-this filter over every function -> gpu_find_functions scope=class_members",
    "/archive_ingest_program": "partly ported: the per-function opcode SHA-256 -> gpu_function_hashes kind=opcode; the rest builds "
                               "upload payloads from DB reads",
    "/batch_string_anchor_report": "string extraction is gpu_list_strings; the report itself joins defined strings to their "
                                   "referencing functions through DB indexes (no arithmetic)",
    "/get_entry_points": "walks all symbols keeping isExternalEntryPoint -- a stored flag, so the cost is the walk itself",
    "/list_program_items": "enumerates one kind of item and pages it (collectMethodNames/ClassNames/Exports/Namespaces); no predicate to compute",
    "/list_bookmarks": "enumerates bookmarks and pages them",
    "/merge_program_documentation": "copies every documented item to the same address in another program: address-keyed DB "
                                    "lookups and writes, nothing data-parallel",
    "/disassemble_function": "getFunctions(start, true) only to find the next function boundary (bounded)",
    "/analyze_for_documentation": "reference iteration bounded to one function body (getReferenceSourceIterator(body))",
    "/create_label": "symbol iterator at one address",
    "/set_global": "findEvictionVictims: symbols in the range being retyped (bounded)",
    "/resolve_duplicate_type": "findAllTypesBySimpleName: one name looked up across data types",
    "(name lookups)": "ServiceUtils.functionsNamedIgnoreCase / searchByNameInAllCategories: one name resolved per call, "
                      "a single equality pass -- cheaper than a host-to-device copy of the names",
    "(struct matching)": "AnalysisService.findMatchingStructType: a few parameter names matched against data type names "
                         "inside per-function completeness/documentation analysis",
}


def ported_map() -> dict[str, list[str]]:
    """upstream path -> [gpu tools] parsed from GpuService.TOOL_MAP."""
    text = SERVICE.read_text(encoding="utf-8")
    out: dict[str, list[str]] = defaultdict(list)
    for m in re.finditer(r'TOOL_MAP\.put\("(gpu_\w+)",\s*List\.of\(([^;]*)\)\);', text):
        gpu = m.group(1)
        for up in re.findall(r'"([^"]+)"', m.group(2)):
            name = up.split(" ")[0]
            out["/" + name].append(gpu)
    return out


def classify() -> list[dict]:
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    ported = ported_map()
    rows = []
    for e in cat["endpoints"]:
        path, category = e["path"], e.get("category", "")
        if path.startswith("/gpu_"):
            continue
        if path in ported:
            rows.append({"path": path, "category": category, "status": "gpu", "detail": ", ".join(sorted(set(ported[path])))})
            continue
        reason = EXPLICIT.get(path) or CATEGORY_REASON.get(category, "")
        rows.append({"path": path, "category": category, "status": "cpu-only" if reason else "UNCLASSIFIED", "detail": reason})
    return rows


def timings() -> dict | None:
    return json.loads(TIMINGS.read_text(encoding="utf-8")) if TIMINGS.exists() else None


def measured(t: dict | None, path: str) -> str:
    v = (t or {}).get("results", {}).get(path)
    if not v:
        return ""
    if v.get("status") == "ok":
        return f"{v['ms']:.1f} ms"
    return v["status"].split(",")[0].split(":")[0]


def render(rows: list[dict]) -> str:
    gpu = [r for r in rows if r["status"] == "gpu"]
    cpu = [r for r in rows if r["status"] == "cpu-only"]
    lines = [
        "# GPU coverage of the ghidra-mcp tool catalog",
        "",
        "Generated by `python -m tools.gpu.coverage --write` from `tests/endpoints.json` and",
        "`GpuService.TOOL_MAP`; `tests/unit/test_gpu_coverage.py` fails if a tool is unclassified.",
        "",
        f"**{len(rows)} upstream tools: {len(gpu)} have a GPU twin, {len(cpu)} have no computation to port.**",
        "Every GPU twin also runs on the CPU (`backend=cpu`), so nothing was lost by porting.",
        "",
        "## Ported (GPU twin available)",
        "",
        "| Upstream tool | GPU tool(s) |",
        "| --- | --- |",
    ]
    for r in sorted(gpu, key=lambda r: r["path"]):
        lines.append(f"| `{r['path'][1:]}` | {', '.join('`' + g + '`' for g in r['detail'].split(', '))} |")
    lines += ["", "## Code review: every program-wide loop outside the GPU tools", "",
              "Every place a non-GPU tool (or a helper it calls) iterates a whole-program collection was read "
              "in context. Verdicts:", "", "| Tool / helper | Verdict |", "| --- | --- |"]
    for path, verdict in REVIEW.items():
        lines.append(f"| `{path.lstrip('/')}` | {verdict} |")
    lines += ["", "Every other non-ported tool touches a bounded number of records per call (one address, one "
              "function, one type, one setting) or is lifecycle/I-O.", ""]
    lines += ["", "## Not ported: nothing a GPU could compute", "",
              "These tools are database edits, single-record lookups, I/O, lifecycle or UI actions, or",
              "wrap Ghidra's native decompiler / SLEIGH disassembler. Moving them to a GPU would only add",
              "a host-to-device copy around a lookup.", ""]
    t = timings()
    if t:
        cpu_paths = {r["path"] for r in cpu}
        res = {p: v for p, v in t["results"].items() if p in cpu_paths}  # tools ported since are not cpu-only
        ok = {p: v for p, v in res.items() if v.get("status") == "ok"}
        ms = sorted(v["ms"] for v in ok.values())
        n_write = sum(1 for v in ok.values() if v.get("dry_run"))
        floors = t.get("http_floor_runs_ms") or [t.get("http_floor_ms", 0)]
        untimed = defaultdict(int)
        for v in res.values():
            if v.get("status") != "ok":
                untimed[v["status"]] += 1
        slow = sorted((p for p, v in ok.items() if v["ms"] > SLOW_MS), key=lambda p: -ok[p]["ms"])
        lines += [
            "### What was measured",
            "",
            f"`python -m tools.gpu.measure_cpu_only` called these tools on {t['binary']} (best of 3, wall time "
            f"including HTTP; write tools as `dry_run=true`, i.e. inside a transaction that is rolled back).",
            "",
            f"- **{len(ok)} of {len(res)} timed** ({len(ok) - n_write} read-only, {n_write} writes). Median "
            f"**{ms[len(ms) // 2]:.1f} ms**. A no-op request (`/check_connection`) took between "
            f"{min(floors):.1f} and {max(floors):.1f} ms across the measurement runs (Windows loopback HTTP), so the "
            f"many 1-20 ms rows are mostly request overhead around a lookup or a single transaction.",
            f"- Every timed tool slower than {SLOW_MS:.0f} ms, and where its time goes (each has its own reason in the table):",
        ]
        for p in slow:
            lines.append(f"  - `{p[1:]}` {ok[p]['ms']:.0f} ms")
        if untimed:
            lines += [f"- **{sum(untimed.values())} not timed**, each named in the table with why:"]
            for status, n in sorted(untimed.items(), key=lambda kv: -kv[1]):
                lines.append(f"  - {n}: {status}")
        else:
            lines += ["- **Every one of them was timed.**"]
        lines += ["",
                  "How the hard ones were reached: GUI-only tools in a real Ghidra GUI started with an isolated "
                  "settings directory (`-Dapplication.settingsdir`, no access to the user's projects); debugger tools "
                  "against a live dbgeng trace of a launched `whoami.exe`; Ghidra Server tools against a scratch "
                  "server bound to 127.0.0.1 (`python -m tools.gpu.measure_live`); tools that refuse a dry run "
                  "(save, export, import, archive, project lifecycle, scripts, analysis) for real on a throwaway "
                  "project. Rows marked `live` in docs/gpu_cpu_only_timings.json say which.", ""]
    lines += ["| Upstream tool | Group | Why | Measured |", "| --- | --- | --- | --- |"]
    for r in sorted(cpu, key=lambda r: (r["category"], r["path"])):
        lines.append(f"| `{r['path'][1:]}` | {r['category']} | {r['detail']} | {measured(t, r['path'])} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rows = classify()
    bad = [r["path"] for r in rows if r["status"] == "UNCLASSIFIED"]
    counts = defaultdict(int)
    for r in rows:
        counts[r["status"]] += 1
    print(dict(counts))
    if bad:
        print("UNCLASSIFIED:", *bad, sep="\n  ")
        return 1
    if args.write:
        OUT.write_text(render(rows), encoding="utf-8", newline="\n")
        print("wrote", OUT.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
