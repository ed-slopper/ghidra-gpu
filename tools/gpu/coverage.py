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
                        "Decompile-derived analyses (gpu_dataflow_slice) run their graph work on the GPU.",
    "/get_functions": "bundles decompiler output (native decompiler, see force_decompile) with indexed reads",
    "/analyze_function_complete": "composite of decompile + indexed reads; its structural part is gpu_control_flow_analysis",
    "/analyze_for_documentation": "composite of decompile + indexed reads",
    "/get_function_documentation": "reads one function's documentation record",
    "/disassemble_function": "Ghidra's SLEIGH disassembler (CPU); output is a listing read",
    "/disassemble_bytes": "Ghidra's SLEIGH disassembler (CPU) on caller-supplied bytes",
    "/run_analysis": "Ghidra's auto-analyzers (Java plug-ins, CPU)",
    "/reanalyze": "Ghidra's auto-analyzers (Java plug-ins, CPU)",
    "/analysis_status": "status read",
    "/read_memory": "returns bytes; nothing to compute",
    "/convert_number": "a single number conversion",
    "/get_function_pcode": "dumps P-code produced by the decompiler (CPU)",
    "/get_language_metadata": "language description read",
    "/merge_program_documentation": "documentation DB writes",
    "/apply_documentation": "documentation DB writes",
    "/get_entry_points": "symbol table read",
    "/get_function_count": "counter read",
    "/list_globals": "indexed symbol read",
    "/list_shadowed_globals": "indexed symbol read",
    "/list_program_items": "indexed listing read",
    "/list_calling_conventions": "compiler-spec read",
    "/get_external_location": "single external-location read",
    "/run_ghidra_script": "runs arbitrary Java/Python scripts in Ghidra",
    "/run_script_inline": "runs arbitrary Java/Python scripts in Ghidra",
    "/analyze_function_completeness": "documentation-quality scoring driven by the decompiler (native, CPU) and naming rules",
    "/batch_string_anchor_report": "joins documentation state to string references (indexed DB reads); raw string "
                                   "extraction itself is gpu_list_strings",
    "/archive_ingest_function": "HTTP upload to an external archive service",
    "/archive_ingest_program": "HTTP upload to an external archive service",
    "/find_data_types": "data type manager name lookup (indexed, small)",
    "/detect_array_bounds": "inspects one address's references and neighbours",
    "/get_assembly_context": "reads a few instructions around one address",
    "/emulate_function": "single run of Ghidra's EmulatorHelper; batched runs are gpu_emulate_function_batch",
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
    lines += ["", "## Not ported: nothing a GPU could compute", "",
              "These tools are database edits, single-record lookups, I/O, lifecycle or UI actions, or",
              "wrap Ghidra's native decompiler / SLEIGH disassembler. Moving them to a GPU would only add",
              "a host-to-device copy around a lookup.", "", "| Upstream tool | Group | Why |", "| --- | --- | --- |"]
    for r in sorted(cpu, key=lambda r: (r["category"], r["path"])):
        lines.append(f"| `{r['path'][1:]}` | {r['category']} | {r['detail']} |")
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
