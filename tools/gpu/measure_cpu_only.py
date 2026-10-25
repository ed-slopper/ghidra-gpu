"""Time every read-only tool the coverage table calls "cpu-only" against a live server.

The claim behind those rows is that the tools have no computation a GPU could take over.
This measures it: each read-only tool is called on a real program (best of 3), and the
wall time lands in docs/gpu_cpu_only_timings.json, which docs/GPU_COVERAGE.md cites.
A GPU call carries a fixed ~1-2 ms launch + transfer floor, so a tool that finishes near
that cannot gain even from infinitely fast compute; slower ones are named with where the
time goes. Write tools run as dry runs (inside a transaction that is rolled back); tools
that refuse a dry run have effects outside the program database and are listed as such.

Usage (server with an analysed program open, e.g. the headless server on 8090):
    python -m tools.gpu.measure_cpu_only --url http://127.0.0.1:8090 --function FUN_18000f650 \\
        --binary "kernel32.dll (x64, 4301 functions)"
"""
from __future__ import annotations

import argparse
import functools
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from tools.gpu import coverage

print = functools.partial(print, flush=True)  # progress shows up in redirected logs

OUT = coverage.REPO / "docs" / "gpu_cpu_only_timings.json"
# Categories that need something this machine may not have: a Ghidra Server, or a debugger
# with a live target. Tools there are only timed when EXPLICIT_ARGS gives them inputs.
NEEDS_CONTEXT = {"debugger", "server"}

# Inputs for tools whose required arguments cannot be guessed from the schema. Placeholders:
# {fn} a function name, {addr} its entry, {base} the image base, {tmp} a scratch directory.
# The enum, struct and property map are created by --setup in the (throwaway) project.
EXPLICIT_ARGS = {
    "/analysis_status": {},
    "/analyze_struct_field_usage": {"address": "{base}", "struct_name": "IMAGE_DOS_HEADER", "max_functions": 5},
    "/get_assembly_context": {"xref_sources": "{addr}", "context_instructions": 5},
    "/get_enum_values": {"enum_name": "GpuMeasureEnum"},
    "/get_struct_layout": {"struct_name": "GpuMeasureStruct"},
    "/get_type_size": {"type_name": "int"},
    "/get_valid_data_types": {"category": "/"},
    "/list_class_members": {"class_name": "GpuMeasureClass"},
    "/list_program_items": {"kind": "exports", "limit": 100},
    "/list_project_files": {"folder": "/"},
    "/list_properties": {"map": "gpu_measure"},
    "/get_property": {"map": "gpu_measure", "address": "{addr}"},
    "/suggest_field_names": {"struct_address": "{base}", "struct_size": 128},
    "/validate_data_type": {"address": "{addr}", "type_name": "dword"},
    "/validate_function_prototype": {"function": "{fn}", "prototype": "int {fn}(int a, int b)"},
    "/list_projects": {"searchDir": "{tmp}"},
    "/get_ui_cursor": {"type": "all"},
    "/debugger/status": {}, "/debugger/traces": {}, "/debugger/list_breakpoints": {}, "/debugger/modules": {},
    "/debugger/registers": {}, "/debugger/launch_offers": {}, "/debugger/stack_trace": {"depth": 5},
    "/debugger/read_memory": {"address": "{addr}", "size": 64},
    "/debugger/static_to_dynamic": {"address": "{addr}"}, "/debugger/dynamic_to_static": {"address": "{addr}"},
    "/tool/running_tools": {},
    "/tool/goto_address": {"address": "{addr}"},
    "/prompt_policy": {"action": "status"},
    # write tools, all run as dry runs (rolled back)
    "/add_function_tag": {"function": "{fn}", "tags": "gpu_measure2"},
    "/add_struct_field": {"struct_name": "GpuMeasureStruct", "field_name": "dwC", "field_type": "dword"},
    "/clone_data_type": {"source_type": "GpuMeasureStruct", "new_name": "GpuMeasureStruct2"},
    "/configure_analyzer": {"name": "Stack", "enabled": "true"},
    "/create_data_type_category": {"category_path": "/gpu_measure2"},
    "/create_derived_type": {"kind": "pointer", "base_type": "GpuMeasureStruct", "name": "PGpuMeasureStruct"},
    "/create_function_signature": {"name": "GpuMeasureSig", "return_type": "int", "parameters": [{"name": "a", "type": "int"}]},
    "/create_label": {"address": "{addr}", "name": "gpu_measure_label"},
    "/create_memory_block": {"name": "gpu_measure", "address": "0x190000000", "size": 4096},
    "/create_struct": {"name": "GpuMeasureNew", "fields": [{"name": "dwA", "type": "dword"}]},
    "/create_union": {"name": "GpuMeasureUnion", "fields": [{"name": "dwA", "type": "dword"}]},
    "/recreate_struct": {"name": "GpuMeasureStruct", "fields": [{"name": "dwA", "type": "dword"}], "force": "true"},
    "/delete_function_tag": {"name": "gpu_measure"},
    "/set_function_tag_comment": {"name": "gpu_measure", "comment": "gpu measure"},
    "/remove_function_tag": {"function": "{fn}", "tags": "gpu_measure"},
    "/import_data_types": {"source": "struct GpuImported {{ int a; int b; }};", "format": "c"},
    "/modify_struct_field": {"struct_name": "GpuMeasureStruct", "field_name": "dwA", "new_name": "dwAA"},
    "/move_data_type_to_category": {"type_name": "GpuMeasureStruct", "category_path": "/gpu_measure"},
    "/set_program_option": {"group": "Program Information", "name": "Executable Location", "value": "x", "type": "string"},
    "/remove_program_option": {"group": "Program Information", "name": "Executable Location"},
    "/set_property": {"map": "gpu_measure", "address": "{addr}", "value": "y"},
    "/remove_property": {"map": "gpu_measure", "address": "{addr}"},
    "/remove_struct_field": {"struct_name": "GpuMeasureStruct", "field_name": "dwB"},
    "/rename_data_type": {"old_name": "GpuMeasureStruct", "new_name": "GpuMeasureStructR"},
    "/rename_function": {"function": "{fn}", "new_name": "ProcessGpuMeasureBuffer"},
    "/resize_struct": {"name": "GpuMeasureStruct", "new_size": 16},
    "/set_function_this_type": {"function_address": "{addr}", "this_type": "GpuMeasureStruct"},
    "/set_global": {"address": "0x1800c7458", "name": "g_dwGpuMeasureResource", "type_name": "dword",
                    "plate_comment": "GPU measurement global, set in a rolled-back dry run to time set_global."},
    "/set_memory_block": {"block": ".data", "write": "true"},
    "/set_variable_type": {"function_address": "{addr}", "variable_name": "local_res10", "new_type": "longlong"},
    "/switch_program": {"program": "kernel32.dll"},
    "/disassemble_bytes": {"start_address": "0x190000000", "length": 5},
    "/create_function": {"address": "0x190000000", "name": "GpuMeasureFunction", "disassemble_first": "true"},
    "/set_function_this_type": {"function_address": "0x180001008", "this_type": "GpuMeasureStruct"},
    "/set_variables": {"function": "{fn}", "variables": {"local_res10": {"type": "longlong"}}},
    "/set_variable_storage": {"function": "{fn}", "variable_name": "local_res10", "storage": "Stack[0x10]:8"},
    "/merge_program_documentation": {"source": "Benchmark.dll", "target": "kernel32.dll"},
    "/apply_data_classification": {"address": "0x1800c7458", "classification": "PRIMITIVE", "name": "g_dwGpuMeasure",
                                   "type_definition": {"type": "dword"}},
}


def setup(url: str, fn: str, addr: str, program: str = "") -> None:
    """Creates the data types / property map EXPLICIT_ARGS refers to. Throwaway projects only."""
    steps = [
        ("/create_enum", {"name": "GpuMeasureEnum", "values": {"A": 1, "B": 2, "C": 4}, "size": 4}),
        ("/create_struct", {"name": "GpuMeasureStruct", "fields": [{"name": "dwA", "type": "dword"}, {"name": "dwB", "type": "dword"}]}),
        ("/create_property_map", {"name": "gpu_measure", "type": "string"}),
        ("/set_property", {"map": "gpu_measure", "address": addr, "value": "x"}),
        ("/add_function_tag", {"function": fn, "tags": "gpu_measure"}),
        ("/create_data_type_category", {"category_path": "/gpu_measure"}),
        # an executable scratch block with a tiny function body for disassemble_bytes / create_function
        ("/create_memory_block", {"name": "gpu_measure_code", "address": "0x190000000", "size": 16, "read": True,
                                  "execute": True, "bytes_hex": "554889e5c3" + "90" * 11}),
        # a second program so merge_program_documentation has a source
        ("/open_program", {"path": "/Benchmark.dll"}),
        ("/switch_program", {"program": "kernel32.dll"}),
        # a __thiscall function for set_function_this_type
        ("/set_function_prototype", {"function_address": "0x180001008", "prototype": "void FUN_180001008(void * this, void * p2)",
                                     "calling_convention": "__thiscall"}),
    ]
    for path, body in steps:
        if program and path not in ("/open_program", "/switch_program"):
            body = {**body, "program": program}
        print("setup", path, call(url, path, "POST", body)[2][:100])


def call(url: str, path: str, method: str, params: dict, dry_run: bool = False) -> tuple[float, bool, str]:
    t = time.perf_counter()
    q = "?dry_run=true" if dry_run else ""
    try:
        if method == "GET":
            req = urllib.request.Request(url + path + "?" + urllib.parse.urlencode(
                {**params, **({"dry_run": "true"} if dry_run else {})}))
        else:
            req = urllib.request.Request(url + path + q, data=json.dumps(params).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read().decode(errors="replace")
        try:
            parsed = json.loads(body)
            ok = not (isinstance(parsed, dict) and parsed.get("error"))
        except ValueError:
            ok = '"error"' not in body[:200]
        return (time.perf_counter() - t) * 1000, ok, body[:160]
    except urllib.error.HTTPError as e:
        return (time.perf_counter() - t) * 1000, False, f"HTTP {e.code}"


def sample_args(tool: dict, fn: str, addr: str, writes: bool = False) -> dict | None:
    args = {}
    for p in tool.get("params", []):
        if not p.get("required"):
            continue
        name, ptype, typ = p["name"], p.get("param_type", ""), p.get("type", "string")
        if ptype == "address" or "address" in name:
            args[name] = addr
        elif ptype == "function_ref" or "function" in name or name in ("name", "target_function"):
            args[name] = fn
        elif typ == "integer":
            args[name] = 10
        elif typ == "boolean":
            args[name] = "false"
        elif name in ("query", "pattern", "filter", "search_term", "search", "mnemonic", "text"):
            args[name] = "Create"
        elif writes:
            args[name] = WRITE_DEFAULTS.get(name, "x")
        else:
            return None  # needs a specific input this sweep cannot invent
    return args


# Plausible values for write tools' required arguments; every write runs as a dry run.
WRITE_DEFAULTS = {
    "new_name": "GpuMeasureName", "name": "GpuMeasureName", "comment": "gpu measure", "type": "int",
    "data_type": "int", "type_name": "int", "value": "1", "tag": "gpu_measure", "tag_name": "gpu_measure",
    "prototype": "int f(int a)", "category": "/gpu_measure", "values": "{\"A\":1}", "fields": "[]",
    "label": "gpu_measure_label", "kind": "PRE", "comment_type": "pre", "size": 4, "length": 4,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--function", required=True, help="a mid-sized function in the open program")
    ap.add_argument("--binary", required=True, help="what was measured, recorded in the output")
    ap.add_argument("--program", default="", help="program to measure when several are open")
    ap.add_argument("--setup", action="store_true", help="create the enum/struct/property map EXPLICIT_ARGS uses "
                    "(writes to the open program: use a throwaway project)")
    ap.add_argument("--only", default="", help="comma-separated path prefixes to (re)measure, e.g. /debugger,/tool/ "
                    "(implies --merge: every other tool keeps its earlier result)")
    ap.add_argument("--merge", action="store_true", help="keep earlier results for tools this server does not serve "
                    "(time the headless server, then the GUI plugin, into one file)")
    args = ap.parse_args()

    with urllib.request.urlopen(args.url + "/mcp/schema", timeout=60) as r:
        schema = {t["path"]: t for t in json.loads(r.read())["tools"]}
    sel = {"program": args.program} if args.program else {}
    with urllib.request.urlopen(args.url + "/get_functions?" + urllib.parse.urlencode({"function": args.function, **sel}), timeout=120) as r:
        addr = "0x" + json.loads(r.read())["address"]
    with urllib.request.urlopen(args.url + "/get_metadata?" + urllib.parse.urlencode(sel), timeout=60) as r:
        meta = json.loads(r.read())
    base = "0x" + str(meta.get("base_address") or meta.get("image_base") or addr.replace("0x", "")).replace("0x", "")
    tmp = str(coverage.REPO / "build")
    if args.setup:
        setup(args.url, args.function, addr, meta.get("program_name", ""))
    only = [p.strip() for p in args.only.split(",") if p.strip()]
    if only:
        args.merge = True
    previous = json.loads(OUT.read_text(encoding="utf-8"))["results"] if args.merge and OUT.exists() else {}

    # The HTTP round trip costs the same for every tool (on Windows it can sit near 15 ms);
    # measure it on a trivial route so per-tool work can be reported above it.
    floor = min(call(args.url, "/check_connection", "GET", {})[0] for _ in range(7))
    print(f"HTTP floor {floor:.2f} ms")
    results = {}
    for row in coverage.classify():
        if row["status"] != "cpu-only":
            continue
        prev = previous.get(row["path"])
        if only and not any(row["path"].startswith(p) for p in only):
            if prev:
                results[row["path"]] = prev
            continue
        tool = schema.get(row["path"])
        if tool is None:
            results[row["path"]] = prev if prev and prev.get("status") in ("ok", "error") else {
                "status": "not served by this server (GUI-only)"}
            continue
        write = not tool.get("read_only")
        if row["path"] in EXPLICIT_ARGS:
            a = {k: (v.format(fn=args.function, addr=addr, base=base, tmp=tmp) if isinstance(v, str) else v)
                 for k, v in EXPLICIT_ARGS[row["path"]].items()}
        elif row["category"] in NEEDS_CONTEXT:
            results[row["path"]] = {"status": "needs a Ghidra Server / live debug target, not timed"}
            continue
        else:
            a = sample_args(tool, args.function, addr, writes=write)
        if a is not None and "program" not in a and any(p["name"] == "program" for p in tool.get("params", [])):
            a["program"] = meta.get("program_name", "")  # explicit, so a second open program never makes it ambiguous
        if a is None:
            results[row["path"]] = {"status": "needs a specific input, not timed"}
            continue
        # Some tools need one of several optional selectors; on an error, retry with the
        # obvious ones (and a numeric value where "Create" was not a number).
        ms, ok, note = call(args.url, row["path"], tool.get("method", "GET"), a, write)
        if not ok:
            names = {p["name"] for p in tool.get("params", [])}
            retry = dict(a)
            for k, v in (("function", args.function), ("address", addr), ("number", "0x1234"), ("value", "0x1234")):
                if k in names:
                    retry[k] = v
            for k in list(retry):
                if retry[k] == "Create" and k in ("value", "number", "text"):
                    retry[k] = "0x1234"
            ms2, ok2, note2 = call(args.url, row["path"], tool.get("method", "GET"), retry, write)
            if ok2:
                a = retry
            ms, ok, note = ms2, ok2, note2
        best = ms
        for _ in range(2):
            ms, ok, note = call(args.url, row["path"], tool.get("method", "GET"), a, write)
            best = min(best, ms)
        if not ok and "dry_run is not supported" in note:
            results[row["path"]] = {"status": "external effect (refuses dry_run: files, save, close, server), not timed"}
            print(f"{row['path']:45} {results[row['path']]}")
            continue
        results[row["path"]] = {"status": "ok" if ok else "error", "ms": round(best, 2)}
        if write:
            results[row["path"]]["dry_run"] = True
        if not ok:
            results[row["path"]]["detail"] = note
        print(f"{row['path']:45} {results[row['path']]}")
    floors = sorted(set((json.loads(OUT.read_text(encoding="utf-8")).get("http_floor_runs_ms", []) if args.merge and OUT.exists() else [])
                        + [round(floor, 2)]))
    OUT.write_text(json.dumps({"binary": args.binary, "function": args.function, "http_floor_ms": round(floor, 2),
                               "http_floor_runs_ms": floors, "results": results},
                              indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(coverage.REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
