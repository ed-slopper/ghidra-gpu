"""Time the tools measure_cpu_only cannot dry-run, against live scratch targets.

Two phases, each merged into docs/gpu_cpu_only_timings.json:

  debugger  -- a GUI server with a live debug trace (e.g. a launched whoami.exe): the
               target-acting tools (breakpoints, step, interrupt, resume) and the
               module-mapped address translation, each once, in an order that keeps the
               target alive until resume runs last.
  gui       -- GUI-only actions that refuse dry_run: debugger/launch, tool/goto_address,
               prompt_policy, and optionally exit_ghidra last (it closes that GUI).
  effects   -- tools that refuse dry_run because they touch files or the project (save,
               close/open, export/import, folders, archive): run FOR REAL, once. Refuses
               unless the server's project lives under --scratch, so it can only ever
               touch a throwaway project.

    python -m tools.gpu.measure_live gui --url http://127.0.0.1:8089 --launch-json launch.json --exit
    python -m tools.gpu.measure_live debugger --url http://127.0.0.1:8089 --module KERNEL32.DLL --function FUN_18000f650
    python -m tools.gpu.measure_live effects --url http://127.0.0.1:8090 --scratch <dir> --import-file <file>
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

from tools.gpu.measure_cpu_only import OUT, call


def get(url: str, path: str) -> object:
    with urllib.request.urlopen(url + path, timeout=120) as r:
        return json.loads(r.read())


def record(results: dict, path: str, ms: float, ok: bool, note: str, live: str) -> None:
    prev = results.get(path, {})
    if not ok and prev.get("status") == "ok" and prev.get("live"):
        print(f"{path:40} kept earlier live timing ({note[:80]})", flush=True)
        return  # a repeat that fails for state reasons (already versioned...) keeps the real timing
    results[path] = {"status": "ok" if ok else "error", "ms": round(ms, 2), "live": live}
    if not ok:
        results[path]["detail"] = note
    print(f"{path:40} {results[path]}", flush=True)


def debugger_phase(a, results: dict) -> None:
    modules = get(a.url, "/debugger/modules")
    mod = next(m for m in modules if os.path.basename(m["name"]).lower() == a.module.lower())
    fn = get(a.url, "/get_functions?function=" + a.function)
    meta = get(a.url, "/get_metadata")
    dyn = hex(int(mod["base"], 16) + int(fn["address"], 16) - int(meta["base_address"], 16))
    live = f"live trace, {a.module} at 0x{mod['base']}"
    steps = [
        ("/debugger/dynamic_to_static", "GET", {"address": dyn}),
        ("/debugger/static_to_dynamic", "GET", {"address": "0x" + fn["address"]}),
        ("/debugger/set_breakpoint", "POST", {"address": dyn}),
        ("/debugger/remove_breakpoint", "POST", {"address": dyn}),
        ("/debugger/step", "POST", {"kind": "into"}),
        ("/debugger/interrupt", "POST", {}),
        ("/debugger/resume", "POST", {}),
    ]
    for path, method, params in steps:
        ms, ok, note = call(a.url, path, method, params)
        record(results, path, ms, ok, note, live)


def gui_phase(a, results: dict) -> None:
    """GUI-only actions that refuse dry_run; exit_ghidra last, since it closes the GUI."""
    fn = get(a.url, "/get_functions?function=" + a.function)
    steps = [] if a.exit_only else [("/tool/goto_address", "POST", {"address": "0x" + fn["address"]}),
                                    ("/prompt_policy", "POST", {"action": "status"})]
    if a.launch_json and not a.exit_only:
        with open(a.launch_json, encoding="utf-8") as fh:
            steps.insert(0, ("/debugger/launch", "POST", json.load(fh)))
    for path, method, params in steps:
        ms, ok, note = call(a.url, path, method, params)
        record(results, path, ms, ok, note, "throwaway GUI instance")
    if a.exit or a.exit_only:
        ms, ok, note = call(a.url, "/exit_ghidra", "POST", {"force": True})
        record(results, "/exit_ghidra", ms, ok, note, "throwaway GUI instance (closed by this call)")


SERVER_A = [
    ("/server/connect", "POST", {}),
    ("/server/status", "GET", {}),
    ("/server/authenticate", "POST", {"username": "pc5", "password": "unused"}),
    ("/server/repository/create", "POST", {"name": "gpurepo"}),
    ("/server/repositories", "GET", {}),
    ("/server/admin/users", "GET", {}),
    ("/server/admin/set_permissions", "POST", {"repo": "gpurepo", "user": "pc5", "access_level": "admin"}),
]
SERVER_B = [
    ("/server/repository/files", "GET", {"repo": "gpurepo", "path": "/"}),
    ("/server/repository/file", "GET", {"repo": "gpurepo", "path": "/{import_name}"}),
    # project-level version control, on a program of the open project
    ("/server/version_control/add", "POST", {"path": "/{import_name}", "comment": "gpu measure", "keep_checked_out": False}),
    ("/server/version_history", "GET", {"path": "/kernel32.dll"}),
    ("/server/version_control/checkout", "POST", {"path": "/kernel32.dll", "exclusive": False}),
    ("/server/checkouts", "GET", {"path": "/kernel32.dll"}),
    ("!/set_comment", "POST", {"address": "0x18000f650", "comment": "gpu measure checkin", "comment_type": "plate",
                               "program": "kernel32.dll"}),
    ("!/save_program", "POST", {"program": "kernel32.dll"}),
    ("/checkin_program", "POST", {"path": "/kernel32.dll", "comment": "gpu measure", "keep_checked_out": True}),
    ("/server/version_control/undo_checkout", "POST", {"path": "/kernel32.dll", "keep": False}),
    ("/server/version_control/checkout", "POST", {"path": "/kernel32.dll", "exclusive": False}),
    ("/server/admin/terminate_checkout", "POST", {"path": "/kernel32.dll", "checkout_id": "1"}),
    ("/server/admin/terminate_all_checkouts", "POST", {"path": "/kernel32.dll"}),
    ("/server/disconnect", "POST", {}),
]


def scripts_phase(a, results: dict) -> None:
    """Script tools (server started with GHIDRA_MCP_ALLOW_SCRIPTS=1, inside Ghidra's own launcher)."""
    steps = [("/run_ghidra_script", "POST", {"script_name": "D:/code/modding/ghidra_12.1.4_PUBLIC/Ghidra/Features/Base/ghidra_scripts/HelloWorldScript.java", "program": a.program}),
             ("/run_script_inline", "POST", {"code": "println(\"gpu measure\");", "program": a.program})]
    for path, method, params in steps:
        ms, ok, note = call(a.url, path, method, params)
        record(results, path, ms, ok, note, "throwaway GUI instance, scripts enabled")


def server_step_c(a, results: dict) -> None:
    """Admin checkout termination needs a project bound to the repository: open it as a shared
    project, check a file out, terminate that checkout (and all), then reopen the local project."""
    live = "scratch Ghidra Server on 127.0.0.1:13100, shared project"
    local = get(a.url, "/mcp/instance_info").get("project_path", "") + ".gpr"
    print("setup open shared", call(a.url, "/open_project", "POST", {"path": "ghidra://127.0.0.1:13100/gpurepo"})[2][:120])
    print("setup checkout", call(a.url, "/server/version_control/checkout", "POST", {"path": "/Benchmark.dll"})[2][:120])
    _, _, body = call(a.url, "/server/checkouts", "GET", {"path": "/Benchmark.dll"})
    print("checkouts:", body[:200])
    try:
        parsed = json.loads(body)
        items = parsed.get("checkouts", []) if isinstance(parsed, dict) else parsed
        cid = str(items[0].get("checkout_id", items[0].get("id"))) if items else "0"
    except (ValueError, AttributeError, IndexError):
        cid = "0"
    ms, ok, note = call(a.url, "/server/admin/terminate_checkout", "POST", {"path": "/Benchmark.dll", "checkout_id": cid})
    record(results, "/server/admin/terminate_checkout", ms, ok, note, live)
    print("setup checkout", call(a.url, "/server/version_control/checkout", "POST", {"path": "/Benchmark.dll"})[2][:120])
    ms, ok, note = call(a.url, "/server/admin/terminate_all_checkouts", "POST", {"path": "/"})
    record(results, "/server/admin/terminate_all_checkouts", ms, ok, note, live)
    ms, ok, note = call(a.url, "/server/repository/create", "POST", {"name": f"gpurepo_{os.getpid()}"})
    record(results, "/server/repository/create", ms, ok, note, live)
    print("setup reopen local", call(a.url, "/open_project", "POST", {"path": local})[2][:120])


def server_phase(a, results: dict) -> None:
    """Against a scratch Ghidra Server (GHIDRA_SERVER_* set on the MCP server). Step b expects a
    file committed to repository 'gpurepo' (analyzeHeadless ghidra://.../gpurepo -import ... -commit)."""
    if a.server_step == "c":
        server_step_c(a, results)
        return
    subst = {"import_name": os.path.basename(a.import_file) if a.import_file else "Benchmark.dll"}
    for path, method, params in (SERVER_A if a.server_step == "a" else SERVER_B):
        body = {k: v.format(**subst) if isinstance(v, str) else v for k, v in params.items()}
        if path.startswith("!"):  # setup step (e.g. a modification so checkin has work), not timed
            print("setup", path[1:], call(a.url, path[1:], method, body)[2][:80], flush=True)
            continue
        ms, ok, note = call(a.url, path, method, body)
        record(results, path, ms, ok, note, "scratch Ghidra Server on 127.0.0.1:13100")


EFFECTS = [
    ("/save_program", "POST", {}),
    ("/save_all_programs", "POST", {}),
    ("/export_program", "POST", {"program_name": "kernel32.dll", "output_dir": "{scratch}", "output_name": "gpu_measure.gzf"}),
    ("/archive_project", "POST", {"output_dir": "{scratch}", "output_name": "gpu_measure.gar"}),
    ("/restore_project", "POST", {"gar_path": "{scratch}/gpu_measure.gar", "parent_dir": "{scratch}", "project_name": "gpu_restored"}),
    ("/create_folder", "POST", {"path": "/gpu_measure"}),
    ("/create_folder", "POST", {"path": "/gpu_measure_dest"}),
    ("/move_folder", "POST", {"sourcePath": "/gpu_measure", "destPath": "/gpu_measure_dest"}),
    ("/import_file", "POST", {"file_path": "{import_file}", "project_folder": "/", "auto_analyze": False}),
    ("/move_file", "POST", {"filePath": "/{import_name}", "destFolder": "/gpu_measure_dest"}),
    ("/move_file", "POST", {"filePath": "/gpu_measure_dest/{import_name}", "destFolder": "/"}),
    ("/import_program", "POST", {"gzf_path": "{scratch}/gpu_measure.gzf", "target_folder": "/gpu_measure_dest", "target_name": "k32_copy"}),
    ("/delete_file", "POST", {"filePath": "/gpu_measure_dest/k32_copy"}),
    ("/close_program", "POST", {"name": "kernel32.dll", "save": False}),
    ("/open_program", "POST", {"path": "/kernel32.dll"}),
    ("/run_ghidra_script", "POST", {"script_name": "HelloWorldScript.java", "program": "kernel32.dll"}),
    ("/run_script_inline", "POST", {"code": "println(\"gpu measure\");", "program": "kernel32.dll"}),
    ("/reanalyze", "POST", {"program": "kernel32.dll"}),
    ("/run_analysis", "POST", {"program": "kernel32.dll"}),
    # project lifecycle last: these switch the server's project
    ("/create_project", "POST", {"parentDir": "{scratch}", "name": "gpu_measure_proj"}),
    ("/close_project", "POST", {}),
    ("/open_project", "POST", {"path": "{project_gpr}"}),
    ("/delete_project", "POST", {"projectPath": "{scratch}/gpu_measure_proj"}),
]


def effects_phase(a, results: dict) -> None:
    info = get(a.url, "/mcp/instance_info")
    project = os.path.normcase(os.path.abspath(info.get("project_path", "")))
    scratch = os.path.normcase(os.path.abspath(a.scratch))
    if not project.startswith(scratch):
        raise SystemExit(f"refusing: project {project} is not under --scratch {scratch}")
    subst = {"scratch": a.scratch.replace("\\", "/"), "import_file": a.import_file.replace("\\", "/"),
             "import_name": os.path.basename(a.import_file),
             "project_gpr": (info.get("project_path", "") + ".gpr").replace("\\", "/")}
    for path, method, params in EFFECTS:
        body = {k: v.format(**subst) if isinstance(v, str) else v for k, v in params.items()}
        ms, ok, note = call(a.url, path, method, body)
        record(results, path, ms, ok, note, "real effect on a throwaway project")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=["debugger", "gui", "effects", "server", "scripts"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--function", default="FUN_18000f650")
    ap.add_argument("--module", default="KERNEL32.DLL")
    ap.add_argument("--program", default="kernel32.dll")
    ap.add_argument("--scratch", default="")
    ap.add_argument("--import-file", default="")
    ap.add_argument("--launch-json", default="", help="gui phase: body for /debugger/launch, timed first")
    ap.add_argument("--server-step", choices=["a", "b", "c"], default="a")
    ap.add_argument("--exit-only", action="store_true", help="gui phase: only time /exit_ghidra")
    ap.add_argument("--exit", action="store_true", help="gui phase: time /exit_ghidra last (closes that GUI)")
    a = ap.parse_args()
    data = json.loads(OUT.read_text(encoding="utf-8"))
    if a.phase == "debugger":
        debugger_phase(a, data["results"])
    elif a.phase == "gui":
        gui_phase(a, data["results"])
    elif a.phase == "server":
        server_phase(a, data["results"])
    elif a.phase == "scripts":
        scripts_phase(a, data["results"])
    else:
        effects_phase(a, data["results"])
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
