"""Time every read-only tool the coverage table calls "cpu-only" against a live server.

The claim behind those rows is that the tools have no computation a GPU could take over.
This measures it: each read-only tool is called on a real program (best of 3), and the
wall time lands in docs/gpu_cpu_only_timings.json, which docs/GPU_COVERAGE.md cites.
A GPU call carries a fixed ~1-2 ms launch + transfer floor, so a tool that finishes near
that cannot gain even from infinitely fast compute; slower ones are named with where the
time goes. Write tools are not timed: each is one database transaction.

Usage (server with an analysed program open, e.g. the headless server on 8090):
    python -m tools.gpu.measure_cpu_only --url http://127.0.0.1:8090 --function FUN_18000f650 \\
        --binary "kernel32.dll (x64, 4301 functions)"
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from tools.gpu import coverage

OUT = coverage.REPO / "docs" / "gpu_cpu_only_timings.json"
NEEDS_CONTEXT = {"debugger", "server", "headless", "project", "utility", "getter", "system"}


def call(url: str, path: str, method: str, params: dict) -> tuple[float, bool, str]:
    t = time.perf_counter()
    try:
        if method == "GET":
            req = urllib.request.Request(url + path + "?" + urllib.parse.urlencode(params))
        else:
            req = urllib.request.Request(url + path, data=json.dumps(params).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read().decode(errors="replace")
        ok = '"error"' not in body[:200]
        return (time.perf_counter() - t) * 1000, ok, body[:160]
    except urllib.error.HTTPError as e:
        return (time.perf_counter() - t) * 1000, False, f"HTTP {e.code}"


def sample_args(tool: dict, fn: str, addr: str) -> dict | None:
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
        else:
            return None  # needs a specific input this sweep cannot invent
    return args


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--function", required=True, help="a mid-sized function in the open program")
    ap.add_argument("--binary", required=True, help="what was measured, recorded in the output")
    args = ap.parse_args()

    with urllib.request.urlopen(args.url + "/mcp/schema", timeout=60) as r:
        schema = {t["path"]: t for t in json.loads(r.read())["tools"]}
    with urllib.request.urlopen(args.url + "/get_functions?" + urllib.parse.urlencode({"function": args.function}), timeout=120) as r:
        addr = "0x" + json.loads(r.read())["address"]

    # The HTTP round trip costs the same for every tool (on Windows it can sit near 15 ms);
    # measure it on a trivial route so per-tool work can be reported above it.
    floor = min(call(args.url, "/check_connection", "GET", {})[0] for _ in range(7))
    print(f"HTTP floor {floor:.2f} ms")
    results = {}
    for row in coverage.classify():
        if row["status"] != "cpu-only":
            continue
        tool = schema.get(row["path"])
        if tool is None:
            results[row["path"]] = {"status": "not served by this server"}
            continue
        if not tool.get("read_only"):
            results[row["path"]] = {"status": "write: one DB transaction, not timed"}
            continue
        if row["category"] in NEEDS_CONTEXT:
            results[row["path"]] = {"status": "needs live debugger / server / project context, not timed"}
            continue
        a = sample_args(tool, args.function, addr)
        if a is None:
            results[row["path"]] = {"status": "needs a specific input, not timed"}
            continue
        # Some tools need one of several optional selectors; on an error, retry with the
        # obvious ones (and a numeric value where "Create" was not a number).
        ms, ok, note = call(args.url, row["path"], tool.get("method", "GET"), a)
        if not ok:
            names = {p["name"] for p in tool.get("params", [])}
            retry = dict(a)
            for k, v in (("function", args.function), ("address", addr), ("number", "0x1234"), ("value", "0x1234")):
                if k in names:
                    retry[k] = v
            for k in list(retry):
                if retry[k] == "Create" and k in ("value", "number", "text"):
                    retry[k] = "0x1234"
            ms2, ok2, note2 = call(args.url, row["path"], tool.get("method", "GET"), retry)
            if ok2:
                a = retry
            ms, ok, note = ms2, ok2, note2
        best = ms
        for _ in range(2):
            ms, ok, note = call(args.url, row["path"], tool.get("method", "GET"), a)
            best = min(best, ms)
        results[row["path"]] = {"status": "ok" if ok else "error", "ms": round(best, 2)}
        if not ok:
            results[row["path"]]["detail"] = note
        print(f"{row['path']:45} {results[row['path']]}")
    OUT.write_text(json.dumps({"binary": args.binary, "function": args.function, "http_floor_ms": round(floor, 2), "results": results},
                              indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(coverage.REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
