"""Every upstream tool is either GPU-ported or carries a reason it is not.

Guards the claim in docs/GPU.md that the whole catalog was considered: a new upstream
tool lands UNCLASSIFIED and fails here until someone ports it or says why not, and
docs/GPU_COVERAGE.md must match what the generator would write today.
"""
from __future__ import annotations

import json

from tools.gpu import coverage


def test_no_upstream_tool_is_unclassified():
    rows = coverage.classify()
    assert [r["path"] for r in rows if r["status"] == "UNCLASSIFIED"] == []


def test_every_mapped_upstream_tool_exists_in_the_catalog():
    paths = {e["path"] for e in json.loads(coverage.CATALOG.read_text(encoding="utf-8"))["endpoints"]}
    missing = sorted(p for p in coverage.ported_map() if p not in paths)
    assert missing == [], f"GpuService.TOOL_MAP names tools the catalog does not have: {missing}"


def test_every_gpu_tool_in_tool_map_is_registered():
    paths = {e["path"] for e in json.loads(coverage.CATALOG.read_text(encoding="utf-8"))["endpoints"]}
    gpu_tools = {g for gs in coverage.ported_map().values() for g in gs}
    assert sorted(g for g in gpu_tools if "/" + g not in paths) == []


def test_coverage_doc_is_current():
    expected = coverage.render(coverage.classify())
    actual = coverage.OUT.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert actual == expected, "docs/GPU_COVERAGE.md is stale: python -m tools.gpu.coverage --write"


def test_every_slow_cpu_only_tool_has_its_own_measured_reason():
    """A tool that measurably takes time must say where the time goes, not inherit a category default."""
    t = coverage.timings()
    assert t is not None, "docs/gpu_cpu_only_timings.json missing: python -m tools.gpu.measure_cpu_only"
    slow = [p for p, v in t["results"].items() if v.get("status") == "ok" and v["ms"] > coverage.SLOW_MS]
    unexplained = sorted(p for p in slow if p not in coverage.EXPLICIT)
    assert unexplained == [], f"slow cpu-only tools with only a category reason: {unexplained}"


def test_timings_cover_only_classified_cpu_only_tools():
    t = coverage.timings()
    cpu_only = {r["path"] for r in coverage.classify() if r["status"] == "cpu-only"}
    assert cpu_only <= set(t["results"]), f"cpu-only tools never measured: {sorted(cpu_only - set(t['results']))}"


def test_review_verdicts_that_say_ported_name_a_mapped_gpu_tool():
    ported = coverage.ported_map()
    for path, verdict in coverage.REVIEW.items():
        if not verdict.startswith("ported"):
            continue
        assert path in ported, f"{path} is reviewed as ported but GpuService.TOOL_MAP does not map it"
        assert any(g in verdict for g in ported[path]), f"{path}: verdict names no GPU tool from TOOL_MAP {ported[path]}"
