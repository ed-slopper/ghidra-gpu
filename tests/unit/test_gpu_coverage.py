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
