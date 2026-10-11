from __future__ import annotations

from pathlib import Path

from ratchet.bench.tau.store import JsonlStore
from ratchet.bench.terminal import report as P


def result(case: str, passed: bool, error: str | None = None) -> dict[str, object]:
    return {"case_id": case, "passed": passed, "error": error}


def write(root: Path, sub: str, harness: str, results: list[dict[str, object]]) -> None:
    store = JsonlStore(root / sub)
    store.put_eval(
        {"type": "eval", "harness_version": harness, "results": results}, eval_id=f"{sub}-{harness}"
    )
    store.put_eval({"type": "evaluator", "version": "x"}, eval_id="ignored")


def test_collect_and_filter(tmp_path: Path) -> None:
    write(tmp_path, "runs/h1/union-trial-1", "h1", [result("a", True), result("b", False)])
    write(tmp_path, "runs/h1/union-trial-2", "h1", [result("a", True), result("b", True)])
    write(tmp_path, "evolve/phase-1/run-1", "h2", [result("a", False, "boom")])
    rows = P.collect(tmp_path)
    assert len(rows) == 3 and {r["source"] for r in rows} >= {"evolve/phase-1/run-1"}
    assert len(P.results_for(rows, "h1")) == 4
    assert len(P.results_for(rows, "h1", ["runs/h1/union-trial-2"])) == 2
    assert len(P.results_for(rows, "h1", ["evolve"])) == 0
    r = P.rate(P.results_for(rows, "h1"), ["a", "b", "c"])
    assert (r.tasks, r.runs, r.passed, r.errors) == (2, 4, 3, 0)
    assert r.per_task == {"a": (2, 2), "b": (1, 2), "c": (0, 0)}
    assert r.value == 0.75 and r.text() == "3/4 runs = 75.0% over 2 tasks"
    bad = P.rate(P.results_for(rows, "h2"), ["a"])
    assert bad.errors == 1 and bad.value == 0.0


def test_empty_rates_and_lifelong() -> None:
    empty = P.rate([], ["a"])
    assert empty.value is None and empty.text() == "no runs"
    half = P.rate([result("a", True), result("a", False)], ["a"])
    full = P.rate([result("a", True)], ["a"])
    assert P.lifelong(half, half, full) == (0.5 + 0.5 + 1.0) / 3
    assert P.lifelong(half, empty, full) is None


def test_tables() -> None:
    half = P.rate([result("a", True), result("a", False)], ["a", "b"])
    empty = P.rate([], ["a"])
    text = P.table([("seed on phase 1", half), ("nothing", empty)])
    assert "| seed on phase 1 | 1/2 | 50.0% | 1 | 0 |" in text and "| nothing | 0/0 | n/a | 0 | 0 |" in text
    per = P.per_task_table([("seed", half)], ["a", "b"])
    assert "| a | 1/2 |" in per and "| b | - |" in per
