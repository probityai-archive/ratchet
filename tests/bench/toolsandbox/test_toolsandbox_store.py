from __future__ import annotations

from pathlib import Path

from ratchet_cases.types import Cost, Outcome, Trace

from ratchet.bench.toolsandbox.cases import ToolSandboxCase
from ratchet.bench.toolsandbox.grading import Graded, Score
from ratchet.bench.toolsandbox.store import RunStore


def _case() -> ToolSandboxCase:
    return ToolSandboxCase("toolsandbox-a", "train", "a", "a", ("STATE_DEPENDENCY",), "a")


def test_cases_round_trip_by_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.write_cases([_case()])
    cases = store.read_cases()
    assert list(cases) == ["toolsandbox-a"]
    assert cases["toolsandbox-a"]["categories"] == ["STATE_DEPENDENCY"]
    assert cases["toolsandbox-a"]["not_run"] is None


def test_the_benchmarks_grade_is_kept_by_run_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    graded = Graded(0.5, 0.5, 0.0, 3, (Score(0, 0.5, 4, "d", "m"),), ())
    trace = Trace("run-1", "toolsandbox-a", "h", "model", [], Outcome(False, ["r"], "", Cost()))
    store.put_graded(_case(), trace, graded)
    by_run = store.graded_by_run()
    assert list(by_run) == ["run-1"]
    assert by_run["run-1"]["graded"]["similarity"] == 0.5
    assert by_run["run-1"]["graded"]["milestones"][0]["snapshot_index"] == 4
    assert (by_run["run-1"]["case_id"], by_run["run-1"]["passed"]) == ("toolsandbox-a", False)


def test_a_fresh_store_has_no_grades(tmp_path: Path) -> None:
    assert RunStore(tmp_path).graded_by_run() == {}
