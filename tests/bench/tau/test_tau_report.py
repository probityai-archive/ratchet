from __future__ import annotations

import json
from pathlib import Path

import pytest

import ratchet.bench.tau.__main__ as cli
from ratchet.bench.tau.report import collect, render
from ratchet.bench.tau.store import JsonlStore


def _eval(harness: str, rows: list[tuple[str, str, bool]], **extra: object) -> dict[str, object]:
    return {
        "type": "eval",
        "harness_version": harness,
        "heldout": {"passed": sum(p for _, s, p in rows if s == "heldout"), "total": 2},
        "train": {"passed": 0, "total": 0},
        "errors": 0,
        "results": [{"case_id": c, "split": s, "passed": p, "error": None} for c, s, p in rows],
        **extra,
    }


def _config(path: Path, **fields: object) -> None:
    path.mkdir(parents=True, exist_ok=True)
    base = {
        "domain": "retail",
        "agent_model": "a",
        "split_record": {"seed": 1, "train": [], "heldout": ["x"]},
    }
    (path / "config.json").write_text(json.dumps({**base, **fields}))


def test_report_counts_every_run_and_lists_stopped_ones(tmp_path: Path) -> None:
    trial = tmp_path / "runs" / "retail" / "h1" / "heldout-trial-1"
    _config(trial, note="first trial")
    JsonlStore(trial).put_eval(_eval("h1", [("x", "heldout", True), ("y", "heldout", False)]))
    evo = tmp_path / "evolve" / "retail" / "run-1"
    _config(evo, improver_model="i", stopped="cut off")
    store = JsonlStore(evo)
    store.put_eval({"type": "evaluator"})
    store.put_eval(
        _eval("h1", [("x", "heldout", True), ("y", "heldout", True)], role="parent_run_2", generation=1)
    )
    store.put_decision(
        {
            "admitted": False,
            "checks": [{"name": "heldout", "passed": False}, {"name": "replay", "passed": True}],
        },
        decision_id="h1->h2",
    )
    gen = {
        "generation": 1,
        "parent": "h1",
        "clusters": [{}],
        "level1": {"candidates": 3, "admitted": 0},
        "note": "n",
    }
    (evo / "result.json").write_text(
        json.dumps({"final_harness": "h1", "active": ["h1", 1], "generations": [gen]})
    )
    _config(tmp_path / "runs" / "retail" / "h1" / "all-trial-9", stopped="rate limits")
    from ratchet.types import Cost, Outcome, Trace

    for rid, cid, ok in [("r1", "x", True), ("r1", "x", True), ("r2", "z", False)]:
        store.put_trace(Trace(rid, cid, "h2", "m", [], Outcome(ok, [], "d", Cost())))
    store.put_trace(Trace("r3", "x", "h1", "m", [], Outcome(True, [], "d", Cost())))
    store.put_eval(_eval("h1", [("w", "train", False)], role="parent_run_1", generation=1))
    _config(tmp_path / "runs" / "airline" / "h9" / "all-trial-1", domain="airline")

    assert {r["source"] for r in collect(tmp_path)} == {
        "runs/retail/h1/heldout-trial-1",
        "evolve/retail/run-1",
    }
    text = render(tmp_path, "retail")
    assert "| `h1` | heldout | 2 | 2 | 4 | 3 | 0 | 75.0% | 50.0% |" in text
    assert "| `h1` | train | 1 | 1 | 1 | 0 | 0 | 0.0% |" in text
    assert "`runs/retail/h1/all-trial-9`: rate limits" in text
    assert "| run-1 | `h1->h2` | False | heldout |" in text
    assert "parent_run_2" in text and "note: n" in text and "airline" not in text
    assert "Note: first trial" in text
    assert "Stopped: cut off" in text
    assert "| `evolve/retail/run-1` | `h2` | 1/1 | 0/1 |" in text
    assert "`h1` | 1/1" not in text


def test_report_command_writes_the_page(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "page.md"
    (tmp_path / "notes-airline.md").write_text("## Status\n\nhand notes\n")
    assert cli.main(["--out", str(tmp_path), "report", "--domain", "airline", "--write", str(out)]) == 0
    assert out.read_text().startswith("# tau-bench airline")
    assert "## Status\n\nhand notes" in out.read_text()
    assert cli.main(["--out", str(tmp_path), "report"]) == 0
    assert capsys.readouterr().out.startswith("# tau-bench retail")
