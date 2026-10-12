from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path
from typing import Any

from ratchet_cases.types import Cost, Outcome, Trace

from ratchet.bench.toolsandbox.cases import ToolSandboxCase
from ratchet.bench.toolsandbox.grading import Graded, Score
from ratchet.bench.toolsandbox.metrics import LOST, NOT_RUN, RAN
from ratchet.bench.toolsandbox.report import (
    Run,
    _split_label,
    _title,
    collect,
    render,
    render_csv,
    rows_for,
)
from ratchet.bench.toolsandbox.store import RunStore

KEY_ERROR = "PermissionError: Please provide 'RAPID_API_KEY' in environment variable."


def _case(name: str, cats: tuple[str, ...], not_run: str | None = None) -> ToolSandboxCase:
    return ToolSandboxCase(f"toolsandbox-{name}", "heldout", name, name, cats, name, not_run)


def _result(case: ToolSandboxCase, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "case_id": case.id,
        "split": case.split,
        "reasons": [],
        "run_id": f"run-{case.name}",
        "error": None,
        "infra_failed": False,
    }
    return {**base, **kw}


def _write_run(root: Path, harness: str = "h1") -> Path:
    cases = [
        _case("ran", ("STATE_DEPENDENCY", "THREE_DISTRACTION_TOOLS")),
        _case("blocked", ("CANONICALIZATION",), KEY_ERROR),
        _case("dropped", ("CANONICALIZATION",)),
        _case("ungraded", ("CANONICALIZATION",)),
        _case("erred", ("INSUFFICIENT_INFORMATION",)),
    ]
    folder = root / "runs" / harness / "all-trial-1"
    store = RunStore(folder)
    store.write_cases(cases)
    graded = Graded(0.75, 0.75, 0.0, 6, (Score(0, 0.75, 5, "d", "m"),), ())
    trace = Trace("run-ran", cases[0].id, harness, "m", [], Outcome(False, ["r"], "", Cost()))
    store.put_graded(cases[0], trace, graded)
    results = [
        _result(cases[0], run_id="run-ran"),
        _result(cases[1], reasons=[f"NOT RUN: {KEY_ERROR}"], run_id="run-blocked"),
        _result(cases[2], infra_failed=True, error="ConnectError: refused", run_id=None),
        _result(cases[3], run_id="run-ungraded"),
        _result(cases[4], error="boom", run_id=None),
    ]
    payload = {
        "type": "eval",
        "harness_version": harness,
        "results": results,
        "config": {},
        "finished": "2026-10-11T18:00:00-0400",
    }
    store.put_eval(payload, eval_id="e1")
    config = {
        "split": "all",
        "trial": 1,
        "agent_model": "agent-m",
        "user_model": "user-m",
        "temperature": 0.0,
        "tool_sandbox_commit": "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1",
        "started": "2026-10-11T17:00:00-0400",
    }
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


def test_rows_join_the_results_with_the_cases_and_the_benchmarks_grades(tmp_path: Path) -> None:
    folder = _write_run(tmp_path)
    store = RunStore(folder)
    rows = {r.name: r for r in rows_for(store, store.read("evals")[0]["report"])}
    assert rows["ran"].status == RAN and rows["ran"].similarity == 0.75 and rows["ran"].turn_count == 6
    assert rows["blocked"].status == NOT_RUN and rows["blocked"].detail == f"NOT RUN: {KEY_ERROR}"
    assert rows["dropped"].status == LOST and "ConnectError" in rows["dropped"].detail
    assert rows["ungraded"].status == LOST and "no grade" in rows["ungraded"].detail
    assert rows["erred"].status == LOST and rows["erred"].detail == "boom"


def test_collect_finds_every_run_and_skips_directories_without_cases(tmp_path: Path) -> None:
    _write_run(tmp_path, "h1")
    _write_run(tmp_path, "h2")
    stray = RunStore(tmp_path / "runs" / "stray" / "x")
    stray.put_eval({"type": "eval", "results": []})
    runs = collect(tmp_path)
    assert [r.report["harness_version"] for r in runs] == ["h1", "h2"]
    assert runs[0].path == "runs/h1/all-trial-1" and runs[0].config["trial"] == 1


def test_collect_tolerates_a_run_without_a_config_and_ignores_other_records(tmp_path: Path) -> None:
    folder = _write_run(tmp_path)
    (folder / "config.json").unlink()
    RunStore(folder).put_eval({"type": "note"})
    runs = collect(tmp_path)
    assert len(runs) == 1 and runs[0].config == {}


def test_the_report_shows_denominators_not_run_reasons_and_lost_scenarios(tmp_path: Path) -> None:
    _write_run(tmp_path)
    text = render(tmp_path)
    assert text.startswith("# ToolSandbox results")
    assert "## Protocol" in text and "## Overview" in text
    assert "| `h1` | all | `agent-m` | 5 | 1 | 1 | 3 |" in text
    assert f"- NOT RUN (1 scenarios): `NOT RUN: {KEY_ERROR}`" in text
    assert "- Lost `dropped`: ConnectError: refused" in text
    assert "tool-sandbox `c8571d785431`" in text
    assert "| `ALL_CATEGORIES` | 5 | 1 | 1 | 3 | 0.750 | 0.150 |" in text
    assert "| `STATE_DEPENDENCY` | 1 | 1 | 0 | 0 | 0.750 | 0.750 |" in text
    assert "toolsandbox-scenarios.csv" in text


def test_a_long_list_of_lost_scenarios_is_cut_and_pointed_at_the_csv(tmp_path: Path) -> None:
    folder = tmp_path / "runs" / "h" / "all-trial-1"
    store = RunStore(folder)
    cases = [_case(f"c{i}", ("CANONICALIZATION",)) for i in range(25)]
    store.write_cases(cases)
    results = [_result(c, infra_failed=True, error="down", run_id=None) for c in cases]
    store.put_eval({"type": "eval", "harness_version": "h", "results": results})
    text = render(tmp_path)
    assert "... and 5 more lost scenarios, listed in the CSV." in text


def test_a_report_with_no_runs_says_so(tmp_path: Path) -> None:
    assert "No run recorded yet." in render(tmp_path)


def test_the_csv_lists_every_scenario_of_every_run(tmp_path: Path) -> None:
    _write_run(tmp_path, "h1")
    rows = list(csv.DictReader(io.StringIO(render_csv(tmp_path))))
    assert len(rows) == 5
    ran = next(r for r in rows if r["scenario"] == "ran")
    assert (ran["status"], ran["similarity"], ran["turns"]) == (RAN, "0.75", "6")
    assert ran["categories"] == "STATE_DEPENDENCY THREE_DISTRACTION_TOOLS"
    blocked = next(r for r in rows if r["scenario"] == "blocked")
    assert blocked["similarity"] == "" and blocked["detail"].startswith("NOT RUN")


def test_the_same_evaluation_under_two_directories_is_counted_once(tmp_path: Path) -> None:
    import shutil

    _write_run(tmp_path / "a")
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    runs = collect(tmp_path)
    assert len(runs) == 1 and runs[0].path.startswith("a/")


def test_the_same_shards_under_two_directories_merge_once(tmp_path: Path) -> None:
    import shutil

    for index in (0, 1):
        folder = tmp_path / "a" / "runs" / "h" / f"all-trial-1-shard-{index}-of-2"
        store = RunStore(folder)
        case = _case(f"c{index}", ("CANONICALIZATION",))
        store.write_cases([case])
        graded = Graded(1.0, 1.0, 0.0, 3, (), ())
        trace = Trace(f"run-c{index}", case.id, "h", "m", [], Outcome(True, [], "", Cost()))
        store.put_graded(case, trace, graded)
        store.put_eval(
            {"type": "eval", "harness_version": "h", "results": [_result(case, run_id=f"run-c{index}")]}
        )
        (folder / "config.json").write_text(
            json.dumps({"split": "all", "trial": 1, "shard": {"index": index, "count": 2}}), encoding="utf-8"
        )
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    runs = collect(tmp_path)
    assert len(runs) == 1
    assert runs[0].path.endswith("(shards 0,1 of 2)")
    assert len(runs[0].rows) == 2 and all(r.status == RAN for r in runs[0].rows)


def test_runs_that_never_finished_are_listed_with_their_reason(tmp_path: Path) -> None:
    _write_run(tmp_path)
    cut = tmp_path / "runs" / "h1" / "all-trial-3"
    cut.mkdir(parents=True)
    (cut / "config.json").write_text(json.dumps({"split": "all", "trial": 3}), encoding="utf-8")
    (cut / "traces.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    evolve = tmp_path / "evolve" / "e1"
    evolve.mkdir(parents=True)
    (evolve / "config.json").write_text(json.dumps({"generations": 1}), encoding="utf-8")
    (evolve / "STOPPED.txt").write_text("killed with the session\n", encoding="utf-8")
    text = render(tmp_path)
    assert "## Runs stopped before they finished" in text
    lines = {line.split("`")[1]: line for line in text.splitlines() if line.startswith("- `")}
    assert lines["evolve/e1"].startswith("- `evolve/e1`: 0 conversations recorded")
    assert lines["evolve/e1"].endswith("killed with the session")
    assert lines["runs/h1/all-trial-3"].startswith("- `runs/h1/all-trial-3`: 2 conversations recorded")
    assert lines["runs/h1/all-trial-3"].endswith("no stop reason recorded")
    assert "runs/h1/all-trial-1`:" not in text
    assert "### Evolve run `evolve/e1`\n\n- Models not recorded.\n- Stopped (`STOPPED.txt`): killed" in text
    (evolve / "result.json").write_text("{}", encoding="utf-8")
    assert "- `evolve/e1`" not in render(tmp_path)
    assert "- Stopped (`STOPPED.txt`): killed with the session" in render(tmp_path)
    (evolve / "STOPPED.txt").unlink()
    assert "- Finished: 0 of 1 planned generations ran." in render(tmp_path)
    shutil.rmtree(tmp_path / "runs")
    assert "No run recorded yet." in render(tmp_path) and "- `evolve/e1`" not in render(tmp_path)
    assert "### Evolve run `evolve/e1`" in render(tmp_path)


def test_an_evolve_run_shows_its_parent_scores_and_its_decision(tmp_path: Path) -> None:
    folder = _write_run(tmp_path)
    evolve = tmp_path / "box" / "evolve" / "smoke-2"
    evolve.mkdir(parents=True)
    config = {"agent_model": "llama", "user_model": "luna", "improver_model": "llama", "generations": 1}
    (evolve / "config.json").write_text(json.dumps(config), encoding="utf-8")
    store = RunStore(evolve)
    parent = {"type": "eval", "harness_version": "h1", "role": "parent_run_1", "generation": 1,
              "heldout": {"passed": 6, "total": 24}, "train": {"passed": 5, "total": 24},
              "results": []}  # fmt: skip
    candidate = {**parent, "harness_version": "h2", "role": "candidate"}
    candidate["heldout"] = {"passed": 10, "total": 24}
    store.put_eval(parent, eval_id="e1")
    store.put_eval(candidate, eval_id="e2")
    checks = [{"name": "heldout", "passed": False}, {"name": "no_regression", "passed": False}]
    decision = {"admitted": False, "candidate": "h2", "checks": checks, "context": {"kind": "procedure"},
                "keyid": "b61d7e39c111ffff"}  # fmt: skip
    store.put_decision(decision, decision_id="h1->h2")
    gen = {"generation": 1, "level1": {"candidates": 6, "evaluated": 1, "admitted": 0}}
    (evolve / "result.json").write_text(json.dumps({"generations": [gen]}), encoding="utf-8")
    text = render(tmp_path)
    section = text[text.index("## Evolve runs") :]
    assert "### Evolve run `box/evolve/smoke-2`" in section
    assert "- Generation 1: 6 candidates proposed, 1 evaluated, 0 admitted." in section
    assert "- 1 decision, 0 admitted, signed with key `b61d7e39c111`." in section
    assert "| 1 | `h1` | 1 | 6/24 | 5/24 | 0 of 0 |" in section
    assert "| 1 | `h2` | procedure | 10/24 | 5/24 | 0 of 0 | rejected | heldout, no_regression |" in section
    assert "| `h1` | all | `agent-m` |" in text and str(folder.relative_to(tmp_path)) in text
    run = Run("box/evolve/smoke-2", config, candidate, [])
    assert _title(run) == "`h2`, evolve candidate, generation 1" and _split_label(run) == "candidate"
    trial = Run(
        "runs/h1/all-trial-1", {"split": "all", "trial": 1, "limit": 10}, {"harness_version": "h1"}, []
    )
    assert _title(trial) == "`h1` on all (trial 1, seeded sample of 10)" and _split_label(trial) == "all"
