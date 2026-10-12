from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ratchet.bench.evolve_report import evolve_stores, render_run, render_runs, status, unique_stores


def _lines(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _result(case: str, split: str, passed: bool, run_id: str | None, lost: bool = False) -> dict[str, Any]:
    return {"case_id": case, "split": split, "passed": passed, "run_id": run_id, "infra_failed": lost}


def _eval(harness: str, role: str, gen: int, results: list[dict[str, Any]]) -> dict[str, Any]:
    count = {
        s: {
            "passed": sum(r["passed"] for r in results if r["split"] == s),
            "total": sum(r["split"] == s for r in results),
        }
        for s in ("heldout", "train")
    }
    return {"type": "eval", "harness_version": harness, "role": role, "generation": gen, **count,
            "results": results}  # fmt: skip


def _decision(candidate: str, failed: list[str], kind: str = "procedure") -> dict[str, Any]:
    checks = [{"name": n, "passed": n not in failed} for n in ("replay", "heldout", "no_regression")]
    return {
        "decision_id": f"p->{candidate}",
        "decision": {"admitted": not failed, "candidate": candidate, "checks": checks,
                     "context": {"kind": kind}, "keyid": "b61d7e39c111" + "0" * 52},
    }  # fmt: skip


def _trace(run_id: str, case: str, harness: str, passed: bool) -> dict[str, Any]:
    return {"run_id": run_id, "case_id": case, "harness_version": harness, "outcome": {"passed": passed}}


def _store(folder: Path, **config: Any) -> Path:
    folder.mkdir(parents=True)
    base = {"agent_model": "a", "user_model": "u", "improver_model": "i", "generations": 2}
    (folder / "config.json").write_text(json.dumps({**base, **config}), encoding="utf-8")
    return folder


def _finished(folder: Path) -> Path:
    _store(folder, started="2026-10-12T03:50:57+0000")
    p1 = [_result("h1", "heldout", False, "r1", lost=True), _result("h2", "heldout", True, "r2"),
          _result("t1", "train", True, "r3")]  # fmt: skip
    p2 = [_result("h1", "heldout", True, "r4"), _result("h2", "heldout", True, "r5")]
    c1 = [_result("h1", "heldout", True, "r6"), _result("h2", "heldout", False, "r7"),
          _result("t1", "train", True, "r8")]  # fmt: skip
    _lines(
        folder / "evals.jsonl",
        [{"eval_id": "x", "report": {"type": "evaluator"}},
         {"eval_id": "e1", "report": _eval("p", "parent_run_1", 1, p1)},
         {"eval_id": "e2", "report": _eval("p", "parent_run_2", 1, p2)},
         {"eval_id": "e3", "report": _eval("c1", "candidate", 1, c1)}],
    )  # fmt: skip
    _lines(folder / "decisions.jsonl", [_decision("c1", ["heldout", "no_regression"]),
                                        _decision("c9", ["replay"], kind="guard")])  # fmt: skip
    _lines(folder / "traces.jsonl", [_trace("retried", "h1", "c1", False)])
    gen = {"generation": 1, "parent": "p", "level1": {"candidates": 4, "evaluated": 1, "admitted": 0},
           "note": "improver calls failed for 1 task"}  # fmt: skip
    (folder / "result.json").write_text(json.dumps({"generations": [gen]}), encoding="utf-8")
    return folder


def test_a_finished_run_lists_parents_candidates_and_signed_decisions(tmp_path: Path) -> None:
    folder = _finished(tmp_path / "box" / "evolve" / "run-3")
    text = "\n".join(render_run(folder, "box/evolve/run-3"))
    assert text.startswith("### Evolve run `box/evolve/run-3`\n\n- agent model `a`, user model `u`, improver")
    assert "- Finished: 1 of 2 planned generations ran." in text
    assert "- Generation 1: 4 candidates proposed, 1 evaluated, 0 admitted. Note: improver calls" in text
    assert "- 2 decisions, 0 admitted, signed with key `b61d7e39c111`." in text
    assert "| 1 | `p` | 1 | 1/2 | 1/1 | 1 of 3 |" in text
    assert "| 1 | `p` | 2 | 2/2 | not run | 0 of 2 |" in text
    assert "| 1 | `c1` | procedure | 1/2 | 1/1 | 0 of 3 | rejected | heldout, no_regression |" in text
    assert "| | `c9` | | not recorded | not recorded | | rejected | replay |" in text
    assert "retried" not in text and "no finished evaluation counts" not in text


def test_a_run_that_did_not_finish_shows_its_reason_and_its_uncounted_conversations(tmp_path: Path) -> None:
    folder = _store(tmp_path / "evolve" / "run-1", split_record={"heldout": ["h1"], "train": ["t1"]})
    _lines(folder / "harnesses.jsonl", [{"version": "p", "parent": None}])
    _lines(
        folder / "evals.jsonl",
        [{"report": _eval("p", "parent_run_1", 1, [_result("h1", "heldout", True, "r1")])}],
    )
    _lines(folder / "traces.jsonl", [_trace("r1", "h1", "p", True), _trace("r2", "h1", "c", True),
                                     _trace("r3", "t1", "c", False), _trace("r3", "t1", "c", False),
                                     _trace("r4", "h1", "p", False)])  # fmt: skip
    text = "\n".join(render_run(folder, "evolve/run-1"))
    assert "- Not finished, and no stop reason is recorded." in text
    assert "- No decision was made." in text and "| | none | | | | | | |" in text
    uncounted = "recorded conversations that no finished evaluation counts"
    assert f"- Candidate `c`: 2 {uncounted} (held-out passed 1/1, train passed 0/1). No decision" in text
    assert f"- Parent `p`: 1 {uncounted} (held-out passed 0/1" in text
    (folder / "config.json").write_text(
        json.dumps({"generations": 1, "stopped": "cut off"}), encoding="utf-8"
    )
    assert status(folder) == "Stopped: cut off"
    assert "- Models not recorded." in render_run(folder, "x")
    (folder / "STOPPED.txt").write_text("killed with\nthe session\n", encoding="utf-8")
    assert status(folder) == "Stopped (`STOPPED.txt`): killed with the session"
    (folder / "FAILED.txt").write_text("HTTP 401 on every improver call\n", encoding="utf-8")
    assert status(folder) == "Failed (`FAILED.txt`): HTTP 401 on every improver call"
    (folder / "NOTE.txt").write_text("40 of 50 lost\n", encoding="utf-8")
    assert "- Note (`NOTE.txt`): 40 of 50 lost" in render_run(folder, "x")


def test_stores_are_found_once_in_start_order_and_filtered_by_domain(tmp_path: Path) -> None:
    late = _store(tmp_path / "evolve" / "a-run", domain="retail", started="2026-10-12T05:32:10+0000")
    early = _store(tmp_path / "box" / "z" / "b-run", domain="retail", started="2026-10-11T17:18:59-0400")
    other = _store(tmp_path / "evolve" / "c-run", domain="airline")
    _store(tmp_path / "runs" / "trial")
    (tmp_path / "runs" / "trial" / "config.json").write_text(json.dumps({"split": "all"}), encoding="utf-8")
    assert evolve_stores(tmp_path) == [early, late, other]
    assert evolve_stores(tmp_path, "retail") == [early, late]
    fuller = tmp_path / "copy" / "a-run"
    fuller.mkdir(parents=True)
    (fuller / "config.json").write_bytes((late / "config.json").read_bytes())
    _lines(fuller / "evals.jsonl", [{"report": {"type": "evaluator"}}])
    assert evolve_stores(tmp_path, "retail") == [early, fuller]
    assert unique_stores([late, fuller, late]) == [fuller]
    text = "\n".join(render_runs(tmp_path, [early]))
    assert text.startswith("### Evolve run `box/z/b-run`")
