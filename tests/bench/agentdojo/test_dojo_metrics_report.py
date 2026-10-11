from __future__ import annotations

from typing import Any

from ratchet.bench.agentdojo.cases import DojoSuite
from ratchet.bench.agentdojo.metrics import Rate, metrics, metrics_from_rows
from ratchet.bench.agentdojo.report import document, overview_row, render
from ratchet.bench.agentdojo.world import ATTACK_SUCCEEDED, UTILITY_MISSED
from ratchet.improve.evaluate import CaseResult, EvalReport
from ratchet.types import Cost


def _row(cid: str, split: str, reasons: list[str], error: str | None = None) -> dict[str, Any]:
    return {"case_id": cid, "split": split, "reasons": reasons, "passed": not reasons and not error,
            "error": error}  # fmt: skip


def test_rates_follow_agentdojo_definitions(banking: DojoSuite) -> None:
    held = banking.by_split("heldout")
    clean = [c.id for c in held if c.kind == "must_succeed"]
    attacked = [c.id for c in held if c.kind == "trap"]
    rows = [
        _row(clean[0], "heldout", []),
        _row(clean[1], "heldout", [UTILITY_MISSED]),
        _row(attacked[0], "heldout", []),
        _row(attacked[1], "heldout", [ATTACK_SUCCEEDED]),
        _row(attacked[2], "heldout", [UTILITY_MISSED, ATTACK_SUCCEEDED]),
        _row(attacked[3], "heldout", [], error="boom"),
        _row(attacked[5], "heldout", [UTILITY_MISSED, "incomplete: model error: HTTP 429"]),
        _row(attacked[4], "train", []),
    ]
    m = metrics_from_rows(rows, banking, "heldout")
    assert (m.utility.hits, m.utility.total, m.utility.errors) == (1, 2, 0)
    ua = m.utility_under_attack
    assert (ua.hits, ua.total, ua.errors) == (2, 3, 2)
    assert (m.attack_success.hits, m.attack_success.total) == (2, 3)
    assert (m.ratchet_passed.hits, m.ratchet_passed.total, m.ratchet_passed.errors) == (2, 5, 2)
    payload = m.to_payload()
    assert payload["targeted_attack_success"]["rate"] == 0.6667
    assert payload["split"] == "heldout"


def test_metrics_read_an_eval_report(banking: DojoSuite) -> None:
    c = banking.by_split("train")[0]
    r = CaseResult(c.id, "train", True, c.kind, True, [], "run-1", Cost())
    report = EvalReport("v", [r], 1.0)
    assert metrics(report, banking, "train").ratchet_passed == Rate(1, 1, 0)


def test_rate_text_and_empty_rate() -> None:
    assert Rate(1, 4, 0).text() == "25.0% (1/4)"
    assert Rate(0, 0, 2).text() == "n/a (0/0, 2 error(s) excluded)"
    assert Rate(0, 0, 0).to_payload()["rate"] is None


def _m(hits: int, total: int, errors: int = 0) -> dict[str, Any]:
    return Rate(hits, total, errors).to_payload()


def test_render_lists_versions_metrics_and_admitted_guards() -> None:
    split = {"heldout": _m(8, 8), "utility": _m(8, 8)}
    block = {"utility": _m(8, 8), "utility_under_attack": _m(29, 32, 1), "targeted_attack_success": _m(0, 32),
             "ratchet_passed": _m(37, 40)}  # fmt: skip
    results = {
        "run_id": "run-x",
        "provenance": {
            "suite": "banking",
            "benchmark_version": "v1.2.2",
            "model": "m",
            "improver_model": "i",
            "endpoint": "e",
            "agentdojo_package_version": "0.1.35",
            "agentdojo_reference_clone_commit": "abc",
            "ratchet_commit": "def",
            "seed": 0,
            "attack": "a",
            "split": {"heldout_users": ["user_task_1"], "heldout_injections": ["injection_task_4"]},
            "cases": {"train": 6, "heldout": 5},
            **split,
        },
        "versions": ["v0", "v1", "v2"],
        "metrics": [
            {
                "version": "v0",
                "eval_id": "eval-0001",
                "role": "baseline",
                "generation": None,
                "heldout": block,
            },
            {"version": "v1", "eval_id": "eval-0009", "role": None, "generation": 1, "heldout": block},
            {
                "version": "v1",
                "eval_id": "eval-0010",
                "role": "parent_run_1",
                "generation": 2,
                "train": block,
            },
        ],
        "admitted": {
            "v1": {
                "parent": "v0",
                "guards": [{"id": "g1", "description": "first guard"}],
                "procedures": [{"name": "p", "when": "w"}],
            },
            "v2": {
                "parent": "v1",
                "guards": [{"id": "g1", "description": "first guard"}],
                "procedures": [{"name": "p", "when": "w"}],
            },
        },
        "notes": ["a note"],
    }
    text = render(results)
    assert (
        "| `v0` | eval-0001 | baseline |  | 100.0% (8/8) | 90.6% (29/32, 1 err) | 0.0% (0/32) "
        "| 92.5% (37/40) |" in text
    )
    assert "| `v1` | eval-0009 |  | 1 |" in text and "eval-0010" not in text
    assert "| `v2` | not evaluated |" in text
    assert text.count("guard `g1`: first guard") == 1 and text.count("procedure `p` when w") == 1
    assert "reference clone commit `abc`" in text and "- a note" in text
    results["versions"] = ["v0"]
    results["provenance"]["agentdojo_reference_clone_commit"] = None
    text = render(results)
    assert "No candidate was admitted" in text and "reference clone" not in text


def test_overview_names_the_newest_version_and_its_latest_heldout_run() -> None:
    block = {"utility": _m(1, 2), "utility_under_attack": _m(3, 4), "targeted_attack_success": _m(1, 4),
             "ratchet_passed": _m(3, 6)}  # fmt: skip
    results = {
        "provenance": {"suite": "slack", "model": "m"},
        "status": "in progress",
        "versions": ["v0", "v1"],
        "generations": [{}],
        "metrics": [
            {"version": "v0", "eval_id": "e1", "role": "baseline", "heldout": block},
            {"version": "v1", "eval_id": "e7", "role": "candidate", "heldout": block},
            {"version": "v1", "eval_id": "e9", "role": "confirm", "heldout": block},
        ],
    }
    row = overview_row(results)
    assert row.startswith("| slack | `m` | in progress, 1 gen | 50.0% (1/2) | 75.0% (3/4) | 25.0% (1/4) ")
    assert "`v1` (e9)" in row
    results["versions"] = ["v0"]
    assert "none promoted | | | |" in overview_row(results)
    results["status"] = "stopped: a long reason"
    assert "| stopped, 1 gen |" in overview_row(results)
    results["status"] = "in progress"
    results["metrics"] = []
    assert overview_row(results).startswith("| slack | `m` | in progress, 1 gen | | | | none promoted")
    assert document([], "pre").startswith("# AgentDojo results\n\npre\n")
