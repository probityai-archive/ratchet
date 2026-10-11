from __future__ import annotations

import io
from typing import Any

import pytest
from rich.console import Console

from ratchet.types import Event, Trace
from ratchet.view.data import decision_info
from ratchet.view.show import clip, decision_view, event_summary, evidence_line, run_view, trace_view

from .fakes import decision_doc, make_trace, scenario


def render(obj: object) -> str:
    c = Console(width=140, record=True, file=io.StringIO())
    c.print(obj)
    return c.export_text()


def ev(kind: str, **payload: Any) -> Event:
    return Event(1, kind, payload, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (ev("model", content="thinking", tool_calls=[{"name": "refund"}, "raw"]), "thinking  -> refund, raw"),
        (ev("model", content="", tool_calls=[]), ""),
        (
            ev("tool_call", call={"name": "refund", "args": {"charge": "ch1", "n": 2}}),
            "refund(charge=ch1, n=2)",
        ),
        (ev("tool_result", result={"ok": True, "data": {"a": 1}}), 'ok {"a": 1}'),
        (ev("tool_result", result={"ok": False, "error": "timeout"}), "ERROR timeout"),
        (ev("guard_block", source="guard", message="no  retry"), "[guard] no retry"),
        (ev("procedure", name="check", how="keyword"), "check (keyword)"),
        (ev("checkpoint", executed=3), "after 3 world call(s)"),
        (ev("final", reason="answered", content="done"), "answered: done"),
    ],
)
def test_event_summary(event: Event, expected: str) -> None:
    assert event_summary(event) == expected


def test_clip() -> None:
    assert clip("a   b", 10) == "a b"
    assert clip("x" * 20, 5) == "xxxx…"


def test_trace_view_lists_events_outcome_and_reasons() -> None:
    t = make_trace(
        "run-1", "refund.t1", False, [("refund", {"charge": "ch1"})], reasons=["charge refunded twice"]
    )
    out = render(trace_view(t))
    assert "FAIL" in out
    assert "refund.t1" in out
    assert "refund(charge=ch1)" in out
    assert "charge refunded twice" in out
    assert "3 model calls" in out
    ok = render(trace_view(make_trace("run-2", "x", True, [])))
    assert "PASS" in ok
    assert "grader:" not in ok


def test_trace_view_accepts_a_trace_with_every_event_kind() -> None:
    events = [
        ev("model", content="hi"),
        ev("procedure", name="n", how="selected"),
        ev("checkpoint", executed=1),
    ]
    t = make_trace("r", "c", True, [])
    t = Trace(t.run_id, t.case_id, t.harness_version, t.model, events, t.outcome)
    assert "selected" in render(trace_view(t))


def test_decision_view_admitted_and_rejected() -> None:
    store, _, _ = scenario()
    admitted = render(decision_view(decision_info(store.decision_docs[0])))
    assert "ADMITTED" in admitted
    assert "why rejected" not in admitted
    assert "ratchet verify" in admitted
    rejected = render(decision_view(decision_info(store.decision_docs[1])))
    assert "REJECTED" in rejected
    assert "why rejected" in rejected
    assert "FAIL" in rejected
    assert "block every refund" in rejected


def test_evidence_lines_for_each_gate() -> None:
    assert "fires on 1/2 target failures, 3 false fire(s)" in evidence_line(
        {
            "name": "replay",
            "evidence": {"reports": [{"targets_hit": 1, "targets_total": 2, "false_positives": [1, 2, 3]}]},
        }
    )
    assert "confirming run pending" in evidence_line(
        {"name": "heldout", "evidence": {"confirm_pending": True, "required": 5}}
    )
    assert "confirming run 4" in evidence_line({"name": "heldout", "evidence": {"confirm_fresh_passed": 4}})
    assert "regressed 1" in evidence_line({"name": "no_regression", "evidence": {"regressed": ["a"]}})
    assert "2 intervention(s)" in evidence_line({"name": "benign_twins", "evidence": {"interventions": 2}})
    assert "1 copied literal(s)" in evidence_line(
        {"name": "overfit_lint", "evidence": {"hits": [{}], "screened": {}}}
    )
    assert "authority grew: tools_granted" in evidence_line(
        {"name": "authority", "evidence": {"tools_granted": ["x"]}}
    )
    assert "subset" in evidence_line({"name": "authority", "evidence": {}})
    assert "model calls 3 -> 4" in evidence_line(
        {"name": "cost", "evidence": {"parent": {"model_calls": 3}, "candidate": {"model_calls": 4}}}
    )
    assert evidence_line({"name": "other"}) == ""


def test_run_view_summarises_cases() -> None:
    payload = {
        "harness_version": "abc",
        "train": {"passed": 1, "total": 2},
        "heldout": {"passed": 0, "total": 1},
        "wall_s": 1.5,
        "results": [
            {"case_id": "a", "split": "train", "kind": "trap", "passed": True, "interventions": 1},
            {
                "case_id": "b",
                "split": "train",
                "kind": "benign_twin",
                "passed": False,
                "interventions": 2,
                "reasons": ["refunded twice"],
            },
            {"case_id": "c", "split": "heldout", "kind": "trap", "passed": False, "reasons": []},
        ],
    }
    out = render(run_view(payload))
    assert "train 1/2" in out
    assert "held-out 0/1" in out
    assert "false interventions 2" in out
    assert "refunded twice" in out
    clean = render(run_view({**payload, "results": [payload["results"][0]]}))  # type: ignore[index]
    assert "false interventions 0" in clean


def test_decision_view_of_unsigned_decision() -> None:
    d = decision_info(decision_doc("p", "c", True, envelope=None))
    assert "receipt none" in render(decision_view(d))
