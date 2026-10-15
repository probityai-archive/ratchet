from __future__ import annotations

import io
from typing import Any

import pytest
from ratchet_cases.types import Event, Trace
from rich.console import Console

from ratchet.view.data import decision_info
from ratchet.view.show import (
    clip,
    decision_view,
    event_summary,
    evidence_line,
    run_view,
    trace_label,
    trace_view,
)

from .fakes import STABLE, decision_doc, fresh_evidence, make_trace, replay_report, scenario


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
    reports = [replay_report("g1"), replay_report("g2", "INCOMPLETE", "3/9 traces checked")]
    composed = replay_report("x", "REJECTED", "9/9 traces checked")
    replay = {"name": "replay", "evidence": {"reports": reports, "composed": composed}}
    line = evidence_line(replay)
    assert "g1: 11/11 checked, CLEARED" in line
    assert "g2: 3/9 checked, INCOMPLETE" in line
    assert "(all guards composed): 9/9 checked, REJECTED" in line
    assert "2 stable recoveries (need 2), 2 protected, 0 broken, parent discordance D=1" in evidence_line(
        {"name": "fresh", "evidence": fresh_evidence(STABLE)}
    )
    assert "C2 pending" in evidence_line(
        {
            "name": "fresh",
            "evidence": fresh_evidence(
                {"a": ["fail", "fail", "pass"], "b": ["fail", "fail", "pass"]}, pending=True
            ),
        }
    )
    assert evidence_line({"name": "fresh", "evidence": {}}) == ""
    assert "regressed 1" in evidence_line({"name": "no_regression", "evidence": {"regressed": ["a"]}})
    assert "2 intervention(s)" in evidence_line({"name": "benign_twins", "evidence": {"interventions": 2}})
    twins = {"interventions": 5, "intervened": ["a", "b", "c", "d"], "benign_twin_runs": 4}
    assert evidence_line({"name": "benign_twins", "evidence": twins}) == (
        "5 intervention(s) on 4 benign twins over 4 twin runs"
    )
    clean = {"interventions": 0, "intervened": [], "benign_twin_runs": 6}
    assert evidence_line({"name": "benign_twins", "evidence": clean}) == "0 intervention(s) over 6 twin runs"
    refused = fresh_evidence({"a": ["pass", "pass", "fail"], "b": ["fail", "fail", "fail"]}, pending=True)
    assert "0 stable recoveries (need 2), 1 protected, 1 broken" in evidence_line(
        {"name": "heldout", "evidence": refused}
    )
    assert evidence_line({"name": "heldout", "evidence": refused}).endswith("; C2 not run (refused after C1)")
    assert "1 copied literal(s)" in evidence_line(
        {"name": "overfit_lint", "evidence": {"hits": [{}], "screened": {}}}
    )
    assert "breaks: tools_granted" in evidence_line(
        {"name": "structure", "evidence": {"violations": ["tools_granted"], "added_guards": 1}}
    )
    assert "additive only" in evidence_line({"name": "structure", "evidence": {}})
    assert "agent turns 3 -> 4" in evidence_line(
        {"name": "cost", "evidence": {"parent": {"model_calls": 3}, "candidate": {"model_calls": 4}}}
    )
    assert evidence_line({"name": "other"}) == ""


def test_decision_view_renders_the_four_runs_and_the_sets_not_a_net_count() -> None:
    store, _, _ = scenario()
    out = render(decision_view(decision_info(store.decision_docs[1])))
    for part in ("P1", "P2", "C1", "C2", "held-out case"):
        assert part in out
    assert "inj.v3" in out
    assert "STABLE RECOVERY" in out
    assert "PROTECTED CASE BROKE" in out
    assert "ord.v3" in out
    assert "still failing" in out
    assert "stable recoveries 1 (need 2)" in out
    assert "broken 1" in out
    assert "parent discordance D=1" in out
    assert "(reported, not a threshold)" in out
    ok = render(decision_view(decision_info(store.decision_docs[0])))
    assert "stable recoveries 2 (need 2)" in ok
    assert "protected 2" in ok
    assert "broken 0" in ok


def test_decision_view_marks_a_pending_confirmation_and_invalid_evidence() -> None:
    ev = {**fresh_evidence({"a": ["fail", "fail", "pass"], "b": ["pass", "pass", "pass"]}, pending=True)}
    ev["invalid_evidence"] = ["b"]
    ev["same_manifest"] = False
    doc = decision_doc("p", "c", False, checks=[{"name": "fresh", "passed": False, "evidence": ev}])
    out = render(decision_view(decision_info(doc)))
    assert "pending" in out
    assert "C2 (the confirming run) has not run yet" in out
    assert "lack valid FRESH evidence" in out
    assert "did not share one case manifest" in out
    assert "NO VALID EVIDENCE" in out


def test_decision_view_renders_replay_coverage_per_guard() -> None:
    reports = [
        replay_report("g1"),
        replay_report("g2", "REJECTED", false_fires=["a@1"], benign_twin_fires=1, errors=0),
    ]
    ev = {"reports": reports, "composed": replay_report("c", "INCOMPLETE", "5/9 traces checked", errors=1),
          "procedures": {"p-1": "NOT APPLICABLE (natural-language trigger)"}}  # fmt: skip
    doc = decision_doc("p", "c", False, checks=[{"name": "replay", "passed": False, "evidence": ev}])
    out = render(decision_view(decision_info(doc)))
    assert "11/11 checked" in out
    assert "CLEARED" in out
    assert "REJECTED" in out
    assert "INCOMPLETE" in out
    assert "5/9 checked" in out
    assert "(all guards composed)" in out
    assert "procedure p-1: replay NOT APPLICABLE" in out


def test_traces_and_runs_are_labelled_fresh_recorded_or_cached() -> None:
    t = make_trace("run-1", "refund.t1", True, [])
    assert trace_label(t).startswith("RECORDED 1970-01-01T")
    assert trace_label(t).endswith("(run was FRESH)")
    cached = Trace(t.run_id, t.case_id, t.harness_version, t.model, [ev("model", cached=True)], t.outcome)
    assert trace_label(cached).endswith("(run was CACHED)")
    empty = Trace(t.run_id, t.case_id, t.harness_version, t.model, [], t.outcome)
    assert "unknown" in trace_label(empty)
    assert "RECORDED 1970" in render(trace_view(t))


def test_run_view_summarises_cases() -> None:
    payload: dict[str, Any] = {
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
    assert "FRESH" in out
    cached = render(run_view({**payload, "results": [{**payload["results"][0], "mode": "CACHED"}]}))
    assert "CACHED" in cached
    clean = render(run_view({**payload, "results": [payload["results"][0]]}))
    assert "false interventions 0" in clean


def test_decision_view_of_unsigned_decision() -> None:
    d = decision_info(decision_doc("p", "c", True, envelope=None))
    assert "receipt none" in render(decision_view(d))
