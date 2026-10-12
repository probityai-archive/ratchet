from __future__ import annotations

import io
from typing import Any

from ratchet_cases.types import Event
from rich.console import Console

from ratchet.view import resume

from .fakes import make_trace


def render(obj: object) -> str:
    c = Console(width=150, record=True, file=io.StringIO())
    c.print(obj)
    return " ".join(c.export_text().replace("│", " ").split())


ALL = {"w1": True, "w2": True, "w3": True}


def saved(**over: Any) -> dict[str, Any]:
    ok = {"result": {"call_id": "r1", "ok": True, "data": {}}}
    lost = {"result": {"call_id": "w1", "ok": False, "error": "timeout"}}
    bad = {"result": {"call_id": "w2", "ok": False, "error": "card declined"}}
    base = {
        "run_id": "run-1", "case_id": "c", "harness_version": "h",
        "events": [{"kind": "tool_result", "payload": p} for p in (ok, lost, bad)],
        "executed": [
            {"name": "tickets_get", "args": {"ticket_id": "T"}, "call_id": "r1"},
            {"name": "payments_refund", "args": {"charge_id": "ch1", "amount_cents": 5}, "call_id": "w1"},
            {"name": "orders_cancel", "args": {"order_id": "o1"}, "call_id": "w2"},
            {"name": "mail_send", "args": {"to": "x"}, "call_id": "w3"},
        ],
        "pending": [],
    }  # fmt: skip
    return {**base, **over}


def test_carried_effects_classify_reads_writes_and_lost_replies() -> None:
    effects = resume.carried_effects(saved(), {"w1": True})
    assert [e.reads for e in effects] == [True, False, False, False]
    assert [e.outcome for e in effects] == [
        "ok",
        "reply lost (timeout)",
        "card declined",
        "no result recorded",
    ]
    assert [e.committed_but_unconfirmed for e in effects] == [False, True, False, False]


def test_reissue_means_same_tool_and_same_arguments_and_says_whether_it_ran() -> None:
    trace = make_trace(
        "run-1", "c", True,
        [("payments_refund", {"amount_cents": 5, "charge_id": "ch1"}), ("orders_cancel", {"order_id": "o1"}),
         ("mail_send", {"to": "other"})],
    )  # fmt: skip
    again = resume.reissued(resume.carried_effects(saved()), trace, after_seq=0)
    assert [r.effect.name for r in again] == ["payments_refund", "orders_cancel"]
    assert all(r.executed and r.outcome == "ok" for r in again)


def test_a_blocked_reissue_is_reported_as_blocked() -> None:
    trace = make_trace("run-1", "c", True, [("payments_refund", {"amount_cents": 5, "charge_id": "ch1"})])
    events = [e for e in trace.events if e.kind != "tool_result"]
    events.insert(1, Event(9, "guard_block", {"call_id": "run-1-c0", "source": "guard"}, 5.0))
    blocked = type(trace)(
        trace.run_id, trace.case_id, trace.harness_version, trace.model, events, trace.outcome
    )
    (r,) = resume.reissued(resume.carried_effects(saved()), blocked, after_seq=0)
    assert not r.executed
    assert r.outcome == "blocked by guard"
    text = render(resume.after(resume.build_report(saved(events=[]), blocked)))
    assert "RE-ISSUED BY THE MODEL" in text
    assert "did not run (blocked by guard)" in text


def test_reports_before_and_after() -> None:
    s = saved(pending=[{"name": "tickets_set_status", "args": {}, "call_id": "p"}])
    before = render(resume.before(resume.build_report(s, commits=ALL)))
    assert "RESUMING" in before
    assert "reply lost (timeout)" in before
    assert "asked for but not yet run when it stopped: tickets_set_status" in before
    trace = make_trace("run-1", "c", False, [("tickets_get", {})], reasons=["no mail sent"])
    after = render(resume.after(resume.build_report(s, trace, ALL)))
    assert after.count("NOT REPEATED") == 3
    assert "case graded FAIL no mail sent" in after


def test_a_run_that_had_committed_nothing() -> None:
    s = saved(executed=[{"name": "tickets_get", "args": {}, "call_id": "r1"}])
    assert "committed no state-changing call" in render(resume.after(resume.build_report(s)))
    assert resume.build_report(s).passed is None


def lost_reply_checkpoint(case_id: str) -> tuple[Any, dict[str, Any]]:
    """A checkpoint built as the worker builds one: the calls run on a real world, its digest recorded."""
    from ratchet_cases.cases import get
    from ratchet_cases.types import ToolCall
    from ratchet_cases.world import World

    case = get(case_id)
    world = World.from_case(case)
    charge = "ch_3MoK8" if case_id.endswith("v1") else "ch_7LbQ2"
    calls = [
        ToolCall("tickets_get", {"ticket_id": case.ticket_id}, "a0"),
        ToolCall(
            "payments_refund",
            {"charge_id": charge, "amount_cents": 100, "idempotency_key": "k", "reason": "r"},
            "b0",
        ),
    ]
    results = [world.call(c) for c in calls]
    events = [
        {"kind": "tool_result", "payload": {"result": {"call_id": r.call_id, "ok": r.ok, "error": r.error}}}
        for r in results
    ]
    saved = {"run_id": "r", "case_id": case_id, "harness_version": "h", "events": events, "pending": [],
             "executed": [{"name": c.name, "args": c.args, "call_id": c.call_id} for c in calls],
             "world_digest": world.digest()}  # fmt: skip
    return case, saved


def test_the_audit_log_decides_whether_a_timeout_committed() -> None:
    from ratchet_cases.world import World

    case, committed = lost_reply_checkpoint("refund_lost_reply.v1")
    commits = resume.audit_commits(committed, lambda: World.from_case(case))
    assert commits == {"a0": False, "b0": True}
    (read, refund) = resume.carried_effects(committed, commits)
    assert refund.outcome == "reply lost (timeout)"
    assert refund.committed is True
    assert refund.committed_but_unconfirmed
    assert read.outcome == "ok"

    case, uncommitted = lost_reply_checkpoint("refund_lost_reply.v2")
    commits = resume.audit_commits(uncommitted, lambda: World.from_case(case))
    assert commits == {"a0": False, "b0": False}
    (_, refund) = resume.carried_effects(uncommitted, commits)
    assert refund.outcome == "timed out, not committed"
    assert refund.committed is False
    assert not refund.committed_but_unconfirmed
    report = resume.build_report(uncommitted, commits=commits)
    assert report.uncommitted == [refund]
    assert report.carried_writes == []
    assert "committed no state-changing call" not in render(resume.after(report))
    assert "NOT COMMITTED" in render(resume.after(report))


def test_a_world_that_does_not_match_the_checkpoint_proves_nothing() -> None:
    from ratchet_cases.world import World

    case, saved = lost_reply_checkpoint("refund_lost_reply.v1")
    saved["world_digest"] = "0" * 64
    assert resume.audit_commits(saved, lambda: World.from_case(case)) is None
    (_, refund) = resume.carried_effects(saved, None)
    assert refund.outcome == "timeout, commit not checked"
    assert refund.committed is None
    assert refund in resume.build_report(saved).carried_writes
