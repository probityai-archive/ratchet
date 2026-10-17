"""Sandbox enforcement, recomputed from the record: every tool call ran only when the harness's allowlist and
every one of its guards let it, and was blocked only for a stated cause. What a container outside the
referee's process let a command reach stays authenticated only, and says why."""

from __future__ import annotations

from typing import Any

from ratchet.improve.receipt import Signer
from ratchet.verify.adapter import AUTHENTICATED_ONLY, AUTHENTICATED_ONLY_LIMITS

from .runs_scenario import HELDOUT, TRAIN, by_kind, item, recommit, run_of, scenario, status, verify

ITEM = "tool_policy"


def forged(role: str, case_id: str, edit: Any) -> tuple[Any, str]:
    """The scenario with one trace edited and its report re-committed and its rows re-derived, as a signer
    who re-signs would, so only the policy is left to disagree."""
    s = scenario()
    _, _, trace = run_of(s, role, case_id)
    edit(trace)
    recommit(s, role, rows=True)
    return s, f"tool_policy_violated:{role}:{case_id}:"


def test_every_recorded_call_obeys_the_harness_policy(signer: Signer) -> None:
    r = verify(signer, scenario())
    assert r.ok, r.reason
    # Three calls a run: the ticket read and the note ran, the refund was blocked as off the allowlist.
    assert item(r, ITEM) == {
        "id": ITEM,
        "status": "recomputed",
        "runs": 10,
        "traces_checked": 10,
        "tool_calls": 30,
    }


def test_without_harnesses_or_traces_the_policy_is_not_supplied(signer: Signer) -> None:
    assert status(verify(signer, scenario(), harnesses=False))[ITEM] == "not_supplied"
    assert status(verify(signer, scenario(), traces=False))[ITEM] == "not_supplied"


def test_a_withheld_trace_leaves_the_policy_partial(signer: Signer) -> None:
    s = scenario()
    s.traces["parent_run_2"] = s.traces["parent_run_2"][:1]
    r = verify(signer, s)
    assert r.ok and item(r, ITEM)["status"] == "partial"


def _refund_ran(trace: dict[str, Any]) -> None:
    """The blocked refund is recorded as having run."""
    block = by_kind(trace, "guard_block")[0]
    block["kind"] = "tool_result"
    block["payload"] = {
        "result": {"call_id": block["payload"]["call_id"], "ok": True, "data": {}, "error": None},
        "decisions": [{"guard_id": "g-noop", "blocked": False, "reason": None}],
    }


def test_a_tool_off_the_allowlist_that_ran_is_contradicted(signer: Signer) -> None:
    s, prefix = forged("parent_run_1", TRAIN, _refund_ran)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}tool_outside_allowlist")


def test_a_call_that_skipped_a_guard_is_contradicted(signer: Signer) -> None:
    def skip(trace: dict[str, Any]) -> None:
        by_kind(trace, "tool_result")[0]["payload"]["decisions"] = []

    s, prefix = forged("candidate_run", HELDOUT[0], skip)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}guards_not_consulted")


def test_a_call_a_guard_blocked_that_ran_anyway_is_contradicted(signer: Signer) -> None:
    def block(trace: dict[str, Any]) -> None:
        by_kind(trace, "tool_result")[0]["payload"]["decisions"][0]["blocked"] = True

    s, prefix = forged("candidate_confirm_run", HELDOUT[1], block)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}blocked_call_executed")


def test_an_allowed_tool_blocked_as_off_the_allowlist_is_contradicted(signer: Signer) -> None:
    def misblock(trace: dict[str, Any]) -> None:
        refund = by_kind(trace, "tool_call")[1]["payload"]["call"]
        refund["name"] = "tickets_get"
        model = by_kind(trace, "model")[0]["payload"]
        next(c for c in model["tool_calls"] if c["call_id"] == refund["call_id"])["name"] = "tickets_get"

    s, prefix = forged("parent_run_2", HELDOUT[0], misblock)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}allowed_tool_blocked_as_off_the_allowlist")


def test_a_guard_block_without_a_blocking_guard_is_contradicted(signer: Signer) -> None:
    def guard(trace: dict[str, Any]) -> None:
        by_kind(trace, "guard_block")[0]["payload"]["source"] = "guard"

    s, prefix = forged("candidate_run", TRAIN, guard)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}guard_block_without_a_blocking_guard")


def test_a_block_of_unknown_source_is_contradicted(signer: Signer) -> None:
    def unknown(trace: dict[str, Any]) -> None:
        by_kind(trace, "guard_block")[0]["payload"]["source"] = "operator"

    s, prefix = forged("candidate_run", TRAIN, unknown)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}block_of_unknown_source")


def test_a_call_left_unanswered_is_contradicted(signer: Signer) -> None:
    def drop_answer(trace: dict[str, Any]) -> None:
        trace["events"].remove(by_kind(trace, "tool_result")[-1])
        for i, e in enumerate(trace["events"]):
            e["seq"] = i

    s, prefix = forged("parent_run_1", HELDOUT[1], drop_answer)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}tool_call_unanswered")


def test_an_answer_to_another_call_is_contradicted(signer: Signer) -> None:
    def misanswer(trace: dict[str, Any]) -> None:
        by_kind(trace, "tool_result")[0]["payload"]["result"]["call_id"] = "other"

    s, prefix = forged("parent_run_1", HELDOUT[0], misanswer)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}answer_without_call")


def test_two_calls_with_one_answer_are_contradicted(signer: Signer) -> None:
    def double(trace: dict[str, Any]) -> None:
        first = by_kind(trace, "tool_call")[0]
        index = trace["events"].index(first)
        trace["events"].insert(index + 1, {**first, "payload": dict(first["payload"])})
        for i, e in enumerate(trace["events"]):
            e["seq"] = i

    s, prefix = forged("parent_run_2", HELDOUT[1], double)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}tool_call_unanswered")


def test_a_report_of_a_harness_not_in_the_decision_is_contradicted(signer: Signer) -> None:
    s = scenario()
    report = s.reports["parent_run_2"]
    report["harness_version"] = "elsewhere"
    for t in s.traces["parent_run_2"]:
        t["harness_version"] = "elsewhere"
    recommit(s, "parent_run_2")
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", "report_of_unknown_harness:parent_run_2")


def test_sandbox_enforcement_is_renamed_to_what_no_record_shows(signer: Signer) -> None:
    r = verify(signer, scenario())
    assert "sandbox_enforcement" not in AUTHENTICATED_ONLY and "sandbox_enforcement" not in status(r)
    assert "container_isolation" in r.authenticated_only
    assert "no launcher records an isolation report" in AUTHENTICATED_ONLY_LIMITS["container_isolation"]
