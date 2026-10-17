"""Capture fidelity, recomputed from the record chain: each supplied trace is the committed one, a well-formed
run of its report's harness, and every report row is what its trace yields."""

from __future__ import annotations

from ratchet.improve.receipt import Signer
from ratchet.verify import Options
from ratchet.verify.adapter import AUTHENTICATED_ONLY

from .runs_scenario import HELDOUT, TRAIN, by_kind, item, recommit, run_of, scenario, status, verify

ITEM = "record_chain"


def test_every_trace_is_the_committed_record_and_yields_its_row(signer: Signer) -> None:
    r = verify(signer, scenario())
    assert r.ok, r.reason
    assert item(r, ITEM) == {"id": ITEM, "status": "recomputed", "runs": 10, "traces_checked": 10}


def test_a_withheld_trace_leaves_the_chain_partial(signer: Signer) -> None:
    s = scenario()
    _, _, trace = run_of(s, "candidate_confirm_run", HELDOUT[0])
    s.traces["candidate_confirm_run"].remove(trace)
    r = verify(signer, s)
    assert r.ok and item(r, ITEM)["status"] == "partial" and item(r, ITEM)["traces_checked"] == 9
    strict = verify(signer, s, options=Options(require_complete=True))
    assert strict.decision == "not_established" and strict.reason.startswith("recomputation_incomplete:")


def test_no_trace_at_all_is_not_supplied(signer: Signer) -> None:
    r = verify(signer, scenario(), traces=False)
    assert r.ok and status(r)[ITEM] == "not_supplied"


def test_a_trace_edited_after_it_was_committed_is_contradicted(signer: Signer) -> None:
    s = scenario()
    _, record, trace = run_of(s, "parent_run_1", TRAIN)
    by_kind(trace, "tool_result")[0]["payload"]["result"]["ok"] = False
    r = verify(signer, s)
    assert (r.decision, r.reason) == (
        "contradicted",
        f"trace_digest_mismatch:parent_run_1:{record['run_id']}",
    )


def test_a_trace_of_a_run_the_report_never_committed_is_contradicted(signer: Signer) -> None:
    s = scenario()
    _, _, trace = run_of(s, "parent_run_2", HELDOUT[0])
    s.traces["parent_run_2"].append({**trace, "run_id": "run-extra"})
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", "trace_not_committed:parent_run_2:run-extra")


def test_a_row_its_trace_does_not_yield_is_contradicted(signer: Signer) -> None:
    """The signer re-signs a report whose row claims a pass its own committed trace records as a fail."""
    s = scenario()
    row, _, trace = run_of(s, "candidate_run", HELDOUT[1])
    assert trace["outcome"]["passed"] is False
    row["passed"], row["reasons"] = True, []
    r = verify(signer, s)
    assert (r.decision, r.reason) == (
        "contradicted",
        f"record_chain_mismatch:candidate_run:{HELDOUT[1]}:passed",
    )


def test_a_hidden_intervention_is_contradicted(signer: Signer) -> None:
    """The allowlist block is cut from the trace and the events renumbered and re-committed; the row still
    counts the intervention it had."""
    s = scenario()
    row, _, trace = run_of(s, "candidate_run", HELDOUT[0])
    assert row["interventions"] == 1
    trace["events"] = [e for e in trace["events"] if e["kind"] != "guard_block"]
    for i, e in enumerate(trace["events"]):
        e["seq"] = i
    recommit(s, "candidate_run")
    r = verify(signer, s)
    reason = f"record_chain_mismatch:candidate_run:{HELDOUT[0]}:interventions"
    assert (r.decision, r.reason) == ("contradicted", reason)


def test_a_dropped_event_breaks_the_numbering(signer: Signer) -> None:
    s = scenario()
    _, _, trace = run_of(s, "parent_run_2", HELDOUT[1])
    trace["events"].remove(by_kind(trace, "tool_result")[0])
    recommit(s, "parent_run_2")
    r = verify(signer, s)
    assert (r.decision, r.reason) == (
        "contradicted",
        f"record_chain_mismatch:parent_run_2:{HELDOUT[1]}:events",
    )


def test_a_tool_call_no_model_turn_asked_for_is_contradicted(signer: Signer) -> None:
    s = scenario()
    _, _, trace = run_of(s, "parent_run_1", HELDOUT[0])
    by_kind(trace, "tool_call")[0]["payload"]["call"]["call_id"] = "injected"
    recommit(s, "parent_run_1")
    r = verify(signer, s)
    reason = f"record_chain_mismatch:parent_run_1:{HELDOUT[0]}:unrequested_call"
    assert (r.decision, r.reason) == ("contradicted", reason)


def test_a_trace_of_another_harness_is_contradicted(signer: Signer) -> None:
    s = scenario()
    _, _, trace = run_of(s, "candidate_confirm_run", HELDOUT[0])
    trace["harness_version"] = s.harnesses["parent"]["version"]
    recommit(s, "candidate_confirm_run")
    r = verify(signer, s)
    reason = f"record_chain_mismatch:candidate_confirm_run:{HELDOUT[0]}:harness_version"
    assert (r.decision, r.reason) == ("contradicted", reason)


def test_a_trace_committed_under_another_case_is_contradicted(signer: Signer) -> None:
    s = scenario()
    _, record, _ = run_of(s, "parent_run_1", HELDOUT[1])
    record["case_id"] = HELDOUT[0]
    # The row of that run is dropped too, so the run records alone carry the swap.
    rows = s.reports["parent_run_1"]["results"]
    s.reports["parent_run_1"]["results"] = [r for r in rows if r["case_id"] != HELDOUT[1]]
    r = verify(signer, s)
    assert (r.decision, r.reason) == (
        "contradicted",
        f"record_chain_mismatch:parent_run_1:{HELDOUT[0]}:case_id",
    )


def test_capture_fidelity_is_no_longer_taken_on_the_signature(signer: Signer) -> None:
    r = verify(signer, scenario())
    assert "capture_fidelity" not in AUTHENTICATED_ONLY and "capture_fidelity" not in r.authenticated_only
    # What the chain cannot show, that no model turn was altered or left out, stays named for what it is.
    assert "model_turns_as_recorded" in r.authenticated_only
