"""Fresh model execution, recomputed as far as records show it: every counted model call is committed once, in
its own run, and equals the supplied trace. Whether an endpoint, not the referee, wrote the turns is
authenticated only, and says why."""

from __future__ import annotations

from ratchet.improve.receipt import Signer
from ratchet.verify import Options, verify_decision
from ratchet.verify.adapter import AUTHENTICATED_ONLY, AUTHENTICATED_ONLY_LIMITS

from .helpers import sign
from .runs_scenario import (
    HELDOUT,
    TRAIN,
    body_for,
    by_kind,
    item,
    recommit,
    run_of,
    scenario,
    status,
    verify,
)

ITEM = "model_call_records"


def test_every_counted_model_call_is_recomputed_from_the_traces(signer: Signer) -> None:
    s = scenario()
    r = verify(signer, s)
    assert r.ok, r.reason
    got = item(r, ITEM)
    assert got["status"] == "recomputed"
    runs = sum(len(rep["runs"]) for rep in s.reports.values())
    assert (got["runs"], got["traces_checked"]) == (runs, runs) == (10, 10)
    assert got["model_calls"] == 30  # three turns in each of ten runs


def test_without_traces_the_commitments_are_checked_but_the_item_is_partial(signer: Signer) -> None:
    r = verify(signer, scenario(), traces=False)
    assert r.ok and item(r, ITEM)["status"] == "partial" and item(r, ITEM)["traces_checked"] == 0


def test_reports_written_before_run_records_leave_the_claim_not_established(signer: Signer) -> None:
    s = scenario()
    for rep in s.reports.values():
        del rep["runs"]
    r = verify(signer, s)
    assert r.ok and status(r)[ITEM] == "not_supplied"
    strict = verify(signer, s, options=Options(require_complete=True))
    assert (strict.decision, strict.reason) == ("not_established", f"recomputation_incomplete:{ITEM}")


def test_a_row_whose_run_is_not_committed_is_not_established(signer: Signer) -> None:
    s = scenario()
    _, record, _ = run_of(s, "candidate_run", HELDOUT[0])
    s.reports["candidate_run"]["runs"].remove(record)
    r = verify(signer, s, traces=False)
    assert (r.decision, r.reason) == ("not_established", f"run_record_missing:candidate_run:{HELDOUT[0]}")


def test_a_forged_call_count_is_contradicted(signer: Signer) -> None:
    s = scenario()
    row, _, _ = run_of(s, "parent_run_1", TRAIN)
    row["fresh_calls"] += 1
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"model_call_count_mismatch:parent_run_1:{TRAIN}")


def test_a_run_counted_in_two_reports_is_contradicted(signer: Signer) -> None:
    """P2 made from P1's run: the run id and its answers appear twice."""
    s = scenario()
    p1_row, p1_record, _ = run_of(s, "parent_run_1", HELDOUT[0])
    p2_row, p2_record, _ = run_of(s, "parent_run_2", HELDOUT[0])
    p2_row["run_id"] = p1_row["run_id"]
    s.reports["parent_run_2"]["runs"][s.reports["parent_run_2"]["runs"].index(p2_record)] = dict(p1_record)
    r = verify(signer, s, traces=False)
    assert (r.decision, r.reason) == (
        "contradicted",
        f"run_reused:parent_run_1:parent_run_2:{p1_row['run_id']}",
    )


def test_an_answer_copied_into_another_run_is_contradicted(signer: Signer) -> None:
    s = scenario()
    _, source, _ = run_of(s, "parent_run_1", HELDOUT[0])
    _, target, _ = run_of(s, "candidate_run", HELDOUT[0])
    target["model_calls"][0] = source["model_calls"][0]
    r = verify(signer, s, traces=False)
    assert (r.decision, r.reason) == ("contradicted", f"model_call_reused:candidate_run:{target['run_id']}")


def test_a_trace_whose_model_turn_differs_from_the_commitment_is_contradicted(signer: Signer) -> None:
    s = scenario()
    _, record, trace = run_of(s, "candidate_confirm_run", HELDOUT[1])
    by_kind(trace, "model")[-1]["payload"]["content"] = "Refunded."
    r = verify(signer, s)
    assert (r.decision, r.reason) == (
        "contradicted",
        f"model_calls_not_the_trace:candidate_confirm_run:{record['run_id']}",
    )


def test_a_cached_answer_counted_as_fresh_is_contradicted(signer: Signer) -> None:
    """The signer re-commits a trace whose answer came from a cache but leaves the row FRESH."""
    s = scenario()
    row, _, trace = run_of(s, "candidate_run", HELDOUT[0])
    by_kind(trace, "model")[0]["payload"]["cached"] = True
    recommit(s, "candidate_run")
    r = verify(signer, s)
    reason = f"cached_calls_not_the_trace:candidate_run:{row['run_id']}"
    assert (r.decision, r.reason) == ("contradicted", reason)


def test_a_fresh_row_that_admits_cached_calls_is_contradicted(signer: Signer) -> None:
    s = scenario()
    row, _, _ = run_of(s, "parent_run_2", HELDOUT[1])
    row["fresh_calls"] -= 1
    row["cached_calls"] += 1
    r = verify(signer, s, traces=False)
    reason = f"fresh_row_with_cached_calls:parent_run_2:{HELDOUT[1]}"
    assert (r.decision, r.reason) == ("contradicted", reason)


def test_malformed_run_records_are_contradicted(signer: Signer) -> None:
    s = scenario()
    s.reports["parent_run_2"]["runs"][0]["trace_digest"] = "not hex"
    r = verify(signer, s, traces=False)
    assert (r.decision, r.reason) == ("contradicted", "run_records_malformed:parent_run_2")


def test_a_row_naming_another_case_than_its_run_is_contradicted(signer: Signer) -> None:
    s = scenario()
    _, record, _ = run_of(s, "parent_run_1", TRAIN)
    record["case_id"] = HELDOUT[0]
    r = verify(signer, s, traces=False)
    assert (r.decision, r.reason) == ("contradicted", f"run_record_case_mismatch:parent_run_1:{TRAIN}")


def test_traces_that_are_not_json_lines_are_contradicted(signer: Signer) -> None:
    s = scenario()
    r = verify_decision(
        sign(body_for(s), signer),
        signer.public_key,
        reports=s.reports,
        traces={"parent_run_1": b"{not json\n"},
    )
    assert (r.decision, r.reason) == ("contradicted", "evidence_not_json:traces:parent_run_1")


def test_the_endpoint_writing_the_turns_stays_authenticated_only_and_says_why(signer: Signer) -> None:
    r = verify(signer, scenario())
    assert "fresh_model_execution" not in AUTHENTICATED_ONLY
    assert "model_turns_as_recorded" in r.authenticated_only
    assert "fresh_model_execution" not in status(r)
    limit = AUTHENTICATED_ONLY_LIMITS["model_turns_as_recorded"]
    assert "no signature" in limit and "referee's key" in limit
    assert (r.packet or {})["scope"]["authenticated_only_limits"]["model_turns_as_recorded"] == limit
