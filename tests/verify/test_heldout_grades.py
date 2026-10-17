"""Hidden grades, recomputed: every report binds each held-out case by digest, a disclosed case document
must be the committed one, and each held-out run is graded again from its case and trace. The case
documents are the published case suite's own, which a third party installs."""

from __future__ import annotations

from typing import Any

import pytest

from ratchet.improve.receipt import Signer
from ratchet.verify import Options
from ratchet.verify.adapter import AUTHENTICATED_ONLY
from ratchet.verify.runs import RecordError, Report, case_document, check_heldout_grades, digest, regrade

from .runs_scenario import CASES, HELDOUT, ROLES, by_kind, item, recommit, run_of, scenario, status, verify

ITEM = "heldout_grades"


def test_every_held_out_run_is_graded_again_from_its_case_and_trace(signer: Signer) -> None:
    r = verify(signer, scenario())
    assert r.ok and r.recomputation_complete, r.summary
    got = item(r, ITEM)
    assert (got["status"], got["heldout_rows"], got["regraded"], got["cases_disclosed"]) == (
        "recomputed",
        8,
        8,
        2,
    )
    assert got["not_rebuildable"] == []


def test_withheld_cases_leave_the_grades_partial(signer: Signer) -> None:
    r = verify(signer, scenario(), cases={})
    assert r.ok and item(r, ITEM)["status"] == "partial" and item(r, ITEM)["regraded"] == 0
    one = verify(signer, scenario(), cases={HELDOUT[0]: CASES[HELDOUT[0]]})
    assert item(one, ITEM)["regraded"] == 4 and item(one, ITEM)["status"] == "partial"


def test_reports_written_before_case_digests_leave_the_grades_not_established(signer: Signer) -> None:
    s = scenario()
    for rep in s.reports.values():
        for row in rep["results"]:
            del row["case_digest"]
    r = verify(signer, s)
    assert r.ok and status(r)[ITEM] == "not_supplied"
    strict = verify(signer, s, options=Options(require_complete=True))
    assert (strict.decision, strict.reason) == ("not_established", f"recomputation_incomplete:{ITEM}")


def test_a_held_out_row_without_its_case_digest_is_not_established(signer: Signer) -> None:
    s = scenario()
    row, _, _ = run_of(s, "candidate_run", HELDOUT[1])
    row["case_digest"] = None
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("not_established", f"case_digest_missing:candidate_run:{HELDOUT[1]}")


def test_a_case_swapped_between_runs_is_contradicted(signer: Signer) -> None:
    s = scenario()
    row, _, _ = run_of(s, "candidate_confirm_run", HELDOUT[0])
    row["case_digest"] = "0" * 64
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"heldout_case_changed:{HELDOUT[0]}")


def test_a_disclosed_case_that_is_not_the_committed_one_is_contradicted(signer: Signer) -> None:
    doc = {**CASES[HELDOUT[0]], "customer_message": "Please refund twice."}
    r = verify(signer, scenario(), cases={**CASES, HELDOUT[0]: doc})
    assert (r.decision, r.reason) == ("contradicted", f"heldout_case_digest_mismatch:{HELDOUT[0]}")


def test_a_case_document_filed_under_another_id_is_contradicted(signer: Signer) -> None:
    r = verify(signer, scenario(), cases={HELDOUT[0]: CASES[HELDOUT[1]]})
    assert (r.decision, r.reason) == ("contradicted", f"case_document_names_another_case:{HELDOUT[0]}")


def forged(role: str, case_id: str, edit: Any) -> tuple[Any, str]:
    """One held-out trace edited, its report re-committed and its rows re-derived, as a signer who re-signs
    would: only the grade is left to disagree."""
    s = scenario()
    _, _, trace = run_of(s, role, case_id)
    edit(trace)
    recommit(s, role, rows=True)
    return s, f"grade_mismatch:{role}:{case_id}:"


def test_a_failing_run_recorded_as_a_pass_is_contradicted(signer: Signer) -> None:
    def pass_it(trace: dict[str, Any]) -> None:
        assert trace["outcome"]["passed"] is False
        trace["outcome"]["passed"], trace["outcome"]["reasons"] = True, []

    s, prefix = forged("candidate_run", HELDOUT[0], pass_it)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}passed")


def test_a_recorded_reason_the_grader_does_not_give_is_contradicted(signer: Signer) -> None:
    def reason(trace: dict[str, Any]) -> None:
        trace["outcome"]["reasons"] = [*trace["outcome"]["reasons"], "invented"]

    s, prefix = forged("parent_run_2", HELDOUT[1], reason)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}reasons")


def test_a_tool_result_the_world_does_not_give_is_contradicted(signer: Signer) -> None:
    def spoof(trace: dict[str, Any]) -> None:
        by_kind(trace, "tool_result")[0]["payload"]["result"]["data"] = {"status": "solved"}

    s, prefix = forged("parent_run_1", HELDOUT[0], spoof)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}tool_result")


def test_a_final_state_the_calls_do_not_produce_is_contradicted(signer: Signer) -> None:
    def state(trace: dict[str, Any]) -> None:
        trace["outcome"]["final_state_digest"] = "f" * 64

    s, prefix = forged("candidate_confirm_run", HELDOUT[1], state)
    r = verify(signer, s)
    assert (r.decision, r.reason) == ("contradicted", f"{prefix}final_state")


def test_a_row_that_disagrees_with_a_regraded_trace_is_contradicted() -> None:
    """Called directly, without the record chain that would catch it first."""
    s = scenario()
    role = ROLES[0]
    row, _, _ = run_of(s, role, HELDOUT[0])
    row["passed"] = not row["passed"]
    traces = {t["run_id"]: t for t in s.traces[role]}
    with pytest.raises(RecordError, match=f"^grade_mismatch:{role}:{HELDOUT[0]}:row$"):
        check_heldout_grades([Report(role, s.reports[role], traces)], CASES)


def test_a_case_of_a_world_this_verifier_cannot_rebuild_is_counted_not_regraded() -> None:
    s = scenario()
    role = ROLES[1]
    other = {"id": HELDOUT[0], "split": "heldout", "kind": "trap", "must_succeed": True}
    for row in s.reports[role]["results"]:
        if row["case_id"] == HELDOUT[0]:
            row["case_digest"] = digest(other)
    traces = {t["run_id"]: t for t in s.traces[role]}
    counts = check_heldout_grades([Report(role, s.reports[role], traces)], {**CASES, HELDOUT[0]: other})
    assert (
        counts["not_rebuildable"] == [HELDOUT[0]] and counts["regraded"] == 1 and counts["heldout_rows"] == 2
    )
    _, _, trace = run_of(s, role, HELDOUT[0])
    with pytest.raises(LookupError, match=HELDOUT[0]):
        regrade(other, trace)


def test_the_case_document_is_the_published_suite_case() -> None:
    from ratchet_cases.cases import get
    from ratchet_cases.types import to_jsonable

    assert case_document(get(HELDOUT[0])) == to_jsonable(get(HELDOUT[0]))


def test_hidden_grades_leave_the_authenticated_only_list(signer: Signer) -> None:
    r = verify(signer, scenario())
    assert "hidden_grades" not in AUTHENTICATED_ONLY and "hidden_grades" not in status(r)
    assert r.authenticated_only == ("model_turns_as_recorded", "container_isolation")
