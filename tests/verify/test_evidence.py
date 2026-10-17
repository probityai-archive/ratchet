"""Recomputation: each gate is derived from stored evidence; a disagreement with the receipt fails closed."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import pytest

from ratchet.canonical import canonical_json
from ratchet.improve.receipt import HUMAN_APPROVAL_TYPE, Signer
from ratchet.types import Guard, Harness
from ratchet.verify import DEFAULT_GATES, Options, report_digest, verify_decision
from ratchet.verify import recompute as rc
from ratchet.verify.adapter import AUTHENTICATED_ONLY

from .helpers import (
    ALWAYS,
    CANDIDATE_HARNESS,
    GUARD_ID,
    HARNESSES,
    NOOP,
    PARENT_HARNESS,
    RECOMPUTED,
    REPLAY_REPORT,
    REPORTS,
    RUN_RECORD_ITEMS,
    SNAPSHOT,
    eval_report,
    payload,
    replay_row,
    sign,
)

AUTHENTICATED = AUTHENTICATED_ONLY
GATE_ITEMS = 10  # signature, payload, admission flag, report digests and the six gates


def run(
    signer: Signer,
    *,
    reports: dict[str, Any] | None = None,
    harnesses: dict[str, Any] | None = None,
    recomputed: dict[str, Any] | None = None,
    approval: bytes | None = None,
    options: Options = Options(),  # noqa: B008
    **overrides: Any,
) -> Any:
    """Sign a receipt whose digests name `reports` (default: the clean scenario) and verify it."""
    stored = REPORTS if reports is None else reports
    body = payload(**overrides)
    body["report_digests"] = overrides.get("report_digests") or {
        n: report_digest(r) for n, r in stored.items()
    }
    return verify_decision(
        sign(body, signer),
        signer.public_key,
        reports=stored,
        harnesses=HARNESSES if harnesses is None else harnesses,
        recomputed=RECOMPUTED if recomputed is None else recomputed,
        approval=approval,
        options=options,
    )


def with_reports(**changes: dict[str, Any]) -> dict[str, Any]:
    return {**REPORTS, **changes}


def status(report: Any) -> dict[str, str]:
    return {i["id"]: i["status"] for i in report.recomputed}


def gates_complete(report: Any) -> bool:
    """Every item but the run records recomputed. These reports carry no run records, so those items are
    not supplied and the recomputation as a whole is incomplete."""
    st = status(report)
    rest = {k: v for k, v in st.items() if k not in RUN_RECORD_ITEMS}
    runs = {st.get(k) for k in RUN_RECORD_ITEMS}
    complete = bool(report.recomputation_complete)
    return set(rest.values()) == {"recomputed"} and runs == {"not_supplied"} and not complete


def failing(check_name: str, evidence: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = payload()["checks"]
    for c in checks:
        if c["name"] == check_name:
            c["passed"] = False
            c["evidence"] = evidence or c["evidence"]
    return checks


# --- what is recomputed, what is not -----------------------------------------------------------------


def test_everything_supplied_recomputes_every_gate_and_lists_what_is_only_authenticated(
    signer: Signer,
) -> None:
    r = run(signer)
    assert r.ok and gates_complete(r)
    assert len(r.recomputed) == GATE_ITEMS + len(RUN_RECORD_ITEMS)
    assert r.authenticated_only == AUTHENTICATED
    assert f"NOT SUPPLIED: {', '.join(RUN_RECORD_ITEMS)}" in r.summary and "AUTHENTICATED ONLY" in r.summary


def test_no_evidence_prints_as_incomplete_never_as_recomputed(signer: Signer) -> None:
    r = verify_decision(sign(payload(), signer), signer.public_key)
    assert r.ok and not r.recomputation_complete
    assert status(r) == {
        "signature": "recomputed",
        "payload_predicate": "recomputed",
        "admission_flag": "recomputed",
        "report_digests": "not_supplied",
        "gate:heldout": "not_supplied",
        "gate:no_regression": "not_supplied",
        "gate:benign_twins": "not_supplied",
        "gate:authority": "not_supplied",
        "gate:replay": "not_supplied",
        "gate:overfit_lint": "not_supplied",
        **dict.fromkeys(RUN_RECORD_ITEMS, "not_supplied"),
    }
    assert "recomputation INCOMPLETE" in r.summary
    assert "every recomputable item recomputed" not in r.summary
    assert r.authenticated_only == AUTHENTICATED


def test_reports_alone_recompute_the_evaluation_gates_but_not_authority_replay_or_overfit(
    signer: Signer,
) -> None:
    r = verify_decision(sign(payload(), signer), signer.public_key, reports=REPORTS)
    st = status(r)
    assert st["report_digests"] == "recomputed"
    assert {st[f"gate:{g}"] for g in ("heldout", "no_regression", "benign_twins")} == {"recomputed"}
    assert {st[f"gate:{g}"] for g in ("authority", "replay", "overfit_lint")} == {"not_supplied"}
    assert not r.recomputation_complete


def test_a_subset_of_reports_is_partial(signer: Signer) -> None:
    some = {k: REPORTS[k] for k in ("parent_run_1", "candidate_run")}
    r = verify_decision(sign(payload(), signer), signer.public_key, reports=some)
    assert status(r)["report_digests"] == "partial" and status(r)["gate:heldout"] == "not_supplied"
    assert "PARTIAL: report_digests" in r.summary


def test_require_complete_demands_every_recomputable_item(signer: Signer) -> None:
    r = verify_decision(sign(payload(), signer), signer.public_key, options=Options(require_complete=True))
    assert (r.ok, r.decision, r.reason) == (
        False,
        "not_established",
        "recomputation_incomplete:report_digests",
    )
    assert not r.recomputation_complete
    reports_without_run_records = run(signer, options=Options(require_complete=True))
    assert reports_without_run_records.reason == f"recomputation_incomplete:{RUN_RECORD_ITEMS[0]}"


# --- held-out gate -------------------------------------------------------------------------------------


def test_one_stable_recovery_is_not_enough(signer: Signer) -> None:
    weak = eval_report("c", ALWAYS | {"h1"})
    r = run(signer, reports=with_reports(candidate_run=weak, candidate_confirm_run=weak))
    assert (r.ok, r.decision, r.reason) == (False, "contradicted", "gate_recompute_mismatch:heldout")


def test_a_protected_case_lost_in_the_confirmation_run_fails(signer: Signer) -> None:
    lost = eval_report("c", (ALWAYS - {"h3"}) | {"h1", "h2"})
    r = run(signer, reports=with_reports(candidate_confirm_run=lost))
    assert r.reason == "gate_recompute_mismatch:heldout"


def test_a_cached_pass_is_not_fresh_evidence(signer: Signer) -> None:
    cached = eval_report("c", ALWAYS | {"h1", "h2"})
    for row in cached["results"]:
        if row["case_id"] == "h2":
            row["mode"] = "CACHED"
    assert run(signer, reports=with_reports(candidate_run=cached)).reason == "gate_recompute_mismatch:heldout"


def test_runs_over_different_case_sets_fail(signer: Signer) -> None:
    short = eval_report("c", ALWAYS | {"h1", "h2"})
    short["results"] = [r for r in short["results"] if r["case_id"] != "b2"]
    assert run(signer, reports=with_reports(candidate_run=short)).reason == "gate_recompute_mismatch:heldout"


def test_a_receipt_that_claims_failure_the_evidence_does_not_show_is_contradicted(signer: Signer) -> None:
    r = run(signer, admitted=False, checks=failing("heldout"))
    assert r.reason == "gate_recompute_mismatch:heldout"


def test_a_verified_rejection_agrees_with_failing_evidence(signer: Signer) -> None:
    weak = eval_report("c", ALWAYS)
    reports = with_reports(candidate_run=weak, candidate_confirm_run=weak)
    r = run(signer, reports=reports, admitted=False, checks=failing("heldout"))
    assert (r.ok, r.admitted, r.admission_verified, gates_complete(r)) == (True, False, False, True)


# --- a refusal made after the first candidate run (no confirming run C2) ------------------------------

WITHOUT_C2 = {k: v for k, v in REPORTS.items() if k != "candidate_confirm_run"}


def refused_after_c1(*gates: str) -> list[dict[str, Any]]:
    checks = payload()["checks"]
    for c in checks:
        if c["name"] in gates:
            c["passed"] = False
    return checks


def test_a_refusal_without_a_confirming_run_recomputes_every_gate_from_the_runs_it_names(
    signer: Signer,
) -> None:
    """The bad fix of the demo: C1 breaks a protected case, regresses a must-succeed case and intervenes on a
    benign twin; C2 never ran. All three gates are recomputed from P1, P2 and C1 alone."""
    c1 = eval_report("c", {"h1", "h2"}, hits={"b2": 2})
    reports = {**WITHOUT_C2, "candidate_run": c1}
    r = run(
        signer,
        reports=reports,
        admitted=False,
        checks=refused_after_c1("heldout", "no_regression", "benign_twins"),
    )
    assert (r.ok, r.reason, gates_complete(r)) == (True, "rejection_receipt_verified", True)
    st = status(r)
    assert {st[f"gate:{g}"] for g in ("heldout", "no_regression", "benign_twins")} == {"recomputed"}
    assert len(r.recomputed) == GATE_ITEMS + len(RUN_RECORD_ITEMS)
    held = next(i for i in r.recomputed if i["id"] == "gate:heldout")["evidence"]
    assert held["confirm_run"] == "not run" and held["stable_recoveries"] == []
    assert held["stable_recoveries_c1"] == ["h1", "h2"] and held["protected_broken"] == ["b1", "b2", "h3"]


def test_without_a_confirming_run_the_held_out_gate_cannot_pass(signer: Signer) -> None:
    """C1 alone clears the held-out rule, but admission needs C2: a receipt claiming a pass is contradicted,
    and a refusal for the missing run agrees."""
    r = run(signer, reports=WITHOUT_C2)
    assert (r.ok, r.reason) == (False, "gate_recompute_mismatch:heldout")
    other_fail = refused_after_c1("heldout", "overfit_lint")
    r = run(
        signer,
        reports=WITHOUT_C2,
        admitted=False,
        checks=other_fail,
        recomputed={**RECOMPUTED, "overfit_lint": {"hits": [{"literal": "x"}]}},
    )
    assert (r.ok, gates_complete(r)) == (True, True)


def test_a_refusal_after_c1_that_claims_a_failure_c1_does_not_show_is_contradicted(signer: Signer) -> None:
    r = run(signer, reports=WITHOUT_C2, admitted=False, checks=refused_after_c1("heldout", "no_regression"))
    assert (r.ok, r.reason) == (False, "gate_recompute_mismatch:no_regression")


def test_a_named_confirming_run_that_is_not_supplied_is_still_missing(signer: Signer) -> None:
    body_digests = {n: report_digest(r) for n, r in REPORTS.items()}
    r = run(signer, reports=WITHOUT_C2, report_digests=body_digests)
    assert {status(r)[f"gate:{g}"] for g in ("heldout", "no_regression", "benign_twins")} == {"not_supplied"}
    assert status(r)["report_digests"] == "partial" and not r.recomputation_complete


def test_the_held_out_recomputation_without_c2_unit() -> None:
    p = eval_report("p", ALWAYS)
    gate = rc.heldout(p, p, eval_report("c", ALWAYS | {"h1", "h2"}), None)
    assert not gate.passed and gate.evidence["confirm_run"] == "not run"
    assert gate.evidence["stable_recoveries_c1"] == ["h1", "h2"] and gate.evidence["protected_broken"] == []
    both = rc.heldout(p, p, eval_report("c", ALWAYS | {"h1", "h2"}), eval_report("c", ALWAYS | {"h1"}))
    assert not both.passed and both.evidence["stable_recoveries"] == ["h1"]
    assert both.evidence["confirm_run"] == "run"


# --- no regression and benign twins ------------------------------------------------------------------


def test_a_must_succeed_regression_is_found(signer: Signer) -> None:
    bad = eval_report("c", (ALWAYS - {"m1"}) | {"h1", "h2"})
    assert (
        run(signer, reports=with_reports(candidate_run=bad)).reason == "gate_recompute_mismatch:no_regression"
    )


def test_a_benign_twin_intervention_is_found(signer: Signer) -> None:
    bad = eval_report("c", ALWAYS | {"h1", "h2"}, hits={"b1": 1})
    assert run(signer, reports=with_reports(candidate_confirm_run=bad)).reason == (
        "gate_recompute_mismatch:benign_twins"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("results"),
        lambda r: r.update(results=[{"passed": True}]),
        lambda r: r.update(results=r["results"] + r["results"][:1]),
        lambda r: r.update(results="none"),
    ],
)
def test_malformed_evaluation_report_is_contradicted(signer: Signer, mutate: Any) -> None:
    bad = copy.deepcopy(REPORTS["candidate_run"])
    mutate(bad)
    r = run(signer, reports=with_reports(candidate_run=bad))
    assert (r.decision, r.reason.split(":")[0]) == ("contradicted", "report_malformed")


def test_a_report_that_is_not_a_json_object_is_contradicted(signer: Signer) -> None:
    raw = b"[1]"
    from ratchet_cases.types import sha256_hex

    r = verify_decision(
        sign(
            payload(report_digests={**payload()["report_digests"], "candidate_run": sha256_hex(raw)}), signer
        ),
        signer.public_key,
        reports={**REPORTS, "candidate_run": raw},
    )
    assert r.reason == "evidence_not_json:report:candidate_run"


def test_the_consumer_names_which_reports_are_the_four_runs(signer: Signer) -> None:
    renamed = {"p1": "a", "p2": "b", "c1": "c", "c2": "d"}
    stored = {
        "a": REPORTS["parent_run_1"],
        "b": REPORTS["parent_run_2"],
        "c": REPORTS["candidate_run"],
        "d": REPORTS["candidate_confirm_run"],
    }
    r = run(signer, reports=stored, options=Options(eval_roles=renamed))
    assert status(r)["gate:heldout"] == "recomputed"
    with_default_roles = run(signer, reports=stored)
    assert status(with_default_roles)["gate:heldout"] == "not_supplied"


# --- authority -------------------------------------------------------------------------------------------


def harness_pair(
    *, tools: list[str] | None = None, prompt: str | None = None, guards: list[Guard] | None = None
) -> tuple[Harness, dict[str, Any]]:
    candidate = Harness.build(
        parent=PARENT_HARNESS.version,
        system_prompt=PARENT_HARNESS.system_prompt + "\nExtra." if prompt is None else prompt,
        procedures=[],
        guards=guards if guards is not None else [*PARENT_HARNESS.guards, Guard(GUARD_ID, "n", NOOP, [])],
        tool_allowlist=tools if tools is not None else ["refund"],
        created_by="improver",
    )
    return candidate, {"parent": asdict(PARENT_HARNESS), "candidate": asdict(candidate)}


def decided(candidate: Harness, **extra: Any) -> dict[str, Any]:
    return {"candidate": candidate.version, "candidate_digest": candidate.digest, **extra}


def test_growth_in_tools_without_a_declared_exception_is_contradicted(signer: Signer) -> None:
    candidate, pair = harness_pair(tools=["refund", "wire"])
    r = run(signer, harnesses=pair, **decided(candidate))
    assert (r.decision, r.reason) == ("contradicted", "authority_recompute_failed_without_exception")


@pytest.mark.parametrize("claim", [True, False])
def test_the_receipt_and_the_recomputation_must_agree_on_authority(signer: Signer, claim: bool) -> None:
    candidate, pair = harness_pair(tools=["refund", "wire"])
    extra: dict[str, Any] = {} if claim else {"admitted": False, "checks": failing("authority")}
    r = run(signer, harnesses=pair, **decided(candidate), **extra)
    if claim:
        assert r.reason == "authority_recompute_failed_without_exception"
    else:
        assert r.ok and status(r)["gate:authority"] == "recomputed"


def test_a_clean_candidate_that_the_receipt_calls_a_failure_is_contradicted(signer: Signer) -> None:
    r = run(signer, admitted=False, checks=failing("authority"))
    assert r.reason == "gate_recompute_mismatch:authority"


def human_approval(human: Signer, candidate: Harness, payload_type: str = HUMAN_APPROVAL_TYPE) -> bytes:
    return human.sign(payload_type, canonical_json({"candidate_digest": candidate.digest, "approver": "ana"}))


def test_authority_growth_covered_by_a_verified_human_approval(signer: Signer, other_signer: Signer) -> None:
    candidate, pair = harness_pair(tools=["refund", "wire"])
    r = run(
        signer,
        harnesses=pair,
        approval=human_approval(other_signer, candidate),
        options=Options(trusted_human_keys=[other_signer.public_key]),
        exceptions=["authority grew under approval by ana"],
        **decided(candidate),
    )
    assert r.ok and gates_complete(r)
    item = next(i for i in r.recomputed if i["id"] == "gate:authority")
    assert item["covered_by"] == "verified human approval"


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("none", "no_approval_supplied"),
        ("garbage", "approval_malformed"),
        ("wrong_type", "approval_type"),
        ("untrusted", "approval_signature"),
        ("other_candidate", "approval_names_another_candidate"),
        ("bad_payload", "approval_payload"),
    ],
)
def test_an_authority_exception_without_a_valid_approval_is_not_established(
    signer: Signer, other_signer: Signer, case: str, reason: str
) -> None:
    candidate, pair = harness_pair(tools=["refund", "wire"])
    human = other_signer
    approvals: dict[str, bytes | None] = {
        "none": None,
        "garbage": b"nope",
        "wrong_type": human_approval(human, candidate, "application/x-other"),
        "untrusted": human_approval(Signer.generate(), candidate),
        "other_candidate": human_approval(human, PARENT_HARNESS),
        "bad_payload": human.sign(HUMAN_APPROVAL_TYPE, b"[1]"),
    }
    r = run(
        signer,
        harnesses=pair,
        approval=approvals[case],
        options=Options(trusted_human_keys=[human.public_key]),
        exceptions=["authority grew under approval by ana"],
        **decided(candidate),
    )
    assert (r.ok, r.decision) == (False, "not_established")
    assert r.reason == f"authority_exception_unverified:{reason}"


def test_harness_documents_must_be_the_decided_pair(signer: Signer) -> None:
    other, pair = harness_pair(prompt=PARENT_HARNESS.system_prompt + "\nSomething else.")
    assert run(signer, harnesses=pair).reason == "harness_not_the_decided_candidate"
    swapped = {"parent": HARNESSES["candidate"], "candidate": HARNESSES["candidate"]}
    assert run(signer, harnesses=swapped).reason == "harness_not_the_decided_parent"
    assert other.digest != CANDIDATE_HARNESS.digest


def test_a_single_harness_is_not_enough(signer: Signer) -> None:
    r = run(signer, harnesses={"parent": HARNESSES["parent"]})
    assert status(r)["gate:authority"] == "not_supplied"


def test_unknown_harness_role_gets_no_verdict(signer: Signer) -> None:
    r = verify_decision(sign(payload(), signer), signer.public_key, harnesses={"grandparent": {}})
    assert (r.decision, r.reason) == ("no_verdict", "invalid_harness_role")


def test_authority_recomputation_unit_cases() -> None:
    parent, cand = asdict(PARENT_HARNESS), asdict(CANDIDATE_HARNESS)
    assert rc.authority(parent, cand).passed
    altered = copy.deepcopy(cand)
    altered["guards"][0]["source"] = "def check(c, v, h):\n    return 'x'\n"
    assert not rc.authority(parent, altered).evidence["guards_kept_in_order"]
    reordered = copy.deepcopy(cand)
    reordered["guards"].reverse()
    assert not rc.authority(parent, reordered).evidence["guards_kept_in_order"]
    rewritten = copy.deepcopy(cand)
    rewritten["system_prompt"] = "Something else entirely."
    assert not rc.authority(parent, rewritten).evidence["prompt_prefix"]
    duplicated = copy.deepcopy(cand)
    duplicated["guards"].append(copy.deepcopy(duplicated["guards"][0]))
    assert not rc.authority(parent, duplicated).evidence["unique_ids"]
    procs = copy.deepcopy(cand)
    procs["procedures"] = [{"id": "p"}]
    assert not rc.authority(asdict(PARENT_HARNESS) | {"procedures": [{"id": "q"}]}, procs).passed
    forged = copy.deepcopy(cand)
    forged["digest"] = "0" * 64
    assert not rc.authority(parent, forged).evidence["digest_valid"]
    stranger = copy.deepcopy(cand)
    stranger["parent"] = "elsewhere"
    assert not rc.authority(parent, stranger).evidence["names_parent"]


def test_harness_digest_is_the_content_address() -> None:
    assert rc.harness_digest(asdict(CANDIDATE_HARNESS)) == CANDIDATE_HARNESS.digest


# --- replay -----------------------------------------------------------------------------------------------


def with_replay(**changes: Any) -> dict[str, Any]:
    return {f"replay:{GUARD_ID}": {**REPLAY_REPORT, **changes}, "overfit_lint": {"hits": []}}


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"passed": False}, "passed"),
        ({"snapshot_digest": "d" * 64}, "snapshot_digest"),
        ({"targets": [{"failure_id": "f1", "hit": True}, {"failure_id": "f2", "hit": False}]}, "targets_hit"),
        ({"targets": [{"failure_id": "f1", "hit": True}]}, "targets_hit"),
        ({"false_positives": [{"run_id": "r1", "seq": 3}]}, "false_fires"),
        ({"errors": [{"error": "boom"}]}, "errors"),
    ],
)
def test_a_recomputed_replay_that_differs_from_the_claim_is_contradicted(
    signer: Signer, changes: dict[str, Any], field: str
) -> None:
    r = run(signer, recomputed=with_replay(**changes))
    assert (r.decision, r.reason) == ("contradicted", f"replay_recompute_mismatch:{GUARD_ID}:{field}")


def test_a_rejected_replay_whose_false_fires_recompute_exactly_agrees() -> None:
    """The row the signer writes (admit.check_replay) names false fires as run_id@seq under false_fires;
    a recomputation that finds the same fires is no contradiction, and a different set is."""
    from ratchet.verify.recompute import replay_matches

    fires = [{"run_id": "r1", "seq": 3}, {"run_id": "r2", "seq": 7}]
    recomputed = {**REPLAY_REPORT, "passed": False, "false_positives": fires}
    row = {**replay_row(), "passed": False, "false_fires": ["r2@7", "r1@3"]}
    assert replay_matches(row, recomputed) == []
    assert replay_matches({**row, "false_fires": ["r1@3"]}, recomputed) == ["false_fires"]


def test_replay_snapshot_must_match_the_receipts_corpus_digest(signer: Signer) -> None:
    r = run(signer, corpus_digests={GUARD_ID: "e" * 64})
    assert r.reason == f"replay_snapshot_mismatch:{GUARD_ID}"


def test_replay_without_its_recorded_inputs_is_not_supplied(signer: Signer) -> None:
    r = run(signer, recomputed={"overfit_lint": {"hits": []}})
    assert status(r)["gate:replay"] == "not_supplied"
    assert "NOT SUPPLIED: gate:replay" in r.summary and not r.recomputation_complete


def test_replay_is_partial_when_only_some_guards_are_recomputed(signer: Signer) -> None:
    second = Guard("g-two", "two", NOOP, [])
    candidate, pair = harness_pair(guards=[*PARENT_HARNESS.guards, Guard(GUARD_ID, "n", NOOP, []), second])
    rows = [replay_row(), {**replay_row(), "guard_id": "g-two"}]
    checks = payload()["checks"]
    checks[0]["evidence"] = {"new_guards": [GUARD_ID, "g-two"], "reports": rows}
    stored = with_reports(**{"replay:g-two": {**REPLAY_REPORT, "guard_id": "g-two"}})
    r = run(
        signer,
        reports=stored,
        harnesses=pair,
        recomputed=with_replay(),
        checks=checks,
        corpus_digests={GUARD_ID: SNAPSHOT, "g-two": SNAPSHOT},
        **decided(candidate),
    )
    assert r.ok and status(r)["gate:replay"] == "partial"
    assert "PARTIAL: gate:replay" in r.summary


def test_a_candidate_with_no_new_guard_has_nothing_to_replay(signer: Signer) -> None:
    candidate, pair = harness_pair(guards=list(PARENT_HARNESS.guards))
    checks = payload()["checks"]
    checks[0]["evidence"] = {"new_guards": [], "reports": []}
    stored = {k: v for k, v in REPORTS.items() if not k.startswith("replay:")}
    r = run(
        signer,
        reports=stored,
        harnesses=pair,
        recomputed={"overfit_lint": {"hits": []}},
        checks=checks,
        corpus_digests={},
        **decided(candidate),
    )
    assert r.ok and gates_complete(r)
    assert next(i for i in r.recomputed if i["id"] == "gate:replay")["guards"] == 0


def test_claimed_new_guards_must_be_the_ones_in_the_harness_diff(signer: Signer) -> None:
    checks = payload()["checks"]
    checks[0]["evidence"] = {"new_guards": ["g-other"], "reports": [{**replay_row(), "guard_id": "g-other"}]}
    assert run(signer, checks=checks).reason == "replay_new_guards_mismatch"


def test_a_replay_check_cannot_pass_over_a_failed_guard_row(signer: Signer) -> None:
    checks = payload()["checks"]
    checks[0]["evidence"] = {"new_guards": [GUARD_ID], "reports": [{**replay_row(), "passed": False}]}
    assert run(signer, checks=checks).reason == "replay_check_passed_with_failed_guard"


def test_recomputed_results_must_be_json_objects(signer: Signer) -> None:
    r = verify_decision(
        sign(payload(), signer),
        signer.public_key,
        harnesses=HARNESSES,
        recomputed={f"replay:{GUARD_ID}": b"[]"},
    )
    assert r.reason == f"evidence_not_json:recomputed:replay:{GUARD_ID}"


# --- overfitting screen --------------------------------------------------------------------------------


def test_recomputed_overfit_hits_contradict_a_clean_claim(signer: Signer) -> None:
    r = run(signer, recomputed={**RECOMPUTED, "overfit_lint": {"hits": ["ord-1001 in prompt"]}})
    assert r.reason == "gate_recompute_mismatch:overfit_lint"


def test_malformed_overfit_result_is_contradicted(signer: Signer) -> None:
    r = run(signer, recomputed={**RECOMPUTED, "overfit_lint": {"hits": "none"}})
    assert r.reason == "evidence_not_json:recomputed:overfit_lint"


def test_overfit_without_its_inputs_is_not_supplied(signer: Signer) -> None:
    assert status(run(signer, recomputed={}))["gate:overfit_lint"] == "not_supplied"


# --- gates the consumer does not require ---------------------------------------------------------------


def test_a_gate_the_consumer_does_not_require_is_not_recomputed(signer: Signer) -> None:
    options = Options(required_gates=("heldout",))
    r = run(signer, options=options)
    assert set(status(r)) == {
        "signature",
        "payload_predicate",
        "admission_flag",
        "report_digests",
        "gate:heldout",
        *RUN_RECORD_ITEMS,
    }
    assert DEFAULT_GATES[0] == "replay"


def test_a_stored_report_that_is_not_json_is_contradicted(signer: Signer) -> None:
    from ratchet_cases.types import sha256_hex

    raw = b"\xff\xfe"
    digests = {**payload()["report_digests"], "parent_run_1": sha256_hex(raw)}
    r = verify_decision(
        sign(payload(report_digests=digests), signer),
        signer.public_key,
        reports={**REPORTS, "parent_run_1": raw},
    )
    assert r.reason == "evidence_not_json:report:parent_run_1"


def test_an_approval_whose_payload_is_not_json_is_not_established(
    signer: Signer, other_signer: Signer
) -> None:
    candidate, pair = harness_pair(tools=["refund", "wire"])
    r = run(
        signer,
        harnesses=pair,
        approval=other_signer.sign(HUMAN_APPROVAL_TYPE, b"\xff"),
        options=Options(trusted_human_keys=[other_signer.public_key]),
        exceptions=["authority grew under approval by ana"],
        **decided(candidate),
    )
    assert r.reason == "authority_exception_unverified:approval_payload"
