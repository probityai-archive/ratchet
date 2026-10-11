from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from ratchet.improve.receipt import Signer
from ratchet.verify import DEFAULT_GATES, PREDICATE_URI, Options, verify_decision

from .helpers import CANDIDATE_HARNESS, HARNESSES, PARENT_HARNESS, RECOMPUTED, REPORTS, b64, payload, sign


def check(envelope: bytes, signer: Signer, **kwargs: Any) -> Any:
    return verify_decision(envelope, signer.public_key, **kwargs)


def strict(**kwargs: Any) -> Options:
    return Options(**kwargs)


def test_valid_receipt_verifies_with_every_report(signer: Signer) -> None:
    r = check(sign(payload(), signer), signer, reports=REPORTS, options=strict(require_reports=True))
    assert (r.ok, r.decision, r.reason) == (True, "supported", "admitted_receipt_verified")
    assert r.admission_verified
    assert (r.keyid, r.parent, r.candidate) == (
        signer.keyid,
        PARENT_HARNESS.version,
        CANDIDATE_HARNESS.version,
    )
    assert r.reports_summary == "5 of 5 named reports checked"
    assert r.packet is not None and len(r.packet["policy_sha256"]) == 64
    assert [c["id"] for c in r.checks][:5] == [
        "envelope",
        "payload_type",
        "signature",
        "predicate",
        "payload_schema",
    ]


def test_valid_receipt_without_reports_reports_the_denominator(signer: Signer) -> None:
    r = check(sign(payload(), signer), signer)
    assert r.ok and r.reports_summary == "0 of 5 named reports checked"
    assert r.packet is not None and r.packet["scope"]["reports"]["unverified"] == [
        "candidate_confirm_run",
        "candidate_run",
        "parent_run_1",
        "parent_run_2",
        "replay:g-new",
    ]


def test_missing_report_fails_when_reports_are_required(signer: Signer) -> None:
    only = {"parent_run_1": REPORTS["parent_run_1"]}
    r = check(sign(payload(), signer), signer, reports=only, options=strict(require_reports=True))
    assert (r.ok, r.decision, r.reason) == (
        False,
        "not_established",
        "report_not_supplied:candidate_confirm_run",
    )


def test_verified_rejection_is_ok_but_not_an_admission(signer: Signer) -> None:
    checks = [{"name": g, "passed": g != "heldout", "evidence": {}} for g in DEFAULT_GATES]
    r = check(sign(payload(admitted=False, checks=checks), signer), signer)
    assert (r.ok, r.reason, r.admitted, r.admission_verified) == (
        True,
        "rejection_receipt_verified",
        False,
        False,
    )


def test_tampered_payload_fails_closed(signer: Signer) -> None:
    good = json.loads(sign(payload(admitted=False, checks=[]), signer))
    forged = payload(admitted=True)
    good["payload"] = b64(json.dumps(forged, sort_keys=True, separators=(",", ":")).encode())
    r = check(json.dumps(good).encode(), signer)
    assert (r.ok, r.decision, r.reason) == (False, "contradicted", "signature_invalid")
    assert r.admitted is None and not r.admission_verified


def test_tampered_signature_fails_closed(signer: Signer) -> None:
    doc = json.loads(sign(payload(), signer))
    sig = bytearray(base64.b64decode(doc["signatures"][0]["sig"]))
    sig[0] ^= 0x01
    doc["signatures"][0]["sig"] = b64(bytes(sig))
    r = check(json.dumps(doc).encode(), signer)
    assert (r.ok, r.decision, r.reason) == (False, "contradicted", "signature_invalid")


def test_wrong_trusted_key_fails_closed(signer: Signer, other_signer: Signer) -> None:
    r = verify_decision(sign(payload(), signer), other_signer.public_key)
    assert (r.ok, r.decision, r.reason) == (False, "contradicted", "untrusted_key")


def test_a_receipt_cannot_name_the_pinned_key_for_a_foreign_signature(
    signer: Signer, other_signer: Signer
) -> None:
    doc = json.loads(sign(payload(), other_signer))
    doc["signatures"][0]["keyid"] = signer.keyid
    r = check(json.dumps(doc).encode(), signer)
    assert r.reason == "signature_invalid"


def test_unsupported_payload_type_is_not_established(signer: Signer) -> None:
    r = check(sign(payload(), signer, "application/vnd.ratchet.human-approval+json"), signer)
    assert (r.ok, r.decision, r.reason) == (False, "not_established", "unsupported_payload_type")


@pytest.mark.parametrize("predicate", ["https://example.invalid/other/v1", None, 7])
def test_unsupported_predicate_is_not_established(signer: Signer, predicate: Any) -> None:
    body = payload(predicateType=predicate)
    r = check(sign(body, signer), signer)
    assert (r.ok, r.decision, r.reason) == (False, "not_established", "unsupported_predicate")


def test_missing_predicate_is_unsupported(signer: Signer) -> None:
    body = payload()
    del body["predicateType"]
    assert check(sign(body, signer), signer).reason == "unsupported_predicate"


def test_signed_payload_that_is_not_an_object_is_not_established(signer: Signer) -> None:
    r = check(signer.sign("application/vnd.ratchet.decision+json", b"[1, 2]"), signer)
    assert (r.decision, r.reason) == ("not_established", "payload_not_json_object")
    r = check(signer.sign("application/vnd.ratchet.decision+json", b"\xff\xfe"), signer)
    assert r.reason == "payload_not_json_object"


def test_report_digest_mismatch_fails_closed(signer: Signer) -> None:
    altered = {**REPORTS["candidate_run"], "harness_version": "forged"}
    r = check(sign(payload(), signer), signer, reports={**REPORTS, "candidate_run": altered})
    assert (r.ok, r.decision, r.reason) == (False, "contradicted", "report_digest_mismatch:candidate_run")


def test_report_the_receipt_does_not_name_fails_closed(signer: Signer) -> None:
    r = check(sign(payload(), signer), signer, reports={"extra": {"x": 1}})
    assert (r.ok, r.reason) == (False, "report_not_named:extra")


def test_reports_may_be_exact_stored_bytes(signer: Signer) -> None:
    from ratchet.types import canonical_json

    raw = {n: canonical_json(r) for n, r in REPORTS.items()}
    assert check(sign(payload(), signer), signer, reports=raw).reports_summary.startswith("5 of 5")
    raw["parent_run_1"] += b" "
    assert check(sign(payload(), signer), signer, reports=raw).reason == "report_digest_mismatch:parent_run_1"


@pytest.mark.parametrize("name", ["", "bad\nname", "bad\x00name", "x" * 201])
def test_unsafe_report_name_is_refused_without_a_verdict(signer: Signer, name: str) -> None:
    r = check(sign(payload(), signer), signer, reports={name: {}})
    assert (r.ok, r.decision, r.reason) == (False, "no_verdict", f"invalid_report_name:{name}")


@pytest.mark.parametrize(
    "field",
    [
        "parent",
        "candidate",
        "candidate_digest",
        "evaluator",
        "corpus_digests",
        "exceptions",
        "admitted",
        "checks",
        "report_digests",
    ],
)
def test_missing_required_field_is_not_established(signer: Signer, field: str) -> None:
    body = payload()
    del body[field]
    r = check(sign(body, signer), signer)
    assert (r.ok, r.decision, r.reason) == (False, "not_established", f"payload_field_invalid:{field}")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluator", {"version": "e1", "components": {"suite_file": None}}),
        ("evaluator", {"version": "e1", "components": {"suite_file": "ABC"}}),
        ("evaluator", {"version": "e1"}),
        ("evaluator", {"version": "", "components": {"suite_file": "a" * 64}}),
        ("evaluator", "e1"),
        ("corpus_digests", {"g": "short"}),
        ("corpus_digests", []),
        ("candidate_digest", "g" * 64),
        ("admitted", "yes"),
        ("checks", [{"name": "replay"}]),
        ("checks", "none"),
        ("report_digests", {}),
        ("report_digests", {"a": "short"}),
        ("report_digests", {"bad\nname": "a" * 64}),
        ("parent", ""),
        ("exceptions", "none"),
        ("exceptions", [{"kind": "x"}]),
    ],
)
def test_malformed_field_is_not_established(signer: Signer, field: str, value: Any) -> None:
    r = check(sign(payload(**{field: value}), signer), signer)
    assert r.decision == "not_established"
    assert r.reason.startswith(f"payload_field_invalid:{field}")


def test_missing_gate_is_not_established(signer: Signer) -> None:
    checks = [{"name": g, "passed": True, "evidence": {}} for g in DEFAULT_GATES if g != "authority"]
    r = check(sign(payload(checks=checks), signer), signer)
    assert (r.decision, r.reason) == ("not_established", "gate_missing:authority")


def test_admitted_with_a_failed_gate_is_contradicted(signer: Signer) -> None:
    checks = [{"name": g, "passed": g != "replay", "evidence": {}} for g in DEFAULT_GATES]
    r = check(sign(payload(admitted=True, checks=checks), signer), signer)
    assert (r.decision, r.reason) == ("contradicted", "admitted_with_failed_gate:replay")


def test_rejected_with_every_gate_passed_is_contradicted(signer: Signer) -> None:
    r = check(sign(payload(admitted=False), signer), signer)
    assert (r.decision, r.reason) == ("contradicted", "rejected_with_all_gates_passed")


def test_the_consumer_chooses_the_required_gates(signer: Signer) -> None:
    checks = [{"name": "replay", "passed": True, "evidence": {}}]
    r = check(sign(payload(checks=checks), signer), signer, options=strict(required_gates=("replay",)))
    assert r.ok
    assert check(sign(payload(checks=checks), signer), signer).reason == "gate_missing:heldout"


@pytest.mark.parametrize(
    "envelope",
    [
        b"not json",
        b"\xff\xfe",
        b"[]",
        b"{}",
        b'{"payloadType": "x", "payload": "", "signatures": []}',
        b'{"payloadType": "x", "payload": "***", "signatures": [{"keyid": "k", "sig": ""}]}',
        b'{"payloadType": "x", "payload": 5, "signatures": [{"keyid": "k", "sig": ""}]}',
        b'{"payloadType": "x", "payload": "", "signatures": [{"keyid": "k", "sig": "***"}]}',
        b'{"payloadType": "x", "payload": "", "signatures": ["bad"]}',
        b'{"payloadType": "", "payload": "", "signatures": [{"keyid": "k", "sig": ""}]}',
    ],
)
def test_malformed_envelope_gets_no_verdict(signer: Signer, envelope: bytes) -> None:
    r = check(envelope, signer)
    assert (r.ok, r.decision) == (False, "no_verdict")
    assert r.reason.startswith("malformed envelope: ")
    assert not r.admission_verified


def test_oversized_envelope_gets_no_verdict(signer: Signer) -> None:
    r = check(b" " * (4 * 1024 * 1024 + 1), signer)
    assert r.decision == "no_verdict" and "exceeds" in r.reason


def test_extra_unknown_signature_is_ignored_when_a_trusted_one_verifies(
    signer: Signer, other_signer: Signer
) -> None:
    doc = json.loads(sign(payload(), signer))
    doc["signatures"].insert(0, json.loads(sign(payload(), other_signer))["signatures"][0])
    assert check(json.dumps(doc).encode(), signer).ok


def test_to_dict_is_json_serialisable(signer: Signer) -> None:
    out = check(sign(payload(), signer), signer, reports=REPORTS).to_dict()
    assert out["recomputation_complete"] is False
    assert json.loads(json.dumps(out))["admission_verified"] is True
    assert out["policy_sha256"] and out["reports"] == "5 of 5 named reports checked"


def test_exceptions_are_carried_through(signer: Signer) -> None:
    r = check(sign(payload(exceptions=["authority grew under approval x"]), signer), signer)
    assert r.ok and r.packet is not None
    assert r.packet["scope"]["exceptions"] == ["authority grew under approval x"]


def test_any_of_several_pinned_keys_verifies_during_a_rotation(signer: Signer, other_signer: Signer) -> None:
    keys = [other_signer.public_key, signer.public_key]
    assert verify_decision(sign(payload(), signer), keys).keyid == signer.keyid
    assert verify_decision(sign(payload(), other_signer), keys).keyid == other_signer.keyid


def test_an_empty_key_list_trusts_nothing(signer: Signer) -> None:
    r = verify_decision(sign(payload(), signer), [])
    assert (r.ok, r.decision, r.reason) == (False, "no_verdict", "no_trusted_key")


def test_decision_object_from_the_admission_lane_verifies_with_the_predicate_added(signer: Signer) -> None:
    from ratchet.improve.admit import Check, Decision

    body = payload()
    decision = Decision(
        parent=body["parent"],
        candidate=body["candidate"],
        candidate_digest=body["candidate_digest"],
        admitted=True,
        checks=[Check(c["name"], c["passed"], c["evidence"]) for c in body["checks"]],
        evaluator=body["evaluator"],
        corpus_digests=body["corpus_digests"],
        report_digests=body["report_digests"],
        decided_at="2026-10-11T12:00:00+00:00",
    )
    signed = sign({**decision.to_payload(), "predicateType": PREDICATE_URI}, signer)
    r = check(signed, signer, reports=REPORTS, harnesses=HARNESSES, recomputed=RECOMPUTED)
    assert (r.ok, r.admission_verified, r.recomputation_complete) == (True, True, True)
    assert check(sign(decision.to_payload(), signer), signer).reason == "unsupported_predicate"
