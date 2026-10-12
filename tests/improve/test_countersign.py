from __future__ import annotations

import base64
import json
from datetime import UTC, datetime

import pytest
from ratchet_cases.types import canonical_json

from ratchet.improve.countersign import countersign_doc, payload_matches
from ratchet.improve.receipt import (
    DECISION_TYPE,
    HUMAN_APPROVAL_TYPE,
    ReceiptError,
    Signer,
    countersign,
    verify,
)

PAYLOAD = {"admitted": False, "parent": "p", "candidate": "c"}


def _stored(signer: Signer) -> dict[str, object]:
    env = signer.sign(DECISION_TYPE, canonical_json(PAYLOAD))
    return {
        **PAYLOAD,
        "_id": "p->c",
        "envelope": base64.b64encode(env).decode("ascii"),
        "keyid": signer.keyid,
    }


def test_countersign_keeps_the_run_signature_and_adds_the_pinned_one(decision_key: Signer) -> None:
    run_key = Signer.generate()
    original = run_key.sign(DECISION_TYPE, canonical_json(PAYLOAD))
    signed = countersign(original, decision_key)
    assert [s["keyid"] for s in json.loads(signed)["signatures"]] == [run_key.keyid, decision_key.keyid]
    assert verify(signed, {decision_key.keyid: decision_key.public_key}) == PAYLOAD
    assert verify(signed, {run_key.keyid: run_key.public_key}) == PAYLOAD
    assert countersign(signed, decision_key) == signed


def test_countersign_refuses_a_non_decision_envelope(decision_key: Signer) -> None:
    other = Signer.generate().sign(HUMAN_APPROVAL_TYPE, b"{}")
    with pytest.raises(ReceiptError, match="only a decision envelope"):
        countersign(other, decision_key)


def test_countersign_doc_makes_a_run_signed_decision_verify_under_the_pinned_key(
    decision_key: Signer,
) -> None:
    doc = _stored(Signer.generate())
    assert payload_matches(doc)
    update = countersign_doc(doc, decision_key, datetime(2026, 10, 11, tzinfo=UTC))
    assert update is not None and update["countersigned"]["keyid"] == decision_key.keyid
    env = base64.b64decode(update["envelope"])
    assert verify(env, {decision_key.keyid: decision_key.public_key})["candidate"] == "c"
    assert countersign_doc({**doc, **update}, decision_key, datetime.now(UTC)) is None


def test_an_edited_decision_does_not_match_its_payload() -> None:
    doc = _stored(Signer.generate())
    assert not payload_matches({**doc, "admitted": True})
