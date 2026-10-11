from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from securesystemslib.dsse import Envelope
from securesystemslib.signer import SSlibKey

from ratchet.improve.receipt import (
    DECISION_TYPE,
    HUMAN_APPROVAL_TYPE,
    ReceiptError,
    Signer,
    pae,
    receipt,
    verify,
)


class _Decision:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_payload(self) -> dict[str, Any]:
        return self.payload


DECISION = _Decision({"admitted": True, "checks": [{"name": "replay", "passed": True, "evidence": {"n": 3}}]})


def test_pae_matches_the_dsse_spec_vector() -> None:
    # Test vector from the DSSE protocol document.
    assert pae("http://example.com/HelloWorld", b"hello world") == (
        b"DSSEv1 29 http://example.com/HelloWorld 11 hello world"
    )


def test_receipt_round_trips() -> None:
    signer = Signer.generate()
    env = receipt(DECISION, signer)
    assert verify(env, {signer.keyid: signer.public_key}) == DECISION.payload
    doc = json.loads(env)
    assert doc["payloadType"] == DECISION_TYPE
    assert doc["signatures"][0]["keyid"] == signer.keyid


def test_tampered_payload_is_rejected() -> None:
    signer = Signer.generate()
    doc = json.loads(receipt(DECISION, signer))
    forged = dict(DECISION.payload, admitted=False)
    doc["payload"] = base64.b64encode(json.dumps(forged).encode()).decode()
    with pytest.raises(ReceiptError, match="no signature verifies"):
        verify(json.dumps(doc).encode(), {signer.keyid: signer.public_key})


def test_type_confusion_is_rejected() -> None:
    signer = Signer.generate()
    doc = json.loads(receipt(DECISION, signer))
    doc["payloadType"] = HUMAN_APPROVAL_TYPE
    with pytest.raises(ReceiptError, match="payloadType"):
        verify(json.dumps(doc).encode(), {signer.keyid: signer.public_key})
    # Even when the verifier is asked for the other type, the signature binds the original type.
    with pytest.raises(ReceiptError, match="no signature verifies"):
        verify(json.dumps(doc).encode(), {signer.keyid: signer.public_key}, HUMAN_APPROVAL_TYPE)


def test_untrusted_key_is_rejected() -> None:
    signer, other = Signer.generate(), Signer.generate()
    env = receipt(DECISION, signer)
    with pytest.raises(ReceiptError, match="no signature verifies"):
        verify(env, {other.keyid: other.public_key})
    # A trusted key listed under the attacker's keyid still has to verify the signature.
    with pytest.raises(ReceiptError, match="no signature verifies"):
        verify(env, {signer.keyid: other.public_key})


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"not json", "not JSON"),
        (b"[1]", "not a JSON object"),
        (json.dumps({"payloadType": DECISION_TYPE, "payload": 5}).encode(), "base64"),
        (json.dumps({"payloadType": DECISION_TYPE, "payload": "!!"}).encode(), "not valid base64"),
        (
            json.dumps({"payloadType": DECISION_TYPE, "payload": "", "signatures": []}).encode(),
            "no signatures",
        ),
        (
            json.dumps({"payloadType": DECISION_TYPE, "payload": "", "signatures": ["x"]}).encode(),
            "no signature verifies",
        ),
    ],
)
def test_malformed_envelopes_are_rejected(raw: bytes, match: str) -> None:
    signer = Signer.generate()
    with pytest.raises(ReceiptError, match=match):
        verify(raw, {signer.keyid: signer.public_key})


def test_non_object_payload_is_rejected() -> None:
    signer = Signer.generate()
    env = signer.sign(DECISION_TYPE, b"[1,2]")
    with pytest.raises(ReceiptError, match="payload is not a JSON object"):
        verify(env, {signer.keyid: signer.public_key})


def test_seed_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    seed = "11" * 32
    a = Signer.from_seed_hex(seed)
    monkeypatch.setenv("RATCHET_SIGNING_KEY", seed)
    assert Signer.from_env().keyid == a.keyid
    monkeypatch.delenv("RATCHET_SIGNING_KEY")
    with pytest.raises(ReceiptError, match="not set"):
        Signer.from_env()
    with pytest.raises(ReceiptError, match="not hex"):
        Signer.from_seed_hex("zz")
    with pytest.raises(ReceiptError, match="32-byte"):
        Signer.from_seed_hex("11")


def test_independent_dsse_implementation_verifies_our_envelope() -> None:
    """An independently written DSSE implementation (a pinned dev dependency) accepts our envelope."""
    signer = Signer.generate()
    env = Envelope.from_dict(json.loads(receipt(DECISION, signer)))
    key = SSlibKey.from_crypto(signer.public_key, keyid=signer.keyid)
    accepted = env.verify([key], 1)
    assert list(accepted) == [signer.keyid]
    assert json.loads(env.pae()[env.pae().index(b"{") :]) == DECISION.payload
