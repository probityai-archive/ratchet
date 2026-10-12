from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from rich.console import Console

from ratchet.improve.receipt import DECISION_TYPE, ReceiptError, Signer
from ratchet.types import canonical_json
from ratchet.view.verify import (
    _independent,
    render_verify,
    trusted_keys,
    verify_decision,
    verify_envelope,
)

from .fakes import decision_doc

SIGNER = Signer.from_seed_hex("11" * 32)
OTHER = Signer.from_seed_hex("22" * 32)


def payload(admitted: bool = True) -> dict[str, Any]:
    doc = decision_doc("p", "c", admitted)
    return {
        k: doc[k]
        for k in ("parent", "candidate", "candidate_digest", "admitted", "checks", "context", "decided_at")
    } | {"type": "decision"}


def signed_doc(signer: Signer = SIGNER, **over: Any) -> dict[str, Any]:
    body = payload()
    env = signer.sign(DECISION_TYPE, canonical_json(body))
    doc = decision_doc("p", "c", True)
    doc.update(body)
    doc["envelope"] = base64.b64encode(env).decode()
    doc["keyid"] = signer.keyid
    doc.update(over)
    return doc


def keys(signer: Signer = SIGNER) -> dict[str, Ed25519PublicKey]:
    return {signer.keyid: signer.public_key}


def render(res: object, name: str = "p->c") -> str:
    c = Console(width=120, record=True, file=io.StringIO())
    c.print(render_verify(res, name))  # type: ignore[arg-type]
    return c.export_text()


def test_valid_receipt_matching_stored_decision() -> None:
    res = verify_decision(signed_doc(), keys())
    assert res.ok
    assert res.summary == "VERIFIED"
    assert "matches the signed payload" in " ".join(res.detail)
    out = render(res)
    assert "VERIFIED" in out
    assert "ADMITTED p -> c" in out


def test_edited_stored_decision_is_tampered() -> None:
    res = verify_decision(signed_doc(admitted=False), keys())
    assert not res.ok
    assert res.summary == "TAMPERED"
    assert "admitted" in res.detail[0]


def test_wrong_key_is_invalid() -> None:
    res = verify_decision(signed_doc(), keys(OTHER))
    assert not res.ok
    assert res.summary == "INVALID"


def test_bad_signature_is_invalid() -> None:
    doc = signed_doc()
    env = base64.b64decode(doc["envelope"]).replace(b'"sig":"', b'"sig":"AAAA')
    doc["envelope"] = base64.b64encode(env).decode()
    assert verify_decision(doc, keys()).summary == "INVALID"


def test_no_trusted_key_is_not_verified() -> None:
    res = verify_decision(signed_doc(), {})
    assert res.summary == "NOT VERIFIED"
    assert "RATCHET_SIGNING_KEY" in res.detail[0]


def test_missing_and_garbled_envelopes() -> None:
    assert verify_decision(signed_doc(envelope=None), keys()).summary == "NO RECEIPT"
    assert verify_decision(signed_doc(envelope="!!!"), keys()).summary == "INVALID"


def test_verify_envelope_without_stored_decision() -> None:
    env = SIGNER.sign(DECISION_TYPE, canonical_json(payload()))
    assert verify_envelope(env, keys()).ok


def test_trusted_keys_from_hex_env_pem_pinned_or_nothing(tmp_path: Path) -> None:
    none = tmp_path / "absent.pub"
    raw = SIGNER.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    assert list(trusted_keys(raw, {}, pinned=none)) == [SIGNER.keyid]
    assert list(trusted_keys(None, {"RATCHET_SIGNING_KEY": "11" * 32}, pinned=none)) == [SIGNER.keyid]
    assert trusted_keys(None, {}, pinned=none) == {}
    with pytest.raises(ReceiptError, match="32 bytes"):
        trusted_keys("abcd", {}, pinned=none)
    pem = tmp_path / "k.pub"
    pem.write_bytes(OTHER.public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))
    assert list(trusted_keys(None, {}, pem, pinned=none)) == [OTHER.keyid]
    assert list(trusted_keys(None, {}, pinned=pem)) == [OTHER.keyid]
    pem.write_text("not a key")
    with pytest.raises(ReceiptError, match="not a PEM"):
        trusted_keys(None, {}, pem)
    with pytest.raises(ReceiptError, match="cannot read"):
        trusted_keys(None, {}, tmp_path / "missing.pub")


def test_the_independent_verifier_must_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    env = SIGNER.sign(DECISION_TYPE, canonical_json(payload()))
    res = verify_envelope(env, keys())
    assert "independent verifier" in " ".join(res.detail)

    class Disagrees:
        keyid = None
        reason = "signature_invalid"

    monkeypatch.setattr("ratchet.verify.dsse.verify_signature", lambda env, trusted: Disagrees())
    bad = verify_envelope(env, keys())
    assert not bad.ok
    assert "independent verifier rejects the signature" in bad.detail[0]
    assert _independent(b"{}", keys()) is not None


def test_render_of_failure_has_no_payload_line() -> None:
    out = render(verify_decision(signed_doc(), {}))
    assert "NOT VERIFIED" in out
    assert "ADMITTED" not in out


def test_a_countersigned_decision_names_the_pinned_key_and_the_run_key() -> None:
    from datetime import UTC, datetime

    from ratchet.improve.countersign import countersign_doc

    doc = signed_doc(OTHER)
    doc.update(countersign_doc(doc, SIGNER, datetime(2026, 10, 11, tzinfo=UTC)) or {})
    res = verify_decision(doc, keys())
    assert res.ok and res.keyid == SIGNER.keyid
    assert f"first signed by per-run key {OTHER.keyid[:12]}" in res.detail[-1]
    assert "countersigned by the pinned key on 2026-10-11" in render(res)
