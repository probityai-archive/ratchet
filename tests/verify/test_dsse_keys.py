from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from securesystemslib.dsse import Envelope as SslibEnvelope
from securesystemslib.signer import SSlibKey  # type: ignore[attr-defined]

from ratchet.improve.receipt import Signer
from ratchet.verify import DEFAULT_KEY_PATH, TrustedKeyError, dsse, load_trusted_key
from ratchet.verify.dsse import keyid_of
from ratchet.verify.keys import raw_b64

from .helpers import payload, sign


def test_pae_matches_the_dsse_spec_vector() -> None:
    assert dsse.pae("http://example.com/HelloWorld", b"hello world") == (
        b"DSSEv1 29 http://example.com/HelloWorld 11 hello world"
    )


def test_keyid_agrees_with_the_signer(signer: Signer) -> None:
    assert dsse.keyid_of(signer.public_key) == signer.keyid == keyid_of(signer.public_key)


def test_envelope_from_the_signer_verifies_under_securesystemslib(signer: Signer) -> None:
    """A second, unrelated DSSE implementation accepts the envelope this lane accepts."""
    raw = sign(payload(), signer)
    env = SslibEnvelope.from_dict(json.loads(raw))
    key = SSlibKey.from_crypto(signer.public_key, keyid=signer.keyid)
    assert list(env.verify([key], 1)) == [signer.keyid]
    assert dsse.verify_signature(dsse.parse(raw), {signer.keyid: signer.public_key}).keyid == signer.keyid


def test_committed_key_loads_and_is_ed25519() -> None:
    key = load_trusted_key()
    assert DEFAULT_KEY_PATH.name == "ratchet-decisions.pub"
    assert len(dsse.keyid_of(key)) == 64
    assert len(raw_b64(key)) == 44


def test_a_receipt_signed_by_another_key_does_not_verify_under_the_committed_key(signer: Signer) -> None:
    from ratchet.verify import verify_decision

    r = verify_decision(sign(payload(), signer), load_trusted_key())
    assert r.reason == "untrusted_key"


def test_missing_key_file(tmp_path: Path) -> None:
    with pytest.raises(TrustedKeyError, match="cannot read"):
        load_trusted_key(tmp_path / "absent.pub")


def test_key_file_that_is_not_pem(tmp_path: Path) -> None:
    path = tmp_path / "k.pub"
    path.write_bytes(b"deadbeef")
    with pytest.raises(TrustedKeyError, match="not a PEM"):
        load_trusted_key(path)


def test_key_file_that_is_not_ed25519(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives.serialization import load_pem_public_key  # noqa: F401

    pem = (
        rsa.generate_private_key(65537, 2048)
        .public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
    path = tmp_path / "rsa.pub"
    path.write_bytes(pem)
    with pytest.raises(TrustedKeyError, match="not an Ed25519"):
        load_trusted_key(path)


def test_signature_with_an_unknown_keyid_is_never_tried(signer: Signer) -> None:
    env = dsse.parse(sign(payload(), signer))
    result = dsse.verify_signature(env, {})
    assert (result.keyid, result.reason) == (None, "untrusted_key")


def test_report_digest_agrees_with_the_admission_lane() -> None:
    from ratchet.improve.admit import report_digest as theirs
    from ratchet.verify import report_digest as ours

    doc = {"type": "eval", "results": [{"case_id": "x", "passed": True}], "n": 1.5, "é": "ü"}
    assert ours(doc) == theirs(doc)
