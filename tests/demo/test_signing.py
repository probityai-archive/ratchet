"""The demo signs with a key it generates, labels it, and never weakens the pinned-key path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ratchet.demo import signing
from ratchet.demo.signing import demo_signer, is_demo_signed, pubkey_path, signing_label
from ratchet.improve.receipt import SIGNING_KEY_ENV, ReceiptError, Signer, receipt
from ratchet.verify.keys import load_trusted_key
from ratchet.view.verify import trusted_keys, verify_envelope


class _Decision:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_payload(self) -> dict[str, object]:
        return self.payload


def test_the_demo_key_writes_only_its_public_half(tmp_path: Path) -> None:
    pem = tmp_path / "store" / "ratchet_demo-decision-key.pub"
    signer = demo_signer(pem)
    text = pem.read_text()
    assert text.startswith("-----BEGIN PUBLIC KEY-----") and "PRIVATE" not in text
    assert signing.key_id(load_trusted_key(pem)) == signer.keyid
    assert [p.name for p in pem.parent.iterdir()] == [pem.name]


def test_each_demo_run_gets_a_fresh_key(tmp_path: Path) -> None:
    assert demo_signer(tmp_path / "a.pub").keyid != demo_signer(tmp_path / "b.pub").keyid


def test_a_demo_decision_verifies_against_the_demo_pem_and_not_the_pinned_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pem = tmp_path / "demo.pub"
    signer = demo_signer(pem)
    env = receipt(
        _Decision({"admitted": True, "context": {"signing_key": signing_label(signer, pem)}}), signer
    )
    ok = verify_envelope(env, trusted_keys(None, {}, pem))
    assert ok.ok and ok.payload is not None and is_demo_signed(ok.payload)
    monkeypatch.delenv(SIGNING_KEY_ENV)
    pinned = verify_envelope(env, trusted_keys(None, {}))
    assert not pinned.ok and pinned.summary == "INVALID"


def test_the_production_signer_still_refuses_a_demo_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signer = demo_signer(tmp_path / "demo.pub")
    monkeypatch.setenv(SIGNING_KEY_ENV, signer.private_key.private_bytes_raw().hex())
    with pytest.raises(ReceiptError, match="is not the pinned decision key"):
        Signer.for_decisions()


def test_the_demo_refuses_a_key_equal_to_the_pinned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision_key: Signer
) -> None:
    monkeypatch.setattr(Signer, "generate", classmethod(lambda cls: decision_key))
    with pytest.raises(ReceiptError, match="equals the pinned decision key"):
        demo_signer(tmp_path / "demo.pub")


def test_the_label_names_the_key_and_the_file(tmp_path: Path) -> None:
    signer = Signer.generate()
    label = signing_label(signer, tmp_path / "x.pub")
    assert label["kind"] == "demo" and label["keyid"] == signer.keyid and label["pubkey_file"] == "x.pub"
    assert "NOT the pinned decision key" in str(label["note"])
    assert is_demo_signed({"context": {"signing_key": label}})
    assert not is_demo_signed({"context": {}}) and not is_demo_signed(json.loads("[]"))


def test_the_public_key_lives_in_the_demo_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(signing.DEMO_DIR_ENV, str(tmp_path))
    assert pubkey_path("ratchet_demo") == tmp_path / "ratchet_demo-decision-key.pub"
    monkeypatch.delenv(signing.DEMO_DIR_ENV)
    assert pubkey_path("ratchet_demo").parent.name == signing.DEFAULT_DEMO_DIR
