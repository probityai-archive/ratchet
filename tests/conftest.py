from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from ratchet.improve import receipt
from ratchet.improve.receipt import SIGNING_KEY_ENV, TRUSTED_KEY_ENV, Signer


@pytest.fixture(autouse=True)
def _no_keyring_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never read the real decision key from the developer's keyring."""
    monkeypatch.setattr(receipt, "_keyring_seed", lambda: "")


@pytest.fixture(autouse=True)
def decision_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Signer:
    """A fresh decision key for every test, set as the signing seed and pinned as the trusted public key."""
    signer = Signer.generate()
    pem = tmp_path / "decision-key.pub"
    pem.write_bytes(signer.public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))
    monkeypatch.setenv(SIGNING_KEY_ENV, signer.private_key.private_bytes_raw().hex())
    monkeypatch.setenv(TRUSTED_KEY_ENV, str(pem))
    return signer
