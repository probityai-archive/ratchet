from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

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


def _inprocess_referee(factory: str, config: Mapping[str, Any], **_options: Any) -> Any:
    from ratchet.improve.referee import RefereeCore, load_factory

    return RefereeCore(load_factory(factory)(config))


@pytest.fixture
def inprocess_referee() -> Callable[..., Any]:
    """Stands in for `ProcessReferee` in a driver's CLI test: the factory the driver names, run in this
    process through the same `RefereeCore`, so the driver's in-process patches (scripted endpoints, fake
    runners) reach the referee too. The process boundary itself is tested in tests/improve/test_referee.py."""
    return _inprocess_referee
