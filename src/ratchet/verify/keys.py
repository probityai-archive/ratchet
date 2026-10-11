"""The pinned decision-verification key: a committed public file, never a key carried inside a receipt."""

from __future__ import annotations

import base64
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_public_key

DEFAULT_KEY_PATH = Path(__file__).resolve().parents[3] / "keys" / "ratchet-decisions.pub"


class TrustedKeyError(ValueError):
    """The key file is missing, not PEM, or not an Ed25519 public key."""


def load_trusted_key(path: Path = DEFAULT_KEY_PATH) -> Ed25519PublicKey:
    """Load the pinned Ed25519 public key from a PEM (SubjectPublicKeyInfo) file."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise TrustedKeyError(f"cannot read trusted key {path}: {exc.strerror}") from exc
    try:
        key = load_pem_public_key(data)
    except ValueError as exc:
        raise TrustedKeyError(f"{path} is not a PEM public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise TrustedKeyError(f"{path} is not an Ed25519 public key")
    return key


def raw_b64(key: Ed25519PublicKey) -> str:
    return base64.b64encode(key.public_bytes(Encoding.Raw, PublicFormat.Raw)).decode("ascii")
