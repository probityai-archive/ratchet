"""Signed receipts: a DSSE envelope over the RFC 8785 canonical JSON of a decision, signed with Ed25519.

The envelope follows the DSSE v1 envelope format: the signature covers PAE(payloadType, payload), never the
raw payload, so a payload cannot be replayed under another type. Signing and verification go through
`ratchet.verify.dsse`, the one DSSE implementation in this package (the `jcs_admit.dsse` binding).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from ratchet.canonical import canonical_json
from ratchet.verify import dsse

DECISION_TYPE = "application/vnd.ratchet.decision+json"
HUMAN_APPROVAL_TYPE = "application/vnd.ratchet.human-approval+json"
SIGNING_KEY_ENV = "RATCHET_SIGNING_KEY"
TRUSTED_KEY_ENV = "RATCHET_DECISION_PUBKEY"
KEYRING_SERVICE = "ratchet"
KEYRING_ACCOUNT = "decision-signing-key"


class ReceiptError(ValueError):
    """An envelope that is malformed, of the wrong type, or not signed by a trusted key."""


class Payload(Protocol):
    def to_payload(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Signer:
    private_key: Ed25519PrivateKey

    @classmethod
    def generate(cls) -> Signer:
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_seed_hex(cls, seed_hex: str) -> Signer:
        try:
            seed = bytes.fromhex(seed_hex.strip())
        except ValueError as exc:
            raise ReceiptError("signing key is not hex") from exc
        if len(seed) != 32:
            raise ReceiptError("signing key must be a 32-byte Ed25519 seed")
        return cls(Ed25519PrivateKey.from_private_bytes(seed))

    @classmethod
    def from_env(cls) -> Signer:
        """Load the seed from RATCHET_SIGNING_KEY. Keys never live in the repository."""
        value = os.environ.get(SIGNING_KEY_ENV)
        if not value:
            raise ReceiptError(f"{SIGNING_KEY_ENV} is not set")
        return cls.from_seed_hex(value)

    @classmethod
    def for_decisions(cls, trusted_path: Path | None = None) -> Signer:
        """The signer for admission decisions: the pinned decision key, or an error.

        The seed comes from RATCHET_SIGNING_KEY, else the keyring (service `ratchet`, account
        `decision-signing-key`). Its public key must equal the pinned key that `ratchet verify` trusts:
        `trusted_path`, else the PEM named by RATCHET_DECISION_PUBKEY, else the repository's
        `keys/ratchet-decisions.pub`. There is no fallback to a fresh key, because a decision signed by an
        unpinned key can never be verified.
        """
        from ratchet.verify import keys  # deferred: ratchet.verify imports this module

        seed = os.environ.get(SIGNING_KEY_ENV) or _keyring_seed()
        if not seed:
            raise ReceiptError(
                f"no decision signing key: set {SIGNING_KEY_ENV} or store keyring "
                f"service={KEYRING_SERVICE} account={KEYRING_ACCOUNT}"
            )
        signer = cls.from_seed_hex(seed)
        path = trusted_path or Path(os.environ.get(TRUSTED_KEY_ENV) or keys.DEFAULT_KEY_PATH)
        try:
            pinned = keys.load_trusted_key(path)
        except keys.TrustedKeyError as exc:
            raise ReceiptError(f"no pinned decision key ({exc}); set {TRUSTED_KEY_ENV} to its PEM") from exc
        if dsse.keyid_of(pinned) != signer.keyid:
            raise ReceiptError(
                f"signing key {signer.keyid[:12]} is not the pinned decision key "
                f"{dsse.keyid_of(pinned)[:12]} ({path})"
            )
        return signer

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    @property
    def keyid(self) -> str:
        return dsse.keyid_of(self.public_key)

    def sign(self, payload_type: str, body: bytes) -> bytes:
        return dsse.sign(payload_type, body, [self.private_key])


def _keyring_seed() -> str:
    """The decision seed from the GNOME keyring; empty when the keyring or the entry is absent."""
    try:
        out = subprocess.run(
            ["secret-tool", "lookup", "service", KEYRING_SERVICE, "account", KEYRING_ACCOUNT],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip()


def receipt(decision: Payload, signer: Signer) -> bytes:
    """A signed DSSE envelope over the canonical JSON of the decision and its evidence."""
    return signer.sign(DECISION_TYPE, canonical_json(decision.to_payload()))


def countersign(envelope: bytes, signer: Signer) -> bytes:
    """Add `signer`'s signature over the same payload, keeping every signature already on the envelope.

    DSSE allows several signatures on one envelope. This lets the pinned decision key vouch for a decision
    that a run signed with a per-run key: the run's own signature and keyid stay as the first record.
    Countersigning twice with the same key returns the envelope unchanged.
    """
    try:
        env = dsse.parse(envelope)
    except dsse.EnvelopeError as exc:
        raise ReceiptError(str(exc)) from exc
    if env.payload_type != DECISION_TYPE:
        raise ReceiptError("only a decision envelope can be countersigned")
    if any(s.keyid == signer.keyid for s in env.signatures):
        return envelope
    doc = json.loads(envelope)
    doc["signatures"].append(dsse.signature_entry(DECISION_TYPE, env.payload, signer.private_key))
    return canonical_json(doc)


def verify(
    envelope: bytes,
    trusted: Mapping[str, Ed25519PublicKey],
    payload_type: str = DECISION_TYPE,
    *,
    threshold: int = 1,
) -> dict[str, Any]:
    """Return the payload when at least `threshold` trusted keys signed it; raise otherwise.

    `trusted` maps keyid to public key. A signature whose keyid is unknown is ignored, never trusted. With
    `threshold=2` a countersigned decision is accepted only when both of its keys verify.
    """
    try:
        env = dsse.parse(envelope)
    except dsse.EnvelopeError as exc:
        raise ReceiptError(str(exc)) from exc
    if env.payload_type != payload_type:
        raise ReceiptError(f"payloadType is {env.payload_type!r}, expected {payload_type!r}")
    result = dsse.verify_signature(env, trusted, threshold)
    if result.reason is not None:
        need = "a trusted key" if threshold == 1 else f"{threshold} trusted keys"
        raise ReceiptError(f"no signature verifies under {need}")
    payload = json.loads(env.payload)
    if not isinstance(payload, dict):
        raise ReceiptError("payload is not a JSON object")
    return payload
