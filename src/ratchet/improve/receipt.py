"""Signed receipts: a DSSE envelope over the canonical JSON of a decision, signed with Ed25519.

The envelope follows the DSSE v1 envelope format: the signature covers
PAE(payloadType, payload), never the raw payload, so a payload cannot be replayed under another type.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from ratchet.types import canonical_json, sha256_hex

DECISION_TYPE = "application/vnd.ratchet.decision+json"
HUMAN_APPROVAL_TYPE = "application/vnd.ratchet.human-approval+json"
SIGNING_KEY_ENV = "RATCHET_SIGNING_KEY"


class ReceiptError(ValueError):
    """An envelope that is malformed, of the wrong type, or not signed by a trusted key."""


class Payload(Protocol):
    def to_payload(self) -> dict[str, Any]: ...


def pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE pre-authentication encoding: "DSSEv1" SP LEN(type) SP type SP LEN(body) SP body."""
    t = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)


def public_key_bytes(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(Encoding.Raw, PublicFormat.Raw)


def key_id(key: Ed25519PublicKey) -> str:
    """The SHA-256 of the raw 32-byte public key, hex. Names the key without carrying it."""
    return sha256_hex(public_key_bytes(key))


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

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    @property
    def keyid(self) -> str:
        return key_id(self.public_key)

    def sign(self, payload_type: str, body: bytes) -> bytes:
        sig = self.private_key.sign(pae(payload_type, body))
        envelope = {
            "payload": base64.b64encode(body).decode("ascii"),
            "payloadType": payload_type,
            "signatures": [{"keyid": self.keyid, "sig": base64.b64encode(sig).decode("ascii")}],
        }
        return canonical_json(envelope)


def receipt(decision: Payload, signer: Signer) -> bytes:
    """A signed DSSE envelope over the canonical JSON of the decision and its evidence."""
    return signer.sign(DECISION_TYPE, canonical_json(decision.to_payload()))


def _b64(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise ReceiptError(f"{label} must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ReceiptError(f"{label} is not valid base64") from exc


def verify(
    envelope: bytes, trusted: Mapping[str, Ed25519PublicKey], payload_type: str = DECISION_TYPE
) -> dict[str, Any]:
    """Return the payload when at least one signature verifies under a trusted key; raise otherwise.

    `trusted` maps keyid to public key. A signature whose keyid is unknown is ignored, never trusted.
    """
    try:
        doc = json.loads(envelope)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptError("envelope is not JSON") from exc
    if not isinstance(doc, dict):
        raise ReceiptError("envelope is not a JSON object")
    if doc.get("payloadType") != payload_type:
        raise ReceiptError(f"payloadType is {doc.get('payloadType')!r}, expected {payload_type!r}")
    body = _b64(doc.get("payload"), "payload")
    signatures = doc.get("signatures")
    if not isinstance(signatures, list) or not signatures:
        raise ReceiptError("envelope carries no signatures")
    message = pae(payload_type, body)
    for entry in signatures:
        if not isinstance(entry, dict):
            continue
        key = trusted.get(str(entry.get("keyid")))
        if key is None:
            continue
        try:
            key.verify(_b64(entry.get("sig"), "sig"), message)
        except InvalidSignature:
            continue
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ReceiptError("payload is not a JSON object")
        return payload
    raise ReceiptError("no signature verifies under a trusted key")
