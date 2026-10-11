"""A DSSE v1 envelope reader and Ed25519 signature check, written independently of the signer.

The signer lives in `ratchet.improve.receipt`. This module shares no code with it, so a defect in one is not
silently a defect in the other: the tests sign with the signer and verify here, and the reverse.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from ratchet.types import sha256_hex

MAX_ENVELOPE_BYTES = 4 * 1024 * 1024


class EnvelopeError(ValueError):
    """The bytes are not a well-formed DSSE envelope. No claim can be read from them."""


@dataclass(frozen=True)
class Signature:
    keyid: str
    sig: bytes


@dataclass(frozen=True)
class Envelope:
    payload_type: str
    payload: bytes
    signatures: tuple[Signature, ...]


def keyid_of(key: Ed25519PublicKey) -> str:
    """The SHA-256 of the raw 32-byte public key, hex. The same naming the signer uses."""
    return sha256_hex(key.public_bytes(Encoding.Raw, PublicFormat.Raw))


def pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE pre-authentication encoding: "DSSEv1" SP LEN(type) SP type SP LEN(body) SP body."""
    t = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)


def _b64(value: object, label: str) -> bytes:
    if not isinstance(value, str):
        raise EnvelopeError(f"{label} must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EnvelopeError(f"{label} is not valid base64") from exc


def _signature(entry: object) -> Signature:
    if not isinstance(entry, dict) or not isinstance(entry.get("keyid"), str):
        raise EnvelopeError("signature entry needs a string keyid and a sig")
    return Signature(entry["keyid"], _b64(entry.get("sig"), "sig"))


def parse(envelope: bytes) -> Envelope:
    """Read the envelope structure. Nothing is trusted yet: the signature is checked by `verify_signature`."""
    if len(envelope) > MAX_ENVELOPE_BYTES:
        raise EnvelopeError(f"envelope exceeds {MAX_ENVELOPE_BYTES} bytes")
    try:
        doc = json.loads(envelope)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvelopeError("envelope is not JSON") from exc
    if not isinstance(doc, dict):
        raise EnvelopeError("envelope is not a JSON object")
    payload_type = doc.get("payloadType")
    if not isinstance(payload_type, str) or not payload_type:
        raise EnvelopeError("envelope has no payloadType")
    entries = doc.get("signatures")
    if not isinstance(entries, list) or not entries:
        raise EnvelopeError("envelope carries no signatures")
    return Envelope(payload_type, _b64(doc.get("payload"), "payload"), tuple(_signature(e) for e in entries))


@dataclass(frozen=True)
class SignatureResult:
    """`keyid` is the verifying trusted key, or None. `reason` is set exactly when `keyid` is None."""

    keyid: str | None
    reason: str | None


def verify_signature(env: Envelope, trusted: Mapping[str, Ed25519PublicKey]) -> SignatureResult:
    """Succeed when one signature verifies under the trusted key it names.

    A signature naming a key that is not trusted is never tried, so a receipt cannot choose its own verifier.
    `untrusted_key` means no signature named a trusted key; `signature_invalid` means one did and failed.
    """
    message = pae(env.payload_type, env.payload)
    named_trusted = False
    for entry in env.signatures:
        key = trusted.get(entry.keyid)
        if key is None:
            continue
        named_trusted = True
        try:
            key.verify(entry.sig, message)
        except InvalidSignature:
            continue
        return SignatureResult(entry.keyid, None)
    return SignatureResult(None, "signature_invalid" if named_trusted else "untrusted_key")
