"""DSSE v1 envelopes for Ratchet, through the `probityai.dsse` binding of the `dsse` crate.

Every pre-authentication encoding, signature and signature check goes through the crate; nothing here
computes PAE or calls an Ed25519 primitive. This module adds what Ratchet's reason codes need around it:
a structural read of the envelope (so a malformed envelope is refused before any key is tried), and the
rule that only a key the caller trusts AND a signature entry names is tried, so a receipt cannot choose
its own verifier and `untrusted_key` stays distinct from `signature_invalid`.

Keys are named by the SHA-256 of their raw 32-byte public key, hex. An independent Python reader of the
same envelopes is kept in the tests (`tests/verify/python_dsse.py`) as a differential check.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from probityai import dsse as native
from probityai import jcs
from ratchet_cases.types import sha256_hex

MAX_ENVELOPE_BYTES = 4 * 1024 * 1024

pae = native.pae


class EnvelopeError(ValueError):
    """The bytes are not a well-formed DSSE envelope. No claim can be read from them."""


@dataclass(frozen=True)
class Signature:
    keyid: str
    sig: bytes


@dataclass(frozen=True)
class Envelope:
    raw: bytes
    payload_type: str
    payload: bytes
    signatures: tuple[Signature, ...]


def raw_public(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(Encoding.Raw, PublicFormat.Raw)


def keyid_of(key: Ed25519PublicKey) -> str:
    """The SHA-256 of the raw 32-byte public key, hex."""
    return sha256_hex(raw_public(key))


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
    """Read the envelope structure. Nothing is trusted yet: signatures are checked by `verify_signature`."""
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
    signatures = tuple(_signature(e) for e in entries)
    return Envelope(envelope, payload_type, _b64(doc.get("payload"), "payload"), signatures)


@dataclass(frozen=True)
class SignatureResult:
    """`keyids` are the trusted keys that verified; `reason` is set exactly when too few did."""

    keyids: tuple[str, ...]
    reason: str | None

    @property
    def keyid(self) -> str | None:
        """The first verifying trusted key, or None."""
        return self.keyids[0] if self.keyids else None


def verify_signature(
    env: Envelope, trusted: Mapping[str, Ed25519PublicKey], threshold: int = 1
) -> SignatureResult:
    """Succeed when at least `threshold` distinct trusted keys, each named by a signature entry, verify.

    A key that no signature entry names is never tried. `untrusted_key` means fewer than `threshold`
    trusted keys are named at all; `signature_invalid` means enough were named and too few verified.
    """
    if threshold < 1:
        raise ValueError("threshold must be at least 1")
    named = {s.keyid for s in env.signatures}
    candidates = [(k, v) for k, v in trusted.items() if k in named]
    if len(candidates) < threshold:
        return SignatureResult((), "untrusted_key")
    verifiers = [native.Ed25519Verifier(raw_public(key), key_id=keyid) for keyid, key in candidates]
    try:
        verified = native.verify_threshold(env.raw, verifiers, threshold)
    except native.DsseError:
        return SignatureResult((), "signature_invalid")
    if verified.payload_type != env.payload_type or verified.payload != env.payload:
        return SignatureResult((), "signature_invalid")
    return SignatureResult(tuple(verified.accepted_keys), None)


def _native_signer(key: Ed25519PrivateKey) -> native.Ed25519Signer:
    return native.Ed25519Signer(key.private_bytes_raw(), key_id=keyid_of(key.public_key()))


def sign(payload_type: str, payload: bytes, keys: Sequence[Ed25519PrivateKey]) -> bytes:
    """A DSSE envelope over `payload` signed by every key in `keys`, as RFC 8785 bytes."""
    if not keys:
        raise ValueError("at least one signing key is needed")
    return jcs.admit(native.sign(payload_type, payload, [_native_signer(k) for k in keys]))


def signature_entry(payload_type: str, payload: bytes, key: Ed25519PrivateKey) -> dict[str, str]:
    """One signature entry by `key` over `payload`, for adding to an envelope that already has others."""
    doc = json.loads(sign(payload_type, payload, [key]))
    entry: dict[str, str] = doc["signatures"][0]
    return entry
