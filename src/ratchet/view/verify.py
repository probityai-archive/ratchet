"""Check a decision's DSSE receipt: the signature, the key, and that the stored decision is the signed one."""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text

from ratchet.improve.receipt import SIGNING_KEY_ENV, ReceiptError, Signer, key_id, verify
from ratchet.types import canonical_json
from ratchet.verify import dsse
from ratchet.verify.keys import DEFAULT_KEY_PATH, TrustedKeyError, load_trusted_key


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    summary: str
    detail: list[str]
    keyid: str | None = None
    payload: Mapping[str, Any] | None = None


def trusted_keys(
    pubkey_hex: str | None,
    env: Mapping[str, str] | None = None,
    pubkey_file: Path | None = None,
    pinned: Path = DEFAULT_KEY_PATH,
) -> dict[str, Ed25519PublicKey]:
    """Keys to trust, first match wins: `--pubkey` (32-byte hex), `--pubkey-file` (PEM), the public half of
    the signing key in the environment, then the repository's pinned decision key. Empty when none exists,
    which makes verification report 'no trusted key'."""
    source = os.environ if env is None else env
    if pubkey_file is not None:
        return _pem(pubkey_file)
    if pubkey_hex:
        try:
            key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex.strip()))
        except ValueError as exc:
            raise ReceiptError(f"public key is not 32 bytes of hex: {exc}") from exc
        return {key_id(key): key}
    if source.get(SIGNING_KEY_ENV):
        key = Signer.from_seed_hex(source[SIGNING_KEY_ENV]).public_key
        return {key_id(key): key}
    return _pem(pinned) if pinned.exists() else {}


def _pem(path: Path) -> dict[str, Ed25519PublicKey]:
    try:
        key = load_trusted_key(path)
    except TrustedKeyError as exc:
        raise ReceiptError(str(exc)) from exc
    return {key_id(key): key}


def _independent(envelope: bytes, trusted: Mapping[str, Ed25519PublicKey]) -> str | None:
    """Re-check with the verifier written apart from the signer; a disagreement is a defect in one of them."""
    try:
        result = dsse.verify_signature(dsse.parse(envelope), trusted)
    except dsse.EnvelopeError as exc:
        return f"independent verifier rejects the envelope: {exc}"
    if result.keyid is None:
        return f"independent verifier rejects the signature: {result.reason}"
    return None


def verify_envelope(
    envelope: bytes, trusted: Mapping[str, Ed25519PublicKey], stored: Mapping[str, Any] | None = None
) -> VerifyResult:
    """Verify an envelope; when `stored` (the decision document) is given, also require that every field
    of the signed payload equals the stored one, so a decision edited after signing is caught."""
    if not trusted:
        return VerifyResult(
            False, "NOT VERIFIED", ["no trusted key: pass --pubkey HEX or set RATCHET_SIGNING_KEY"]
        )
    try:
        payload = verify(envelope, trusted)
    except ReceiptError as exc:
        return VerifyResult(False, "INVALID", [str(exc)])
    other = _independent(envelope, trusted)
    if other is not None:
        return VerifyResult(False, "INVALID", [other])
    detail = [
        "DSSE envelope signature valid under a trusted Ed25519 key",
        "the independent verifier (ratchet.verify.dsse, shares no code with the signer) agrees",
    ]
    if stored is not None:
        diff = sorted(k for k, v in payload.items() if canonical_json(stored.get(k)) != canonical_json(v))
        if diff:
            return VerifyResult(
                False,
                "TAMPERED",
                [f"stored decision differs from the signed payload in: {', '.join(diff)}"],
                payload=payload,
            )
        detail.append("stored decision matches the signed payload field for field")
    return VerifyResult(True, "VERIFIED", detail, payload=payload)


def verify_decision(doc: Mapping[str, Any], trusted: Mapping[str, Ed25519PublicKey]) -> VerifyResult:
    raw = doc.get("envelope")
    if not raw:
        return VerifyResult(False, "NO RECEIPT", ["the decision carries no signed envelope"])
    try:
        envelope = base64.b64decode(str(raw), validate=True)
    except (binascii.Error, ValueError):
        return VerifyResult(False, "INVALID", ["the stored envelope is not valid base64"])
    res = verify_envelope(envelope, trusted, doc)
    return VerifyResult(res.ok, res.summary, res.detail, doc.get("keyid"), res.payload)


def render_verify(res: VerifyResult, decision_id: str) -> RenderableType:
    style = "green" if res.ok else "red"
    lines = [Text.assemble((f" {res.summary} ", f"bold white on {style}"), f"  {decision_id}")]
    lines += [Text(f"  {'✓' if res.ok else '✗'} {d}", style=style) for d in res.detail]
    if res.payload is not None:
        lines.append(
            Text(
                f"  decision: {'ADMITTED' if res.payload.get('admitted') else 'REJECTED'} "
                f"{res.payload.get('parent')} -> {res.payload.get('candidate')}",
                style="bold",
            )
        )
        lines.append(Text(f"  candidate digest {res.payload.get('candidate_digest')}", style="dim"))
    if res.keyid:
        lines.append(Text(f"  key {res.keyid}", style="dim"))
    out = Text("\n").join(lines)
    return Panel(out, border_style=style)
