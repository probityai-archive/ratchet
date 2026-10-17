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

from ratchet.canonical import canonical_json
from ratchet.demo.signing import is_demo_signed
from ratchet.improve.receipt import SIGNING_KEY_ENV, ReceiptError, Signer, verify
from ratchet.verify.dsse import keyid_of
from ratchet.verify.keys import DEFAULT_KEY_PATH, TrustedKeyError, load_trusted_key
from ratchet.verify.receipt import Options, VerifyReport
from ratchet.verify.receipt import verify_decision as recompute_receipt
from ratchet.view.evidence import Evidence


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    summary: str
    detail: list[str]
    keyid: str | None = None
    payload: Mapping[str, Any] | None = None
    recompute: VerifyReport | None = None


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
        return {keyid_of(key): key}
    if source.get(SIGNING_KEY_ENV):
        key = Signer.from_seed_hex(source[SIGNING_KEY_ENV]).public_key
        return {keyid_of(key): key}
    return _pem(pinned) if pinned.exists() else {}


def _pem(path: Path) -> dict[str, Ed25519PublicKey]:
    try:
        key = load_trusted_key(path)
    except TrustedKeyError as exc:
        raise ReceiptError(str(exc)) from exc
    return {keyid_of(key): key}


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
    detail = ["DSSE envelope signature valid under a trusted Ed25519 key, checked by the dsse crate"]
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


def verify_decision(
    doc: Mapping[str, Any],
    trusted: Mapping[str, Ed25519PublicKey],
    evidence: Evidence | None = None,
    options: Options = Options(),  # noqa: B008 - frozen dataclass, safe as a default
) -> VerifyResult:
    """Check the stored receipt. With `evidence`, the pinned verifier also recomputes what it can from it
    (`ratchet.verify.verify_decision`). The signature double-check runs first and both must pass."""
    res = _signature_check(doc, trusted)
    if evidence is None or not res.ok:
        return res
    envelope = base64.b64decode(str(doc["envelope"]))
    report = recompute_receipt(
        envelope,
        list(trusted.values()),
        reports=evidence.reports,
        harnesses=evidence.harnesses,
        recomputed=evidence.recomputed,
        options=options,
        traces=evidence.traces,
    )
    detail = [*res.detail, f"recomputation verdict: {report.decision} {report.reason}"]
    if not report.ok:
        return VerifyResult(False, "CONTRADICTED" if report.decision == "contradicted" else "NOT ESTABLISHED",
                            detail, res.keyid, res.payload, report)  # fmt: skip
    return VerifyResult(True, "VERIFIED", detail, res.keyid, res.payload, report)


def _signature_check(doc: Mapping[str, Any], trusted: Mapping[str, Ed25519PublicKey]) -> VerifyResult:
    raw = doc.get("envelope")
    if not raw:
        return VerifyResult(False, "NO RECEIPT", ["the decision carries no signed envelope"])
    try:
        envelope = base64.b64decode(str(raw), validate=True)
    except (binascii.Error, ValueError):
        return VerifyResult(False, "INVALID", ["the stored envelope is not valid base64"])
    res = verify_envelope(envelope, trusted, doc)
    keyid, detail = doc.get("keyid"), res.detail
    counter = doc.get("countersigned")
    if isinstance(counter, Mapping) and counter.get("keyid") in trusted:
        keyid = counter["keyid"]
        detail = [
            *detail,
            f"countersigned by the pinned key on {str(counter.get('at', ''))[:10]}; "
            f"first signed by per-run key {str(doc.get('keyid', ''))[:12]}",
        ]
    return VerifyResult(res.ok, res.summary, detail, keyid, res.payload)


def recompute_lines(r: VerifyReport) -> list[Text]:
    """RECOMPUTED apart from AUTHENTICATED ONLY, then anything not supplied, then the reason code."""
    done = [i for i in r.recomputed if i["status"] == "recomputed"]
    partial = [i["id"] for i in r.recomputed if i["status"] == "partial"]
    missing = [i["id"] for i in r.recomputed if i["status"] == "not_supplied"]
    good = r.ok
    lines = [
        Text(""),
        Text.assemble(
            ("  reason code ", "dim"), (f"{r.decision}: {r.reason}", "bold green" if good else "bold red")
        ),
        Text.assemble(
            (f"  RECOMPUTED ({len(done)} of {len(r.recomputed)})  ", "bold green"),
            (", ".join(str(i["id"]) for i in done) or "none", "green"),
        ),
    ]
    if partial:
        lines.append(Text(f"  PARTIAL  {', '.join(partial)}", style="yellow"))
    if missing:
        lines.append(Text(f"  NOT SUPPLIED  {', '.join(missing)}", style="yellow"))
    lines.append(
        Text.assemble(
            ("  AUTHENTICATED ONLY (signed, not re-derived)  ", "bold yellow"),
            (", ".join(r.authenticated_only) or "none", "yellow"),
        )
    )
    level = "every recomputable item recomputed" if r.recomputation_complete else "recomputation INCOMPLETE"
    lines.append(Text(f"  {r.reports_summary}; {level}", style="dim"))
    return lines


def render_verify(res: VerifyResult, decision_id: str) -> RenderableType:
    style = "green" if res.ok else "red"
    lines = [Text.assemble((f" {res.summary} ", f"bold white on {style}"), f"  {decision_id}")]
    for i, d in enumerate(res.detail):
        # the signature checks passed whenever a recomputation ran; only its own verdict can be the failure
        good = res.ok or (res.recompute is not None and i < len(res.detail) - 1)
        lines.append(Text(f"  {'✓' if good else '✗'} {d}", style="green" if good else "red"))
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
    if res.payload is not None and is_demo_signed(res.payload):
        lines.append(Text("  signed by a DEMO key generated by `ratchet demo`, not the pinned decision key",
                          style="bold yellow"))  # fmt: skip
    if res.recompute is not None:
        lines += recompute_lines(res.recompute)
    out = Text("\n").join(lines)
    return Panel(out, border_style=style)
