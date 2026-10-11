"""Verify a Ratchet admission receipt with the pinned public verifier and an independently pinned key.

`verify_decision` builds a `probity-verify` case from the envelope bytes (and any stored reports), pins the
trusted key and the predicate in the consumer policy, and asks `probity_verify.adjudicate` for the verdict
using the `ratchet_decision/v1` adapter in `ratchet.verify.adapter`. Every failure is a specific reason code.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from probity_verify import CaseError, adjudicate  # type: ignore[import-untyped]

from ratchet.types import canonical_json, sha256_hex
from ratchet.verify import adapter
from ratchet.verify.dsse import keyid_of
from ratchet.verify.keys import raw_b64

NO_VERDICT = "no_verdict"


@dataclass(frozen=True)
class VerifyReport:
    """The outcome of verifying one receipt.

    `ok` means the receipt is authentic and consistent: signed by the pinned key, of the supported type and
    predicate, with its gate record agreeing with its `admitted` flag, every supplied piece of evidence
    matching what the receipt names, and every gate that could be recomputed agreeing with the receipt. A
    rejection receipt can be `ok`. Use `admission_verified` to ask whether the candidate was admitted on a
    receipt that checks out.

    `recomputed` lists what this verifier derived from bytes, each item `recomputed`, `partial` or
    `not_supplied`. `authenticated_only` lists what the receipt asserts and this verifier cannot re-derive.
    `ok` never implies `recomputation_complete`: print `summary`, which keeps the two apart.

    `decision` is the `probity-verify` verdict (`supported`, `contradicted`, `not_established`), or
    `no_verdict` when the input was malformed and the verifier refused to judge it.
    """

    decision: str
    reason: str
    admitted: bool | None = None
    keyid: str | None = None
    parent: str | None = None
    candidate: str | None = None
    reports_named: int = 0
    reports_verified: tuple[str, ...] = ()
    checks: tuple[dict[str, Any], ...] = ()
    recomputed: tuple[dict[str, Any], ...] = ()
    authenticated_only: tuple[str, ...] = ()
    recomputation_complete: bool = False
    packet: dict[str, Any] | None = field(default=None, compare=False, repr=False)

    @property
    def ok(self) -> bool:
        return self.decision == "supported"

    @property
    def admission_verified(self) -> bool:
        return self.ok and self.admitted is True

    @property
    def reports_summary(self) -> str:
        return f"{len(self.reports_verified)} of {self.reports_named} named reports checked"

    @property
    def summary(self) -> str:
        """One block that never lets missing evidence read as recomputed."""
        done = [i["id"] for i in self.recomputed if i["status"] == "recomputed"]
        partial = [i["id"] for i in self.recomputed if i["status"] == "partial"]
        missing = [i["id"] for i in self.recomputed if i["status"] == "not_supplied"]
        level = "recomputation INCOMPLETE"
        if self.recomputation_complete:
            level = "every recomputable item recomputed"
        lines = [f"{self.decision} {self.reason} ({level})"]
        lines.append(f"  RECOMPUTED ({len(done)} of {len(self.recomputed)}): {', '.join(done) or 'none'}")
        if partial:
            lines.append(f"  PARTIAL: {', '.join(partial)}")
        if missing:
            lines.append(f"  NOT SUPPLIED: {', '.join(missing)}")
        if self.authenticated_only:
            lines.append(f"  AUTHENTICATED ONLY (not re-derived): {', '.join(self.authenticated_only)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "admission_verified": self.admission_verified,
            "recomputation_complete": self.recomputation_complete,
            "decision": self.decision,
            "reason": self.reason,
            "admitted": self.admitted,
            "keyid": self.keyid,
            "parent": self.parent,
            "candidate": self.candidate,
            "reports": self.reports_summary,
            "recomputed": list(self.recomputed),
            "authenticated_only": list(self.authenticated_only),
            "checks": list(self.checks),
            "policy_sha256": (self.packet or {}).get("policy_sha256"),
        }


def report_digest(report: Mapping[str, Any]) -> str:
    """The digest a receipt names for a stored report: SHA-256 of its canonical JSON."""
    return sha256_hex(canonical_json(dict(report)))


def _bytes(doc: Mapping[str, Any] | bytes) -> bytes:
    return doc if isinstance(doc, bytes) else canonical_json(dict(doc))


def _write(root: Path, relative: str, data: bytes) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "sha256": sha256_hex(data), "length": len(data)}


Doc = Mapping[str, Any] | bytes


@dataclass(frozen=True)
class Options:
    """What the consumer pins: set by the verifying party, never read from the receipt."""

    required_gates: tuple[str, ...] = adapter.DEFAULT_GATES
    eval_roles: Mapping[str, str] = field(default_factory=lambda: dict(adapter.DEFAULT_EVAL_ROLES))
    require_reports: bool = False
    require_complete: bool = False
    trusted_human_keys: Sequence[Ed25519PublicKey] = ()


def _slot(name: str) -> str:
    return sha256_hex(name.encode())[:16]


def _artifacts(
    root: Path,
    envelope: bytes,
    reports: Mapping[str, Doc],
    harnesses: Mapping[str, Doc],
    recomputed: Mapping[str, Doc],
    approval: bytes | None,
) -> dict[str, Any]:
    out = {"envelope": _write(root, "envelope.json", envelope)}
    groups = ((adapter.REPORT_ROLE, "reports", reports), (adapter.RECOMPUTED_ROLE, "recomputed", recomputed))
    for prefix, folder, docs in groups:
        for name, doc in docs.items():
            out[f"{prefix}{name}"] = _write(root, f"{folder}/{_slot(name)}.json", _bytes(doc))
    for name, doc in harnesses.items():
        out[f"{adapter.HARNESS_ROLE}{name}"] = _write(root, f"harness/{name}.json", _bytes(doc))
    if approval is not None:
        out["approval"] = _write(root, "approval.json", approval)
    return out


def _build(
    root: Path,
    envelope: bytes,
    keys: Sequence[Ed25519PublicKey],
    evidence: Mapping[str, Mapping[str, Doc]],
    approval: bytes | None,
    options: Options,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = _artifacts(
        root,
        envelope,
        evidence.get("reports", {}),
        evidence.get("harnesses", {}),
        evidence.get("recomputed", {}),
        approval,
    )
    case_id = f"ratchet-decision-{artifacts['envelope']['sha256'][:16]}"
    case = {"schema_version": adapter.CASE_SCHEMA, "case_id": case_id, "artifacts": artifacts}
    policy = {
        "schema_version": adapter.POLICY_SCHEMA,
        "assessments": {
            case_id: {
                "claim_type": adapter.CLAIM_TYPE,
                "bindings": {"envelope": artifacts["envelope"]["sha256"]},
                "trusted_keys": {keyid_of(k): raw_b64(k) for k in keys},
                "trusted_human_keys": {keyid_of(k): raw_b64(k) for k in options.trusted_human_keys},
                "predicate_type": adapter.PREDICATE_URI,
                "required_gates": list(options.required_gates),
                "eval_roles": dict(options.eval_roles),
                "require_reports": options.require_reports,
                "require_complete": options.require_complete,
            }
        },
    }
    return case, policy


def _from_packet(packet: dict[str, Any]) -> VerifyReport:
    scope = packet["scope"]
    reports = scope.get("reports", {})
    return VerifyReport(
        decision=packet["decision"],
        reason=packet["reason"],
        admitted=scope.get("admitted"),
        keyid=scope.get("keyid"),
        parent=scope.get("parent"),
        candidate=scope.get("candidate"),
        reports_named=reports.get("named", 0),
        reports_verified=tuple(reports.get("verified", ())),
        checks=tuple(packet["checks"]),
        recomputed=tuple(scope.get("recomputed", ())),
        authenticated_only=tuple(scope.get("authenticated_only", ())),
        recomputation_complete=bool(scope.get("recomputation_complete")),
        packet=packet,
    )


def _bad_names(*groups: Mapping[str, Doc]) -> str | None:
    names = [n for g in groups for n in g]
    return next((n for n in names if not adapter.REPORT_NAME.fullmatch(n)), None)


def verify_decision(
    envelope_bytes: bytes,
    trusted_public_key: Ed25519PublicKey | Sequence[Ed25519PublicKey],
    *,
    reports: Mapping[str, Doc] | None = None,
    harnesses: Mapping[str, Doc] | None = None,
    recomputed: Mapping[str, Doc] | None = None,
    approval: bytes | None = None,
    options: Options = Options(),  # noqa: B008 - frozen dataclass, safe as a default
) -> VerifyReport:
    """Verify a signed admission receipt. Never raises on bad evidence: it returns a failing report.

    `trusted_public_key` (one key, or several during a rotation) must come from somewhere other than the
    receipt (see `load_trusted_key`). Evidence is optional and each kind buys recomputation:

    - `reports`: stored reports by the name the receipt gives them (a mapping, hashed by its canonical JSON,
      or the exact stored bytes). Each must match its named digest. The four evaluation reports named by
      `options.eval_roles` also let the held-out, no-regression and benign-twin gates be recomputed.
    - `harnesses`: `{"parent": ..., "candidate": ...}` harness documents; lets the authority gate be
      recomputed.
    - `recomputed`: results the caller derived from the recorded inputs: `"replay:<guard id>"` replay report
      payloads and `"overfit_lint"` as `{"hits": [...]}`; compared with the receipt's claims.
    - `approval`: the human approval envelope, needed only when authority grew by declared exception.

    Anything not supplied is reported as not supplied, never as recomputed.
    """
    keys = (
        [trusted_public_key] if isinstance(trusted_public_key, Ed25519PublicKey) else list(trusted_public_key)
    )
    if not keys:
        return VerifyReport(NO_VERDICT, "no_trusted_key")
    evidence = {
        "reports": dict(reports or {}),
        "harnesses": dict(harnesses or {}),
        "recomputed": dict(recomputed or {}),
    }
    bad = _bad_names(evidence["reports"], evidence["recomputed"])
    if bad is not None:
        return VerifyReport(NO_VERDICT, f"invalid_report_name:{bad}")
    if set(evidence["harnesses"]) - {"parent", "candidate"}:
        return VerifyReport(NO_VERDICT, "invalid_harness_role")
    with tempfile.TemporaryDirectory(prefix="ratchet-verify-") as tmp:
        root = Path(os.path.realpath(tmp))
        case, policy = _build(root, envelope_bytes, keys, evidence, approval, options)
        try:
            packet = adjudicate(case, policy, root, adapters={adapter.CLAIM_TYPE: adapter.adjudicate})
        except CaseError as exc:
            return VerifyReport(NO_VERDICT, str(exc))
    return _from_packet(packet)
