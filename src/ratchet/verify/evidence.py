"""The evidence steps that follow a verified signature: report digests, recomputed gates, authority,
replay and the overfitting screen. Each step appends one item to `Ctx.items`:

- `recomputed`: this verifier derived the result from the stored evidence and it matches the receipt.
- `partial`: some, not all, of the inputs were supplied; what was supplied matched.
- `not_supplied`: the inputs were not given, so the receipt's claim stands on its signature alone.

A step that finds a disagreement raises `Refuse`, which ends the verdict with a specific reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ratchet.verify import dsse, recompute

HUMAN_APPROVAL_TYPE = "application/vnd.ratchet.human-approval+json"
EXCEPTION_PREFIX = "authority grew under"


class Refuse(Exception):
    """A step ends the verdict. `failed` names the probity check recorded for it."""

    def __init__(self, decision: str, reason: str, failed: str) -> None:
        super().__init__(reason)
        self.decision, self.reason, self.failed = decision, reason, failed


@dataclass
class Ctx:
    payload: dict[str, Any]
    claimed: dict[str, bool]
    required: list[str]
    reports: dict[str, bytes]
    harnesses: dict[str, bytes]
    recomputed: dict[str, bytes]
    approval: bytes | None
    eval_roles: dict[str, str]
    human_keys: dict[str, Any]
    require_reports: bool
    items: list[dict[str, Any]] = field(default_factory=list)
    scope: dict[str, Any] = field(default_factory=dict)

    def item(self, name: str, status: str, **detail: Any) -> None:
        self.items.append({"id": name, "status": status, **detail})


def _load(raw: bytes, label: str) -> dict[str, Any]:
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Refuse("contradicted", f"evidence_not_json:{label}", label) from exc
    if not isinstance(doc, dict):
        raise Refuse("contradicted", f"evidence_not_json:{label}", label)
    return doc


def check_report_digests(ctx: Ctx) -> None:
    named: dict[str, str] = ctx.payload["report_digests"]
    unnamed = sorted(set(ctx.reports) - set(named))
    if unnamed:
        raise Refuse("contradicted", f"report_not_named:{unnamed[0]}", "reports")
    for name in sorted(ctx.reports):
        if hashlib.sha256(ctx.reports[name]).hexdigest() != named[name]:
            raise Refuse("contradicted", f"report_digest_mismatch:{name}", "reports")
    verified, unverified = sorted(ctx.reports), sorted(set(named) - set(ctx.reports))
    ctx.scope["reports"] = {"named": len(named), "verified": verified, "unverified": unverified}
    if unverified and ctx.require_reports:
        raise Refuse("not_established", f"report_not_supplied:{unverified[0]}", "reports")
    status = "recomputed" if not unverified else "partial" if verified else "not_supplied"
    ctx.item("report_digests", status, checked=len(verified), named=len(named))


def _agree(ctx: Ctx, gate: str, result: recompute.Gate) -> None:
    if ctx.claimed[gate] != result.passed:
        raise Refuse("contradicted", f"gate_recompute_mismatch:{gate}", f"gate:{gate}")
    ctx.item(f"gate:{gate}", "recomputed", passed=result.passed, evidence=result.evidence)


def recompute_eval_gates(ctx: Ctx) -> None:
    gates = [g for g in ("heldout", "no_regression", "benign_twins") if g in ctx.required]
    names = [ctx.eval_roles[r] for r in ("p1", "p2", "c1", "c2")]
    if not all(n in ctx.reports for n in names):
        for g in gates:
            ctx.item(f"gate:{g}", "not_supplied", needs=names)
        return
    try:
        p1, p2, c1, c2 = (_load(ctx.reports[n], f"report:{n}") for n in names)
        results = {
            "heldout": lambda: recompute.heldout(p1, p2, c1, c2),
            "no_regression": lambda: recompute.no_regression([p1, p2], [c1, c2]),
            "benign_twins": lambda: recompute.benign_twins([c1, c2]),
        }
        for g in gates:
            _agree(ctx, g, results[g]())
    except ValueError as exc:
        raise Refuse("contradicted", f"report_malformed:{exc}", "reports") from exc


def _approval_covers(ctx: Ctx, candidate_digest: str) -> str | None:
    """None when a human approval verifies for exactly this candidate; otherwise the reason it does not."""
    if ctx.approval is None:
        return "authority_exception_unverified:no_approval_supplied"
    try:
        env = dsse.parse(ctx.approval)
    except dsse.EnvelopeError:
        return "authority_exception_unverified:approval_malformed"
    if env.payload_type != HUMAN_APPROVAL_TYPE:
        return "authority_exception_unverified:approval_type"
    if dsse.verify_signature(env, ctx.human_keys).keyid is None:
        return "authority_exception_unverified:approval_signature"
    try:
        body = json.loads(env.payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "authority_exception_unverified:approval_payload"
    if not isinstance(body, dict):
        return "authority_exception_unverified:approval_payload"
    if body.get("candidate_digest") != candidate_digest:
        return "authority_exception_unverified:approval_names_another_candidate"
    return None


def _harness_pair(ctx: Ctx) -> tuple[dict[str, Any], dict[str, Any]]:
    parent, candidate = (_load(ctx.harnesses[r], f"harness:{r}") for r in ("parent", "candidate"))
    p = ctx.payload
    if candidate.get("digest") != p["candidate_digest"] or candidate.get("version") != p["candidate"]:
        raise Refuse("contradicted", "harness_not_the_decided_candidate", "harness")
    if parent.get("version") != p["parent"]:
        raise Refuse("contradicted", "harness_not_the_decided_parent", "harness")
    return parent, candidate


def recompute_authority(ctx: Ctx) -> None:
    if len(ctx.harnesses) < 2:
        ctx.item("gate:authority", "not_supplied", needs=["harness:parent", "harness:candidate"])
        return
    parent, candidate = _harness_pair(ctx)
    result = recompute.authority(parent, candidate)
    ctx.scope["new_guards"] = result.evidence["new_guards"]
    if "authority" not in ctx.required:
        return
    if result.passed or not ctx.claimed["authority"]:
        _agree(ctx, "authority", result)
        return
    excepted = any(str(e).startswith(EXCEPTION_PREFIX) for e in ctx.payload["exceptions"])
    if not excepted:
        raise Refuse("contradicted", "authority_recompute_failed_without_exception", "gate:authority")
    if problem := _approval_covers(ctx, ctx.payload["candidate_digest"]):
        raise Refuse("not_established", problem, "gate:authority")
    ctx.item("gate:authority", "recomputed", passed=False, covered_by="verified human approval")


def recompute_replay(ctx: Ctx) -> None:
    if "replay" not in ctx.required:
        return
    check = next(c for c in ctx.payload["checks"] if c["name"] == "replay")
    evidence = check.get("evidence") or {}
    new = list(evidence.get("new_guards") or [])
    rows = {r.get("guard_id"): r for r in evidence.get("reports") or [] if isinstance(r, dict)}
    if "new_guards" in ctx.scope and ctx.scope["new_guards"] != new:
        raise Refuse("contradicted", "replay_new_guards_mismatch", "gate:replay")
    if any(rows.get(g, {}).get("passed") is not True for g in new) and check["passed"]:
        raise Refuse("contradicted", "replay_check_passed_with_failed_guard", "gate:replay")
    supplied = [g for g in new if f"replay:{g}" in ctx.recomputed]
    for gid in supplied:
        report = _load(ctx.recomputed[f"replay:{gid}"], f"recomputed:replay:{gid}")
        if bad := recompute.replay_matches(rows.get(gid, {}), report):
            raise Refuse("contradicted", f"replay_recompute_mismatch:{gid}:{bad[0]}", "gate:replay")
        if ctx.payload["corpus_digests"].get(gid) != report.get("snapshot_digest"):
            raise Refuse("contradicted", f"replay_snapshot_mismatch:{gid}", "gate:replay")
    known = "new_guards" in ctx.scope
    if known and not new:
        ctx.item("gate:replay", "recomputed", guards=0)
    elif known and len(supplied) == len(new):
        ctx.item("gate:replay", "recomputed", guards=len(new))
    elif supplied:
        ctx.item("gate:replay", "partial", checked=len(supplied), guards=len(new))
    else:
        ctx.item("gate:replay", "not_supplied", needs=[f"recomputed:replay:{g}" for g in new])


def recompute_overfit(ctx: Ctx) -> None:
    if "overfit_lint" not in ctx.required:
        return
    raw = ctx.recomputed.get("overfit_lint")
    if raw is None:
        ctx.item("gate:overfit_lint", "not_supplied", needs=["recomputed:overfit_lint"])
        return
    hits = _load(raw, "recomputed:overfit_lint").get("hits")
    if not isinstance(hits, list):
        raise Refuse("contradicted", "evidence_not_json:recomputed:overfit_lint", "gate:overfit_lint")
    _agree(ctx, "overfit_lint", recompute.Gate(not hits, {"hits": len(hits)}))
