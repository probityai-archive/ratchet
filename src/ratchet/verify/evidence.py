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
from pathlib import Path
from typing import Any

from ratchet.verify import dsse, recompute, runs

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
    sessions: dict[str, bytes] = field(default_factory=dict)  # a k-session bundle's files, by file name
    traces: dict[str, bytes] = field(default_factory=dict)  # JSON lines of trace documents, by report name
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


def _eval_report_names(ctx: Ctx) -> list[str]:
    """The evaluation reports the three gates rest on. A decision whose signed digests name the parent runs
    and the first candidate run but no confirming run was refused after C1, and C2 never ran: its gates are
    recomputed from the three runs it names. A named run that was not supplied is still missing."""
    names = [ctx.eval_roles[r] for r in ("p1", "p2", "c1", "c2")]
    named = ctx.payload["report_digests"]
    if all(n in named for n in names[:3]) and names[3] not in named:
        return names[:3]
    return names


def recompute_eval_gates(ctx: Ctx) -> None:
    gates = [g for g in ("heldout", "no_regression", "benign_twins") if g in ctx.required]
    names = _eval_report_names(ctx)
    if not all(n in ctx.reports for n in names):
        for g in gates:
            ctx.item(f"gate:{g}", "not_supplied", needs=names)
        return
    try:
        p1, p2, c1, *rest = (_load(ctx.reports[n], f"report:{n}") for n in names)
        c2 = rest[0] if rest else None
        cands = [c1] if c2 is None else [c1, c2]
        results = {
            "heldout": lambda: recompute.heldout(p1, p2, c1, c2),
            "no_regression": lambda: recompute.no_regression([p1, p2], cands),
            "benign_twins": lambda: recompute.benign_twins(cands),
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
    claimed = next(c for c in ctx.payload["checks"] if c["name"] == "authority").get("evidence") or {}
    if isinstance(claimed.get("extended"), dict):
        _recompute_extended(ctx, parent, candidate, claimed["extended"])
        return
    excepted = any(str(e).startswith(EXCEPTION_PREFIX) for e in ctx.payload["exceptions"])
    if not excepted:
        raise Refuse("contradicted", "authority_recompute_failed_without_exception", "gate:authority")
    if problem := _approval_covers(ctx, ctx.payload["candidate_digest"]):
        raise Refuse("not_established", problem, "gate:authority")
    ctx.item("gate:authority", "recomputed", passed=False, covered_by="verified human approval")


BUNDLE_FILES = ("manifest.json", "executions.jsonl", "verdict.json")


def _recompute_extended(
    ctx: Ctx, parent: dict[str, Any], candidate: dict[str, Any], claimed: dict[str, Any]
) -> None:
    """An added tool or edited procedure admitted on k-session evidence: recompute the bundle's verdict and
    check it compares exactly this parent and candidate. That the candidate's runs exercised each change is
    read from the signed evidence, since the runs' traces are not supplied here."""
    import tempfile

    from ratchet.improve import sessions

    if "refused" in claimed:
        raise Refuse("contradicted", "admitted_with_refused_extended_change", "gate:authority")
    if set(ctx.sessions) != set(BUNDLE_FILES):
        ctx.item("gate:authority", "not_supplied", needs=[f"sessions:{n}" for n in BUNDLE_FILES])
        return
    with tempfile.TemporaryDirectory(prefix="ratchet-sessions-") as tmp:
        for name, raw in ctx.sessions.items():
            (Path(tmp) / name).write_bytes(raw)
        try:
            same, verdict = sessions.verify_bundle(Path(tmp))
            protocol = sessions.load_bundle(Path(tmp)).protocol
        except sessions.BundleError as exc:
            raise Refuse("contradicted", f"sessions_bundle_refused:{exc}", "gate:authority") from exc
    if not same or verdict["verdict"] != "admitted":
        raise Refuse("contradicted", "sessions_verdict_not_admitted", "gate:authority")
    if protocol.digest != claimed.get("protocol_digest"):
        raise Refuse("contradicted", "sessions_bundle_not_the_named_one", "gate:authority")
    arms = dict(protocol.arms)
    if arms["candidate"] != candidate.get("version") or any(
        arms[a] != parent.get("version") for a in sessions.UNCHANGED
    ):
        raise Refuse("contradicted", "sessions_bundle_compares_other_harnesses", "gate:authority")
    ctx.item(
        "gate:authority",
        "partial",
        covered_by="recomputed k-session verdict",
        authenticated_only=["tools_called", "procedures_used"],
    )


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


# ---------------------------------------------------------------------------------------------------------
# Run records: the claims that used to rest on the evaluator's signature alone
# ---------------------------------------------------------------------------------------------------------


def _traces(ctx: Ctx, name: str) -> dict[str, dict[str, Any]] | None:
    """The trace documents supplied for one report, by run id; None when none were supplied."""
    raw = ctx.traces.get(name)
    if raw is None:
        return None
    label = f"traces:{name}"
    out: dict[str, dict[str, Any]] = {}
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise Refuse("contradicted", f"evidence_not_json:{label}", label) from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Refuse("contradicted", f"evidence_not_json:{label}", label) from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("run_id"), str) or doc["run_id"] in out:
            raise Refuse("contradicted", f"evidence_not_json:{label}", label)
        out[doc["run_id"]] = doc
    return out


def _run_reports(ctx: Ctx, item: str) -> list[runs.Report] | None:
    """The evaluation reports with their traces, or None after recording `item` as not supplied: a report
    is missing, or it was written before reports carried run records."""
    names = _eval_report_names(ctx)
    missing = [n for n in names if n not in ctx.reports]
    if missing:
        ctx.item(item, "not_supplied", needs=[f"report:{n}" for n in missing])
        return None
    reports = [runs.Report(n, _load(ctx.reports[n], f"report:{n}"), _traces(ctx, n)) for n in names]
    legacy = [r.name for r in reports if not runs.has_runs(r)]
    if legacy:
        ctx.item(item, "not_supplied", needs=[f"run records in report:{n}" for n in legacy])
        return None
    return reports


def _records(item: str, check: Any, *args: Any) -> dict[str, Any]:
    try:
        result: dict[str, Any] = check(*args)
    except runs.RecordError as exc:
        raise Refuse(exc.decision, exc.reason, item) from exc
    return result


MODEL_CALLS = "model_call_records"


def recompute_model_calls(ctx: Ctx) -> None:
    """Fresh model execution, as far as records show it: every counted call committed once, in its own run,
    uncached, and equal to the supplied trace. `partial` while a run's trace is not supplied."""
    reports = _run_reports(ctx, MODEL_CALLS)
    if reports is None:
        return
    counts = _records(MODEL_CALLS, runs.check_model_calls, reports)
    status = "recomputed" if counts["traces_checked"] == counts["runs"] else "partial"
    ctx.item(MODEL_CALLS, status, **counts)


RECORD_CHAIN = "record_chain"


def recompute_record_chain(ctx: Ctx) -> None:
    """Capture fidelity, as far as records show it: the supplied traces are the committed ones, each a
    well-formed run, and each report row is what its trace yields. `partial` while a committed run's trace
    is not supplied; not supplied when no trace is."""
    reports = _run_reports(ctx, RECORD_CHAIN)
    if reports is None:
        return
    if all(r.traces is None for r in reports):
        ctx.item(RECORD_CHAIN, "not_supplied", needs=[f"traces:{r.name}" for r in reports])
        return
    counts = _records(RECORD_CHAIN, runs.check_record_chain, reports)
    status = "recomputed" if counts["traces_checked"] == counts["runs"] else "partial"
    ctx.item(RECORD_CHAIN, status, **counts)
