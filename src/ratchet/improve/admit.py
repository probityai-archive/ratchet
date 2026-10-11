"""The admission rule. A candidate harness is admitted only when every check holds; each check records
the counts, case ids and trace ids it rests on, so the signed receipt carries the whole argument.

1. Replay: every new guard fires on its target failures and on zero previously passing traces.
2. Held-out: the held-out pass count strictly rises (cases the improver never saw).
3. No regression: no must-succeed case that passed under the parent fails under the candidate.
4. Authority does not grow: the tool allowlist is a subset of the parent's and every parent guard is kept
   byte for byte, unless a human-signed approval names this exact candidate digest. Checked here, in
   code, against the two harnesses themselves; no prompt can talk its way past it.
5. Cost is recorded. It is not a gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ratchet.improve.evaluate import EvalReport
from ratchet.improve.receipt import HUMAN_APPROVAL_TYPE, ReceiptError, verify
from ratchet.improve.replay import ReplayReport
from ratchet.types import Harness

GATES = ("replay", "heldout", "no_regression", "authority")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    evidence: dict[str, Any]


@dataclass(frozen=True)
class Decision:
    parent: str
    candidate: str
    candidate_digest: str
    admitted: bool
    checks: list[Check]
    context: dict[str, Any] = field(default_factory=dict)
    decided_at: str = ""

    def check(self, name: str) -> Check:
        return next(c for c in self.checks if c.name == name)

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "decision",
            "parent": self.parent,
            "candidate": self.candidate,
            "candidate_digest": self.candidate_digest,
            "admitted": self.admitted,
            "checks": [{"name": c.name, "passed": c.passed, "evidence": c.evidence} for c in self.checks],
            "context": self.context,
            "decided_at": self.decided_at,
        }


def check_replay(parent: Harness, candidate: Harness, replays: Mapping[str, ReplayReport]) -> Check:
    old = {g.id for g in parent.guards}
    new = [g.id for g in candidate.guards if g.id not in old]
    rows = []
    ok = True
    for gid in new:
        rep = replays.get(gid)
        if rep is None:
            ok = False
            rows.append({"guard_id": gid, "passed": False, "missing": True})
            continue
        ok = ok and rep.passed
        rows.append({
            "guard_id": gid, "passed": rep.passed,
            "targets_hit": sum(1 for t in rep.targets if t["hit"]), "targets_total": len(rep.targets),
            "target_failures": [t["failure_id"] for t in rep.targets],
            "false_positives": [f"{fp['run_id']}@{fp['seq']}" for fp in rep.false_positives],
            "errors": len(rep.errors), "traces_replayed": rep.traces, "checks": rep.checks,
        })  # fmt: skip
    return Check("replay", ok, {"new_guards": new, "reports": rows})


def check_heldout(parent_eval: EvalReport, cand_eval: EvalReport) -> Check:
    p, c = parent_eval.by_case(), cand_eval.by_case()
    p_ids = {k for k, r in p.items() if r.split == "heldout"}
    c_ids = {k for k, r in c.items() if r.split == "heldout"}
    before, after = parent_eval.passed("heldout"), cand_eval.passed("heldout")
    same = p_ids == c_ids and bool(p_ids)
    return Check(
        "heldout",
        same and len(after) > len(before),
        {
            "cases": len(c_ids),
            "same_case_set": same,
            "parent_passed": len(before),
            "candidate_passed": len(after),
            "gained": sorted(after - before),
            "lost": sorted(before - after),
            "candidate_runs": sorted(r.run_id for k, r in c.items() if k in c_ids and r.run_id),
        },
    )


def check_no_regression(parent_eval: EvalReport, cand_eval: EvalReport) -> Check:
    c = cand_eval.by_case()
    must = [r for r in parent_eval.results if r.must_succeed]
    regressed = sorted(r.case_id for r in must if r.passed and not (c.get(r.case_id) and c[r.case_id].passed))
    return Check(
        "no_regression",
        not regressed,
        {
            "must_succeed_cases": len(must),
            "passed_before": sum(1 for r in must if r.passed),
            "regressed": regressed,
            "regressed_runs": [c[k].run_id for k in regressed if k in c],
        },
    )


def _approval(
    approval: bytes | None, trusted: Mapping[str, Ed25519PublicKey], candidate: Harness
) -> tuple[bool, str]:
    if approval is None:
        return False, "no human approval supplied"
    try:
        body = verify(approval, trusted, HUMAN_APPROVAL_TYPE)
    except ReceiptError as exc:
        return False, f"human approval rejected: {exc}"
    if body.get("candidate_digest") != candidate.digest:
        return False, "human approval names a different candidate"
    return True, f"human approval by {body.get('approver', 'unnamed')}"


def check_authority(
    parent: Harness,
    candidate: Harness,
    approval: bytes | None = None,
    trusted_humans: Mapping[str, Ed25519PublicKey] | None = None,
) -> Check:
    granted = sorted(set(candidate.tool_allowlist) - set(parent.tool_allowlist))
    kept = {g.id: g.source for g in candidate.guards}
    removed = sorted(g.id for g in parent.guards if g.id not in kept)
    altered = sorted(g.id for g in parent.guards if g.id in kept and kept[g.id] != g.source)
    grows = bool(granted or removed or altered)
    evidence: dict[str, Any] = {
        "tools_granted": granted,
        "guards_removed": removed,
        "guards_altered": altered,
        "parent_tools": sorted(parent.tool_allowlist),
        "candidate_tools": sorted(candidate.tool_allowlist),
        "digest_valid": candidate.verify_digest(),
    }
    ok = not grows
    if grows:
        ok, evidence["override"] = _approval(approval, trusted_humans or {}, candidate)
    return Check("authority", ok and candidate.verify_digest(), evidence)


def check_cost(parent_eval: EvalReport, cand_eval: EvalReport) -> Check:
    p, c = parent_eval.cost(), cand_eval.cost()
    return Check(
        "cost",
        True,
        {
            "gate": False,
            "parent": {"model_calls": p.model_calls, "input_tokens": p.input_tokens,
                       "output_tokens": p.output_tokens, "tool_calls": p.tool_calls, "wall_s": p.wall_s},
            "candidate": {"model_calls": c.model_calls, "input_tokens": c.input_tokens,
                          "output_tokens": c.output_tokens, "tool_calls": c.tool_calls, "wall_s": c.wall_s},
        },
    )  # fmt: skip


def admit(
    parent: Harness,
    candidate: Harness,
    *,
    parent_eval: EvalReport,
    cand_eval: EvalReport,
    replays: Mapping[str, ReplayReport],
    approval: bytes | None = None,
    trusted_humans: Mapping[str, Ed25519PublicKey] | None = None,
    context: Mapping[str, Any] | None = None,
) -> Decision:
    if candidate.parent != parent.version:
        raise ValueError(f"candidate {candidate.version} is not a child of {parent.version}")
    checks = [
        check_replay(parent, candidate, replays),
        check_heldout(parent_eval, cand_eval),
        check_no_regression(parent_eval, cand_eval),
        check_authority(parent, candidate, approval, trusted_humans),
        check_cost(parent_eval, cand_eval),
    ]
    admitted = all(c.passed for c in checks if c.name in GATES)
    return Decision(
        parent=parent.version,
        candidate=candidate.version,
        candidate_digest=candidate.digest,
        admitted=admitted,
        checks=checks,
        context=dict(context or {}),
        decided_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
