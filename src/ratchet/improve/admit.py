"""The admission rule, as adjudicated in docs/CONTRACT.md. A candidate is admitted only when every gate
holds; each check records the counts, case ids and run ids it rests on, so the signed receipt carries the
whole argument.

1. replay: every new guard is CLEARED by an exhaustive replay over recorded pre-call views: it fires on
   every target failure at or before the causing step and on zero passing and zero benign-twin traces.
2. heldout: the parent ran twice fresh on the held-out cases; D is the number of cases whose outcome
   differs between those runs. The candidate's fresh held-out pass count must beat the better parent run by
   at least max(2, D + 1), and a second fresh candidate run must confirm it.
3. no_regression: no must-succeed case that passed in either parent run fails in either candidate run.
4. benign_twins: zero harness interventions on held-out benign-twin cases in either candidate run.
5. overfit_lint: the candidate's new text carries no id, amount or name copied from training traces.
6. authority: the tool allowlist is a subset of the parent's and every parent guard is kept byte for byte,
   unless a human-signed approval names this exact candidate digest. Enforced here, in code.
7. cost: recorded and reported, never a gate.

Only FRESH results count; a CACHED result is excluded and listed. Every decision reports the recovery rate
and the false-intervention rate together, names the case ids the improver could see, the evaluator version
with its component digests, the corpus snapshot digest of every replay, the digest of every report it rests
on, and every exception met on the way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ratchet.improve.evaluate import CaseResult, EvalReport
from ratchet.improve.overfit import Literals, lint
from ratchet.improve.receipt import HUMAN_APPROVAL_TYPE, ReceiptError, verify
from ratchet.improve.replay import ReplayReport
from ratchet.types import Harness, canonical_json, sha256_hex

GATES = ("replay", "heldout", "no_regression", "benign_twins", "overfit_lint", "authority")


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
    rates: dict[str, Any] = field(default_factory=dict)
    visible_case_ids: list[str] = field(default_factory=list)
    evaluator: dict[str, Any] = field(default_factory=dict)
    corpus_digests: dict[str, str] = field(default_factory=dict)
    report_digests: dict[str, str] = field(default_factory=dict)
    exceptions: list[str] = field(default_factory=list)
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
            "rates": self.rates,
            "visible_case_ids": self.visible_case_ids,
            "evaluator": self.evaluator,
            "corpus_digests": self.corpus_digests,
            "report_digests": self.report_digests,
            "exceptions": self.exceptions,
            "context": self.context,
            "decided_at": self.decided_at,
        }


def report_digest(payload: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(dict(payload)))


def _heldout(e: EvalReport) -> dict[str, CaseResult]:
    return {r.case_id: r for r in e.results if r.split == "heldout"}


def _fresh_passes(e: EvalReport) -> set[str]:
    return {k for k, r in _heldout(e).items() if r.passed and r.mode == "FRESH"}


def check_replay(parent: Harness, candidate: Harness, replays: Mapping[str, ReplayReport]) -> Check:
    old = {g.id for g in parent.guards}
    new = [g.id for g in candidate.guards if g.id not in old]
    rows: list[dict[str, Any]] = []
    for gid in new:
        rep = replays.get(gid)
        if rep is None:
            rows.append({"guard_id": gid, "passed": False, "missing": True})
            continue
        rows.append({
            "guard_id": gid, "passed": rep.passed, "verdict": rep.verdict, "coverage": rep.coverage,
            "snapshot_digest": rep.snapshot,
            "targets_hit": sum(1 for t in rep.targets if t["hit"]), "targets_total": len(rep.targets),
            "target_failures": [t["failure_id"] for t in rep.targets],
            "false_positives": [f"{fp['run_id']}@{fp['seq']}" for fp in rep.false_positives],
            "benign_twin_fires": rep.benign_fires, "errors": len(rep.errors), "checks": rep.checks,
        })  # fmt: skip
    return Check("replay", all(r["passed"] for r in rows), {"new_guards": new, "reports": rows})


def check_heldout(
    parent_runs: Sequence[EvalReport], cand_eval: EvalReport, confirm_eval: EvalReport | None
) -> Check:
    a, b = parent_runs
    ids_a, ids_b, ids_c = set(_heldout(a)), set(_heldout(b)), set(_heldout(cand_eval))
    discordant = sorted(k for k in ids_a & ids_b if _heldout(a)[k].passed != _heldout(b)[k].passed)
    parent_counts = [len(_fresh_passes(a)), len(_fresh_passes(b))]
    best = max(parent_counts)
    margin = max(2, len(discordant) + 1)
    need = best + margin
    fresh = _fresh_passes(cand_eval)
    confirm = _fresh_passes(confirm_eval) if confirm_eval is not None else None
    same = (
        bool(ids_a)
        and ids_a == ids_b == ids_c
        and (confirm_eval is None or set(_heldout(confirm_eval)) == ids_a)
    )
    initial_ok = same and len(fresh) >= need
    confirmed = confirm is not None and len(confirm) >= need
    before = _fresh_passes(a) | _fresh_passes(b)
    excluded = sorted(k for e in (cand_eval, confirm_eval) if e for k, r in _heldout(e).items()
                      if r.passed and r.mode != "FRESH")  # fmt: skip
    return Check(
        "heldout",
        initial_ok and confirmed,
        {
            "cases": len(ids_c),
            "same_case_set": same,
            "parent_fresh_passed": parent_counts,
            "discordant": discordant,
            "margin": margin,
            "required": need,
            "candidate_fresh_passed": len(fresh),
            "initial_ok": initial_ok,
            "confirm_fresh_passed": None if confirm is None else len(confirm),
            "confirm_pending": confirm_eval is None,
            "confirmed": confirmed,
            "gained": sorted(fresh - before),
            "lost": sorted(before - fresh),
            "excluded_non_fresh": excluded,
            "parent_runs": [sorted(r.run_id for r in _heldout(e).values() if r.run_id) for e in (a, b)],
            "candidate_runs": [
                sorted(r.run_id for r in _heldout(e).values() if r.run_id)
                for e in (cand_eval, confirm_eval)
                if e
            ],
        },
    )


def check_no_regression(parent_runs: Sequence[EvalReport], cand_runs: Sequence[EvalReport]) -> Check:
    passed_before = {r.case_id for e in parent_runs for r in e.results if r.must_succeed and r.passed}
    failed_now = {
        r.case_id: r.run_id for e in cand_runs for r in e.results if r.must_succeed and not r.passed
    }
    regressed = sorted(k for k in failed_now if k in passed_before)
    return Check(
        "no_regression",
        not regressed,
        {
            "must_succeed_passed_in_either_parent_run": len(passed_before),
            "regressed": regressed,
            "regressed_runs": [failed_now[k] for k in regressed],
        },
    )


def check_benign_twins(cand_runs: Sequence[EvalReport]) -> Check:
    twins = [r for e in cand_runs for r in _heldout(e).values() if r.kind == "benign_twin"]
    hit = [r for r in twins if r.interventions > 0]
    return Check(
        "benign_twins",
        not hit,
        {
            "heldout_benign_twin_runs": len(twins),
            "intervened": sorted({r.case_id for r in hit}),
            "intervened_runs": sorted(r.run_id for r in hit if r.run_id),
            "interventions": sum(r.interventions for r in twins),
        },
    )


def rates(parent_eval: EvalReport, cand_eval: EvalReport) -> dict[str, Any]:
    """Recovery rate and false-intervention rate on held-out cases, reported together."""

    def of(e: EvalReport) -> dict[str, Any]:
        held = list(_heldout(e).values())
        traps = [r for r in held if r.kind == "trap"]
        twins = [r for r in held if r.kind == "benign_twin"]
        recovered = sum(1 for r in traps if r.passed and r.mode == "FRESH")
        intervened = sum(1 for r in twins if r.interventions > 0)
        return {
            "traps": len(traps),
            "traps_passed": recovered,
            "recovery_rate": round(recovered / len(traps), 4) if traps else None,
            "benign_twins": len(twins),
            "benign_twins_intervened": intervened,
            "false_intervention_rate": round(intervened / len(twins), 4) if twins else None,
        }

    return {"parent": of(parent_eval), "candidate": of(cand_eval)}


def check_overfit(parent: Harness, candidate: Harness, literals: Literals) -> Check:
    hits = lint(parent, candidate, literals)
    screened = {"ids": len(literals.ids), "amounts": len(literals.amounts), "names": len(literals.names)}
    return Check("overfit_lint", not hits, {"hits": hits, "screened": screened})


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


def _cost_of(e: EvalReport) -> dict[str, Any]:
    c = e.cost()
    return {"model_calls": c.model_calls, "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
            "tool_calls": c.tool_calls, "wall_s": c.wall_s}  # fmt: skip


def check_cost(parent_eval: EvalReport, cand_eval: EvalReport) -> Check:
    return Check(
        "cost", True, {"gate": False, "parent": _cost_of(parent_eval), "candidate": _cost_of(cand_eval)}
    )


def _exceptions(
    reports: Mapping[str, EvalReport], replays: Mapping[str, ReplayReport], authority: Check
) -> list[str]:
    out: list[str] = []
    for name, e in reports.items():
        out += [f"{name}: run error on {r.case_id}: {r.error}" for r in e.results if r.error]
        out += [f"{name}: {r.case_id} is {r.mode}, excluded" for r in e.results if r.mode != "FRESH"]
    for gid, rep in replays.items():
        if rep.unfaithful:
            out.append(
                f"replay {gid}: {len(rep.unfaithful)} trace(s) whose rebuilt world differs from the record"
            )
        out += [f"replay {gid}: {n}" for n in rep.notes]
    if "override" in authority.evidence and authority.passed:
        out.append(f"authority grew under {authority.evidence['override']}")
    return out


def admit(
    parent: Harness,
    candidate: Harness,
    *,
    parent_runs: tuple[EvalReport, EvalReport],
    cand_eval: EvalReport,
    confirm_eval: EvalReport | None,
    replays: Mapping[str, ReplayReport],
    literals: Literals,
    visible_case_ids: Sequence[str],
    evaluator: Mapping[str, Any],
    approval: bytes | None = None,
    trusted_humans: Mapping[str, Ed25519PublicKey] | None = None,
    context: Mapping[str, Any] | None = None,
) -> Decision:
    """`parent_runs` are the parent's two fresh runs (the first over every case, the second at least over
    the held-out cases); `confirm_eval` is the candidate's confirming fresh held-out run, or None before it
    ran."""
    if candidate.parent != parent.version:
        raise ValueError(f"candidate {candidate.version} is not a child of {parent.version}")
    cand_runs = [e for e in (cand_eval, confirm_eval) if e is not None]
    authority = check_authority(parent, candidate, approval, trusted_humans)
    checks = [
        check_replay(parent, candidate, replays),
        check_heldout(parent_runs, cand_eval, confirm_eval),
        check_no_regression(parent_runs, cand_runs),
        check_benign_twins(cand_runs),
        check_overfit(parent, candidate, literals),
        authority,
        check_cost(parent_runs[0], cand_eval),
    ]
    named = {"parent_run_1": parent_runs[0], "parent_run_2": parent_runs[1], "candidate_run": cand_eval}
    if confirm_eval is not None:
        named["candidate_confirm_run"] = confirm_eval
    digests = {k: report_digest(e.to_payload()) for k, e in named.items()}
    digests |= {f"replay:{gid}": report_digest(r.to_payload()) for gid, r in replays.items()}
    return Decision(
        parent=parent.version,
        candidate=candidate.version,
        candidate_digest=candidate.digest,
        admitted=all(c.passed for c in checks if c.name in GATES),
        checks=checks,
        rates=rates(parent_runs[0], cand_eval),
        visible_case_ids=sorted(visible_case_ids),
        evaluator=dict(evaluator),
        corpus_digests={gid: r.snapshot for gid, r in replays.items()},
        report_digests=digests,
        exceptions=_exceptions(named, replays, authority),
        context=dict(context or {}),
        decided_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def awaiting_confirmation(d: Decision) -> bool:
    """Every gate holds except the held-out confirmation, which has not run yet."""
    h = d.check("heldout").evidence
    others = all(c.passed for c in d.checks if c.name in GATES and c.name != "heldout")
    return others and bool(h["initial_ok"]) and bool(h["confirm_pending"])
