"""The admission rule, ruling 2 of docs/CONTRACT.md. Admission and activation are separate: this module
decides and records; activation happens elsewhere, only after the signed receipt is stored. Each check
records the counts, case ids and run ids it rests on, so the receipt carries the whole argument.

1. replay: every added guard, alone and composed with every other guard of the candidate, is CLEARED by an
   exhaustive replay over recorded pre-call views. Errors or partial coverage mean INCOMPLETE, never
   clearance. Procedures with natural-language triggers are NOT APPLICABLE to replay.
2. heldout: the parent ran twice fresh (P1, P2) and the immutable candidate twice fresh (C1, C2) on the same
   held-out manifest. At least two stable recoveries (fail in P1 and P2, pass in C1 and C2), and every
   protected case (pass in P1 or P2) passes in both C1 and C2. Parent discordance D is reported, not used.
3. no_regression: no must-succeed case that passed in either parent run fails in either candidate run.
4. benign_twins: zero harness interventions on benign twins, in every split, in either candidate run.
5. overfit_lint: no fixture-specific id, name or amount copied from training traces.
6. authority: additive deltas only. Parent guards and procedures unchanged and in order, the parent prompt a
   byte prefix, ids unique, tools a subset; anything else needs a separately signed human exception.
7. cost: recorded and reported, never a gate.

Only FRESH results count: a CACHED or RECORDED result, or a missing one, is invalid evidence for its case.
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
from ratchet.verify.adapter import DEFAULT_GATES, PREDICATE_URI

GATES = DEFAULT_GATES


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
            "predicateType": PREDICATE_URI,
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


def check_replay(
    parent: Harness, candidate: Harness, replays: Mapping[str, ReplayReport], composed: ReplayReport | None
) -> Check:
    old_g = {g.id for g in parent.guards}
    new = [g.id for g in candidate.guards if g.id not in old_g]
    old_p = {p.id for p in parent.procedures}
    procs = [p.id for p in candidate.procedures if p.id not in old_p]
    rows: list[dict[str, Any]] = []
    for gid in new:
        rep = replays.get(gid)
        if rep is None:
            rows.append({"guard_id": gid, "passed": False, "verdict": "MISSING"})
            continue
        rows.append({
            "guard_id": gid, "passed": rep.passed, "verdict": rep.verdict, "coverage": rep.coverage,
            "snapshot_digest": rep.snapshot,
            "targets_hit": sum(1 for t in rep.targets if t["hit"]), "targets_total": len(rep.targets),
            "target_failures": [t["failure_id"] for t in rep.targets],
            "false_fires": [f"{fp['run_id']}@{fp['seq']}" for fp in rep.false_positives],
            "benign_twin_fires": rep.benign_fires, "errors": len(rep.errors), "checks": rep.checks,
        })  # fmt: skip
    composed_row: dict[str, Any] | None = None
    if new:
        composed_row = (
            {"verdict": "MISSING", "passed": False}
            if composed is None
            else {"verdict": composed.verdict, "passed": composed.passed, "coverage": composed.coverage,
                  "false_fires": len(composed.false_positives), "errors": len(composed.errors),
                  "snapshot_digest": composed.snapshot}
        )  # fmt: skip
    ok = all(r["passed"] for r in rows) and (composed_row is None or bool(composed_row["passed"]))
    return Check(
        "replay",
        ok,
        {
            "new_guards": new,
            "reports": rows,
            "composed": composed_row,
            "procedures": {pid: "NOT APPLICABLE (natural-language trigger)" for pid in procs},
        },
    )


def _outcome(e: EvalReport | None, case_id: str) -> bool | None:
    """True or False for a FRESH result; None when the result is missing or not FRESH (invalid evidence)."""
    if e is None:
        return None
    r = _heldout(e).get(case_id)
    if r is None or r.mode != "FRESH" or r.error or r.infra_failed:
        return None
    return r.passed


def check_fresh(
    parent_runs: Sequence[EvalReport], cand_eval: EvalReport, confirm_eval: EvalReport | None
) -> Check:
    p1, p2 = parent_runs
    manifest = sorted(_heldout(p1))
    same = bool(manifest) and all(sorted(_heldout(e)) == manifest for e in (p2, cand_eval))
    same = same and (confirm_eval is None or sorted(_heldout(confirm_eval)) == manifest)
    vectors = {k: [_outcome(e, k) for e in (p1, p2, cand_eval, confirm_eval)] for k in manifest}
    invalid = {k for k, v in vectors.items() if any(x is None for x in v[:3])}
    if confirm_eval is not None:
        invalid |= {k for k, v in vectors.items() if v[3] is None}

    def stable_in(cols: tuple[int, ...]) -> list[str]:
        return sorted(k for k, v in vectors.items() if v[0] is False and v[1] is False
                      and all(v[c] is True for c in cols))  # fmt: skip

    protected = sorted(k for k, v in vectors.items() if v[0] is True or v[1] is True)
    broken_c1 = sorted(k for k in protected if vectors[k][2] is not True)
    broken_c2 = [k for k in protected if vectors[k][3] is not True] if confirm_eval is not None else []
    broken = sorted(set(broken_c1) | set(broken_c2))
    discordant = sorted(k for k, v in vectors.items() if None not in v[:2] and v[0] != v[1])
    stable, stable_c1 = stable_in((2, 3)), stable_in((2,))
    initial_ok = same and not invalid and len(stable_c1) >= 2 and not broken_c1
    passed = initial_ok and confirm_eval is not None and len(stable) >= 2 and not broken
    label = {True: "pass", False: "fail", None: "invalid"}
    return Check(
        "heldout",
        passed,
        {
            "manifest": len(manifest),
            "same_manifest": same,
            "order": ["P1", "P2", "C1", "C2"],
            "outcomes": {k: [label[x] for x in v] for k, v in vectors.items()},
            "stable_recoveries": stable,
            "stable_recoveries_c1": stable_c1,
            "protected": protected,
            "protected_broken": broken,
            "parent_discordance": len(discordant),
            "discordant": discordant,
            "invalid_evidence": sorted(invalid),
            "initial_ok": initial_ok,
            "confirm_pending": confirm_eval is None,
            "runs": {
                name: sorted(r.run_id for r in _heldout(e).values() if r.run_id)
                for name, e in (("P1", p1), ("P2", p2), ("C1", cand_eval), ("C2", confirm_eval))
                if e is not None
            },
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
    """Zero harness interventions on benign twins in every split either candidate run covered."""
    twins = [r for e in cand_runs for r in e.results if r.kind == "benign_twin"]
    hit = [r for r in twins if r.interventions > 0]
    return Check(
        "benign_twins",
        not hit,
        {
            "benign_twin_runs": len(twins),
            "by_split": {
                sp: sum(1 for r in twins if r.split == sp) for sp in sorted({r.split for r in twins})
            },
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


def check_structure(
    parent: Harness,
    candidate: Harness,
    approval: bytes | None = None,
    trusted_humans: Mapping[str, Ed25519PublicKey] | None = None,
) -> Check:
    """Additive deltas only: parent guards and procedures unchanged and in order, the parent prompt a byte
    prefix, every id unique, the tool allowlist a subset of the parent's."""
    granted = sorted(set(candidate.tool_allowlist) - set(parent.tool_allowlist))
    ids = [g.id for g in candidate.guards] + [p.id for p in candidate.procedures]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    violations = [
        name
        for name, bad in (
            ("tools_granted", bool(granted)),
            ("guards_changed_or_reordered", candidate.guards[: len(parent.guards)] != parent.guards),
            (
                "procedures_changed_or_reordered",
                candidate.procedures[: len(parent.procedures)] != parent.procedures,
            ),
            ("prompt_prefix_lost", not candidate.system_prompt.startswith(parent.system_prompt)),
            ("duplicate_ids", bool(duplicates)),
            ("digest_invalid", not candidate.verify_digest()),
        )
        if bad
    ]
    evidence: dict[str, Any] = {
        "violations": violations,
        "tools_granted": granted,
        "duplicate_ids": duplicates,
        "parent_tools": sorted(parent.tool_allowlist),
        "candidate_tools": sorted(candidate.tool_allowlist),
        "added_guards": len(candidate.guards) - len(parent.guards),
        "added_procedures": len(candidate.procedures) - len(parent.procedures),
        "prompt_bytes_added": len(candidate.system_prompt.encode()) - len(parent.system_prompt.encode()),
    }
    ok = not violations
    if violations and "digest_invalid" not in violations:
        ok, evidence["exception"] = _approval(approval, trusted_humans or {}, candidate)
    return Check("authority", ok, evidence)


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
    if "exception" in authority.evidence and authority.passed:
        out.append(f"structural exception: {authority.evidence['exception']}")
    return out


def admit(
    parent: Harness,
    candidate: Harness,
    *,
    parent_runs: tuple[EvalReport, EvalReport],
    cand_eval: EvalReport,
    confirm_eval: EvalReport | None,
    replays: Mapping[str, ReplayReport],
    composed: ReplayReport | None,
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
    authority = check_structure(parent, candidate, approval, trusted_humans)
    checks = [
        check_replay(parent, candidate, replays, composed),
        check_fresh(parent_runs, cand_eval, confirm_eval),
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
    if composed is not None:
        digests["replay:composed"] = report_digest(composed.to_payload())
    return Decision(
        parent=parent.version,
        candidate=candidate.version,
        candidate_digest=candidate.digest,
        admitted=all(c.passed for c in checks if c.name in GATES),
        checks=checks,
        rates=rates(parent_runs[0], cand_eval),
        visible_case_ids=sorted(visible_case_ids),
        evaluator=dict(evaluator),
        corpus_digests={gid: r.snapshot for gid, r in replays.items()}
        | ({"composed": composed.snapshot} if composed is not None else {}),
        report_digests=digests,
        exceptions=_exceptions(named, replays, authority),
        context=dict(context or {}),
        decided_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def awaiting_confirmation(d: Decision) -> bool:
    """Every gate holds on C1, and only the confirming run C2 is missing."""
    f = d.check("heldout").evidence
    others = all(c.passed for c in d.checks if c.name in GATES and c.name != "heldout")
    return others and bool(f["initial_ok"]) and bool(f["confirm_pending"])
