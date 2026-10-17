"""Stochastic policy v3: the fresh worker-evaluation gate for nondeterministic executions (docs/CONTRACT.md).

Every arm runs 40 fresh executions of every declared case, split prospectively into two panels of 20 (an
execution's panel is fixed by its repetition index before any outcome exists). Arms are interleaved by a
predeclared randomized schedule and share a predeclared seed schedule. A cohort is qualified by a parent
reference arm and two identity-control arms before any candidate in it can be admitted, and only a cohort of
at least `min_cohort_cases` cases can be qualified at all (version v3.1; see `cohort_size`).

This policy is a tolerance-based empirical policy. It does not establish zero regression, statistical
significance or universal reliability, and a decision made under it never claims zero regressions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any

from ratchet_cases.types import sha256_hex

from ratchet.canonical import canonical_json
from ratchet.types import Harness

POLICY: dict[str, Any] = {
    "name": "stochastic",
    "version": "v3.1",
    "runs_per_arm": 40,
    "panels": 2,
    "panel_size": 20,
    "high_pass_min": 30,
    "low_pass_max": 10,
    "recovery_panel_margin": 8,
    "min_stable_recoveries": 2,
    "max_stable_regressions": 0,
    "max_case_decline": 8,
    "max_aggregate_decline_points": 5.0,
    "min_cohort_cases": 8,
}
RUNS: int = 40
PANELS: int = 2


def panel_of(rep: int) -> int:
    """Panel membership is a function of the repetition index alone, fixed before outcomes exist."""
    return rep % PANELS


@dataclass(frozen=True)
class Execution:
    arm: str
    case_id: str
    rep: int
    panel: int
    seed: int | None
    passed: bool | None  # None: missing or invalid (infrastructure lost, cached, errored)
    run_id: str | None = None
    interventions: int = 0
    attempts: int = 1
    note: str = ""
    endpoint: str = ""


@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    kind: str
    must_succeed: bool


@dataclass
class Arm:
    name: str
    harness_version: str
    executions: list[Execution] = field(default_factory=list)

    def of(self, case_id: str) -> list[Execution]:
        return [e for e in self.executions if e.case_id == case_id]

    def score(self, case_id: str, panel: int | None = None) -> int:
        return sum(1 for e in self.of(case_id) if e.passed and (panel is None or e.panel == panel))

    def invalid(self, case_id: str) -> int:
        return sum(1 for e in self.of(case_id) if e.passed is None)

    def complete(self, case_id: str) -> bool:
        rows = self.of(case_id)
        return len(rows) == RUNS and all(e.passed is not None for e in rows)


def classify(score: int) -> str:
    if score >= POLICY["high_pass_min"]:
        return "high"
    if score <= POLICY["low_pass_max"]:
        return "low"
    return "intermediate"


def _mean(arm: Arm, cases: Sequence[CaseInfo], panel: int | None) -> float:
    size = RUNS if panel is None else POLICY["panel_size"]
    return 100.0 * sum(arm.score(c.case_id, panel) / size for c in cases) / len(cases) if cases else 0.0


def compare(parent: Arm, other: Arm, cases: Sequence[CaseInfo]) -> dict[str, Any]:
    """Every per-case and aggregate quantity of the policy, for one arm against the parent reference."""
    rows = []
    for c in cases:
        p, o = parent.score(c.case_id), other.score(c.case_id)
        pp = [parent.score(c.case_id, j) for j in range(PANELS)]
        op = [other.score(c.case_id, j) for j in range(PANELS)]
        pc, oc = classify(p), classify(o)
        recovery = (
            pc == "low"
            and oc == "high"
            and all(op[j] - pp[j] >= POLICY["recovery_panel_margin"] for j in range(PANELS))
        )
        rows.append({
            "case_id": c.case_id, "kind": c.kind, "must_succeed": c.must_succeed,
            "parent": p, "parent_panels": pp, "arm": o, "arm_panels": op,
            "parent_class": pc, "arm_class": oc, "class_changed": pc != oc,
            "stable_recovery": recovery, "stable_regression": pc == "high" and oc == "low",
            "decline": p - o, "decline_ok": p - o <= POLICY["max_case_decline"],
            "invalid": {"parent": parent.invalid(c.case_id), "arm": other.invalid(c.case_id)},
            "complete": parent.complete(c.case_id) and other.complete(c.case_id),
        })  # fmt: skip
    aggregate = {
        "pooled": {"parent": _mean(parent, cases, None), "arm": _mean(other, cases, None)},
        **{
            f"panel_{j}": {"parent": _mean(parent, cases, j), "arm": _mean(other, cases, j)}
            for j in range(PANELS)
        },
    }
    for v in aggregate.values():
        v["decline_points"] = round(v["parent"] - v["arm"], 4)
        v["ok"] = v["decline_points"] <= POLICY["max_aggregate_decline_points"]
        v["parent"], v["arm"] = round(v["parent"], 4), round(v["arm"], 4)
    recoveries = [r["case_id"] for r in rows if r["stable_recovery"]]
    regressions = [r["case_id"] for r in rows if r["stable_regression"]]
    incomplete = [r["case_id"] for r in rows if not r["complete"]]
    non_regression = (
        not regressions and all(r["decline_ok"] for r in rows) and all(v["ok"] for v in aggregate.values())
        and not incomplete and bool(rows)
    )  # fmt: skip
    return {
        "cases": rows,
        "aggregate": aggregate,
        "stable_recoveries": recoveries,
        "stable_regressions": regressions,
        "case_declines_over_limit": [r["case_id"] for r in rows if not r["decline_ok"]],
        "incomplete_cases": incomplete,
        "non_regression_checks_pass": non_regression,
        "must_succeed": [r for r in rows if r["must_succeed"]],
    }


def min_cohort_cases(policy: Mapping[str, Any]) -> int:
    """The smallest case count at which the aggregate tolerance is no stricter than the per-case tolerance.

    One case may decline by `max_case_decline` of 40, and nothing stops all of it falling in one panel of
    20. Spread over n cases, that moves the panel mean by 100 * min(max_case_decline, panel_size) /
    (panel_size * n) points. The aggregate limit accepts it only when n is at least this value (8 under
    the v3 tolerances). Below it, the aggregate check refuses a decline the per-case tolerance accepts, and
    for one case it refuses a single lost execution in a panel: it measures one run's noise, not drift."""
    worst = min(policy["max_case_decline"], policy["panel_size"])
    points = Fraction(100 * worst, policy["panel_size"])
    return math.ceil(points / Fraction(str(policy["max_aggregate_decline_points"])))


def cohort_size(cases: Sequence[Any]) -> dict[str, Any]:
    """Whether the declared case manifest is large enough for the v3 tolerances to mean anything. It reads
    the manifest only, never an outcome, so it is fixed before the cohort runs."""
    ok = len(cases) >= POLICY["min_cohort_cases"]
    return {
        "cases": len(cases),
        "min_cases": POLICY["min_cohort_cases"],
        "ok": ok,
        "reason": None if ok else "cohort_too_small",
    }


def qualify(parent: Arm, controls: Sequence[Arm], cases: Sequence[CaseInfo]) -> dict[str, Any]:
    """The cohort is QUALIFIED only when it has at least `min_cohort_cases` cases and each identity control
    passes every non-regression check against the parent reference and shows fewer than two stable
    recoveries. A smaller cohort is TOO_SMALL whatever its controls did; they are still compared and
    reported, so the run stands as a path check."""
    results = []
    for arm in controls:
        cmp = compare(parent, arm, cases)
        ok = (
            cmp["non_regression_checks_pass"]
            and len(cmp["stable_recoveries"]) < POLICY["min_stable_recoveries"]
        )
        results.append({"arm": arm.name, "qualifies": ok, **cmp})
    size = cohort_size(cases)
    qualified = len(controls) == 2 and all(r["qualifies"] for r in results)
    status = "TOO_SMALL" if not size["ok"] else "QUALIFIED" if qualified else "UNQUALIFIED"
    return {"status": status, "size": size, "controls": results}


def effective_change(parent: Harness, candidate: Harness) -> dict[str, bool]:
    """What the worker can actually see or run differently; ids, provenance and parents do not count."""
    return {
        "system_prompt": candidate.system_prompt != parent.system_prompt,
        "procedures": [(p.name, p.when, p.steps) for p in candidate.procedures]
        != [(p.name, p.when, p.steps) for p in parent.procedures],
        "guards": [g.source for g in candidate.guards] != [g.source for g in parent.guards],
        "tool_permissions": sorted(candidate.tool_allowlist) != sorted(parent.tool_allowlist),
    }


def schedule(arms: Sequence[str], cases: Sequence[str], seed: int) -> list[tuple[str, str, int]]:
    """The predeclared randomized interleaving: every (arm, case, repetition) once, shuffled by `seed`, so
    endpoint drift falls on every arm alike."""
    order = [(a, c, r) for r in range(RUNS) for c in cases for a in arms]
    random.Random(seed).shuffle(order)
    return order


def schedule_digest(order: Sequence[tuple[str, str, int]]) -> str:
    return sha256_hex(canonical_json([list(x) for x in order]))


Runner = Callable[[str, str, int], Execution]


def candidate_gate(
    cohort: Mapping[str, Any], cmp: Mapping[str, Any], benign_interventions: int, eligible: bool
) -> dict[str, Any]:
    checks = {
        "cohort_size": cohort_size(cmp["cases"])["ok"],
        "cohort_qualified": cohort["status"] == "QUALIFIED",
        "effective_change": eligible,
        "min_stable_recoveries": len(cmp["stable_recoveries"]) >= POLICY["min_stable_recoveries"],
        "no_stable_regressions": not cmp["stable_regressions"],
        "per_case_decline_within_tolerance": not cmp["case_declines_over_limit"],
        "aggregate_decline_within_tolerance": all(v["ok"] for v in cmp["aggregate"].values()),
        "all_executions_valid": not cmp["incomplete_cases"],
        "zero_benign_twin_interventions": benign_interventions == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def execution_record(e: Execution) -> dict[str, Any]:
    return asdict(e)
