"""Paired comparison of two harnesses run FRESH on the same cases with the same model.

Inputs are `EvalReport`s from `ratchet.improve.evaluate`, one per repeat per harness, each produced by an
independent fresh run. Nothing here reruns, filters or drops a case: every case in the selection appears in
every count, and a run that errored counts as a failure and is also counted separately.

Definitions (per split, per model):
- pass count of an arm in repeat r: cases passed in that repeat, out of the split's case count.
- noise D of an arm: cases whose outcome differs between that arm's repeats (0 with one repeat).
- discordant pairs, repeat-matched: over every repeat r, cases where base failed and candidate passed (b)
  and where base passed and candidate failed (c); the exact two-sided binomial p over the pooled b and c
  (under no effect each discordant pair is equally likely to go either way).
- strict pairs: a case counts as passed by an arm only when every repeat passed (the admission rule's
  repeat semantics); b and c are then counted once per case.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

from ratchet.improve.evaluate import CaseResult, EvalReport


@dataclass(frozen=True)
class CaseRow:
    case_id: str
    split: str
    kind: str
    must_succeed: bool
    base: tuple[bool, ...]
    cand: tuple[bool, ...]
    base_interventions: tuple[int, ...]
    cand_interventions: tuple[int, ...]
    base_errors: int
    cand_errors: int


@dataclass(frozen=True)
class ArmStats:
    passes: tuple[int, ...]
    strict_passes: int
    noise_d: int
    errors: int
    benign_twin_interventions: int
    must_succeed_failures: int
    non_fresh: int


@dataclass(frozen=True)
class SplitSummary:
    split: str
    cases: int
    repeats: int
    base: ArmStats
    cand: ArmStats
    b: int
    c: int
    p_value: float
    strict_b: int
    strict_c: int
    strict_p_value: float
    by_kind: dict[str, dict[str, Any]]

    def to_payload(self) -> dict[str, Any]:
        def arm(a: ArmStats) -> dict[str, Any]:
            return {
                "passes": list(a.passes),
                "strict_passes": a.strict_passes,
                "noise_d": a.noise_d,
                "errors": a.errors,
                "benign_twin_interventions": a.benign_twin_interventions,
                "must_succeed_failures": a.must_succeed_failures,
                "non_fresh": a.non_fresh,
            }

        return {
            "split": self.split,
            "cases": self.cases,
            "repeats": self.repeats,
            "base": arm(self.base),
            "cand": arm(self.cand),
            "b": self.b,
            "c": self.c,
            "p_value": self.p_value,
            "strict_b": self.strict_b,
            "strict_c": self.strict_c,
            "strict_p_value": self.strict_p_value,
            "by_kind": self.by_kind,
        }


def discordant_pair_p(b: int, c: int) -> float:
    """Exact two-sided binomial p for b of b + c discordant pairs at 1/2: doubled tail, capped at 1."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts cannot be negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / float(1 << n)
    return min(1.0, 2 * tail)


def _check_reports(reports: Sequence[EvalReport], label: str) -> None:
    if not reports:
        raise ValueError(f"{label}: at least one report is required")
    ids = [r.case_id for r in reports[0].results]
    for rep in reports[1:]:
        if [r.case_id for r in rep.results] != ids:
            raise ValueError(f"{label}: every repeat must cover the same cases in the same order")


def paired_rows(base: Sequence[EvalReport], cand: Sequence[EvalReport]) -> list[CaseRow]:
    """One row per case, with every repeat of both arms. Both arms must cover identical case lists."""
    _check_reports(base, "base")
    _check_reports(cand, "cand")
    if len(base) != len(cand):
        raise ValueError("both arms need the same number of repeats")
    base_ids = [r.case_id for r in base[0].results]
    if base_ids != [r.case_id for r in cand[0].results]:
        raise ValueError("both arms must cover the same cases in the same order")
    rows: list[CaseRow] = []
    for i, case_id in enumerate(base_ids):
        b: list[CaseResult] = [rep.results[i] for rep in base]
        c: list[CaseResult] = [rep.results[i] for rep in cand]
        first = b[0]
        rows.append(
            CaseRow(
                case_id=case_id,
                split=first.split,
                kind=first.kind,
                must_succeed=first.must_succeed,
                base=tuple(r.passed for r in b),
                cand=tuple(r.passed for r in c),
                base_interventions=tuple(r.interventions for r in b),
                cand_interventions=tuple(r.interventions for r in c),
                base_errors=sum(1 for r in b if r.error),
                cand_errors=sum(1 for r in c if r.error),
            )
        )
    return rows


def _arm(rows: Sequence[CaseRow], which: str, non_fresh: int) -> ArmStats:
    runs = [r.base if which == "base" else r.cand for r in rows]
    ivs = [r.base_interventions if which == "base" else r.cand_interventions for r in rows]
    repeats = len(runs[0]) if runs else 0
    return ArmStats(
        passes=tuple(sum(1 for o in runs if o[k]) for k in range(repeats)),
        strict_passes=sum(1 for o in runs if all(o)),
        noise_d=sum(1 for o in runs if len(set(o)) > 1),
        errors=sum(r.base_errors if which == "base" else r.cand_errors for r in rows),
        benign_twin_interventions=sum(
            sum(iv) for r, iv in zip(rows, ivs, strict=True) if r.kind == "benign_twin"
        ),
        must_succeed_failures=sum(
            sum(1 for ok in o if not ok) for r, o in zip(rows, runs, strict=True) if r.must_succeed
        ),
        non_fresh=non_fresh,
    )


def _discordance(rows: Sequence[CaseRow]) -> tuple[int, int, int, int]:
    b = sum(1 for r in rows for x, y in zip(r.base, r.cand, strict=True) if not x and y)
    c = sum(1 for r in rows for x, y in zip(r.base, r.cand, strict=True) if x and not y)
    sb = sum(1 for r in rows if not all(r.base) and all(r.cand))
    sc = sum(1 for r in rows if all(r.base) and not all(r.cand))
    return b, c, sb, sc


def _non_fresh(reports: Sequence[EvalReport], split: str) -> int:
    return sum(1 for rep in reports for r in rep.results if r.split == split and r.mode != "FRESH")


def summarize(base: Sequence[EvalReport], cand: Sequence[EvalReport], split: str) -> SplitSummary:
    """Counts for one split. Raises when the split has no cases, so an empty table is never reported."""
    rows = [r for r in paired_rows(base, cand) if r.split == split]
    if not rows:
        raise ValueError(f"no cases in split {split!r}")
    b, c, sb, sc = _discordance(rows)
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({r.kind for r in rows}):
        sub = [r for r in rows if r.kind == kind]
        kb, kc, _, _ = _discordance(sub)
        by_kind[kind] = {
            "cases": len(sub),
            "base_passes": [sum(1 for r in sub if r.base[k]) for k in range(len(sub[0].base))],
            "cand_passes": [sum(1 for r in sub if r.cand[k]) for k in range(len(sub[0].cand))],
            "b": kb,
            "c": kc,
        }
    return SplitSummary(
        split=split,
        cases=len(rows),
        repeats=len(rows[0].base),
        base=_arm(rows, "base", _non_fresh(base, split)),
        cand=_arm(rows, "cand", _non_fresh(cand, split)),
        b=b,
        c=c,
        p_value=discordant_pair_p(b, c),
        strict_b=sb,
        strict_c=sc,
        strict_p_value=discordant_pair_p(sb, sc),
        by_kind=by_kind,
    )
