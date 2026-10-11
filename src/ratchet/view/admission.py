"""Reading an admission decision's evidence: the replay gate per guard, and the four paired runs.

The fresh gate is shown as the per-case outcome vector over P1, P2 (the parent run twice) and C1, C2 (the
candidate run twice), with the stable recoveries, the protected cases and any case that broke, never as a
net count. The replay gate is shown as `k/N checked` with its verdict: CLEARED needs N/N, a miss or a false
fire on complete evidence is REJECTED, and an error or partial coverage is INCOMPLETE.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

RUNS = ("P1", "P2", "C1", "C2")
Role = Literal["broken", "stable", "protected", "unrecovered", "invalid"]
ROLE_ORDER: tuple[Role, ...] = ("broken", "invalid", "stable", "protected", "unrecovered")
STABLE_NEEDED = 2
_COVERAGE = re.compile(r"(\d+)/(\d+)")


@dataclass(frozen=True)
class ReplayRow:
    """One guard's replay (or the composed guard runtime's) with its coverage and verdict."""

    guard_id: str
    verdict: str
    checked: int | None
    total: int | None
    targets_hit: int | None
    targets_total: int | None
    false_fires: int
    benign_fires: int
    errors: int

    @property
    def coverage(self) -> str:
        """`k/N checked`, the words the replay views use."""
        return "no coverage recorded" if self.checked is None else f"{self.checked}/{self.total} checked"

    @property
    def summary(self) -> str:
        return f"{self.coverage}, {self.verdict}"


def parse_coverage(text: Any) -> tuple[int | None, int | None]:
    m = _COVERAGE.search(str(text or ""))
    return (int(m[1]), int(m[2])) if m else (None, None)


def _row(guard_id: str, rep: Mapping[str, Any]) -> ReplayRow:
    checked, total = parse_coverage(rep.get("coverage"))
    errors = rep.get("errors", 0)
    fires = rep.get("false_fires", rep.get("false_positives", 0))
    return ReplayRow(
        guard_id=guard_id,
        verdict=str(rep.get("verdict") or ("CLEARED" if rep.get("passed") else "REJECTED")),
        checked=checked,
        total=total,
        targets_hit=rep.get("targets_hit"),
        targets_total=rep.get("targets_total"),
        false_fires=len(fires) if isinstance(fires, list) else int(fires or 0),
        benign_fires=int(rep.get("benign_twin_fires", 0)),
        errors=len(errors) if isinstance(errors, list) else int(errors or 0),
    )


def replay_rows(evidence: Mapping[str, Any]) -> list[ReplayRow]:
    """One row per added guard, then the composed guard runtime when there is one."""
    rows = [_row(str(r.get("guard_id")), r) for r in evidence.get("reports") or []]
    composed = evidence.get("composed")
    if composed:
        rows.append(_row("(all guards composed)", composed))
    return rows


def procedures_not_applicable(evidence: Mapping[str, Any]) -> list[str]:
    return sorted(evidence.get("procedures") or {})


@dataclass(frozen=True)
class FreshRow:
    case_id: str
    outcomes: tuple[str, str, str, str]  # "pass", "fail", "invalid" or "pending", in P1 P2 C1 C2 order
    role: Role


@dataclass(frozen=True)
class FreshView:
    """The paired-recovery gate, case by case."""

    rows: list[FreshRow]
    stable: list[str]
    protected: list[str]
    broken: list[str]
    invalid: list[str]
    discordance: int
    confirm_pending: bool
    same_manifest: bool
    needed: int = STABLE_NEEDED

    @property
    def stable_c1(self) -> list[str]:
        return [r.case_id for r in self.rows if r.outcomes[:3] == ("fail", "fail", "pass")]


def _role(case: str, ev: Mapping[str, Any], outcomes: tuple[str, ...]) -> Role:
    if case in (ev.get("protected_broken") or []):
        return "broken"
    if "invalid" in outcomes or case in (ev.get("invalid_evidence") or []):
        return "invalid"
    if case in (ev.get("stable_recoveries") or []):
        return "stable"
    if case in (ev.get("protected") or []):
        return "protected"
    return "unrecovered"


def fresh_view(evidence: Mapping[str, Any]) -> FreshView | None:
    """The four outcome vectors and the sets drawn from them; None when the decision records no fresh gate."""
    outcomes = evidence.get("outcomes")
    if not isinstance(outcomes, Mapping):
        return None
    pending = bool(evidence.get("confirm_pending"))
    rows: list[FreshRow] = []
    for case, vec in outcomes.items():
        v = [str(x) for x in vec] + ["pending"] * 4
        four = (v[0], v[1], v[2], "pending" if pending else v[3])
        rows.append(FreshRow(str(case), four, _role(str(case), evidence, four)))
    rows.sort(key=lambda r: (ROLE_ORDER.index(r.role), r.case_id))
    return FreshView(
        rows=rows,
        stable=list(evidence.get("stable_recoveries") or []),
        protected=list(evidence.get("protected") or []),
        broken=list(evidence.get("protected_broken") or []),
        invalid=list(evidence.get("invalid_evidence") or []),
        discordance=int(evidence.get("parent_discordance", 0)),
        confirm_pending=pending,
        same_manifest=bool(evidence.get("same_manifest", True)),
    )


# -- reasons ----------------------------------------------------------------------------------------------


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def check_reasons(check: Mapping[str, Any]) -> list[str]:
    """Why one admission check failed, in the words of the demo ("blocks 4 past successes")."""
    if check.get("passed"):
        return []
    name, ev = str(check.get("name")), check.get("evidence") or {}
    if name == "replay":
        return _replay_reasons(ev)
    if name == "fresh":
        return _fresh_reasons(ev)
    if name == "no_regression":
        n = len(ev.get("regressed") or [])
        return [f"regresses {_plural(n, 'must-succeed case', 'must-succeed cases')}"]
    if name == "benign_twins":
        n = len(ev.get("intervened") or [])
        return [f"intervenes on {_plural(n, 'held-out benign twin', 'held-out benign twins')}"]
    if name == "overfit_lint":
        hits = ev.get("hits") or []
        shown = ", ".join(str(h.get("literal")) for h in hits[:3])
        return [f"copies {len(hits)} literal(s) from training traces ({shown})"]
    if name == "structure":
        return _structure_reasons(ev)
    return [f"{name} failed"]


def _replay_reasons(ev: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for rep in ev.get("reports") or []:
        if rep.get("verdict") == "MISSING" or rep.get("missing"):
            out.append(f"no replay recorded for {rep.get('guard_id')}")
            continue
        row = _row(str(rep.get("guard_id")), rep)
        if row.verdict == "INCOMPLETE":
            out.append(f"replay INCOMPLETE ({row.coverage}, {_plural(row.errors, 'error', 'errors')})")
        passing = row.false_fires - row.benign_fires
        if passing > 0:
            out.append(f"blocks {_plural(passing, 'past success', 'past successes')}")
        if row.benign_fires:
            out.append(f"fires on {_plural(row.benign_fires, 'benign twin', 'benign twins')}")
        miss = (row.targets_total or 0) - (row.targets_hit or 0)
        if miss > 0:
            out.append(f"misses {miss} of {row.targets_total} target failures")
    composed = ev.get("composed")
    if composed and not composed.get("passed"):
        row = _row("composed", composed)
        out.append(f"composed guards: {row.summary}")
    return out or ["replay failed"]


def _fresh_reasons(ev: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    if not ev.get("same_manifest", True):
        out.append("the four runs did not use the same case manifest")
    invalid = ev.get("invalid_evidence") or []
    if invalid:
        out.append(f"{_plural(len(invalid), 'case lacks', 'cases lack')} valid FRESH evidence")
    c1 = len(ev.get("stable_recoveries_c1") or [])
    stable = ev.get("stable_recoveries") or []
    broken = ev.get("protected_broken") or []
    if broken:
        out.append("protected case broke: " + ", ".join(str(b) for b in broken[:3]))
    if ev.get("confirm_pending"):
        out.append(
            f"stable recoveries so far {c1}, needs {STABLE_NEEDED}; awaiting the confirming run C2"
            if c1 < STABLE_NEEDED
            else "awaiting the confirming run C2"
        )
    elif len(stable) < STABLE_NEEDED:
        out.append(f"{len(stable)} stable recoveries over C1 and C2, needs {STABLE_NEEDED}")
    return out or ["paired recovery not shown"]


def _structure_reasons(ev: Mapping[str, Any]) -> list[str]:
    names = {
        "tools_granted": "grants tools: " + ", ".join(ev.get("tools_granted") or []),
        "guards_changed_or_reordered": "changes or reorders existing guards",
        "procedures_changed_or_reordered": "changes or reorders existing procedures",
        "prompt_prefix_lost": "rewrites the existing prompt",
        "duplicate_ids": "duplicate ids: " + ", ".join(ev.get("duplicate_ids") or []),
        "digest_invalid": "digest does not match content",
    }
    return [names.get(v, v) for v in ev.get("violations") or []] or ["structure rule broken"]
