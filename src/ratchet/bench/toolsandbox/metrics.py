"""The benchmark's own category scores, with every denominator.

The benchmark reports the mean similarity of a model per scenario category (a scenario carries several
categories) and over all scenarios. These are those means, computed with the benchmark's own bookkeeping:

- a scenario counts in each category it carries, with one exception the benchmark makes: the 3-distractor
  scenarios that the scrambling augmentations are built on are not counted under `THREE_DISTRACTION_TOOLS`
  when they carry a scrambling tag, so that category compares with `TEN_DISTRACTION_TOOLS`;
- a scenario that crashed counts as similarity 0 in the benchmark's runner.

Ratchet adds the rows the benchmark cannot have. A scenario NOT RUN (it needs a key that is absent) or lost
to the model endpoint after the predeclared retries has no similarity; it stays in the count `n` and is shown
separately, never as a pass or a fail. Each category therefore shows two means: over the scenarios that ran
(`ran` of `n`), and over all `n` with the others counted as 0, the benchmark's own treatment of a scenario
that produced no score.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ratchet.bench.toolsandbox.grading import EXACT, PASS_SIMILARITY

ALL = "ALL_CATEGORIES"
THREE = "THREE_DISTRACTION_TOOLS"
SCRAMBLED = frozenset(
    {
        "TOOL_NAME_SCRAMBLED",
        "TOOL_DESCRIPTION_SCRAMBLED",
        "ARG_DESCRIPTION_SCRAMBLED",
        "ARG_TYPE_SCRAMBLED",
        "ARG_NAME_SCRAMBLED",
    }
)
HEADLINE = ("STATE_DEPENDENCY", "CANONICALIZATION", "INSUFFICIENT_INFORMATION")
ORDER = (
    ALL,
    *HEADLINE,
    "SINGLE_TOOL_CALL",
    "MULTIPLE_TOOL_CALL",
    "SINGLE_USER_TURN",
    "MULTIPLE_USER_TURN",
    "NO_DISTRACTION_TOOLS",
    THREE,
    "TEN_DISTRACTION_TOOLS",
    "ALL_TOOLS_AVAILABLE",
    "TOOL_NAME_SCRAMBLED",
    "TOOL_DESCRIPTION_SCRAMBLED",
    "ARG_DESCRIPTION_SCRAMBLED",
    "ARG_TYPE_SCRAMBLED",
    "ARG_NAME_SCRAMBLED",
)
RAN, NOT_RUN, LOST = "ran", "not_run", "lost"


@dataclass(frozen=True)
class Row:
    """One scenario's result in one run. Scores are None unless `status` is `ran`."""

    case_id: str
    name: str
    categories: tuple[str, ...]
    split: str
    status: str
    similarity: float | None = None
    milestone_similarity: float | None = None
    minefield_similarity: float | None = None
    turn_count: int | None = None
    detail: str = ""

    @property
    def ran(self) -> bool:
        return self.status == RAN

    @property
    def passed(self) -> bool:
        return self.similarity is not None and self.similarity >= PASS_SIMILARITY

    @property
    def exact(self) -> bool:
        return self.similarity is not None and self.similarity >= EXACT


def counted_categories(row: Row) -> list[str]:
    """The categories a scenario counts in: its own (less the one exclusion), then `ALL_CATEGORIES`."""
    scrambled = bool(SCRAMBLED & set(row.categories))
    own = [c for c in row.categories if not (c == THREE and scrambled)]
    return [*own, ALL]


def group(rows: Iterable[Row]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = {}
    for row in rows:
        for category in counted_categories(row):
            out.setdefault(category, []).append(row)
    return out


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_category(rows: Sequence[Row]) -> dict[str, Any]:
    ran = [r for r in rows if r.ran]

    def means(field: str) -> tuple[float | None, float | None]:
        values = [float(getattr(r, field)) for r in ran]
        return _mean(values), (sum(values) / len(rows) if rows else None)

    similarity_ran, similarity_all = means("similarity")
    milestone_ran, milestone_all = means("milestone_similarity")
    minefield_ran, minefield_all = means("minefield_similarity")
    return {
        "n": len(rows),
        "ran": len(ran),
        "not_run": sum(1 for r in rows if r.status == NOT_RUN),
        "lost": sum(1 for r in rows if r.status == LOST),
        "similarity_ran": similarity_ran,
        "similarity_all": similarity_all,
        "milestone_ran": milestone_ran,
        "milestone_all": milestone_all,
        "minefield_ran": minefield_ran,
        "minefield_all": minefield_all,
        "turns_ran": _mean([float(r.turn_count) for r in ran if r.turn_count is not None]),
        "passed": sum(1 for r in ran if r.passed),
        "exact": sum(1 for r in ran if r.exact),
    }


def summarize(rows: Sequence[Row]) -> dict[str, dict[str, Any]]:
    """Per category, in the report's order, then any category not in the list."""
    grouped = group(rows)
    keys = [c for c in ORDER if c in grouped] + sorted(c for c in grouped if c not in ORDER)
    return {c: summarize_category(grouped[c]) for c in keys}
