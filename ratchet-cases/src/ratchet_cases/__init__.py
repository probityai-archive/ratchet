"""Agent cases graded on final state: a simulated company, deterministic faults, exact graders.

`SUITE_DIGEST` is computed from the shipped cases when the package is imported; `released_digest()` is
the digest this version was released with. The two are equal in every release, and a test enforces it.
"""

from __future__ import annotations

from ratchet_cases.cases import (
    CASE_KINDS,
    SPLITS,
    Case,
    CaseKind,
    Expect,
    Family,
    all_cases,
    by_family,
    by_kind,
    by_split,
    families,
    get,
    grade,
    suite_digest,
)
from ratchet_cases.cases.registry import released_digests
from ratchet_cases.cases.solvers import NAIVE, REFERENCE, REFUSER, Habits, solve
from ratchet_cases.world import Fault, World

__version__ = "0.1.0"

SUITE_DIGEST = suite_digest()


def released_digest(version: str = __version__) -> str:
    """The suite digest recorded for `version` in `DIGESTS.json`. Raises KeyError for an unknown version."""
    return released_digests()[version]


__all__ = [
    "CASE_KINDS",
    "NAIVE",
    "REFERENCE",
    "REFUSER",
    "SPLITS",
    "SUITE_DIGEST",
    "Case",
    "CaseKind",
    "Expect",
    "Family",
    "Fault",
    "Habits",
    "World",
    "__version__",
    "all_cases",
    "by_family",
    "by_kind",
    "by_split",
    "families",
    "get",
    "grade",
    "released_digest",
    "solve",
    "suite_digest",
]
