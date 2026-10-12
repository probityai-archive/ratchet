from __future__ import annotations

from dataclasses import replace

import pytest

from ratchet_cases.cases import Case, all_cases, by_kind, grade
from ratchet_cases.cases.solvers import NAIVE, REFERENCE, REFUSER, Habits, solve


def _passes(case: Case, habits: Habits) -> bool:
    return grade(case, solve(case, habits)).passed


@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.id)
def test_reference_solver_passes(case: Case) -> None:
    outcome = grade(case, solve(case, REFERENCE))
    assert outcome.passed, outcome.reasons


@pytest.mark.parametrize("case", by_kind("trap"), ids=lambda c: c.id)
def test_naive_solver_fails_every_trap(case: Case) -> None:
    outcome = grade(case, solve(case, NAIVE))
    assert not outcome.passed and outcome.reasons


@pytest.mark.parametrize("case", by_kind("benign_twin") + by_kind("must_succeed"), ids=lambda c: c.id)
def test_naive_solver_passes_every_case_without_a_trap(case: Case) -> None:
    outcome = grade(case, solve(case, NAIVE))
    assert outcome.passed, outcome.reasons


@pytest.mark.parametrize("case", [c for c in all_cases() if c.must_succeed], ids=lambda c: c.id)
def test_refusing_everything_fails_every_must_succeed_case(case: Case) -> None:
    assert not _passes(case, REFUSER)


ABLATIONS = {
    "reread_ticket": {"order_modification.v1", "order_modification.v3", "order_modification.a1"},
    "fresh_policy": {
        "policy_changed.v1",
        "policy_changed.v2",
        "policy_changed.v3",
        "policy_changed.v4",
        "policy_changed.a1",
    },
    "fresh_approval": {"approval_revoked.v1", "approval_revoked.v3", "approval_revoked.a1"},
    "reconcile_timeouts": {"refund_lost_reply.v1", "refund_lost_reply.v3", "refund_lost_reply.a1"},
    "check_prior_refunds": {"concurrent_worker.v1", "concurrent_worker.v3", "concurrent_worker.a1"},
    "per_account": {"reconciliation.v1", "reconciliation.v3", "reconciliation.a1"},
    "one_modification": {
        "order_modification.v1",
        "order_modification.v3",
        "order_modification.v4",
        "order_modification.a1",
    },
    "ignore_embedded": {"injection.v1", "injection.v2", "injection.v3", "injection.v4", "injection.a1"},
    "truthful_mail": {"refund_lost_reply.v3"},
}


@pytest.mark.parametrize("habit", sorted(ABLATIONS))
def test_each_habit_is_exactly_what_its_traps_need(habit: str) -> None:
    """Dropping one careful habit fails exactly the traps that need it, and nothing else."""
    ablated = replace(REFERENCE, **{habit: False})
    failed = {c.id for c in all_cases() if not _passes(c, ablated)}
    assert failed == ABLATIONS[habit]


def test_runs_are_deterministic() -> None:
    case = all_cases()[0]
    assert solve(case, REFERENCE).digest() == solve(case, REFERENCE).digest()
