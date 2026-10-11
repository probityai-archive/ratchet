from __future__ import annotations

import pytest

from ratchet.bench.tau.metrics import by_task, pass_hat_k, summary
from ratchet.bench.tau.tasks import DEFAULT_SEED, case_id, default_cases, split_cases, split_record, tau_tasks


def test_split_is_seeded_and_disjoint() -> None:
    a = split_cases(5, 50, seed=1, n_train=10)
    b = split_cases(5, 50, seed=1, n_train=10)
    c = split_cases(5, 50, seed=2, n_train=10)
    assert a == b and a != c
    train = [x for x in a if x.split == "train"]
    held = [x for x in a if x.split == "heldout"]
    assert len(train) == 10 and all(x.tau_split == "train" for x in train)
    assert [x.index for x in held] == [0, 1, 2, 3, 4] and all(x.tau_split == "test" for x in held)
    assert all(x.must_succeed and x.kind == "must_succeed" and x.family == "tau-retail" for x in a)
    with pytest.raises(ValueError):
        split_cases(5, 50, n_train=51)


def test_default_cases_cover_the_whole_test_split() -> None:
    cases = default_cases()
    assert len(tau_tasks("test")) == 115 and len(tau_tasks("train")) == 500
    assert sum(c.split == "heldout" for c in cases) == 115
    assert sum(c.split == "train" for c in cases) == 40
    rec = split_record(cases, DEFAULT_SEED)
    assert rec["seed"] == DEFAULT_SEED and len(rec["heldout"]) == 115 and len(rec["digest"]) == 64
    assert rec == split_record(default_cases(), DEFAULT_SEED)
    assert case_id("test", 7) == "tau-retail-test-007"


def test_pass_hat_k_matches_the_combinatorial_definition() -> None:
    runs = {"a": [True, True], "b": [True, False], "c": [False, False]}
    assert pass_hat_k(runs, 1) == pytest.approx(0.5)
    assert pass_hat_k(runs, 2) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        pass_hat_k(runs, 3)
    with pytest.raises(ValueError):
        pass_hat_k(runs, 0)
    with pytest.raises(ValueError):
        pass_hat_k({}, 1)


def test_summary_and_grouping() -> None:
    rows = [
        {"case_id": "a", "passed": True},
        {"case_id": "a", "passed": False},
        {"case_id": "b", "passed": True},
    ]
    grouped = by_task(rows)
    assert grouped == {"a": [True, False], "b": [True]}
    s = summary(grouped, [1, 2])
    assert s == {"tasks": 2, "trials_per_task": 1, "runs": 3, "successes": 2, "pass^1": 0.75}


def test_heldout_subset_is_seeded_and_bounded() -> None:
    a = split_cases(115, 500, seed=3, n_train=5, n_heldout=10)
    held = [c.index for c in a if c.split == "heldout"]
    assert len(held) == 10 and held == sorted(held) and len(set(held)) == 10
    assert a == split_cases(115, 500, seed=3, n_train=5, n_heldout=10)
    assert sum(c.split == "heldout" for c in default_cases(n_train=5, n_heldout=7)) == 7
    with pytest.raises(ValueError):
        split_cases(115, 500, n_heldout=0)
    with pytest.raises(ValueError):
        default_cases("airline", n_heldout=5)
