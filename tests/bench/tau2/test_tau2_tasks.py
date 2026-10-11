from __future__ import annotations

import random

import pytest

from ratchet.bench.tau2.tasks import (
    DEFAULT_SEED,
    case_id,
    domain_cases,
    load_tasks,
    task_splits,
    trial_seeds,
    trial_units,
)


def test_base_sets_have_the_benchmark_sizes() -> None:
    assert {d: len(load_tasks(d)) for d in ("retail", "airline", "telecom")} == {
        "retail": 114,
        "airline": 50,
        "telecom": 114,
    }


def test_test_tasks_are_heldout_and_train_tasks_are_train() -> None:
    cases = domain_cases("retail")
    splits = task_splits("retail")
    held = {c.task_id for c in cases if c.split == "heldout"}
    assert held == set(splits["test"]) and len(held) == 40
    assert {c.task_id for c in cases if c.split == "train"} == set(splits["train"])
    assert cases[0].id == case_id("retail", "0") == "tau2-retail-0" and cases[0].must_succeed


def test_named_tasks_keep_their_order_and_unknown_ids_are_refused() -> None:
    assert [c.task_id for c in domain_cases("airline", ["3", "1"])] == ["3", "1"]
    with pytest.raises(ValueError, match="unknown airline task ids"):
        domain_cases("airline", ["999"])


def test_trial_seeds_are_tau2_bench_seeds() -> None:
    random.seed(DEFAULT_SEED)
    expected = [random.randint(0, 1000000) for _ in range(4)]
    assert trial_seeds(DEFAULT_SEED, 4) == expected


def test_trial_units_are_trial_major_with_seeds() -> None:
    cases = domain_cases("airline", ["0", "1"])
    units = trial_units(cases, 2, seed=5)
    assert [(u.task_id, u.trial) for u in units] == [("0", 0), ("1", 0), ("0", 1), ("1", 1)]
    assert [u.seed for u in units] == [trial_seeds(5, 2)[0]] * 2 + [trial_seeds(5, 2)[1]] * 2
