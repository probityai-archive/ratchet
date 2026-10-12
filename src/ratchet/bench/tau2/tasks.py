"""tau2-bench tasks as Ratchet cases, with the benchmark's own train / test split and trial seeds.

A domain's `base` task set is the full set a reportable run covers (retail 114, airline 50, telecom 114
tasks at the pinned commit). tau2-bench divides each base set into a `train` and a `test` split
(`split_tasks.json`); a Ratchet case is `train` when tau2-bench calls its task train and `heldout` when it
calls it test, so the improver never reads a test task. Every task asks the agent to finish the customer's
legitimate work, so every case is `must_succeed`.

Trial seeds are tau2-bench's: `random.seed(seed)` then one `randint(0, 1000000)` per trial, so trial t of a
run here uses the user-simulator seed trial t of `tau2 run` uses.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from ratchet_cases.types import Split

import ratchet.bench.tau2  # noqa: F401  # sets TAU2_DATA_DIR before tau2-bench is imported

Domain = Literal["retail", "airline", "telecom"]
DOMAINS: tuple[Domain, ...] = ("retail", "airline", "telecom")
TASK_SET = "base"
DEFAULT_SEED = 300  # tau2-bench's DEFAULT_SEED


@dataclass(frozen=True)
class Tau2Case:
    """One tau2-bench task, and for a trial unit its trial index and user-simulator seed."""

    id: str
    task_id: str
    domain: Domain
    split: Split
    trial: int = 0
    seed: int | None = None
    family: str = ""
    must_succeed: bool = True
    kind: str = "must_succeed"


def case_id(domain: Domain, task_id: str) -> str:
    return f"tau2-{domain}-{task_id}"


def load_tasks(domain: Domain, task_set: str = TASK_SET) -> list[Any]:
    """tau2-bench's tasks for `domain` (its `base` split by default), in the benchmark's order."""
    from tau2.runner.helpers import get_tasks  # type: ignore[import-untyped]  # tau2-bench ships no types

    return list(get_tasks(domain, task_split_name=task_set))


def task_splits(domain: Domain) -> dict[str, list[str]]:
    from tau2.runner.helpers import load_task_splits

    splits = load_task_splits(domain) or {}
    return {str(k): [str(t) for t in v] for k, v in splits.items()}


def domain_cases(domain: Domain, task_ids: Sequence[str] | None = None) -> list[Tau2Case]:
    """Every base task of `domain` as a case (or the named ones, in the order given)."""
    tasks = [str(t.id) for t in load_tasks(domain)]
    test = set(task_splits(domain).get("test", []))
    if task_ids is not None:
        unknown = [t for t in task_ids if t not in set(tasks)]
        if unknown:
            raise ValueError(f"unknown {domain} task ids: {unknown}")
        tasks = list(task_ids)
    return [
        Tau2Case(
            case_id(domain, t), t, domain, "heldout" if t in test else "train", family=family_of(domain, t)
        )
        for t in tasks
    ]


def family_of(domain: Domain, task_id: str) -> str:
    """A case family: the issue category for telecom (`[mobile_data_issue]...`), else the domain."""
    if task_id.startswith("[") and "]" in task_id:
        return f"tau2-{domain}-{task_id[1 : task_id.index(']')]}"
    return f"tau2-{domain}"


def trial_seeds(seed: int, trials: int) -> list[int]:
    """tau2-bench's per-trial seeds for a run seeded with `seed`."""
    rng = random.Random(seed)
    return [rng.randint(0, 1000000) for _ in range(trials)]


def trial_units(cases: Sequence[Tau2Case], trials: int, seed: int = DEFAULT_SEED) -> list[Tau2Case]:
    """Every case at every trial, trial-major, as tau2-bench orders a run."""
    seeds = trial_seeds(seed, trials)
    return [replace(c, trial=t, seed=seeds[t]) for t in range(trials) for c in cases]
