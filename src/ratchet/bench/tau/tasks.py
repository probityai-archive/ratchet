"""tau-bench tasks (retail and airline) as Ratchet cases, with a recorded train / held-out split.

Retail. Held-out is tau-bench's own retail test split, all 115 tasks, so the held-out pass^k is the number
the benchmark reports. Train is a seeded sample of tau-bench's retail train split (500 tasks, disjoint from
test).

Airline. tau-bench ships one airline split, the 50 test tasks (`tau_bench/envs/airline/tasks_test.py`; the
environment refuses any other split). They are divided with a fixed seed into train and held-out. A run over
both sides together is a run over the whole benchmark, which is how the full-set pass^k is reported.

The improver reads train traces only, so it never sees a held-out task, its instruction, or its gold actions.
Every task asks the agent to finish the customer's legitimate work (a database change, an answer, or both),
so every case is `must_succeed`: refusing or transferring a task the agent could do fails it.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Literal

from ratchet.types import Split

Domain = Literal["retail", "airline"]
TauSplit = Literal["train", "test"]
DOMAINS: tuple[Domain, ...] = ("retail", "airline")
DEFAULT_SEED = 20261011
DEFAULT_TRAIN: dict[Domain, int] = {"retail": 40, "airline": 20}
# Records a guard may need, per domain: the tables keyed by ids that appear in a conversation. Catalogue
# tables (retail products, airline flights) are left out of the view; tools return them on request.
VIEW_TABLES: dict[Domain, tuple[str, ...]] = {
    "retail": ("users", "orders"),
    "airline": ("users", "reservations"),
}


@dataclass(frozen=True)
class TauCase:
    """One tau-bench task. `index` is its position in tau-bench's `tau_split` task list for `domain`."""

    id: str
    split: Split
    tau_split: TauSplit
    index: int
    domain: Domain = "retail"
    must_succeed: bool = True
    kind: str = "must_succeed"


def case_id(tau_split: TauSplit, index: int, domain: Domain = "retail") -> str:
    return f"tau-{domain}-{tau_split}-{index:03d}"


def tau_tasks(tau_split: TauSplit, domain: Domain = "retail") -> list[Any]:
    if domain == "airline":
        if tau_split != "test":
            raise ValueError("tau-bench airline has only the test split")
        from tau_bench.envs.airline.tasks_test import TASKS  # type: ignore[import-untyped]  # no type info

        return list(TASKS)
    if tau_split == "test":
        from tau_bench.envs.retail.tasks_test import TASKS_TEST  # type: ignore[import-untyped]  # no type info

        return list(TASKS_TEST)
    from tau_bench.envs.retail.tasks_train import TASKS_TRAIN  # type: ignore[import-untyped]  # no type info

    return list(TASKS_TRAIN)


def split_cases(
    n_test: int,
    n_train_pool: int,
    *,
    seed: int = DEFAULT_SEED,
    n_train: int = 40,
    n_heldout: int | None = None,
) -> list[TauCase]:
    """Retail: `n_train` indices drawn from the train split with `seed`; held-out is the whole test split, or
    `n_heldout` test indices drawn with `seed + 1` when an experiment's run budget cannot cover all of it."""
    if not 0 <= n_train <= n_train_pool:
        raise ValueError(f"n_train must be between 0 and {n_train_pool}")
    if n_heldout is not None and not 0 < n_heldout <= n_test:
        raise ValueError(f"n_heldout must be between 1 and {n_test}")
    train = sorted(random.Random(seed).sample(range(n_train_pool), n_train))
    held = (
        range(n_test)
        if n_heldout is None
        else sorted(random.Random(seed + 1).sample(range(n_test), n_heldout))
    )
    cases = [TauCase(case_id("train", i), "train", "train", i) for i in train]
    cases += [TauCase(case_id("test", i), "heldout", "test", i) for i in held]
    return cases


def split_single(
    n_tasks: int, domain: Domain, *, seed: int = DEFAULT_SEED, n_train: int = 20
) -> list[TauCase]:
    """`n_train` of the `n_tasks` test tasks, drawn with `seed`, are train; the rest are held-out."""
    if not 0 <= n_train <= n_tasks:
        raise ValueError(f"n_train must be between 0 and {n_tasks}")
    train = set(random.Random(seed).sample(range(n_tasks), n_train))
    return [
        TauCase(case_id("test", i, domain), "train" if i in train else "heldout", "test", i, domain)
        for i in range(n_tasks)
    ]


def default_cases(
    domain: Domain = "retail",
    *,
    seed: int = DEFAULT_SEED,
    n_train: int | None = None,
    n_heldout: int | None = None,
) -> list[TauCase]:
    n = DEFAULT_TRAIN[domain] if n_train is None else n_train
    if domain == "airline":
        if n_heldout is not None:
            raise ValueError("airline's held-out side is the 50 tasks minus train; n_heldout does not apply")
        return split_single(len(tau_tasks("test", "airline")), "airline", seed=seed, n_train=n)
    tests, pool = len(tau_tasks("test")), len(tau_tasks("train"))
    return split_cases(tests, pool, seed=seed, n_train=n, n_heldout=n_heldout)


def split_record(cases: list[TauCase], seed: int) -> dict[str, Any]:
    """The split as data: seed, ids per side, and a digest of the whole assignment."""
    # The id names the domain, so the rows (and the digest) need no separate domain field.
    rows = [{"id": c.id, "split": c.split, "tau_split": c.tau_split, "index": c.index} for c in cases]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "seed": seed,
        "train": [c.id for c in cases if c.split == "train"],
        "heldout": [c.id for c in cases if c.split == "heldout"],
        "digest": digest,
    }
