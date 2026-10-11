"""tau-bench's reliability metric, pass^k, over repeated trials of the same tasks.

For a task run n times with c successes, the chance that k runs drawn without replacement all succeed is
C(c, k) / C(n, k); pass^k is its mean over tasks (tau-bench, Yao et al. 2024, section 5). pass^1 is the mean
success rate. A trial that crashed counts as a failure, never as a missing run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import comb
from typing import Any


def pass_hat_k(successes: Mapping[str, list[bool]], k: int) -> float:
    """pass^k over tasks; every task must have at least k trials."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not successes:
        raise ValueError("no tasks")
    total = 0.0
    for task, runs in successes.items():
        n = len(runs)
        if n < k:
            raise ValueError(f"task {task} has {n} trials, fewer than k={k}")
        total += comb(sum(runs), k) / comb(n, k)
    return total / len(successes)


def by_task(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[bool]]:
    """Group result rows (`case_id`, `passed`) into per-task success lists, in row order."""
    out: dict[str, list[bool]] = {}
    for r in rows:
        out.setdefault(str(r["case_id"]), []).append(bool(r["passed"]))
    return out


def summary(successes: Mapping[str, list[bool]], ks: Iterable[int]) -> dict[str, Any]:
    trials = min((len(v) for v in successes.values()), default=0)
    out: dict[str, Any] = {
        "tasks": len(successes),
        "trials_per_task": trials,
        "runs": sum(len(v) for v in successes.values()),
        "successes": sum(sum(v) for v in successes.values()),
    }
    for k in ks:
        if 1 <= k <= trials:
            out[f"pass^{k}"] = round(pass_hat_k(successes, k), 4)
    return out
