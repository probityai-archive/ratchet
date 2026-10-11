"""Pass rates from every recorded run, with their denominators.

A task's score is the fraction of its runs that passed; a task set's pass rate is passed runs over all runs
(each task with the same number of runs, so this is the mean of the task scores). A run that crashed counts as
a failed run, never as a missing one. Every number is printed with the counts it came from.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ratchet.bench.tau.store import JsonlStore


@dataclass(frozen=True)
class Rate:
    tasks: int
    runs: int
    passed: int
    per_task: dict[str, tuple[int, int]]
    errors: int

    @property
    def value(self) -> float | None:
        return self.passed / self.runs if self.runs else None

    def text(self) -> str:
        if not self.runs:
            return "no runs"
        return (
            f"{self.passed}/{self.runs} runs = {100 * self.passed / self.runs:.1f}% over {self.tasks} tasks"
        )


def collect(root: Path) -> list[dict[str, Any]]:
    """Every recorded evaluation report under `root` (trials and evolve runs), with its source directory."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(root).rglob("evals.jsonl")):
        for rec in JsonlStore(path.parent).read("evals"):
            rep = rec["report"]
            if rep.get("type") == "eval" and "results" in rep:
                rows.append({**rep, "source": str(path.parent.relative_to(root)), "eval_id": rec["eval_id"]})
    return rows


def results_for(rows: Iterable[Mapping[str, Any]], harness: str, sources: Sequence[str] = ()) -> list[Any]:
    """Every per-case result of `harness`, optionally only from sources starting with one of `sources`."""
    out: list[Any] = []
    for rep in rows:
        if rep["harness_version"] != harness:
            continue
        if sources and not any(str(rep["source"]).startswith(s) for s in sources):
            continue
        out.extend(rep["results"])
    return out


def rate(results: Iterable[Mapping[str, Any]], case_ids: Iterable[str]) -> Rate:
    wanted = set(case_ids)
    per: dict[str, list[bool]] = {c: [] for c in sorted(wanted)}
    errors = 0
    for r in results:
        if r["case_id"] in wanted:
            per[r["case_id"]].append(bool(r["passed"]))
            errors += 1 if r.get("error") else 0
    counts = {c: (sum(v), len(v)) for c, v in per.items()}
    return Rate(
        tasks=sum(1 for v in per.values() if v),
        runs=sum(n for _, n in counts.values()),
        passed=sum(p for p, _ in counts.values()),
        per_task=counts,
        errors=errors,
    )


def lifelong(phase1: Rate, transfer: Rate, final: Rate) -> float | None:
    """Unweighted mean of the three staged pass rates, when all three exist."""
    values = [phase1.value, transfer.value, final.value]
    if any(v is None for v in values):
        return None
    return sum(v for v in values if v is not None) / 3


def table(rows: Sequence[tuple[str, Rate]]) -> str:
    lines = ["| Measurement | Passed / runs | Pass rate | Tasks | Run errors |", "|---|---|---|---|---|"]
    for label, r in rows:
        pct = f"{100 * r.value:.1f}%" if r.value is not None else "n/a"
        lines.append(f"| {label} | {r.passed}/{r.runs} | {pct} | {r.tasks} | {r.errors} |")
    return "\n".join(lines)


def per_task_table(columns: Sequence[tuple[str, Rate]], case_ids: Sequence[str]) -> str:
    head = "| Task | " + " | ".join(label for label, _ in columns) + " |"
    lines = [head, "|---|" + "---|" * len(columns)]
    for c in case_ids:
        cells = []
        for _, r in columns:
            p, n = r.per_task.get(c, (0, 0))
            cells.append(f"{p}/{n}" if n else "-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
