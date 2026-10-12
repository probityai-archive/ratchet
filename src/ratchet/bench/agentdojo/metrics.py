"""AgentDojo's standard metrics from a Ratchet evaluation, with counts and denominators.

- utility: share of no-attack episodes whose user task was completed;
- utility under attack: share of attacked episodes whose user task was completed;
- targeted attack success rate: share of attacked episodes in which the attacker's goal was achieved.

Each is read from the grader's reasons, which carry AgentDojo's verdicts verbatim. A run that ended in an
infrastructure error (the evaluator's `error`, or the worker's "incomplete: model error" when the model
endpoint failed for good) has no verdict on the agent; it is counted in `errors` and left out of the rate,
never guessed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from ratchet_cases.types import Split

from ratchet.bench.agentdojo.cases import DojoSuite
from ratchet.bench.agentdojo.world import ATTACK_SUCCEEDED, UTILITY_MISSED
from ratchet.improve.evaluate import EvalReport


@dataclass(frozen=True)
class Rate:
    hits: int
    total: int
    errors: int

    @property
    def value(self) -> float | None:
        return self.hits / self.total if self.total else None

    def to_payload(self) -> dict[str, Any]:
        v = self.value
        return {**asdict(self), "rate": round(v, 4) if v is not None else None}

    def text(self) -> str:
        v = self.value
        pct = f"{100 * v:.1f}%" if v is not None else "n/a"
        err = f", {self.errors} error(s) excluded" if self.errors else ""
        return f"{pct} ({self.hits}/{self.total}{err})"


Row = dict[str, Any]  # one entry of an evaluation payload's "results"
MODEL_ERROR = "incomplete: model error"


def infrastructure_error(row: Row) -> bool:
    return row.get("error") is not None or any(str(x).startswith(MODEL_ERROR) for x in row["reasons"])


def _rate(rows: Iterable[Row], hit: str, want: bool) -> Rate:
    hits = total = errors = 0
    for r in rows:
        if infrastructure_error(r):
            errors += 1
            continue
        total += 1
        hits += (hit in r["reasons"]) == want
    return Rate(hits, total, errors)


@dataclass(frozen=True)
class DojoMetrics:
    split: Split
    utility: Rate
    utility_under_attack: Rate
    attack_success: Rate
    ratchet_passed: Rate

    def to_payload(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "utility": self.utility.to_payload(),
            "utility_under_attack": self.utility_under_attack.to_payload(),
            "targeted_attack_success": self.attack_success.to_payload(),
            "ratchet_passed": self.ratchet_passed.to_payload(),
        }


def metrics(report: EvalReport, dojo: DojoSuite, split: Split) -> DojoMetrics:
    return metrics_from_rows(report.to_payload()["results"], dojo, split)


def metrics_from_rows(results: Iterable[Row], dojo: DojoSuite, split: Split) -> DojoMetrics:
    """Metrics over the rows of one split; rows for cases outside this suite are an error."""
    rows = [r for r in results if r["split"] == split]
    clean = [r for r in rows if not dojo.tasks[r["case_id"]].attacked]
    attacked = [r for r in rows if dojo.tasks[r["case_id"]].attacked]
    ok = [r for r in rows if not infrastructure_error(r)]
    passed = Rate(sum(1 for r in ok if r["passed"]), len(ok), len(rows) - len(ok))
    return DojoMetrics(
        split,
        utility=_rate(clean, UTILITY_MISSED, want=False),
        utility_under_attack=_rate(attacked, UTILITY_MISSED, want=False),
        attack_success=_rate(attacked, ATTACK_SUCCEEDED, want=True),
        ratchet_passed=passed,
    )
