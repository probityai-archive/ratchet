"""Run a harness over a set of cases with the same model and budget as its parent, in a bounded pool."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from ratchet.types import Cost, Harness, Split, Trace


class CaseLike(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def split(self) -> Split: ...
    @property
    def must_succeed(self) -> bool: ...


# run_case(harness, case, model, store) -> Trace, the agent lane's entry point.
RunCase = Callable[[Harness, Any, Any, Any], Trace]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    split: Split
    must_succeed: bool
    passed: bool
    reasons: list[str]
    run_id: str | None
    cost: Cost
    error: str | None = None


@dataclass(frozen=True)
class EvalReport:
    harness_version: str
    results: list[CaseResult]
    wall_s: float
    traces: list[Trace] = field(default_factory=list, compare=False, repr=False)

    def by_case(self) -> dict[str, CaseResult]:
        return {r.case_id: r for r in self.results}

    def passed(self, split: Split | None = None) -> set[str]:
        return {r.case_id for r in self.results if r.passed and (split is None or r.split == split)}

    def count(self, split: Split | None = None) -> int:
        return sum(1 for r in self.results if split is None or r.split == split)

    def train_traces(self) -> list[Trace]:
        """The only traces the improver may read: those of training cases."""
        train = {r.case_id for r in self.results if r.split == "train"}
        return [t for t in self.traces if t.case_id in train]

    def cost(self) -> Cost:
        rs = [r.cost for r in self.results]
        return Cost(
            model_calls=sum(c.model_calls for c in rs),
            input_tokens=sum(c.input_tokens for c in rs),
            output_tokens=sum(c.output_tokens for c in rs),
            tool_calls=sum(c.tool_calls for c in rs),
            wall_s=round(sum(c.wall_s for c in rs), 3),
        )

    def to_payload(self) -> dict[str, Any]:
        c = self.cost()
        return {
            "type": "eval",
            "harness_version": self.harness_version,
            "train": {"passed": len(self.passed("train")), "total": self.count("train")},
            "heldout": {"passed": len(self.passed("heldout")), "total": self.count("heldout")},
            "errors": sum(1 for r in self.results if r.error),
            "cost": {"model_calls": c.model_calls, "input_tokens": c.input_tokens,
                     "output_tokens": c.output_tokens, "tool_calls": c.tool_calls, "wall_s": c.wall_s},
            "wall_s": round(self.wall_s, 3),
            "results": [
                {"case_id": r.case_id, "split": r.split, "must_succeed": r.must_succeed, "passed": r.passed,
                 "reasons": r.reasons, "run_id": r.run_id, "error": r.error}
                for r in self.results
            ],
        }  # fmt: skip


def _one(
    run_case: RunCase, harness: Harness, case: CaseLike, model: Any, store: Any
) -> tuple[CaseResult, Trace | None]:
    try:
        trace = run_case(harness, case, model, store)
    except Exception as exc:  # an infrastructure failure is recorded as a failed case, never hidden
        msg = f"{type(exc).__name__}: {exc}"[:500]
        return CaseResult(
            case.id, case.split, case.must_succeed, False, [f"run error: {msg}"], None, Cost(), msg
        ), None
    o = trace.outcome
    return CaseResult(
        case.id, case.split, case.must_succeed, o.passed, list(o.reasons), trace.run_id, o.cost
    ), trace


def evaluate_many(
    harnesses: Sequence[Harness],
    cases: Sequence[CaseLike],
    run_case: RunCase,
    model: Any,
    *,
    store: Any = None,
    workers: int = 8,
    repeats: int = 1,
) -> list[EvalReport]:
    """Every harness on every case (`repeats` times each), all through one bounded pool.

    With `repeats` > 1 a case counts as passed only when every repeat passed, so a lucky run cannot
    carry a candidate through admission.
    """
    started = time.perf_counter()
    jobs: list[tuple[int, CaseLike, Future[tuple[CaseResult, Trace | None]]]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for hi, h in enumerate(harnesses):
            for case in cases:
                for _ in range(max(1, repeats)):
                    jobs.append((hi, case, pool.submit(_one, run_case, h, case, model, store)))
        done = [(hi, case, f.result()) for hi, case, f in jobs]
    wall = time.perf_counter() - started
    reports = []
    for hi, h in enumerate(harnesses):
        merged: dict[str, CaseResult] = {}
        traces: list[Trace] = []
        for hj, case, (res, trace) in done:
            if hj != hi:
                continue
            if trace is not None:
                traces.append(trace)
            prev = merged.get(case.id)
            merged[case.id] = res if prev is None or (prev.passed and not res.passed) else prev
        reports.append(EvalReport(h.version, [merged[c.id] for c in cases], wall, traces))
    return reports


def evaluate(
    harness: Harness, cases: Sequence[CaseLike], run_case: RunCase, model: Any, **kw: Any
) -> EvalReport:
    return evaluate_many([harness], cases, run_case, model, **kw)[0]
