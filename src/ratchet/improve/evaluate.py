"""Run a harness over a set of cases with the same model and budget as its parent, in a bounded pool.

Every run gets a fresh run id, so a re-run of the same harness on the same case is a new run, never a
resumed or deduplicated old one. Each result records how many of its model calls were answered fresh and
how many came from a cache (a model event whose payload says `"cached": true`), and how many times the
harness intervened (a `guard_block` event for any reason other than malformed arguments).

Modes, labelled on every result: FRESH (every model call was answered fresh; the only mode admission
counts), CACHED (at least one byte-identical cached response; debugging only), RECORDED (a stored fresh
result shown again, with the time it was recorded).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Protocol

from ratchet_cases.types import Cost, Split, Trace

from ratchet.types import Harness
from ratchet.verify.runs import case_digest, run_record

Mode = Literal["FRESH", "RECORDED", "CACHED"]
CaseKind = str  # "trap", "benign_twin" or "must_succeed", as the case suite labels them


class CaseLike(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def split(self) -> Split: ...
    @property
    def must_succeed(self) -> bool: ...
    @property
    def kind(self) -> CaseKind: ...
    @property
    def family(self) -> str: ...


class RunCase(Protocol):
    """The agent lane's `run_case(harness, case, model, store, *, run_id)`."""

    def __call__(self, harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Trace: ...


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    split: Split
    must_succeed: bool
    kind: CaseKind
    passed: bool
    reasons: list[str]
    run_id: str | None
    cost: Cost
    interventions: int = 0
    fresh_calls: int = 0
    cached_calls: int = 0
    error: str | None = None
    mode: Mode = "FRESH"
    recorded_at: str | None = None
    attempts: int = 1
    infra_failed: bool = False
    case_digest: str | None = None  # SHA-256 of the case document the run graded (ratchet.verify.runs)


INFRA_REASON = "incomplete: model error"
INFRA_RETRIES = 2  # predeclared: a run lost to the model endpoint is re-run at most this many times


def is_infra_failure(trace: Trace) -> bool:
    """The run never completed because the model endpoint failed: no evidence about the harness either way."""
    return any(r.startswith(INFRA_REASON) for r in trace.outcome.reasons)


def interventions(trace: Trace) -> int:
    return sum(1 for e in trace.events if e.kind == "guard_block" and e.payload.get("source") != "arguments")


def model_calls(trace: Trace) -> tuple[int, int]:
    """(fresh, cached) model calls in the trace."""
    cached = sum(1 for e in trace.events if e.kind == "model" and e.payload.get("cached") is True)
    total = sum(1 for e in trace.events if e.kind == "model")
    return total - cached, cached


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

    def recorded(self, recorded_at: str) -> EvalReport:
        """This report shown again from storage: every result is labelled RECORDED with its timestamp."""
        results = [replace(r, mode="RECORDED", recorded_at=recorded_at) for r in self.results]
        return replace(self, results=results)

    def modes(self) -> dict[str, int]:
        out = {"FRESH": 0, "RECORDED": 0, "CACHED": 0}
        for r in self.results:
            out[r.mode] += 1
        return out

    def cost(self) -> Cost:
        rs = [r.cost for r in self.results]
        return Cost(
            model_calls=sum(c.model_calls for c in rs),
            input_tokens=sum(c.input_tokens for c in rs),
            output_tokens=sum(c.output_tokens for c in rs),
            tool_calls=sum(c.tool_calls for c in rs),
            wall_s=round(sum(c.wall_s for c in rs), 3),
        )

    def run_records(self) -> list[dict[str, Any]]:
        """What the report commits about every run it holds a trace for (`ratchet.verify.runs`)."""
        return sorted((run_record(asdict(t)) for t in self.traces), key=lambda r: str(r["run_id"]))

    def to_payload(self) -> dict[str, Any]:
        c = self.cost()
        return {
            "type": "eval",
            "harness_version": self.harness_version,
            "train": {"passed": len(self.passed("train")), "total": self.count("train")},
            "heldout": {"passed": len(self.passed("heldout")), "total": self.count("heldout")},
            "errors": sum(1 for r in self.results if r.error),
            "infra_failed": sum(1 for r in self.results if r.infra_failed),
            "infra_retries": sum(r.attempts - 1 for r in self.results),
            "fresh_calls": sum(r.fresh_calls for r in self.results),
            "cached_calls": sum(r.cached_calls for r in self.results),
            "modes": self.modes(),
            "cost": {
                "model_calls": c.model_calls,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "tool_calls": c.tool_calls,
                "wall_s": c.wall_s,
            },
            "wall_s": round(self.wall_s, 3),
            "results": [
                {
                    "case_id": r.case_id,
                    "split": r.split,
                    "kind": r.kind,
                    "must_succeed": r.must_succeed,
                    "passed": r.passed,
                    "reasons": r.reasons,
                    "run_id": r.run_id,
                    "interventions": r.interventions,
                    "fresh_calls": r.fresh_calls,
                    "cached_calls": r.cached_calls,
                    "mode": r.mode,
                    "attempts": r.attempts,
                    "infra_failed": r.infra_failed,
                    "recorded_at": r.recorded_at,
                    "error": r.error,
                    "case_digest": r.case_digest,
                }
                for r in self.results
            ],
            "runs": self.run_records(),
        }


def _one(
    run_case: RunCase, harness: Harness, case: CaseLike, model: Any, store: Any
) -> tuple[CaseResult, Trace | None]:
    """One case. A run lost to the model endpoint (an exception, or an incomplete run whose model call
    failed) is re-run under the predeclared limit; a completed run, pass or fail, is never re-run. A case
    still lost after the limit is marked `infra_failed`: invalid evidence, never a pass and never a fail."""
    msg = ""
    trace: Trace | None = None
    attempts = 0
    graded = case_digest(case)
    while attempts <= INFRA_RETRIES:
        attempts += 1
        run_id = f"run-{harness.version}-{uuid.uuid4().hex[:12]}"
        try:
            trace = run_case(harness, case, model, store, run_id=run_id)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"[:500]
            trace = None
            continue
        if not is_infra_failure(trace):
            break
    if trace is None:
        failed = CaseResult(
            case.id, case.split, case.must_succeed, case.kind, False, [f"run error: {msg}"], None, Cost(),
            error=msg, attempts=attempts, infra_failed=True, case_digest=graded,
        )  # fmt: skip
        return failed, None
    o = trace.outcome
    fresh, cached = model_calls(trace)
    result = CaseResult(
        case.id, case.split, case.must_succeed, case.kind, o.passed, list(o.reasons), trace.run_id, o.cost,
        interventions(trace), fresh, cached, mode="CACHED" if cached else "FRESH", attempts=attempts,
        infra_failed=is_infra_failure(trace), case_digest=graded,
    )  # fmt: skip
    return result, trace


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
