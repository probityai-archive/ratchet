"""Counterfactual replay: would this guard have stopped the recorded failures, and only them?

For every recorded tool call the guard sees exactly what it would have seen live: the call, the state view
before the call, and the event history before it. No model is called. The guard passes when it blocks each
target failure at or before the step that caused it, and blocks nothing in any passing trace. Every check
runs in one batch through the sandboxed guard runtime, so a guard costs one process start, not one per call.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ratchet.improve.diagnose import FailurePoint
from ratchet.improve.sim import Simulator, Walk
from ratchet.types import Event, Guard, ToolCall, Trace


@dataclass(frozen=True)
class GuardInput:
    call: ToolCall
    view: dict[str, Any]
    history: list[Event]


@dataclass(frozen=True)
class Verdict:
    block: str | None
    error: str | None = None


class GuardRunner(Protocol):
    """Runs untrusted guard source over many inputs in the sandbox, one verdict per input, in order."""

    def check_batch(self, guard: Guard, inputs: Sequence[GuardInput]) -> list[Verdict]: ...


class WalkCache:
    """Walk each trace once and share the walk across every guard replayed against it."""

    def __init__(self, simulator: Simulator) -> None:
        self.simulator = simulator
        self._walks: dict[str, Walk] = {}
        self.walk_s = 0.0

    def walk(self, trace: Trace) -> Walk:
        hit = self._walks.get(trace.run_id)
        if hit is None:
            started = time.perf_counter()
            hit = self.simulator.walk(trace)
            self.walk_s += time.perf_counter() - started
            self._walks[trace.run_id] = hit
        return hit


@dataclass(frozen=True)
class ReplayReport:
    guard_id: str
    passed: bool
    targets: list[dict[str, Any]]
    false_positives: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    unfaithful: list[str]
    traces: int
    checks: int
    check_s: float
    walk_s: float
    notes: list[str] = field(default_factory=list)

    @property
    def checks_per_s(self) -> float:
        return self.checks / self.check_s if self.check_s > 0 else float(self.checks)

    def to_payload(self) -> dict[str, Any]:
        return {
            "guard_id": self.guard_id,
            "passed": self.passed,
            "targets_hit": sum(1 for t in self.targets if t["hit"]),
            "targets_total": len(self.targets),
            "targets": self.targets,
            "false_positives": self.false_positives,
            "errors": self.errors,
            "unfaithful": self.unfaithful,
            "traces": self.traces,
            "checks": self.checks,
            "check_s": round(self.check_s, 6),
            "walk_s": round(self.walk_s, 6),
            "checks_per_s": round(self.checks_per_s, 1),
            "notes": self.notes,
        }


def replay(
    guard: Guard,
    targets: Sequence[FailurePoint],
    failing: Mapping[str, Trace],
    passing: Iterable[Trace],
    walks: WalkCache,
    runner: GuardRunner,
) -> ReplayReport:
    """Replay `guard` over the target failures (by run id in `failing`) and every passing trace."""
    walk_before = walks.walk_s
    inputs: list[GuardInput] = []
    # (role, run_id, case_id, seq) for each input, in order.
    where: list[tuple[str, str, str, int]] = []
    unfaithful: list[str] = []
    notes: list[str] = []

    target_runs = sorted({t.run_id for t in targets})
    for run_id in target_runs:
        trace = failing.get(run_id)
        if trace is None:
            notes.append(f"target trace {run_id} not supplied")
            continue
        horizon = max(t.seq for t in targets if t.run_id == run_id)
        w = walks.walk(trace)
        if not w.faithful:
            unfaithful.append(run_id)
        for s in w.steps:
            if s.seq <= horizon:
                inputs.append(GuardInput(s.call, s.view, s.history))
                where.append(("target", run_id, trace.case_id, s.seq))

    n_passing = 0
    for trace in passing:
        n_passing += 1
        w = walks.walk(trace)
        if not w.faithful:
            unfaithful.append(trace.run_id)
        for s in w.steps:
            if s.executed:  # a call an existing guard already blocked cannot be a new false positive
                inputs.append(GuardInput(s.call, s.view, s.history))
                where.append(("passing", trace.run_id, trace.case_id, s.seq))

    started = time.perf_counter()
    verdicts = runner.check_batch(guard, inputs) if inputs else []
    check_s = time.perf_counter() - started
    if len(verdicts) != len(inputs):
        raise RuntimeError(f"guard runtime returned {len(verdicts)} verdicts for {len(inputs)} inputs")

    first_block: dict[str, int] = {}
    false_positives: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for (role, run_id, case_id, seq), v in zip(where, verdicts, strict=True):
        if v.error is not None:
            errors.append({"run_id": run_id, "seq": seq, "error": v.error[:300]})
            continue
        if v.block is None:
            continue
        if role == "target":
            first_block[run_id] = min(seq, first_block.get(run_id, seq))
        else:
            false_positives.append(
                {"run_id": run_id, "case_id": case_id, "seq": seq, "reason": v.block[:300]}
            )

    target_rows = []
    for t in targets:
        fb = first_block.get(t.run_id)
        target_rows.append(
            {"failure_id": t.id, "case_id": t.case_id, "failing_seq": t.seq, "first_block_seq": fb,
             "hit": fb is not None and fb <= t.seq}
        )  # fmt: skip
    passed = bool(target_rows) and all(r["hit"] for r in target_rows) and not false_positives and not errors
    return ReplayReport(
        guard_id=guard.id,
        passed=passed,
        targets=target_rows,
        false_positives=false_positives,
        errors=errors,
        unfaithful=sorted(set(unfaithful)),
        traces=len(target_runs) + n_passing,
        checks=len(inputs),
        check_s=check_s,
        walk_s=walks.walk_s - walk_before,
        notes=notes,
    )
