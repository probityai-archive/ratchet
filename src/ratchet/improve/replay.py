"""Counterfactual replay: would this guard have stopped the recorded failures, and only them?

For every recorded tool call the guard sees exactly what it would have seen live: the call, the state view
before the call, and the event history before it. No model is called. The guard passes when it blocks each
target failure at or before the step that caused it, and blocks nothing in any passing trace. Every check
runs in one batch through the sandboxed guard runtime, so a guard costs one process start, not one per call.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ratchet_cases.types import Event, ToolCall, Trace, sha256_hex

from ratchet.canonical import canonical_json
from ratchet.improve.diagnose import FailurePoint
from ratchet.improve.sim import Simulator, Walk, call_of
from ratchet.types import Guard


@dataclass(frozen=True)
class GuardInput:
    call: ToolCall
    view: dict[str, Any]
    history: list[Event]


@dataclass(frozen=True)
class Verdict:
    block: str | None
    error: str | None = None

    def __post_init__(self) -> None:
        # A failed check must say why: an empty error, or the text of a missing field, cannot be diagnosed
        # from the record.
        if self.error is not None and not (isinstance(self.error, str) and self.error.strip()):
            raise ValueError(f"a failed guard check must carry its error message, got {self.error!r}")


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

    def has(self, run_id: str) -> bool:
        return run_id in self._walks

    def walk_by_id(self, run_id: str) -> Walk:
        return self._walks[run_id]


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
    traces_checked: int = 0
    snapshot: str = ""

    @property
    def verdict(self) -> str:
        """CLEARED, REJECTED (a miss or a false fire on complete evidence) or INCOMPLETE (a runtime error, a
        timeout, a missing target trace or partial coverage: such evidence can never clear a guard)."""
        if self.errors or self.traces_checked < self.traces or self.notes:
            return "INCOMPLETE"
        return "CLEARED" if self.passed else "REJECTED"

    def sanitized(self) -> dict[str, Any]:
        """The only form of a protected (evaluator-side) replay the proposer may see: counts, no case ids,
        run ids, inputs or guard-returned strings."""
        return {
            "verdict": self.verdict,
            "targets_hit": sum(1 for t in self.targets if t["hit"]),
            "targets_total": len(self.targets),
            "false_fires": len(self.false_positives),
            "benign_twin_fires": self.benign_fires,
            "errors": len(self.errors),
            "coverage": self.coverage,
        }

    @property
    def coverage(self) -> str:
        return f"{self.traces_checked}/{self.traces} traces checked"

    @property
    def benign_fires(self) -> int:
        return sum(1 for fp in self.false_positives if fp.get("benign_twin"))

    @property
    def checks_per_s(self) -> float:
        return self.checks / self.check_s if self.check_s > 0 else float(self.checks)

    def to_payload(self) -> dict[str, Any]:
        return {
            "guard_id": self.guard_id,
            "passed": self.passed,
            "verdict": self.verdict,
            "coverage": self.coverage,
            "exhaustive": self.traces_checked == self.traces,
            "snapshot_digest": self.snapshot,
            "targets_hit": sum(1 for t in self.targets if t["hit"]),
            "targets_total": len(self.targets),
            "targets": self.targets,
            "false_positives": self.false_positives,
            "benign_twin_fires": self.benign_fires,
            "errors": self.errors,
            "unfaithful": self.unfaithful,
            "traces": self.traces,
            "checks": self.checks,
            "check_s": round(self.check_s, 6),
            "walk_s": round(self.walk_s, 6),
            "checks_per_s": round(self.checks_per_s, 1),
            "notes": self.notes,
        }


def trace_digest(trace: Trace) -> str:
    return sha256_hex(canonical_json(asdict(trace)))


def snapshot_digest(traces: Iterable[Trace]) -> str:
    """The digest of a declared trace snapshot: every (run id, trace digest) pair, sorted."""
    return sha256_hex(canonical_json(sorted([t.run_id, trace_digest(t)] for t in traces)))


def _tools(trace: Trace) -> set[str]:
    return {call_of(e)["name"] for e in trace.events if e.kind == "tool_call"}


def similarity_order(traces: Sequence[Trace], targets: Sequence[Trace], tool: str | None) -> list[Trace]:
    """Most target-like first (tool-set overlap, then use of the failing tool), so a guard that is too
    broad meets its counterexample early."""
    ref = set().union(*(_tools(t) for t in targets)) if targets else set()

    def score(t: Trace) -> tuple[float, int, str]:
        mine = _tools(t)
        jac = len(mine & ref) / len(mine | ref) if mine | ref else 0.0
        return (-jac, -(tool in mine), t.run_id)

    return sorted(traces, key=score)


def replay(
    guards: Guard | Sequence[Guard],
    targets: Sequence[FailurePoint],
    failing: Mapping[str, Trace],
    passing: Iterable[Trace],
    walks: WalkCache,
    runner: GuardRunner,
    benign: Iterable[Trace] = (),
    *,
    chunk: int = 32,
) -> ReplayReport:
    """Replay one guard, or a composed set of guards (a call is blocked when any of them blocks), over the
    target failures, every passing trace and every benign-twin trace, using each call's recorded pre-call
    view. The replay always completes: every false fire is recorded, and CLEARED needs "N/N traces checked"
    with no runtime error. Clearance traces run in similarity order, `chunk` traces per sandbox batch.
    """
    group = [guards] if isinstance(guards, Guard) else list(guards)
    gid = group[0].id if len(group) == 1 else "composed:" + "+".join(g.id for g in group)
    walk_before = walks.walk_s
    unfaithful: list[str] = []
    notes: list[str] = []
    check_s = 0.0
    checks = 0
    first_block: dict[str, int] = {}
    false_positives: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def combined(inputs: list[GuardInput]) -> list[Verdict]:
        per_guard = [runner.check_batch(g, inputs) if inputs else [] for g in group]
        for vs in per_guard:
            if len(vs) != len(inputs):
                raise RuntimeError(f"guard runtime returned {len(vs)} verdicts for {len(inputs)} inputs")
        out = []
        for i in range(len(inputs)):
            vs = [pg[i] for pg in per_guard]
            err = next((v.error for v in vs if v.error is not None), None)
            block = next((v.block for v in vs if v.block is not None), None)
            out.append(Verdict(None, err) if err is not None else Verdict(block))
        return out

    def run(batch: list[tuple[str, Trace]]) -> None:
        nonlocal check_s, checks
        inputs: list[GuardInput] = []
        where: list[tuple[str, str, str, int]] = []
        for role, trace in batch:
            w = walks.walk(trace)
            if not w.faithful:
                unfaithful.append(trace.run_id)
            horizon = max((t.seq for t in targets if t.run_id == trace.run_id), default=0)
            for st in w.steps:
                if role == "target" and st.seq > horizon:
                    continue
                if role != "target" and not st.executed:
                    continue  # a call an existing guard already blocked cannot be a new false positive
                inputs.append(GuardInput(st.call, st.view, st.history))
                where.append((role, trace.run_id, trace.case_id, st.seq))
        started = time.perf_counter()
        verdicts = combined(inputs)
        check_s += time.perf_counter() - started
        checks += len(inputs) * len(group)
        for (role, run_id, case_id, seq), v in zip(where, verdicts, strict=True):
            if v.error is not None:
                errors.append({"run_id": run_id, "seq": seq, "error": v.error[:300]})
            elif v.block is not None and role == "target":
                first_block[run_id] = min(seq, first_block.get(run_id, seq))
            elif v.block is not None:
                fp = {"run_id": run_id, "case_id": case_id, "seq": seq, "reason": v.block[:300]}
                false_positives.append({**fp, "benign_twin": role == "benign"})

    target_traces: list[Trace] = []
    for run_id in sorted({t.run_id for t in targets}):
        trace = failing.get(run_id)
        if trace is None:
            notes.append(f"target trace {run_id} not supplied")
        else:
            target_traces.append(trace)
    clear: list[tuple[str, Trace]] = []
    seen = {t.run_id for t in target_traces}
    for role, traces in (("benign", benign), ("passing", passing)):
        for trace in traces:
            if trace.run_id not in seen:
                seen.add(trace.run_id)
                clear.append((role, trace))
    roles = {t.run_id: r for r, t in clear}
    tool = targets[0].tool if targets else None
    ordered = [(roles[t.run_id], t) for t in similarity_order([t for _, t in clear], target_traces, tool)]
    total = len(target_traces) + len(ordered)

    run([("target", t) for t in target_traces])
    checked = len(target_traces)
    target_rows = []
    for t in targets:
        fb = first_block.get(t.run_id)
        target_rows.append({"failure_id": t.id, "case_id": t.case_id, "failing_seq": t.seq,
                            "first_block_seq": fb, "hit": fb is not None and fb <= t.seq})  # fmt: skip
    for i in range(0, len(ordered), max(1, chunk)):
        batch = ordered[i : i + max(1, chunk)]
        run(batch)
        checked += len(batch)
    targets_ok = bool(target_rows) and all(r["hit"] for r in target_rows)
    exhaustive = checked == total
    passed = exhaustive and targets_ok and not false_positives and not errors and not notes
    return ReplayReport(
        guard_id=gid,
        passed=passed,
        targets=target_rows,
        false_positives=false_positives,
        errors=errors,
        unfaithful=sorted(set(unfaithful)),
        traces=total,
        checks=checks,
        check_s=check_s,
        walk_s=walks.walk_s - walk_before,
        notes=notes,
        traces_checked=checked,
        snapshot=snapshot_digest([*target_traces, *(t for _, t in ordered)]),
    )
