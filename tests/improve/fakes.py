"""A tiny billing world, trace builder, simulator, guard runner and scripted model for the improver's tests.

The guard runner here executes guard source in-process. It exists for tests only; production replay goes
through the sandboxed runtime.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ratchet_cases.types import Cost, Event, Outcome, ToolCall, ToolResult, Trace

from ratchet.improve.replay import GuardInput, Verdict
from ratchet.improve.sim import Step, Walk, result_ids, tool_call_of
from ratchet.types import Guard


@dataclass
class World:
    case_id: str
    charges: dict[str, int]
    timeout_first_refund: bool
    refunds: list[dict[str, Any]] = field(default_factory=list)
    refund_attempts: int = 0

    @classmethod
    def for_case(cls, case_id: str) -> World:
        charge = "ch_" + case_id.rsplit("-", 1)[-1]
        return cls(case_id, {charge: 50}, timeout_first_refund=case_id.startswith("timeout"))

    def call(self, c: ToolCall) -> ToolResult:
        if c.name == "payments_refund":
            self.refund_attempts += 1
            self.refunds.append({"charge_id": c.args["charge_id"], "status": "succeeded"})
            if self.timeout_first_refund and self.refund_attempts == 1:
                return ToolResult(c.call_id, False, None, "timeout")
            return ToolResult(c.call_id, True, {"status": "succeeded"})
        if c.name == "payments_list_refunds":
            return ToolResult(
                c.call_id,
                True,
                {"refunds": [r for r in self.refunds if r["charge_id"] == c.args["charge_id"]]},
            )
        return ToolResult(c.call_id, True, {"charges": self.charges})

    def view(self) -> dict[str, Any]:
        return {"charges": dict(self.charges), "refunds": copy.deepcopy(self.refunds)}

    def reasons(self) -> list[str]:
        out = []
        for ch in self.charges:
            n = sum(1 for r in self.refunds if r["charge_id"] == ch and r["status"] == "succeeded")
            if n > 1:
                out.append(f"duplicate refund for charge {ch}")
            if n == 0:
                out.append(f"customer not refunded for charge {ch}")
        return out


def _blocked_by(
    guards: Sequence[Guard], call: ToolCall, view: dict[str, Any], history: list[Event]
) -> str | None:
    for g in guards:
        scope: dict[str, Any] = {}
        exec(g.source, scope)
        reason = scope["check"](call, view, history)
        if reason is not None:
            return f"blocked by guard {g.id}: {reason}"
    return None


def make_trace(
    case_id: str,
    calls: Sequence[tuple[str, dict[str, Any]]],
    run_id: str,
    version: str = "v0",
    guards: Sequence[Guard] = (),
) -> Trace:
    """Run a fixed call script against the fake world, the way the worker records it: each call carries
    the view the guards saw, and a call a guard blocks gets a `guard_block` event and never executes."""
    world = World.for_case(case_id)
    events: list[Event] = []

    def add(kind: Any, payload: dict[str, Any]) -> None:
        events.append(Event(len(events), kind, payload, 0.0))

    add("model", {"content": "thinking"})
    for i, (name, args) in enumerate(calls):
        c = ToolCall(name, dict(args), f"{run_id}-c{i}")
        view, history = world.view(), list(events)
        add("tool_call", {"call": {"name": c.name, "args": c.args, "call_id": c.call_id}, "view": view})
        reason = _blocked_by(guards, c, view, history)
        if reason is not None:
            gid = reason.split(":", 1)[0].removeprefix("blocked by guard ")
            add("guard_block", {"call_id": c.call_id, "source": "guard", "message": reason, "guard_id": gid,
                                "tool": c.name})  # fmt: skip
            continue
        r = world.call(c)
        add("tool_result", {"result": {"call_id": r.call_id, "ok": r.ok, "data": r.data, "error": r.error}})
    add("final", {"content": "done"})
    reasons = world.reasons()
    outcome = Outcome(not reasons, reasons, "d", Cost(model_calls=len(calls) + 1, tool_calls=len(calls)))
    return Trace(run_id, case_id, version, "scripted", events, outcome)


def bad_timeout(n: int, run: str | None = None) -> Trace:
    ch = f"ch_{n}"
    calls = [
        ("payments_get_charge", {"charge_id": ch}),
        ("payments_refund", {"charge_id": ch}),
        ("payments_refund", {"charge_id": ch}),
    ]
    return make_trace(f"timeout-{n}", calls, run or f"bad-{n}")


def good_timeout(n: int, run: str | None = None) -> Trace:
    ch = f"ch_{n}"
    return make_trace(
        f"timeout-{n}",
        [("payments_refund", {"charge_id": ch}), ("payments_list_refunds", {"charge_id": ch})],
        run or f"good-{n}",
    )


def good_plain(n: int, run: str | None = None) -> Trace:
    ch = f"ch_{n}"
    return make_trace(
        f"plain-{n}",
        [("payments_get_charge", {"charge_id": ch}), ("payments_refund", {"charge_id": ch})],
        run or f"plain-{n}",
    )


def lazy(n: int, run: str | None = None) -> Trace:
    return make_trace(f"plain-{n}", [("payments_get_charge", {"charge_id": f"ch_{n}"})], run or f"lazy-{n}")


class FakeSimulator:
    """Rebuilds the world from the case id and re-applies each executed call, grading after each one."""

    def __init__(self, grade: bool = True) -> None:
        self.grade = grade
        self.walked: list[str] = []

    def walk(self, trace: Trace) -> Walk:
        self.walked.append(trace.run_id)
        world = World.for_case(trace.case_id)
        done = result_ids(trace.events)
        initial = tuple(world.reasons()) if self.grade else None
        steps = []
        for i, e in enumerate(trace.events):
            if e.kind != "tool_call":
                continue
            c = tool_call_of(e)
            view = world.view()
            executed = c.call_id in done
            if executed:
                world.call(c)
            reasons = tuple(world.reasons()) if self.grade else None
            steps.append(Step(e.seq, c, view, trace.events[:i], executed, reasons))
        return Walk(trace.run_id, steps, initial, True)


class LocalRunner:
    """Tests only: runs guard source in this process. Production uses the sandboxed runtime."""

    def __init__(self) -> None:
        self.batches = 0

    def check_batch(self, guard: Guard, inputs: Sequence[GuardInput]) -> list[Verdict]:
        self.batches += 1
        scope: dict[str, Any] = {}
        try:
            exec(guard.source, scope)
        except Exception as exc:
            return [Verdict(None, f"load: {exc}") for _ in inputs]
        out = []
        for gi in inputs:
            try:
                r = scope["check"](gi.call, gi.view, gi.history)
                out.append(Verdict(None if r is None else str(r)))
            except Exception as exc:
                out.append(Verdict(None, f"{type(exc).__name__}: {exc}"))
        return out


@dataclass(frozen=True)
class FakeTurn:
    content: str | None
    input_tokens: int = 7
    output_tokens: int = 3


class ScriptedChat:
    """Answers each request with the next scripted reply (a string, or a dict sent as JSON)."""

    def __init__(self, replies: Sequence[str | dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> FakeTurn:
        self.requests.append([dict(m) for m in messages])
        if not self.replies:
            raise RuntimeError("script exhausted")
        r = self.replies.pop(0)
        return FakeTurn(r if isinstance(r, str) else json.dumps(r))

    def prompts(self) -> str:
        return "\n".join(str(m.get("content")) for req in self.requests for m in req)


NO_DUP_GUARD = """
def check(call, view, history):
    if call.name != "payments_refund":
        return None
    charge = call.args["charge_id"]
    if any(r["charge_id"] == charge and r["status"] == "succeeded" for r in view["refunds"]):
        return "a refund for this charge already succeeded at the provider"
    return None
"""

BLOCK_ALL_REFUNDS = """
def check(call, view, history):
    return "no refunds" if call.name == "payments_refund" else None
"""

CRASHING_GUARD = """
def check(call, view, history):
    return view["missing"]
"""


@dataclass(frozen=True)
class FakeCase:
    id: str
    split: Any
    must_succeed: bool = True
    kind: str = "trap"

    @property
    def family(self) -> str:
        return self.id.rsplit("-", 1)[0]


def fake_cases() -> list[FakeCase]:
    """Timeout cases are traps a duplicate-refund guard fixes; plain cases are their benign twins."""
    return [
        FakeCase("timeout-1", "train"), FakeCase("timeout-2", "train"),
        FakeCase("plain-3", "train", kind="benign_twin"),
        FakeCase("timeout-4", "heldout"), FakeCase("timeout-5", "heldout"),
        FakeCase("plain-6", "heldout", kind="benign_twin"),
    ]  # fmt: skip


class FakeRunCase:
    """Stands in for the agent: a fixed naive script per case family, run under the harness's guards.

    timeout-N: get_charge, refund (times out but commits), refund again: a duplicate unless a guard stops it.
    plain-N:   get_charge, refund: passes unless a guard blocks the refund.
    omit-N:    get_charge, and a refund only when the harness has a procedure.
    """

    def __init__(self, explode: str | None = None) -> None:
        self.explode = explode
        self.calls = 0

    def __call__(self, harness: Any, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
        self.calls += 1
        if case.id == self.explode:
            raise RuntimeError("model endpoint down")
        ch = {"charge_id": "ch_" + case.id.rsplit("-", 1)[-1]}
        script: list[tuple[str, dict[str, Any]]] = [("payments_get_charge", ch), ("payments_refund", ch)]
        if case.id.startswith("timeout"):
            script.append(("payments_refund", ch))
        elif case.id.startswith("omit") and not harness.procedures:
            script = script[:1]
        return make_trace(case.id, script, f"{run_id}-{case.id}", harness.version, harness.guards)


def without_views(trace: Trace) -> Trace:
    """The trace as an agent that records no view snapshots would have written it."""
    events = [
        Event(e.seq, e.kind, {k: v for k, v in e.payload.items() if k != "view"}, e.ts) for e in trace.events
    ]
    return Trace(trace.run_id, trace.case_id, trace.harness_version, trace.model, events, trace.outcome)
