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

from ratchet.improve.replay import GuardInput, Verdict
from ratchet.improve.sim import Step, Walk, result_ids, tool_call_of
from ratchet.types import Cost, Event, Guard, Outcome, ToolCall, ToolResult, Trace


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
        if c.name == "refund":
            self.refund_attempts += 1
            self.refunds.append({"charge_id": c.args["charge_id"], "status": "succeeded"})
            if self.timeout_first_refund and self.refund_attempts == 1:
                return ToolResult(c.call_id, False, None, "timeout")
            return ToolResult(c.call_id, True, {"status": "succeeded"})
        if c.name == "list_refunds":
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


def make_trace(
    case_id: str, calls: Sequence[tuple[str, dict[str, Any]]], run_id: str, version: str = "v0"
) -> Trace:
    world = World.for_case(case_id)
    events: list[Event] = []

    def add(kind: Any, payload: dict[str, Any]) -> None:
        events.append(Event(len(events), kind, payload, 0.0))

    add("model", {"content": "thinking"})
    for i, (name, args) in enumerate(calls):
        c = ToolCall(name, dict(args), f"{run_id}-c{i}")
        add("tool_call", {"name": c.name, "args": c.args, "call_id": c.call_id})
        r = world.call(c)
        add("tool_result", {"call_id": r.call_id, "ok": r.ok, "data": r.data, "error": r.error})
    add("final", {"content": "done"})
    reasons = world.reasons()
    outcome = Outcome(not reasons, reasons, "d", Cost(model_calls=len(calls) + 1, tool_calls=len(calls)))
    return Trace(run_id, case_id, version, "scripted", events, outcome)


def bad_timeout(n: int, run: str | None = None) -> Trace:
    ch = f"ch_{n}"
    calls = [("get_charge", {"charge_id": ch}), ("refund", {"charge_id": ch}), ("refund", {"charge_id": ch})]
    return make_trace(f"timeout-{n}", calls, run or f"bad-{n}")


def good_timeout(n: int, run: str | None = None) -> Trace:
    ch = f"ch_{n}"
    return make_trace(
        f"timeout-{n}",
        [("refund", {"charge_id": ch}), ("list_refunds", {"charge_id": ch})],
        run or f"good-{n}",
    )


def good_plain(n: int, run: str | None = None) -> Trace:
    ch = f"ch_{n}"
    return make_trace(
        f"plain-{n}", [("get_charge", {"charge_id": ch}), ("refund", {"charge_id": ch})], run or f"plain-{n}"
    )


def lazy(n: int, run: str | None = None) -> Trace:
    return make_trace(f"plain-{n}", [("get_charge", {"charge_id": f"ch_{n}"})], run or f"lazy-{n}")


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
    if call.name != "refund":
        return None
    charge = call.args["charge_id"]
    if any(r["charge_id"] == charge and r["status"] == "succeeded" for r in view["refunds"]):
        return "a refund for this charge already succeeded at the provider"
    return None
"""

BLOCK_ALL_REFUNDS = """
def check(call, view, history):
    return "no refunds" if call.name == "refund" else None
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


def fake_cases() -> list[FakeCase]:
    """Timeout cases fail until a guard exists; plain cases always pass; one plain case refuses-to-run."""
    return [
        FakeCase("timeout-1", "train"), FakeCase("timeout-2", "train"), FakeCase("plain-3", "train"),
        FakeCase("timeout-4", "heldout"), FakeCase("timeout-5", "heldout"), FakeCase("plain-6", "heldout"),
    ]  # fmt: skip


class FakeRunCase:
    """Stands in for the agent: any guard fixes timeouts; `block_plain` breaks plain cases."""

    def __init__(self, block_plain: bool = False, explode: str | None = None) -> None:
        self.block_plain = block_plain
        self.explode = explode
        self.calls = 0

    def __call__(self, harness: Any, case: Any, model: Any, store: Any) -> Trace:
        self.calls += 1
        n = int(case.id.rsplit("-", 1)[-1])
        run = f"{harness.version}-{case.id}-{self.calls}"
        if case.id == self.explode:
            raise RuntimeError("model endpoint down")
        if case.id.startswith("timeout"):
            t = good_timeout(n, run) if harness.guards else bad_timeout(n, run)
        else:
            t = lazy(n, run) if self.block_plain and harness.guards else good_plain(n, run)
        return Trace(t.run_id, t.case_id, harness.version, t.model, t.events, t.outcome)
