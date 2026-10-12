"""A minimal reference loop and the trace file format.

`run_case` drives one case with any `Act` callable: the model is handed the conversation and the tool
schemas and returns one OpenAI-style assistant message. Each tool call it makes is executed by the world,
and the run ends when the model answers without a tool call or the step budget runs out. There are no
guards, procedures or retries here: this is the bare agent a harness is compared against.

A trace file holds one JSON object per line: the trace, the package version and the suite digest it was
run under. `regrade` rebuilds the final state by replaying, in order, every tool call the world executed
(a `tool_call` event that has a matching `tool_result`), checks it reaches the recorded final-state digest,
and grades it again. Harness traces with guard blocks replay the same way, because a blocked call has a
`tool_call` event and no `tool_result`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ratchet_cases.cases.grade import grade
from ratchet_cases.cases.model import Case
from ratchet_cases.cases.registry import get, suite_digest
from ratchet_cases.types import Cost, Event, EventKind, Outcome, ToolCall, Trace, to_jsonable
from ratchet_cases.world.core import World

JSON = dict[str, Any]
Act = Callable[[list[JSON], list[JSON]], JSON]

SYSTEM_PROMPT = (
    "You are a customer-support agent for a small online company. Resolve the ticket you are given by "
    "calling the tools. Every tool call changes or reads the company's real systems. When the work is done, "
    "reply with a short summary and no tool calls."
)
HARNESS_VERSION = "ratchet-cases-reference-loop"


class TraceFileError(ValueError):
    """A trace line that is not a trace this package wrote."""


@dataclass
class _Recorder:
    events: list[Event]

    def add(self, kind: EventKind, payload: JSON) -> None:
        self.events.append(Event(seq=len(self.events), kind=kind, payload=payload, ts=time.time()))


def _tool_calls(reply: JSON) -> list[JSON]:
    calls = reply.get("tool_calls") or []
    if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
        raise TypeError("tool_calls must be a list of objects")
    return calls


def _arguments(call: JSON) -> tuple[str, str, JSON | str]:
    """(call id, tool name, parsed arguments or the reason they did not parse)."""
    call_id = str(call.get("id") or "")
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw = function.get("arguments", "{}")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as err:
        return call_id, name, f"invalid_arguments: not JSON ({err.msg})"
    if not isinstance(args, dict):
        return call_id, name, "invalid_arguments: not a JSON object"
    return call_id, name, args


def _execute(world: World, recorder: _Recorder, call: JSON, step: int, index: int) -> JSON:
    """Run one tool call and return the tool message for the model."""
    call_id, name, args = _arguments(call)
    call_id = call_id or f"call_{step}_{index}"
    if isinstance(args, str):
        return {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"error": args})}
    tool_call = ToolCall(name=name, args=args, call_id=call_id)
    recorder.add("tool_call", {"call": to_jsonable(tool_call)})
    result = world.call(tool_call)
    recorder.add("tool_result", {"result": to_jsonable(result)})
    body = {"ok": True, "data": result.data} if result.ok else {"ok": False, "error": result.error}
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(body, sort_keys=True)}


def run_case(case: Case, act: Act, *, model: str = "", max_steps: int = 40) -> Trace:
    """Run one case to the end and return its graded trace."""
    world = World.from_case(case)
    tools = world.tools()
    messages: list[JSON] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ticket {case.ticket_id}\n\n{case.customer_message}"},
    ]
    recorder = _Recorder([])
    reason = "step budget"
    content: str | None = None
    for step in range(max_steps):
        reply = act(messages, tools)
        calls = _tool_calls(reply)
        content = reply.get("content")
        recorder.add(
            "model",
            {"model": model, "content": content, "tool_calls": calls, "usage": reply.get("usage", {})},
        )
        messages.append({"role": "assistant", "content": content, "tool_calls": calls or None})
        if not calls:
            reason = "answered"
            break
        messages.extend(_execute(world, recorder, call, step, i) for i, call in enumerate(calls))
    recorder.add("final", {"reason": reason, "content": content})
    provisional = Trace(
        run_id="",
        case_id=case.id,
        harness_version=HARNESS_VERSION,
        model=model,
        events=recorder.events,
        outcome=Outcome(passed=False, reasons=[], final_state_digest=""),
    )
    outcome = grade(case, world, provisional)
    return Trace(
        run_id=f"{case.id}@{int(time.time())}",
        case_id=case.id,
        harness_version=HARNESS_VERSION,
        model=model,
        events=recorder.events,
        outcome=outcome,
    )


# --- trace files --------------------------------------------------------------------------------------


def trace_line(trace: Trace, version: str) -> str:
    """One trace as a JSON line, with the package version and suite digest it ran under."""
    record = {"ratchet_cases": version, "suite_digest": suite_digest(), "trace": to_jsonable(trace)}
    return json.dumps(record, sort_keys=True)


def trace_from_json(data: JSON) -> Trace:
    """Rebuild a `Trace` from its JSON form."""
    try:
        outcome = data["outcome"]
        return Trace(
            run_id=str(data["run_id"]),
            case_id=str(data["case_id"]),
            harness_version=str(data["harness_version"]),
            model=str(data["model"]),
            events=[
                Event(int(e["seq"]), e["kind"], dict(e["payload"]), float(e["ts"])) for e in data["events"]
            ],
            outcome=Outcome(
                passed=bool(outcome["passed"]),
                reasons=list(outcome["reasons"]),
                final_state_digest=str(outcome["final_state_digest"]),
                cost=Cost(**outcome.get("cost", {})),
            ),
        )
    except (KeyError, TypeError, ValueError) as err:
        raise TraceFileError(f"not a trace: {err!r}") from err


def read_traces(path: Path) -> Iterator[tuple[JSON, Trace]]:
    """Every (record, trace) in a trace file, in file order."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise TraceFileError(f"{path}:{number}: not JSON ({err.msg})") from err
            if not isinstance(record, dict) or "trace" not in record:
                raise TraceFileError(f"{path}:{number}: no trace object")
            yield record, trace_from_json(record["trace"])


def replay(case: Case, events: Iterable[Event]) -> World:
    """The world after every tool call the original run executed, in the order it executed them."""
    events = list(events)
    executed = {e.payload.get("result", {}).get("call_id") for e in events if e.kind == "tool_result"}
    world = World.from_case(case)
    for event in events:
        if event.kind != "tool_call":
            continue
        call = event.payload.get("call", {})
        if call.get("call_id") not in executed:
            continue
        world.call(ToolCall(name=str(call["name"]), args=dict(call["args"]), call_id=str(call["call_id"])))
    return world


@dataclass(frozen=True)
class Regraded:
    case_id: str
    kind: str
    split: str
    outcome: Outcome
    recorded_passed: bool
    digest_matches: bool


def regrade(trace: Trace) -> Regraded:
    """Grade a recorded trace again from its replayed final state."""
    case = get(trace.case_id)
    world = replay(case, trace.events)
    outcome = grade(case, world, trace)
    return Regraded(
        case_id=case.id,
        kind=case.kind,
        split=case.split,
        outcome=outcome,
        recorded_passed=trace.outcome.passed,
        digest_matches=outcome.final_state_digest == trace.outcome.final_state_digest,
    )
