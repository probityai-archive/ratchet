"""Reconstruct what a guard would have seen at every recorded tool call, without calling a model.

A `Simulator` walks a recorded trace and yields, for each `tool_call` event, the state view before the call,
the event history before it, and (when it can grade intermediate states) the grader's reasons after it.
Replay and diagnosis both stand on this: replay asks "would the guard have blocked here", diagnosis asks
"after which call did the failure first hold".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ratchet.types import Event, ToolCall, Trace


@dataclass(frozen=True)
class Step:
    seq: int
    call: dict[str, Any]
    view: dict[str, Any]
    history: list[dict[str, Any]]
    executed: bool
    reasons_after: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Walk:
    run_id: str
    steps: list[Step]
    initial_reasons: tuple[str, ...] | None
    faithful: bool
    note: str = ""


class Simulator(Protocol):
    def walk(self, trace: Trace) -> Walk: ...


def call_of(event: Event) -> dict[str, Any]:
    """The `ToolCall` carried by a `tool_call` event, as JSON."""
    p = event.payload.get("call", event.payload)
    return {
        "name": str(p.get("name", "")),
        "args": dict(p.get("args") or {}),
        "call_id": str(p.get("call_id", "")),
    }


def tool_call_of(event: Event) -> ToolCall:
    c = call_of(event)
    return ToolCall(name=c["name"], args=c["args"], call_id=c["call_id"])


def result_ids(events: Iterable[Event]) -> set[str]:
    out: set[str] = set()
    for e in events:
        if e.kind == "tool_result":
            p = e.payload.get("result", e.payload)
            out.add(str(p.get("call_id", "")))
    return out


def event_json(event: Event) -> dict[str, Any]:
    return asdict(event)


def last_seq(trace: Trace) -> int:
    return trace.events[-1].seq if trace.events else 0


class RecordedViews:
    """Uses view snapshots the agent recorded on each `tool_call` event (payload key `view`).

    No world is rebuilt; intermediate grading is unavailable, so `reasons_after` is None.
    """

    def walk(self, trace: Trace) -> Walk:
        done = result_ids(trace.events)
        steps: list[Step] = []
        faithful = True
        for i, e in enumerate(trace.events):
            if e.kind != "tool_call":
                continue
            view = e.payload.get("view")
            if not isinstance(view, dict):
                faithful = False
                view = {}
            c = call_of(e)
            steps.append(
                Step(e.seq, c, view, [event_json(x) for x in trace.events[:i]], executed=c["call_id"] in done)
            )
        note = "" if faithful else "some tool_call events carry no view snapshot"
        return Walk(trace.run_id, steps, None, faithful, note)
