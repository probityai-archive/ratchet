"""Rebuild a recorded conversation step by step, with the benchmark's grade after every step.

`ToolSandboxSimulator` is the improver's `Simulator` for this benchmark. Given a recorded trace it builds a
fresh world for the scenario, replays the tool calls the world executed, and yields for each recorded
`tool_call` the state view the guard saw, the event history before it, and the reasons the benchmark's
milestone and minefield grader gives for the conversation up to and including that call. `diagnose` reads
those reasons to find the step after which a failure first held and kept holding: a milestone that is
unmet from the start and never met is an omission, a state that was right and then broken is the call that
broke it.

The simulated user is not asked again. `ReplayUser` says what the user said in the recording (the
observation of each executed `respond`), and ends the conversation where the recording shows the user ended
it. A replay is `faithful` when it ends in the state the trace recorded; if the user's messages were not all
captured (a simulated user that retried a malformed tool call writes messages the recording does not show),
the digest differs and the walk says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole

from ratchet.bench.toolsandbox.cases import scenarios
from ratchet.bench.toolsandbox.grading import reasons_for
from ratchet.bench.toolsandbox.world import RESPOND, ToolSandboxWorld
from ratchet.improve.sim import Step, Walk, tool_call_of
from ratchet.types import Event, Trace

END = ""  # a reply of "" ends the conversation
CASE_PREFIX = "toolsandbox-"


class ReplayUser(BaseRole):
    """The benchmark's user role answering from a list. A reply of `END` ends the conversation with the
    user's own `end_conversation` tool, and so does an exhausted list, so a replay always terminates."""

    role_type = RoleType.USER

    def __init__(self, replies: Sequence[str] = (END,)) -> None:
        self.replies = list(replies)
        self.heard: list[str] = []
        self.turns = 0

    def respond(self, ending_index: int | None = None) -> None:
        last = self.get_messages(ending_index=ending_index)[-1]
        if (last.sender, last.recipient) == (RoleType.AGENT, RoleType.USER):
            self.heard.append(last.content)
        self.turns += 1
        reply = self.replies.pop(0) if self.replies else END
        if reply == END:
            tid = f"call_user_{self.turns}"
            code = (
                f"{tid}_parameters = {{}}\n{tid}_response = end_conversation(**{tid}_parameters)\n"
                f"print(repr({tid}_response))"
            )
            message = Message(sender=RoleType.USER, recipient=RoleType.EXECUTION_ENVIRONMENT, content=code)
        else:
            message = Message(sender=RoleType.USER, recipient=RoleType.AGENT, content=reply)
        self.add_messages([message])


def executed_ids(events: Sequence[Event]) -> set[str]:
    """Call ids of the tool calls the world executed (those with a `tool_result` event)."""
    ids: set[str] = set()
    for e in events:
        if e.kind == "tool_result":
            ids.add(str(e.payload.get("result", e.payload).get("call_id", "")))
    return ids


def user_replies(events: Sequence[Event]) -> list[str]:
    """What the user said after each executed `respond`, in order; `END` where the user ended it."""
    sent = {
        e.payload["call"]["call_id"]
        for e in events
        if e.kind == "tool_call" and e.payload["call"]["name"] == RESPOND
    }
    replies: list[str] = []
    for e in events:
        if e.kind != "tool_result":
            continue
        result: dict[str, Any] = e.payload.get("result", e.payload)
        if str(result.get("call_id")) in sent and result.get("ok") and isinstance(result.get("data"), dict):
            replies.append(str(result["data"].get("observation", "")))
    return replies


class ToolSandboxSimulator:
    """`Simulator` for the benchmark: replays a trace against a fresh world and grades after every call."""

    def walk(self, trace: Trace) -> Walk:
        name = trace.case_id.removeprefix(CASE_PREFIX)
        scenario = scenarios().get(name)
        if scenario is None:
            return Walk(trace.run_id, [], None, False, f"no scenario named {name!r}")
        replies = user_replies(trace.events)
        world = ToolSandboxWorld(scenario, name, lambda: ReplayUser(replies))
        try:
            done = executed_ids(trace.events)
            initial = tuple(reasons_for(world.grade()))
            steps: list[Step] = []
            for i, event in enumerate(trace.events):
                if event.kind != "tool_call":
                    continue
                call = tool_call_of(event)
                view = world.view().to_json()
                executed = call.call_id in done
                if executed:
                    world.call(call)
                steps.append(
                    Step(event.seq, call, view, trace.events[:i], executed, tuple(reasons_for(world.grade())))
                )
            faithful = (
                not trace.outcome.final_state_digest or world.digest() == trace.outcome.final_state_digest
            )
        finally:
            world.close()
        note = "" if faithful else "the replay ended in a different state than the trace recorded"
        return Walk(trace.run_id, steps, initial, faithful, note)
