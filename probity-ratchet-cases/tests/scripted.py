"""A scripted `Act` that replays a solver's exact tool calls through the reference loop."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ratchet_cases.cases import Case
from ratchet_cases.cases.solvers import Habits, Solver
from ratchet_cases.run import Act
from ratchet_cases.types import ToolCall, ToolResult
from ratchet_cases.world.core import AGENT, World

JSON = dict[str, Any]


def solver_calls(case: Case, habits: Habits, monkeypatch: pytest.MonkeyPatch) -> list[ToolCall]:
    """Every agent tool call the scripted solver makes on `case`, in order."""
    world = World.from_case(case)
    calls: list[ToolCall] = []
    original = world.call

    def recording(call: ToolCall, actor: str = AGENT) -> ToolResult:
        if actor == AGENT:
            calls.append(call)
        return original(call, actor)

    monkeypatch.setattr(world, "call", recording)
    Solver(case, world, habits).run()
    return calls


def replaying(calls: list[ToolCall]) -> Act:
    """An agent that makes `calls` one per turn, then answers."""
    pending = list(calls)

    def act(messages: list[JSON], tools: list[JSON]) -> JSON:
        assert tools, "the loop must offer the world's tools"
        if not pending:
            return {"content": "done", "usage": {"input_tokens": 3, "output_tokens": 1}}
        call = pending.pop(0)
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.args)},
                }
            ],
        }

    return act


def answer_at_once(messages: list[JSON], tools: list[JSON]) -> JSON:
    """An agent that never calls a tool."""
    return {"content": "I cannot help with that."}
