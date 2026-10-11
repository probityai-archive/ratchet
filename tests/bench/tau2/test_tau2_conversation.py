from __future__ import annotations

import json
from typing import Any

from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.bench.tau2.conversation import (
    CLOSED,
    GREETING,
    ConversationModel,
    as_respond,
    conversation_ended,
    to_tau2_messages,
)
from ratchet.bench.tau2.world import RESPOND, RESPOND_SPEC


def _tool(call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload)}


def _respond_call(call_id: str, text: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": RESPOND, "arguments": json.dumps({"content": text})},
    }


def test_transcript_matches_tau2_bench_agent_view() -> None:
    lookup = {"id": "c1", "type": "function", "function": {"name": "get_user_details", "arguments": "{}"}}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Ticket tau2-retail-0\n\nI need help."},
        {"role": "assistant", "content": None, "tool_calls": [_respond_call("r0", "Your name?")]},
        _tool("r0", {"ok": True, "data": {"observation": "Ann.", "done": False}}),
        {"role": "assistant", "content": None, "tool_calls": [lookup]},
        _tool("c1", {"ok": True, "data": {"observation": '{"id": 1}', "done": False}}),
        {"role": "assistant", "content": None, "tool_calls": [_respond_call("r1", "Hi")]},
        _tool("r1", {"ok": False, "error": "blocked by guard g: no"}),
        {"role": "assistant", "content": None, "tool_calls": [{**lookup, "id": "c2"}]},
        _tool("c2", {"ok": False, "error": "tool not allowed"}),
        {"role": "assistant", "content": None, "tool_calls": [{**lookup, "id": "c3"}]},
        {"role": "tool", "tool_call_id": "c3", "content": "not json"},
    ]
    out = to_tau2_messages(messages)
    assert out[0] == {"role": "system", "content": "policy"}
    assert out[1] == {"role": "assistant", "content": GREETING}
    assert out[2] == {"role": "user", "content": "I need help."}
    assert out[3] == {"role": "assistant", "content": "Your name?"}
    assert out[4] == {"role": "user", "content": "Ann."}
    assert out[5]["tool_calls"] == [lookup]
    assert out[6] == {"role": "tool", "tool_call_id": "c1", "content": '{"id": 1}'}
    assert out[8] == {"role": "system", "content": "Your message was not delivered: blocked by guard g: no"}
    assert out[10]["content"] == "Error: tool not allowed"
    assert out[12]["content"] == "not json"


def test_conversation_ended_reads_the_last_tool_result() -> None:
    assert not conversation_ended([])
    assert not conversation_ended([{"role": "user", "content": "x"}])
    assert conversation_ended([_tool("r", {"ok": True, "data": {"done": True}})])
    assert not conversation_ended([_tool("r", {"ok": True, "data": {"done": False}})])
    assert not conversation_ended([{"role": "tool", "tool_call_id": "r", "content": "[1]"}])


def test_text_becomes_respond_and_tool_calls_pass_through() -> None:
    inner = ScriptedModel(
        [scripted_turn("Hello there"), scripted_turn(None, [("a", {}), ("b", {"x": 1})])], name="m"
    )
    model = ConversationModel(inner)
    assert model.name == "m"
    tools: list[dict[str, Any]] = [RESPOND_SPEC, {"type": "function", "function": {"name": "a"}}]
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "Ticket x\n\nhi"},
    ]
    first = model.complete(msgs, tools)
    assert [c.name for c in first.tool_calls] == [RESPOND] and first.tool_calls[0].args == {
        "content": "Hello there"
    }
    assert first.tool_calls[0].call_id == "respond_0"
    sent_tools = inner.requests[0][1]
    assert [t["function"]["name"] for t in sent_tools] == ["a"]
    second = model.complete(msgs, tools)
    assert [c.name for c in second.tool_calls] == ["a", "b"]
    ended = [*msgs, _tool("r", {"ok": True, "data": {"done": True}})]
    closing = model.complete(ended, tools)
    assert closing.finish_reason == CLOSED and not closing.tool_calls and len(inner.requests) == 2


def test_as_respond_keeps_usage() -> None:
    turn = as_respond(scripted_turn(None, input_tokens=7, output_tokens=3), "r9")
    assert turn.tool_calls[0].args == {"content": ""} and turn.input_tokens == 7 and turn.output_tokens == 3
