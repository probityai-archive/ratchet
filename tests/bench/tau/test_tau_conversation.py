from __future__ import annotations

import json

from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.bench.tau.conversation import (
    CLOSED,
    OUT_OF_ACTIONS,
    ConversationModel,
    actions_taken,
    conversation_ended,
    first_call_only,
    to_tau_messages,
)
from ratchet.bench.tau.world import RESPOND, RESPOND_SPEC

LOOKUP = {"type": "function", "function": {"name": "get_order_details", "parameters": {}}}


def _tool(call_id: str, ok: bool, data: object = None, error: str | None = None) -> dict[str, object]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps({"ok": ok, "data": data, "error": error}),
    }


def _respond_msg(call_id: str, text: str) -> dict[str, object]:
    raw = {
        "id": call_id,
        "type": "function",
        "function": {"name": RESPOND, "arguments": json.dumps({"content": text})},
    }
    return {"role": "assistant", "content": None, "tool_calls": [raw]}


def _lookup_msg(call_id: str) -> dict[str, object]:
    raw = {"id": call_id, "type": "function", "function": {"name": "get_order_details", "arguments": "{}"}}
    return {"role": "assistant", "content": "checking", "tool_calls": [raw]}


TRANSCRIPT = [
    {"role": "system", "content": "policy"},
    {"role": "user", "content": "Ticket tau-retail-test-000\n\nHi, I want an exchange."},
    _respond_msg("respond_0", "Your email?"),
    _tool("respond_0", True, {"observation": "a@b.c", "done": False}),
    _lookup_msg("call_1"),
    _tool("call_1", True, {"observation": "{order}", "done": False}),
    _lookup_msg("call_2"),
    _tool("call_2", False, None, "blocked by guard g1: confirm first"),
    _respond_msg("respond_3", "Shall I?"),
    _tool("respond_3", False, None, "blocked by guard g2: no"),
    _lookup_msg("call_4"),
    _tool("call_4", True, {"steps": "1. do"}),
]


def test_transcript_is_rewritten_as_tau_bench_sees_it() -> None:
    out = to_tau_messages(TRANSCRIPT)
    assert out[0] == {"role": "system", "content": "policy"}
    assert out[1] == {"role": "user", "content": "Hi, I want an exchange."}
    assert out[2] == {"role": "assistant", "content": "Your email?"}
    assert out[3] == {"role": "user", "content": "a@b.c"}
    assert out[4]["tool_calls"][0]["id"] == "call_1"
    assert out[5] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "get_order_details",
        "content": "{order}",
    }
    assert out[7]["content"] == "Error: blocked by guard g1: confirm first"
    assert out[8] == {"role": "assistant", "content": "Shall I?"}
    assert out[9] == {"role": "system", "content": "Your message was not delivered: blocked by guard g2: no"}
    assert out[11]["content"] == json.dumps({"steps": "1. do"})


def test_unparseable_tool_content_passes_through() -> None:
    msgs = [{"role": "system", "content": "p"}, {"role": "user", "content": "hi"}, _lookup_msg("c"),
            {"role": "tool", "tool_call_id": "c", "content": "raw text"}]  # fmt: skip
    assert to_tau_messages(msgs)[3]["content"] == "raw text"
    listed = [*msgs[:3], {"role": "tool", "tool_call_id": "c", "content": "[1]"}]
    assert to_tau_messages(listed)[3]["content"] == "[1]"


def test_ended_and_action_counting() -> None:
    assert not conversation_ended([])
    assert not conversation_ended(TRANSCRIPT[:2])
    assert not conversation_ended(TRANSCRIPT)
    assert conversation_ended([*TRANSCRIPT, _tool("x", True, {"observation": "###STOP###", "done": True})])
    assert actions_taken(TRANSCRIPT) == 5


def test_text_reply_becomes_a_respond_call_and_respond_is_hidden_from_the_model() -> None:
    inner = ScriptedModel([scripted_turn("Hello! Your email?")], name="m")
    model = ConversationModel(inner)
    turn = model.complete(TRANSCRIPT[:2], [LOOKUP, RESPOND_SPEC])
    assert model.name == "m"
    assert [c.name for c in turn.tool_calls] == [RESPOND]
    assert turn.tool_calls[0].args == {"content": "Hello! Your email?"}
    assert turn.tool_calls[0].call_id == "respond_0"
    sent_messages, sent_tools = inner.requests[0]
    assert [t["function"]["name"] for t in sent_tools] == ["get_order_details"]
    assert sent_messages[1]["content"] == "Hi, I want an exchange."


def test_only_the_first_tool_call_is_kept() -> None:
    turn = scripted_turn(None, [("get_order_details", {"order_id": "1"}), ("get_user_details", {})])
    turn = first_call_only(turn)
    assert [c.name for c in turn.tool_calls] == ["get_order_details"]
    assert len(turn.message["tool_calls"]) == 1
    single = scripted_turn(None, [("get_order_details", {})])
    assert first_call_only(single) is single


def test_closing_turns_call_no_model() -> None:
    inner = ScriptedModel([], name="m")
    model = ConversationModel(inner, max_actions=5)
    ended = [*TRANSCRIPT, _tool("x", True, {"observation": "###STOP###", "done": True})]
    turn = model.complete(ended, [])
    assert turn.finish_reason == CLOSED and not turn.tool_calls and turn.input_tokens == 0
    out = model.complete(TRANSCRIPT, [])
    assert out.finish_reason == OUT_OF_ACTIONS
    assert inner.requests == []
