"""The tau-bench conversation protocol on top of Ratchet's worker loop.

Ratchet's worker loop treats a reply without tool calls as the end of the run. In tau-bench a reply without
tool calls is a message to the customer, and the run ends when the customer does. `ConversationModel` sits
between the worker and the agent's chat model and translates, in both directions, so that:

- the agent model sees exactly tau-bench's tool-calling transcript: the policy as the system prompt, the
  customer's words as user messages, its own replies as plain assistant text, tau-bench's tool
  observations as tool messages, and tau-bench's tools without `respond`;
- the worker sees every reply to the customer as a call to the `respond` tool, so it passes through the
  allowlist and every guard like any other action, and the customer's answer is recorded in the trace;
- tau-bench's limits hold: one tool call per turn (the first, as tau-bench's agent takes it) and at most
  `max_actions` agent actions; after the conversation ends or the budget runs out, the wrapper closes the
  run with a final turn that calls no model.

The wrapper keeps no state between calls: everything is derived from the messages, so one instance serves
every run of a parallel pool.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from ratchet.agent.model import ChatModel, Message, ModelTurn, ToolSpec
from ratchet.bench.tau.world import RESPOND

CLOSED = "conversation_end"
OUT_OF_ACTIONS = "action_budget"
_TICKET = re.compile(r"\ATicket [^\n]*\n\n")


def _parse(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        return {}
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _calls(message: Message) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    return [c for c in calls if isinstance(c, dict)]


def _name(call: dict[str, Any]) -> str:
    return str((call.get("function") or {}).get("name") or "")


def _respond_text(call: dict[str, Any]) -> str:
    args = _parse((call.get("function") or {}).get("arguments"))
    return str(args.get("content") or "")


def _observation(result: dict[str, Any], raw: Any) -> str:
    """What tau-bench would show the agent for a tool result the worker serialised."""
    data = result.get("data")
    if isinstance(data, dict) and "observation" in data:
        return str(data["observation"])
    if isinstance(data, dict):
        return json.dumps(data)
    if result.get("error") is not None:
        return f"Error: {result['error']}"
    return str(raw)


def to_tau_messages(messages: Sequence[Message]) -> list[Message]:
    """The worker's transcript rewritten as tau-bench's tool-calling transcript."""
    out: list[Message] = []
    names: dict[str, str] = {}
    respond_ids: set[str] = set()
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "user" and i == 1:
            out.append({"role": "user", "content": _TICKET.sub("", str(m.get("content") or ""))})
        elif role == "assistant" and _calls(m) and all(_name(c) == RESPOND for c in _calls(m)):
            for c in _calls(m):
                respond_ids.add(str(c.get("id")))
            out.append({"role": "assistant", "content": "\n".join(_respond_text(c) for c in _calls(m))})
        elif role == "assistant":
            for c in _calls(m):
                names[str(c.get("id"))] = _name(c)
            out.append(dict(m))
        elif role == "tool" and m.get("tool_call_id") in respond_ids:
            result = _parse(m.get("content"))
            if result.get("ok") and isinstance(result.get("data"), dict):
                out.append({"role": "user", "content": str(result["data"].get("observation", ""))})
            else:
                note = result.get("error") or m.get("content")
                out.append({"role": "system", "content": f"Your message was not delivered: {note}"})
        elif role == "tool":
            call_id = str(m.get("tool_call_id"))
            content = _observation(_parse(m.get("content")), m.get("content"))
            out.append(
                {"role": "tool", "tool_call_id": call_id, "name": names.get(call_id, ""), "content": content}
            )
        else:
            out.append(dict(m))
    return out


def conversation_ended(messages: Sequence[Message]) -> bool:
    """True when the last tool result says the conversation is over."""
    if not messages or messages[-1].get("role") != "tool":
        return False
    data = _parse(messages[-1].get("content")).get("data")
    return isinstance(data, dict) and data.get("done") is True


def actions_taken(messages: Sequence[Message]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


def closing_turn(reason: str) -> ModelTurn:
    return ModelTurn(
        message={"role": "assistant", "content": None},
        content=None,
        tool_calls=[],
        finish_reason=reason,
        input_tokens=0,
        output_tokens=0,
        latency_s=0.0,
    )


def as_respond(turn: ModelTurn, call_id: str) -> ModelTurn:
    """A plain-text reply turned into a call of the `respond` tool."""
    from ratchet.types import ToolCall

    content = turn.content or ""
    raw = {
        "id": call_id,
        "type": "function",
        "function": {"name": RESPOND, "arguments": json.dumps({"content": content})},
    }
    return ModelTurn(
        message={"role": "assistant", "content": None, "tool_calls": [raw]},
        content=None,
        tool_calls=[ToolCall(RESPOND, {"content": content}, call_id)],
        finish_reason=turn.finish_reason,
        input_tokens=turn.input_tokens,
        output_tokens=turn.output_tokens,
        latency_s=turn.latency_s,
    )


def first_call_only(turn: ModelTurn) -> ModelTurn:
    """tau-bench's agent executes only the first tool call of a turn; the others are dropped."""
    if len(turn.tool_calls) <= 1:
        return turn
    first = turn.tool_calls[0]
    message = dict(turn.message)
    message["tool_calls"] = _calls(turn.message)[:1]
    errors = {k: v for k, v in turn.arg_errors.items() if k == first.call_id}
    return ModelTurn(
        message=message,
        content=turn.content,
        tool_calls=[first],
        finish_reason=turn.finish_reason,
        input_tokens=turn.input_tokens,
        output_tokens=turn.output_tokens,
        latency_s=turn.latency_s,
        arg_errors=errors,
    )


class ConversationModel:
    """A `ChatModel` that runs tau-bench's conversation protocol over an inner tool-calling model."""

    def __init__(self, inner: ChatModel, max_actions: int = 30) -> None:
        self.inner = inner
        self.max_actions = max_actions

    @property
    def name(self) -> str:
        return self.inner.name

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        if conversation_ended(messages):
            return closing_turn(CLOSED)
        taken = actions_taken(messages)
        if taken >= self.max_actions:
            return closing_turn(OUT_OF_ACTIONS)
        visible = [t for t in tools if (t.get("function") or {}).get("name") != RESPOND]
        turn = self.inner.complete(to_tau_messages(messages), visible)
        if not turn.tool_calls:
            return as_respond(turn, f"respond_{taken}")
        return first_call_only(turn)
