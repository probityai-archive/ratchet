"""The tau2-bench conversation protocol on top of Ratchet's worker loop.

Ratchet's worker loop treats a reply without tool calls as the end of the run. In tau2-bench a reply
without tool calls is a message to the customer, and the run ends when the orchestrator says so.
`ConversationModel` sits between the worker and the agent's chat model and translates, both ways:

- the agent model sees tau2-bench's half-duplex agent transcript: the harness prompt as the system
  message, the orchestrator's greeting and the customer's words as assistant and user messages, its own
  replies as plain assistant text, tool observations as tool messages, and the domain's tools without
  `respond`;
- the worker sees every reply to the customer as a call to the `respond` tool, so it passes through the
  allowlist and every guard like any other action, and the customer's answer is recorded in the trace.

A turn may call several tools, as tau2-bench's agent may; each runs in order. Once the conversation has
ended, the wrapper closes the run with a final turn that calls no model. The wrapper keeps no state
between calls: everything is derived from the messages, so one instance serves every run of a pool.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from ratchet_cases.types import ToolCall

from ratchet.agent.model import ChatModel, Message, ModelTurn, ToolSpec
from ratchet.bench.tau2.world import RESPOND

CLOSED = "conversation_end"
GREETING = "Hi! How can I help you today?"  # tau2-bench's DEFAULT_FIRST_AGENT_MESSAGE
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
    return str(_parse((call.get("function") or {}).get("arguments")).get("content") or "")


def _observation(result: dict[str, Any], raw: Any) -> str:
    """What tau2-bench shows the agent for a tool result the worker serialised."""
    data = result.get("data")
    if isinstance(data, dict) and "observation" in data:
        return str(data["observation"])
    if result.get("error") is not None:
        return f"Error: {result['error']}"
    return str(raw)


def to_tau2_messages(messages: Sequence[Message]) -> list[Message]:
    """The worker's transcript rewritten as tau2-bench's agent transcript."""
    out: list[Message] = []
    respond_ids: set[str] = set()
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "user" and i == 1:
            out.append({"role": "assistant", "content": GREETING})
            out.append({"role": "user", "content": _TICKET.sub("", str(m.get("content") or ""))})
        elif role == "assistant" and _calls(m) and all(_name(c) == RESPOND for c in _calls(m)):
            respond_ids.update(str(c.get("id")) for c in _calls(m))
            out.append({"role": "assistant", "content": "\n".join(_respond_text(c) for c in _calls(m))})
        elif role == "tool" and m.get("tool_call_id") in respond_ids:
            result = _parse(m.get("content"))
            if result.get("ok") and isinstance(result.get("data"), dict):
                out.append({"role": "user", "content": str(result["data"].get("observation", ""))})
            else:
                note = result.get("error") or m.get("content")
                out.append({"role": "system", "content": f"Your message was not delivered: {note}"})
        elif role == "tool":
            content = _observation(_parse(m.get("content")), m.get("content"))
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": content})
        else:
            out.append(dict(m))
    return out


def conversation_ended(messages: Sequence[Message]) -> bool:
    """True when the last tool result says the conversation is over."""
    if not messages or messages[-1].get("role") != "tool":
        return False
    data = _parse(messages[-1].get("content")).get("data")
    return isinstance(data, dict) and data.get("done") is True


def closing_turn() -> ModelTurn:
    return ModelTurn(
        message={"role": "assistant", "content": None},
        content=None,
        tool_calls=[],
        finish_reason=CLOSED,
        input_tokens=0,
        output_tokens=0,
        latency_s=0.0,
    )


def as_respond(turn: ModelTurn, call_id: str) -> ModelTurn:
    """A plain-text reply turned into a call of the `respond` tool."""
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


class ConversationModel:
    """A `ChatModel` that runs tau2-bench's conversation protocol over an inner tool-calling model."""

    def __init__(self, inner: ChatModel) -> None:
        self.inner = inner

    @property
    def name(self) -> str:
        return self.inner.name

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        if conversation_ended(messages):
            return closing_turn()
        visible = [t for t in tools if (t.get("function") or {}).get("name") != RESPOND]
        turn = self.inner.complete(to_tau2_messages(messages), visible)
        if not turn.tool_calls:
            taken = sum(1 for m in messages if m.get("role") == "assistant")
            return as_respond(turn, f"respond_{taken}")
        return turn
