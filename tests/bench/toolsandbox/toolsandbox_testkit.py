"""Shared helpers for the benchmark adapter tests: the real scenarios with a scripted user and agent."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from ratchet.agent.model import Message as ChatMessage
from ratchet.agent.model import ModelTurn, ToolSpec, scripted_turn
from ratchet.bench.toolsandbox.cases import scenarios
from ratchet.bench.toolsandbox.replay import END, ReplayUser
from ratchet.bench.toolsandbox.world import ToolSandboxWorld, UserFactory

MISSING_KEY = "PermissionError: Please provide 'RAPID_API_KEY' in environment variable."


def user_factory(replies: Sequence[str] = (END,)) -> UserFactory:
    return lambda: ReplayUser(replies)


def make_world(name: str, replies: Sequence[str] = (END,)) -> ToolSandboxWorld:
    return ToolSandboxWorld(scenarios()[name], name, user_factory(replies))


class Steady:
    """A chat model that answers every request with the same turn."""

    name = "steady"

    def __init__(self, turn: ModelTurn | None = None) -> None:
        self.turn = turn or scripted_turn("I cannot help with that.")
        self.requests = 0

    def complete(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> ModelTurn:
        self.requests += 1
        return self.turn


def chat_completion(
    content: str | None = None, tool_calls: Sequence[tuple[str, str, dict[str, Any]]] = ()
) -> dict[str, Any]:
    """A chat-completions body: text, or tool calls given as (id, name, arguments)."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, n, a in tool_calls
        ]
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))
