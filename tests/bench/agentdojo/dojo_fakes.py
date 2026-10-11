"""Offline stand-ins for the agent and improver models in the AgentDojo adapter tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ratchet.agent.model import Message, ModelTurn, ToolSpec, scripted_turn


class GroundTruthModel:
    """Plays a fixed list of tool calls one per turn, then answers with `final`. Stateless across runs:
    the next turn is chosen by counting the tool results already in the transcript, so it is safe to share
    across threads and cases."""

    def __init__(self, plan: dict[str, list[tuple[str, dict[str, Any]]]], final: str = "Done.") -> None:
        self.plan = plan
        self.final = final

    @property
    def name(self) -> str:
        return "ground-truth"

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        prompt = next(str(m["content"]) for m in messages if m["role"] == "user")
        calls = next((v for k, v in self.plan.items() if k in prompt), [])
        done = sum(1 for m in messages if m["role"] == "tool")
        if done < len(calls):
            return scripted_turn(None, [calls[done]], id_prefix=f"t{done}-")
        return scripted_turn(self.final)


class JsonImprover:
    """An improver that always answers with the same JSON object (no candidates, no new instructions)."""

    name = "json-improver"

    def __init__(self, reply: str = '{"mechanism": "m", "candidates": [], "text": ""}') -> None:
        self.reply = reply

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        return scripted_turn(self.reply)
