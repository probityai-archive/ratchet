"""The improver's narrow view of a chat model: send messages, get one JSON object back, count the cost."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ratchet.types import Cost

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class Turn(Protocol):
    @property
    def content(self) -> str | None: ...
    @property
    def input_tokens(self) -> int: ...
    @property
    def output_tokens(self) -> int: ...


class Chat(Protocol):
    """Structurally the agent lane's `ChatModel`: any OpenAI-compatible client satisfies it."""

    def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> Turn: ...


class ImproverModelError(RuntimeError):
    """The model did not return a usable JSON object after the allowed attempts."""


def extract_json(text: str) -> dict[str, Any]:
    """The first JSON object in `text`: a fenced block if present, else the outermost braces."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for raw in candidates:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in model output")


@dataclass
class Meter:
    """Running cost of the improver's own model calls, kept apart from the agent's cost."""

    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, turn: Turn) -> None:
        self.model_calls += 1
        self.input_tokens += turn.input_tokens
        self.output_tokens += turn.output_tokens

    def cost(self) -> Cost:
        return Cost(self.model_calls, self.input_tokens, self.output_tokens)


def ask_json(model: Chat, system: str, user: str, meter: Meter, attempts: int = 2) -> dict[str, Any]:
    """Ask for one JSON object. A reply that does not parse is sent back once with the parse error."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last = ""
    for _ in range(attempts):
        turn = model.complete(messages, [])
        meter.add(turn)
        text = turn.content or ""
        try:
            return extract_json(text)
        except ValueError as exc:
            last = str(exc)
            messages += [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": f"That was not a single JSON object ({exc}). Reply with JSON only.",
                },
            ]
    raise ImproverModelError(f"model gave no JSON object after {attempts} attempts: {last}")
