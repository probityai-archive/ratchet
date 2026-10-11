"""The user simulator, pointed at an OpenAI-compatible endpoint.

`EndpointUser` is `tool_sandbox`'s own simulated user (`OpenAIAPIUser`: its prompt construction, its
`end_conversation` tool, its message handling), with two changes to how it reaches its model and none to what
it says to it. The stock class builds its client for the default OpenAI host and a fixed model name; this one
takes the endpoint, key and model of the run. And the model call runs outside the context gate, so one slow
reply does not stop the other worlds in the pool.

Two things a non-OpenAI server does differently are absorbed here. Tool-call ids that are not Python names
(vLLM's `chatcmpl-tool-...`) are rewritten, because the benchmark pastes the id into the code it executes.
A reply with neither text nor a tool call is retried like a dropped connection, because the stock class would
stop on an assertion.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from typing import Any

import httpx
import openai
from openai import OpenAI
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser  # type: ignore[import-untyped]  # no py.typed

from ratchet.bench.toolsandbox.gate import GATE

_NOT_NAME = re.compile(r"\W")
TRANSIENT: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class EmptyReply(Exception):
    """The endpoint answered with no text and no tool call."""


RETRYABLE: tuple[type[BaseException], ...] = (*TRANSIENT, EmptyReply)


def python_name(raw: str) -> str:
    """`raw` as a valid Python identifier, so the benchmark can paste it into the code it executes."""
    name = _NOT_NAME.sub("_", raw)
    return name if name and not name[0].isdigit() else f"call_{name}"


def with_retries(
    fn: Callable[[], Any],
    *,
    attempts: int = 12,
    cap_s: float = 60.0,
    budget_s: float = 900.0,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> Any:
    """`fn()` retried on transient errors with full-jitter exponential backoff inside a time budget."""
    draw = rng or random.Random()
    slept = 0.0
    for attempt in range(attempts):
        try:
            return fn()
        except RETRYABLE:
            delay = draw.uniform(0.0, min(cap_s, 2.0 * 2**attempt))
            if attempt + 1 == attempts or slept + delay > budget_s:
                raise
            sleep(delay)
            slept += delay
    raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises


class EndpointUser(OpenAIAPIUser):  # type: ignore[misc]  # base class is untyped
    """The benchmark's simulated user, answering from `model` at `base_url`."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        # The base constructor builds a client for the default OpenAI host and needs OPENAI_API_KEY; skipped
        # on purpose.
        self.model_name = model
        self.temperature = temperature
        self.sleep: Callable[[float], None] = time.sleep
        self.openai_client = OpenAI(
            base_url=base_url, api_key=api_key, max_retries=0, timeout=300.0, http_client=http_client
        )

    def teardown(self) -> None:
        self.openai_client.close()

    def model_inference(self, openai_messages: Any, openai_tools: Any) -> Any:
        extra: dict[str, Any] = {} if self.temperature is None else {"temperature": self.temperature}

        def once() -> Any:
            with GATE.released():
                response = self.openai_client.chat.completions.create(
                    model=self.model_name, messages=openai_messages, tools=openai_tools, **extra
                )
            message = response.choices[0].message
            if message.tool_calls:
                for call in message.tool_calls:
                    call.id = python_name(call.id)
            elif not message.content:
                raise EmptyReply("the user simulator's model answered with neither text nor a tool call")
            return response

        return with_retries(once, sleep=self.sleep)
