"""Chat-completions client for any OpenAI-compatible endpoint, plus a scripted stand-in for tests.

The endpoint, key and model come from the environment (`RATCHET_BASE_URL`, `RATCHET_API_KEY`,
`RATCHET_MODEL`). The key is held only in the request header: it is excluded from every repr, error
message and record this module produces.

Retry policy: a transport failure (connect error, read timeout, broken stream) and HTTP 429 or 5xx are
retried with full-jitter exponential backoff inside a bounded budget (attempt count and total sleep). Any
other 4xx is a caller error and is raised at once, never retried.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from ratchet.types import Cost, ToolCall

Message = dict[str, Any]
ToolSpec = dict[str, Any]


class ModelError(Exception):
    """The model call failed for good: a non-retryable response, or the retry budget ran out."""


class ModelHTTPError(ModelError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"model endpoint returned HTTP {status}: {detail}")
        self.status = status


class ModelTransportError(ModelError):
    pass


class ModelResponseError(ModelError):
    """The endpoint answered 2xx with a body that is not a chat completion."""


@dataclass(frozen=True)
class ModelTurn:
    """One assistant turn. `message` is the assistant message to append to the transcript verbatim."""

    message: Message
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    # call_id -> reason, for tool calls whose `arguments` were not a JSON object.
    arg_errors: dict[str, str] = field(default_factory=dict)


class ChatModel(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn: ...


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_s: float = 120.0
    max_attempts: int = 5
    max_backoff_total_s: float = 60.0
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 16.0
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ModelConfig:
        source = os.environ if env is None else env
        missing = [k for k in ("RATCHET_BASE_URL", "RATCHET_API_KEY", "RATCHET_MODEL") if not source.get(k)]
        if missing:
            raise ModelError(f"missing environment variables: {', '.join(missing)}")
        return cls(
            base_url=source["RATCHET_BASE_URL"].rstrip("/"),
            model=source["RATCHET_MODEL"],
            api_key=source["RATCHET_API_KEY"],
        )


def _retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


def _retry_after_s(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _detail(response: httpx.Response) -> str:
    text = response.text
    return text[:300] if text else response.reason_phrase


def _parse_tool_calls(raw_calls: Any) -> tuple[list[ToolCall], dict[str, str]]:
    calls: list[ToolCall] = []
    errors: dict[str, str] = {}
    for i, raw in enumerate(raw_calls or []):
        fn = raw.get("function") or {}
        call_id = str(raw.get("id") or f"call_{i}")
        name = str(fn.get("name") or "")
        arguments = fn.get("arguments")
        args: dict[str, Any] = {}
        if isinstance(arguments, dict):
            args = arguments
        elif arguments in (None, ""):
            args = {}
        else:
            try:
                parsed = json.loads(arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                errors[call_id] = f"arguments are not valid JSON: {exc}"
            else:
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    errors[call_id] = "arguments must be a JSON object"
        calls.append(ToolCall(name=name, args=args, call_id=call_id))
    return calls, errors


def parse_completion(body: Any, latency_s: float) -> ModelTurn:
    """Turn a chat-completions response body into a `ModelTurn`."""
    try:
        choice = body["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelResponseError(f"response has no choices[0].message: {exc!r}") from exc
    if not isinstance(message, dict):
        raise ModelResponseError("choices[0].message is not an object")
    usage = body.get("usage") or {}
    calls, errors = _parse_tool_calls(message.get("tool_calls"))
    content = message.get("content")
    # Keep only the fields a follow-up request needs; providers add extras some peers reject.
    clean: Message = {"role": "assistant", "content": content if isinstance(content, str) else None}
    if message.get("tool_calls"):
        clean["tool_calls"] = message["tool_calls"]
    return ModelTurn(
        message=clean,
        content=content if isinstance(content, str) else None,
        tool_calls=calls,
        finish_reason=str(choice.get("finish_reason") or ""),
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        latency_s=latency_s,
        arg_errors=errors,
    )


class ModelClient:
    """Synchronous chat-completions client with tool calling and a bounded retry budget."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._http = httpx.Client(timeout=config.timeout_s, transport=transport)
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._calls = 0
        self._input = 0
        self._output = 0
        self._wall = 0.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ModelClient:
        return cls(ModelConfig.from_env(env))

    @property
    def name(self) -> str:
        return self._config.model

    def cost(self) -> Cost:
        """Totals over every successful call this client has made."""
        return Cost(
            model_calls=self._calls, input_tokens=self._input, output_tokens=self._output, wall_s=self._wall
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ModelClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        payload: dict[str, Any] = {"model": self._config.model, "messages": list(messages)}
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        if self._config.temperature is not None:
            payload["temperature"] = self._config.temperature
        started = time.monotonic()
        body = self._post_with_retry(payload)
        latency = time.monotonic() - started
        turn = parse_completion(body, latency)
        self._calls += 1
        self._input += turn.input_tokens
        self._output += turn.output_tokens
        self._wall += latency
        return turn

    def _backoff(self, attempt: int, hint: float | None) -> float:
        ceiling = min(self._config.backoff_cap_s, self._config.backoff_base_s * (2**attempt))
        delay = self._rng.uniform(0.0, ceiling)
        if hint is not None:
            delay = max(delay, min(hint, self._config.backoff_cap_s))
        return delay

    def _post_with_retry(self, payload: dict[str, Any]) -> Any:
        url = f"{self._config.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        slept = 0.0
        last: ModelError | None = None
        for attempt in range(self._config.max_attempts):
            hint: float | None = None
            try:
                response = self._http.post(url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                last = ModelTransportError(f"{type(exc).__name__}: {exc}")
            else:
                if response.is_success:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise ModelResponseError(f"response body is not JSON: {exc}") from exc
                if not _retryable_status(response.status_code):
                    raise ModelHTTPError(response.status_code, _detail(response))
                last = ModelHTTPError(response.status_code, _detail(response))
                hint = _retry_after_s(response)
            if attempt + 1 == self._config.max_attempts:
                break
            delay = self._backoff(attempt, hint)
            if slept + delay > self._config.max_backoff_total_s:
                break
            self._sleep(delay)
            slept += delay
        assert last is not None
        raise ModelError(f"retry budget exhausted after {attempt + 1} attempts: {last}") from last


class ScriptExhaustedError(ModelError):
    pass


def scripted_turn(
    content: str | None = None,
    tool_calls: Sequence[tuple[str, dict[str, Any]]] = (),
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    id_prefix: str = "c",
) -> ModelTurn:
    """Build a `ModelTurn` from plain values, the shape a recorded or hand-written script uses."""
    calls = [ToolCall(name=n, args=dict(a), call_id=f"{id_prefix}{i}") for i, (n, a) in enumerate(tool_calls)]
    message: Message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": c.call_id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.args)},
            }
            for c in calls
        ]
    return ModelTurn(
        message=message,
        content=content,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=0.0,
    )


class ScriptedModel:
    """Replays a fixed list of turns in order and records every request it was given."""

    def __init__(self, turns: Sequence[ModelTurn], name: str = "scripted") -> None:
        self._turns = list(turns)
        self._name = name
        self.requests: list[tuple[list[Message], list[ToolSpec]]] = []

    @property
    def name(self) -> str:
        return self._name

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        self.requests.append((json.loads(json.dumps(list(messages))), list(tools)))
        index = len(self.requests) - 1
        if index >= len(self._turns):
            raise ScriptExhaustedError(f"script has {len(self._turns)} turns; turn {index + 1} was requested")
        return self._turns[index]
