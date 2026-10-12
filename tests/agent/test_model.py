from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from ratchet_cases.types import ToolCall

from ratchet.agent.model import (
    ModelClient,
    ModelConfig,
    ModelError,
    ModelHTTPError,
    ModelResponseError,
    ScriptedModel,
    ScriptExhaustedError,
    parse_completion,
    scripted_turn,
)

SECRET = "sk-test-secret-value"


def _config(**over: Any) -> ModelConfig:
    base: dict[str, Any] = {
        "base_url": "http://model.test/v1",
        "model": "m1",
        "api_key": SECRET,
        "max_attempts": 4,
    }
    base.update(over)
    return ModelConfig(**base)


def _completion(content: str | None = "hi", tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content, "refusal": None}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }


def _client(handler: Any, **over: Any) -> tuple[ModelClient, list[float]]:
    sleeps: list[float] = []
    client = ModelClient(_config(**over), transport=httpx.MockTransport(handler), sleep=sleeps.append)
    return client, sleeps


def test_from_env_reads_all_three_and_hides_key() -> None:
    cfg = ModelConfig.from_env(
        {"RATCHET_BASE_URL": "http://x/v1/", "RATCHET_API_KEY": SECRET, "RATCHET_MODEL": "gpt"}
    )
    assert cfg.base_url == "http://x/v1" and cfg.model == "gpt" and cfg.api_key == SECRET
    assert SECRET not in repr(cfg)


def test_from_env_names_missing_variables_without_values() -> None:
    with pytest.raises(ModelError, match="RATCHET_MODEL") as info:
        ModelConfig.from_env({"RATCHET_BASE_URL": "http://x", "RATCHET_API_KEY": SECRET})
    assert SECRET not in str(info.value)


def test_from_env_uses_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATCHET_BASE_URL", "http://env/v1")
    monkeypatch.setenv("RATCHET_API_KEY", SECRET)
    monkeypatch.setenv("RATCHET_MODEL", "envmodel")
    client = ModelClient.from_env()
    assert client.name == "envmodel"
    client.close()


def test_config_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError):
        _config(max_attempts=0)


def test_complete_sends_tools_and_parses_tool_calls() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://model.test/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {SECRET}"
        seen.append(json.loads(request.content))
        call = {"id": "t1", "type": "function", "function": {"name": "refund", "arguments": '{"amount": 5}'}}
        return httpx.Response(200, json=_completion(None, [call]))

    client, _ = _client(handler, temperature=0.0)
    tools = [{"type": "function", "function": {"name": "refund", "parameters": {"type": "object"}}}]
    with client:
        turn = client.complete([{"role": "user", "content": "x"}], tools)
    assert seen[0]["tools"] == tools and seen[0]["tool_choice"] == "auto" and seen[0]["temperature"] == 0.0
    assert turn.tool_calls == [ToolCall(name="refund", args={"amount": 5}, call_id="t1")]
    assert turn.input_tokens == 12 and turn.output_tokens == 7 and turn.finish_reason == "tool_calls"
    assert "refusal" not in turn.message and turn.message["tool_calls"][0]["id"] == "t1"
    cost = client.cost()
    assert (cost.model_calls, cost.input_tokens, cost.output_tokens) == (1, 12, 7)
    assert cost.wall_s >= 0.0


def test_complete_without_tools_omits_tool_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "tools" not in body and "tool_choice" not in body and "temperature" not in body
        return httpx.Response(200, json=_completion("done"))

    client, _ = _client(handler)
    turn = client.complete([{"role": "user", "content": "x"}], [])
    assert turn.content == "done" and turn.tool_calls == []


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retries_429_and_5xx_then_succeeds(status: int) -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(status, text="busy", headers={"retry-after": "0.25"})
        return httpx.Response(200, json=_completion())

    client, sleeps = _client(handler)
    assert client.complete([], []).content == "hi"
    assert attempts["n"] == 3 and len(sleeps) == 2 and all(s >= 0.25 for s in sleeps)


def test_retries_transport_errors() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("refused", request=request)
        if attempts["n"] == 2:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=_completion())

    client, sleeps = _client(handler)
    assert client.complete([], []).content == "hi" and len(sleeps) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_never_retries_4xx(status: int) -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(status, text="bad request")

    client, sleeps = _client(handler)
    with pytest.raises(ModelHTTPError) as info:
        client.complete([], [])
    assert info.value.status == status and attempts["n"] == 1 and sleeps == []
    assert SECRET not in str(info.value)


def test_retry_budget_is_bounded_by_attempts() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(502, text="")

    client, sleeps = _client(handler, max_attempts=3)
    with pytest.raises(ModelError, match="after 3 attempts"):
        client.complete([], [])
    assert attempts["n"] == 3 and len(sleeps) == 2 and client.cost().model_calls == 0


def test_retry_budget_is_bounded_by_total_sleep() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "10"})

    client, sleeps = _client(handler, max_attempts=50, max_backoff_total_s=25.0, backoff_cap_s=10.0)
    with pytest.raises(ModelError, match="retry budget exhausted"):
        client.complete([], [])
    assert sum(sleeps) <= 25.0 and len(sleeps) == 2


def test_unparseable_retry_after_falls_back_to_jitter() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})
        return httpx.Response(200, json=_completion())

    client, sleeps = _client(handler, backoff_base_s=0.5)
    client.complete([], [])
    assert len(sleeps) == 1 and 0.0 <= sleeps[0] <= 0.5


def test_non_json_success_body_is_a_response_error() -> None:
    client, _ = _client(lambda r: httpx.Response(200, text="<html>"))
    with pytest.raises(ModelResponseError, match="not JSON"):
        client.complete([], [])


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": "text"}]}, []])
def test_malformed_completion_is_a_response_error(body: Any) -> None:
    with pytest.raises(ModelResponseError):
        parse_completion(body, 0.0)


def test_bad_tool_arguments_are_reported_per_call() -> None:
    calls: list[dict[str, Any]] = [
        {"id": "a", "function": {"name": "f", "arguments": "{not json"}},
        {"id": "b", "function": {"name": "g", "arguments": "[1, 2]"}},
        {"id": "c", "function": {"name": "h", "arguments": ""}},
        {"id": "d", "function": {"name": "k", "arguments": {"x": 1}}},
        {"function": {"name": "m", "arguments": "{}"}},
    ]
    turn = parse_completion(_completion(None, calls) | {"usage": None}, 0.1)
    assert set(turn.arg_errors) == {"a", "b"}
    assert [c.args for c in turn.tool_calls] == [{}, {}, {}, {"x": 1}, {}]
    assert turn.tool_calls[4].call_id == "call_4"
    assert turn.input_tokens == 0 and turn.content is None


def test_scripted_model_replays_and_records() -> None:
    turns = [scripted_turn(tool_calls=[("lookup", {"id": 1})]), scripted_turn("bye")]
    model = ScriptedModel(turns, name="s")
    first = model.complete([{"role": "user", "content": "q"}], [{"type": "function"}])
    assert model.name == "s" and first.tool_calls[0].name == "lookup"
    assert first.message["tool_calls"][0]["function"]["arguments"] == '{"id": 1}'
    assert model.complete([], []).content == "bye"
    assert model.requests[0][0] == [{"role": "user", "content": "q"}]
    with pytest.raises(ScriptExhaustedError):
        model.complete([], [])
