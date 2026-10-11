from __future__ import annotations

import json
import random
from typing import Any

import httpx
import openai
import pytest
from tool_sandbox.common.execution_context import ExecutionContext
from toolsandbox_testkit import chat_completion, mock_client

from ratchet.bench.toolsandbox.gate import GATE
from ratchet.bench.toolsandbox.user import (
    EmptyReply,
    EndpointUser,
    python_name,
    with_retries,
)

END_TOOL = {"type": "function", "function": {"name": "end_conversation", "parameters": {}}}


def _json(body: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=body)


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("call_1", "call_1"),
        ("chatcmpl-tool-abc123", "chatcmpl_tool_abc123"),
        ("9lives", "call_9lives"),
        ("", "call_"),
        ("a b.c", "a_b_c"),
    ],
)
def test_python_name_makes_a_valid_identifier(raw: str, want: str) -> None:
    assert python_name(raw) == want
    assert python_name(raw).isidentifier()


def test_with_retries_returns_the_first_success_without_sleeping() -> None:
    slept: list[float] = []
    assert with_retries(lambda: 7, sleep=slept.append) == 7
    assert slept == []


def test_with_retries_waits_out_transient_errors() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise EmptyReply("again")
        return "ok"

    assert with_retries(flaky, sleep=slept.append, rng=random.Random(1)) == "ok"
    assert calls["n"] == 3
    assert len(slept) == 2


def test_with_retries_gives_up_after_the_attempt_limit() -> None:
    calls = {"n": 0}

    def always() -> None:
        calls["n"] += 1
        raise EmptyReply("never")

    with pytest.raises(EmptyReply):
        with_retries(always, attempts=3, sleep=lambda _s: None, rng=random.Random(1))
    assert calls["n"] == 3


def test_with_retries_gives_up_when_the_waiting_budget_is_spent() -> None:
    calls = {"n": 0}

    def always() -> None:
        calls["n"] += 1
        raise EmptyReply("never")

    with pytest.raises(EmptyReply):
        with_retries(always, attempts=50, budget_s=0.0, sleep=lambda _s: None, rng=random.Random(1))
    assert calls["n"] == 1


def test_with_retries_does_not_retry_a_caller_error() -> None:
    calls = {"n": 0}

    def bad() -> None:
        calls["n"] += 1
        raise ValueError("a bug")

    with pytest.raises(ValueError, match="a bug"):
        with_retries(bad, sleep=lambda _s: None)
    assert calls["n"] == 1


def _user(handler: Any, temperature: float | None = None) -> EndpointUser:
    return EndpointUser(
        "http://endpoint.test/v1", "dummy-key", "the-model", temperature, mock_client(handler)
    )


def test_a_text_reply_comes_back_unchanged() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _json(chat_completion("Please turn it off."))

    user = _user(handler)
    with GATE.held(ExecutionContext()):
        response = user.model_inference([{"role": "user", "content": "hi"}], openai.NOT_GIVEN)
    assert response.choices[0].message.content == "Please turn it off."
    assert seen[0]["model"] == "the-model"
    assert "temperature" not in seen[0]
    assert "tools" not in seen[0]


def test_the_temperature_is_sent_only_when_set() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _json(chat_completion("ok"))

    with GATE.held(ExecutionContext()):
        _user(handler, 0.0).model_inference([], openai.NOT_GIVEN)
    assert seen[0]["temperature"] == 0.0


def test_a_tool_call_id_that_is_not_a_python_name_is_rewritten() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _json(chat_completion(None, [("chatcmpl-tool-9f", "end_conversation", {})]))

    with GATE.held(ExecutionContext()):
        response = _user(handler).model_inference([], [END_TOOL])
    call = response.choices[0].message.tool_calls[0]
    assert call.id == "chatcmpl_tool_9f"
    assert call.function.name == "end_conversation"


def test_the_model_call_runs_outside_the_gate() -> None:
    held_during_call: list[bool] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        held_during_call.append(GATE._lock.locked())
        return _json(chat_completion("ok"))

    with GATE.held(ExecutionContext()):
        _user(handler).model_inference([], openai.NOT_GIVEN)
        assert GATE._lock.locked()
    assert held_during_call == [False]


def test_an_empty_reply_is_retried_until_the_endpoint_answers() -> None:
    replies = [chat_completion(None), chat_completion(""), chat_completion("finally")]
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json(replies.pop(0))

    user = _user(handler)
    user.sleep = sleeps.append
    with GATE.held(ExecutionContext()):
        response = user.model_inference([], openai.NOT_GIVEN)
    assert response.choices[0].message.content == "finally"
    assert len(sleeps) == 2


def test_a_rate_limit_is_retried_and_a_bad_request_is_not() -> None:
    answers = [httpx.Response(429, json={"error": {"message": "slow down"}}), _json(chat_completion("ok"))]

    def limited(_request: httpx.Request) -> httpx.Response:
        return answers.pop(0)

    user = _user(limited)
    user.sleep = lambda _s: None
    with GATE.held(ExecutionContext()):
        assert user.model_inference([], openai.NOT_GIVEN).choices[0].message.content == "ok"

    def refused(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad"}})

    with GATE.held(ExecutionContext()), pytest.raises(openai.BadRequestError):
        _user(refused).model_inference([], openai.NOT_GIVEN)


def test_a_reply_that_calls_a_tool_the_user_does_not_have_is_retried() -> None:
    replies = [
        chat_completion(None, [("c1", "get_contacts", {})]),
        chat_completion(None, [("c2", "end_conversation", {})]),
    ]
    sleeps: list[float] = []
    tools = [END_TOOL]

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json(replies.pop(0))

    user = _user(handler)
    user.sleep = sleeps.append
    with GATE.held(ExecutionContext()):
        response = user.model_inference([], tools)
    assert response.choices[0].message.tool_calls[0].function.name == "end_conversation"
    assert len(sleeps) == 1


def test_without_an_offered_tool_list_any_tool_call_is_left_to_the_benchmark() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _json(chat_completion(None, [("c1", "anything", {})]))

    with GATE.held(ExecutionContext()):
        response = _user(handler).model_inference([], openai.NOT_GIVEN)
    assert response.choices[0].message.tool_calls[0].function.name == "anything"
