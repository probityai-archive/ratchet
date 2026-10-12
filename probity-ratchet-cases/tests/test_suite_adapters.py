from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from ratchet_cases.adapters.openai_compat import Endpoint, EndpointError, parse_completion
from ratchet_cases.adapters.python_callable import AdapterError, load

COMPLETION = {
    "choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
}


def test_flags_win_over_the_environment_and_both_are_required() -> None:
    env = {"RATCHET_BASE_URL": "http://env/v1/", "RATCHET_MODEL": "env-model", "RATCHET_API_KEY": "k"}
    assert Endpoint.from_env(environ=env) == Endpoint("http://env/v1", "env-model", "k")
    assert Endpoint.from_env(base_url="http://flag", model="m", environ=env).base_url == "http://flag"
    with pytest.raises(EndpointError, match="RATCHET_MODEL"):
        Endpoint.from_env(environ={"RATCHET_BASE_URL": "http://x"})


def test_the_request_carries_the_key_only_in_its_header() -> None:
    req = Endpoint("http://x/v1", "m", "secret").request([{"role": "user", "content": "q"}], [])
    assert req.full_url == "http://x/v1/chat/completions" and req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer secret"
    assert isinstance(req.data, bytes) and b"secret" not in req.data
    assert json.loads(req.data)["temperature"] == 0.0
    assert Endpoint("http://x", "m").request([], []).get_header("Authorization") is None


def test_a_completion_becomes_one_assistant_message_with_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        return io.BytesIO(json.dumps(COMPLETION).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    message = Endpoint("http://x", "m")([], [])
    assert message["content"] == "hi" and message["usage"] == {"input_tokens": 7, "output_tokens": 2}


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("http://x", 429, "slow down", {}, None),  # type: ignore[arg-type]
        urllib.error.URLError("refused"),
        TimeoutError(),
    ],
)
def test_endpoint_failures_are_endpoint_errors(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def fail(request: urllib.request.Request, timeout: float) -> Any:
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(EndpointError):
        Endpoint("http://x", "m")([], [])


def test_a_body_that_is_not_a_completion_is_refused() -> None:
    with pytest.raises(EndpointError, match="not a chat completion"):
        parse_completion({"choices": []})
    assert parse_completion({"choices": [{"message": {}}]})["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
    }


def test_a_python_callable_loads_by_module_and_attribute() -> None:
    assert load("json:dumps") is json.dumps
    assert load("json.decoder:JSONDecoder.decode") is json.decoder.JSONDecoder.decode


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("json", "expected module:function"),
        ("no_such_module_xyz:act", "cannot import"),
        ("json:nothing_here", "has no attribute"),
        ("json:__name__", "is not callable"),
    ],
)
def test_a_bad_python_spec_says_what_is_wrong(spec: str, message: str) -> None:
    with pytest.raises(AdapterError, match=message):
        load(spec)
