"""An `Act` over any OpenAI-compatible chat-completions endpoint, using only the standard library.

Flags win over the environment: `RATCHET_BASE_URL`, `RATCHET_API_KEY` and `RATCHET_MODEL` fill whatever is
not given. The key is sent in the Authorization header and is never written to a trace.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

JSON = dict[str, Any]


class EndpointError(RuntimeError):
    """The endpoint refused the request or returned something that is not a chat completion."""


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    api_key: str = ""
    temperature: float = 0.0
    timeout_s: float = 120.0

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> Endpoint:
        env = os.environ if environ is None else environ
        url = base_url or env.get("RATCHET_BASE_URL", "")
        name = model or env.get("RATCHET_MODEL", "")
        if not url or not name:
            raise EndpointError("set --base-url and --model, or RATCHET_BASE_URL and RATCHET_MODEL")
        return cls(base_url=url.rstrip("/"), model=name, api_key=api_key or env.get("RATCHET_API_KEY", ""))

    def request(self, messages: list[JSON], tools: list[JSON]) -> urllib.request.Request:
        body = {"model": self.model, "messages": messages, "tools": tools, "temperature": self.temperature}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )

    def __call__(self, messages: list[JSON], tools: list[JSON]) -> JSON:
        try:
            with urllib.request.urlopen(self.request(messages, tools), timeout=self.timeout_s) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as err:
            raise EndpointError(f"HTTP {err.code} from {self.base_url}") from err
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            raise EndpointError(f"no usable response from {self.base_url}: {err}") from err
        return parse_completion(payload)


def parse_completion(payload: Any) -> JSON:
    """The assistant message of a chat completion, with token usage under `usage`."""
    try:
        message = dict(payload["choices"][0]["message"])
    except (KeyError, IndexError, TypeError) as err:
        raise EndpointError(f"not a chat completion: {err!r}") from err
    usage = payload.get("usage") or {}
    message["usage"] = {
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
    }
    return message
