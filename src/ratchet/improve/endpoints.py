"""The worker model's endpoint configuration, passed by reference: a JSON file and the SHA-256 of its RFC 8785
canonical JSON (`endpoint_config_digest`).

A driver names the file and its digest; the referee opens the file in its own process, refuses it when the
digest differs, and builds the worker clients there. The driver builds its own training-run clients from the
same file, so both halves of the evaluator name one model configuration, and every receipt that carries the
digest names exactly the document a third party can hash.

    {"endpoints": ["http://host:8801/v1", ...], "model": "gpt-oss-120b", "temperature": 0.0,
     "seed": 7,               # deterministic: one fixed seed on every request, endpoints in round robin
     "seed_base": 1000,       # stochastic: repetition r is sent with seed seed_base + r
     "api_key_env": "...",    # optional: the environment variable holding the key (never the key itself)
     "serving": {...}}        # optional: how the endpoint was served, part of what the digest names
"""

from __future__ import annotations

import itertools
import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ratchet_cases.types import Trace, sha256_hex

from ratchet.canonical import canonical_json


class EndpointsError(ValueError):
    """The endpoint configuration is missing, malformed, or not the one its digest names."""


def config_digest(doc: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(dict(doc)))


@dataclass(frozen=True)
class Endpoints:
    """A loaded endpoint configuration and its digest."""

    doc: dict[str, Any]
    digest: str

    @property
    def urls(self) -> list[str]:
        return list(self.doc["endpoints"])

    @property
    def model(self) -> str:
        return str(self.doc["model"])

    def model_config(self) -> dict[str, Any]:
        """What the evaluator identity names: the document and its digest, never a key."""
        return {**self.doc, "endpoint_config_digest": self.digest}

    def reference(self, path: Path) -> dict[str, str]:
        """What a driver passes to the referee: the file and the digest it must have."""
        return {"file": str(path), "digest": self.digest}

    def client(self, url: str, seed: int | None) -> Any:
        from ratchet.agent.model import ModelClient, ModelConfig, SeededTransport

        key_env = self.doc.get("api_key_env")
        key = os.environ.get(str(key_env), "") if key_env else "unused"
        temperature = self.doc.get("temperature")
        return ModelClient(
            ModelConfig(
                base_url=url,
                model=self.model,
                api_key=key or "unused",
                temperature=None if temperature is None else float(temperature),
            ),
            transport=None if seed is None else SeededTransport(seed),
        )


def _check(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise EndpointsError("the endpoint configuration is not a JSON object")
    urls = doc.get("endpoints")
    if not isinstance(urls, list) or not urls or not all(isinstance(u, str) and u for u in urls):
        raise EndpointsError("endpoints must be a non-empty list of URLs")
    if not isinstance(doc.get("model"), str) or not doc["model"]:
        raise EndpointsError("model must be a non-empty string")
    for name in ("seed", "seed_base"):
        if name in doc and (not isinstance(doc[name], int) or isinstance(doc[name], bool)):
            raise EndpointsError(f"{name} must be an integer")
    if "api_key" in doc:
        raise EndpointsError("the configuration names a key; name its environment variable (api_key_env)")
    return doc


def parse(doc: Any, digest: str | None = None) -> Endpoints:
    checked = _check(doc)
    actual = config_digest(checked)
    if digest is not None and digest != actual:
        raise EndpointsError(f"endpoint configuration digest {actual[:12]} is not the named {digest[:12]}")
    return Endpoints(checked, actual)


def load(path: Path, digest: str | None = None) -> Endpoints:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EndpointsError(f"cannot read the endpoint configuration {path}: {exc}") from exc
    return parse(doc, digest)


def resolve(reference: Mapping[str, Any]) -> Endpoints:
    """The referee's side: the file a driver named, checked against the digest it named."""
    if not isinstance(reference.get("file"), str) or not isinstance(reference.get("digest"), str):
        raise EndpointsError("an endpoint reference names a file and a digest")
    return load(Path(reference["file"]), reference["digest"])


RunCase = Callable[..., Trace]


class RoundRobin:
    """One client per endpoint, used in turn: a `RunCase` that ignores the model it is handed. `seeded` (the
    deterministic worker) requires the configuration's fixed seed and sends it with every request; unseeded,
    a seed the configuration names is still sent."""

    def __init__(self, endpoints: Endpoints, run_case: RunCase, *, seeded: bool = True) -> None:
        seed = endpoints.doc.get("seed")
        if seed is None and seeded:
            raise EndpointsError("a deterministic configuration names a seed")
        self.clients = [endpoints.client(u, None if seed is None else int(seed)) for u in endpoints.urls]
        self._next = itertools.count(1)
        self._lock = threading.Lock()
        self._run_case = run_case

    def __call__(self, harness: Any, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
        with self._lock:
            i = next(self._next)
        return self._run_case(harness, case, self.clients[i % len(self.clients)], store, run_id=run_id)

    def close(self) -> None:
        for c in self.clients:
            c.close()


class PerRepetition:
    """The stochastic worker: repetition `r` of every arm goes to a client sending seed `seed_base + r`, so
    the arms share one seed schedule. With a replicas file, the endpoint list is re-read before every
    execution, so replicas of the same served model join as they answer; each execution records its URL."""

    def __init__(
        self,
        endpoints: Endpoints,
        replicas: Path | None = None,
        make: Callable[[str, int], Any] | None = None,
    ) -> None:
        if endpoints.doc.get("seed_base") is None:
            raise EndpointsError("a stochastic configuration names seed_base")
        self.endpoints = endpoints
        self.seed_base = int(endpoints.doc["seed_base"])
        self.replicas = replicas
        self._make = make or (lambda url, seed: endpoints.client(url, seed))
        self._clients: dict[tuple[str, int], Any] = {}
        self._lock = threading.Lock()

    def live(self) -> Sequence[str]:
        if self.replicas is not None and self.replicas.is_file():
            listed = [x.strip() for x in self.replicas.read_text(encoding="utf-8").split() if x.strip()]
            if listed:
                return listed
        return self.endpoints.urls

    def client_for(self, index: int, rep: int) -> tuple[str, Any]:
        live = self.live()
        url = live[index % len(live)]
        with self._lock:
            if (url, rep) not in self._clients:
                self._clients[(url, rep)] = self._make(url, self.seed_base + rep)
            return url, self._clients[(url, rep)]

    def close(self) -> None:
        for c in self._clients.values():
            c.close()
