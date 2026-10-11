"""Fixtures for the store lane. Integration tests need the local Atlas container (docker/compose.yaml)."""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pymongo import MongoClient

from ratchet.store import Store
from ratchet.store.embed import EMBEDDING_DIMS, InputType

ROOT = Path(__file__).resolve().parents[2]
LOCAL_URI = "mongodb://ratchet:ratchet-local-dev@127.0.0.1:27018/?directConnection=true"
_WORD = re.compile(r"[a-z0-9_]+")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: needs the local Atlas container (docker/compose.yaml)")
    config.addinivalue_line("markers", "live: calls the real Voyage API; needs VOYAGE_API_KEY")


class HashEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words are close in cosine distance."""

    model = "test-hash-1024"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        self.calls.append((input_type, len(texts)))
        return [self._one(t) for t in texts]

    @staticmethod
    def _one(text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIMS
        for word in _WORD.findall(text.lower()):
            h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
            vec[h % EMBEDDING_DIMS] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


@pytest.fixture
def hash_embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture(scope="session")
def atlas_uri() -> str:
    uri = os.environ.get("RATCHET_TEST_MONGODB_URI", LOCAL_URI)

    def up() -> bool:
        client: MongoClient[dict[str, Any]] = MongoClient(uri, serverSelectionTimeoutMS=2000)
        try:
            client.admin.command("ping")
        except Exception:
            return False
        finally:
            client.close()
        return True

    if not up():
        compose = ROOT / "docker" / "compose.yaml"
        subprocess.run(["docker", "compose", "-f", str(compose), "up", "-d", "--wait"], check=False)
    assert up(), "local Atlas is not reachable; run: docker compose -f docker/compose.yaml up -d --wait"
    return uri


@pytest.fixture
def store(atlas_uri: str, hash_embedder: HashEmbedder) -> Iterator[Store]:
    name = f"ratchet_test_{uuid.uuid4().hex[:10]}"
    s = Store(atlas_uri, name, embedder=hash_embedder)
    yield s
    s._client.drop_database(name)
    s.close()


def make_client(uri: str) -> MongoClient[dict[str, Any]]:
    return MongoClient(uri, tz_aware=True)
