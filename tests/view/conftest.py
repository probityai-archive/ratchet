"""Fixtures for the views' integration tests, which need the local Atlas container (docker/compose.yaml)."""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pymongo import MongoClient

from ratchet.store import Store

ROOT = Path(__file__).resolve().parents[2]
LOCAL_URI = "mongodb://ratchet:ratchet-local-dev@127.0.0.1:27018/?directConnection=true"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: needs the local Atlas container (docker/compose.yaml)")


def _up(uri: str) -> bool:
    client: MongoClient[dict[str, Any]] = MongoClient(uri, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except Exception:
        return False
    finally:
        client.close()
    return True


@pytest.fixture(scope="session")
def atlas_uri() -> str:
    uri = os.environ.get("RATCHET_TEST_MONGODB_URI", LOCAL_URI)
    if not _up(uri):
        compose = ROOT / "docker" / "compose.yaml"
        subprocess.run(
            ["docker", "compose", "-f", str(compose), "up", "-d", "--wait"], check=False, timeout=240
        )
    if not _up(uri):
        pytest.fail("local Atlas is not reachable; run: docker compose -f docker/compose.yaml up -d --wait")
    return uri


@pytest.fixture
def atlas_db(atlas_uri: str) -> Iterator[tuple[str, str, Store]]:
    name = f"ratchet_view_{uuid.uuid4().hex[:10]}"
    store = Store(atlas_uri, name)
    yield atlas_uri, name, store
    store._client.drop_database(name)
    store.close()
