"""Fixtures for the Atlas probes. Integration tests need the hackathon sandbox and fail fast without it."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from pymongo import MongoClient
from pymongo.database import Database

from ratchet.atlas_features.common import ProbeUnavailableError, connect, probe_db


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: needs the hackathon Atlas sandbox")


@pytest.fixture(scope="session")
def sandbox_client() -> Iterator[MongoClient[dict[str, Any]]]:
    try:
        client = connect(timeout_ms=2000)
    except ProbeUnavailableError as exc:
        pytest.fail(f"Atlas sandbox is required for this test: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="session")
def sandbox_db(sandbox_client: MongoClient[dict[str, Any]]) -> Database[dict[str, Any]]:
    return probe_db(sandbox_client)
