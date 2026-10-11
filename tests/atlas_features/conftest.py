"""Fixtures for the Atlas probes. Integration tests need the hackathon sandbox and fail fast without it."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from pymongo import MongoClient
from pymongo.database import Database

from ratchet.atlas_features.common import ProbeUnavailableError, connect, probe_db


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "sandbox: runs against the hackathon Atlas cluster; opt in with -m sandbox"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Sandbox tests need live secrets: they run only on request (`-m sandbox`), never error otherwise."""
    selected = config.getoption("-m") or ""
    opted_out = [i for i in items if "sandbox" in i.keywords and "sandbox" not in selected]
    config.hook.pytest_deselected(items=opted_out)
    items[:] = [i for i in items if i not in opted_out]


@pytest.fixture(scope="session")
def sandbox_client() -> Iterator[MongoClient[dict[str, Any]]]:
    try:
        client = connect()
    except ProbeUnavailableError as exc:
        pytest.fail(f"Atlas sandbox is required for this test: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="session")
def sandbox_db(sandbox_client: MongoClient[dict[str, Any]]) -> Database[dict[str, Any]]:
    return probe_db(sandbox_client)
