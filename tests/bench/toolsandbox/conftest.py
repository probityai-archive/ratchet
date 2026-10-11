"""Fixtures for the benchmark adapter tests. Everything runs offline: the scenarios ship with the package."""

from __future__ import annotations

import pytest
from toolsandbox_testkit import MISSING_KEY

from ratchet.bench.toolsandbox.cases import ToolSandboxCase, build_cases, scenarios


@pytest.fixture(scope="session")
def all_cases() -> list[ToolSandboxCase]:
    scenarios()  # build once for the session
    return build_cases(error=MISSING_KEY)


@pytest.fixture(autouse=True)
def no_rapid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAPID_API_KEY", raising=False)
