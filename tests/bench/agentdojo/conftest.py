"""Shared fixtures for the AgentDojo adapter tests. Everything runs offline: AgentDojo ships its data."""

from __future__ import annotations

import warnings

import pytest

with warnings.catch_warnings():
    # google-genai, imported by AgentDojo's pipeline package, touches typing._UnionGenericAlias at import
    # time, which Python 3.14 deprecates. It is third-party code we cannot change; nothing else is ignored.
    warnings.filterwarnings("ignore", "'_UnionGenericAlias' is deprecated", DeprecationWarning)
    import agentdojo.agent_pipeline  # noqa: F401

from ratchet.bench.agentdojo.cases import DojoSuite, build_suite


@pytest.fixture(scope="session")
def banking() -> DojoSuite:
    return build_suite("banking", max_users=2)
