"""The verifier is our pinned public package, not a copy: assert the exact pins and that they run."""

from __future__ import annotations

import json
import sys
from importlib import metadata

import pytest
from agent_evidence_vectors.run_vectors import main as vectors_main  # type: ignore[import-untyped]

PROBITY_VERIFY_COMMIT = "b920366397cb131232163706864a70723b867367"


def test_probity_verify_is_pinned_to_the_full_commit() -> None:
    direct = json.loads(metadata.distribution("probity-verify").read_text("direct_url.json") or "{}")
    assert direct["url"] == "https://github.com/probityai/probity-verify"
    assert direct["vcs_info"]["commit_id"] == PROBITY_VERIFY_COMMIT


def test_agent_evidence_vectors_is_pinned_from_pypi() -> None:
    assert metadata.version("agent-evidence-vectors") == "0.17.9"


def test_pinned_vector_harness_self_test_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["agent-evidence-vectors", "--self-test"])
    assert vectors_main() == 0
