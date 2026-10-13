"""The verifier is our pinned public package, not a copy: assert the exact pins and that they run."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

import pytest
from agent_evidence_vectors.run_vectors import main as vectors_main  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
# The PyPI wheel of probity-verify 0.1.0; its package files are byte-identical to the source at commit
# b920366397cb131232163706864a70723b867367, which the verifier was pinned to before the release.
PROBITY_VERIFY_WHEEL_SHA256 = "18e0f99fb21701cd2d5eb312d99700d7cf800e8cdf9dc8e5d3608bc315cf1453"


def test_probity_verify_is_the_hash_pinned_release_from_pypi() -> None:
    dist = metadata.distribution("probity-verify")
    assert dist.version == "0.1.0"
    assert dist.read_text("direct_url.json") is None  # from the index, not a git checkout or a path
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    entry = lock[lock.index('name = "probity-verify"\nversion = "0.1.0"') :]
    entry = entry[: entry.index("\n[[package]]")]
    assert 'source = { registry = "https://pypi.org/simple" }' in entry
    assert f'hash = "sha256:{PROBITY_VERIFY_WHEEL_SHA256}"' in entry


def test_the_demo_image_build_installs_no_system_package() -> None:
    dockerfile = (ROOT / "docker/demo.Dockerfile").read_text(encoding="utf-8")
    assert "dnf " not in dockerfile and "apt-get" not in dockerfile and "microdnf" not in dockerfile


def test_agent_evidence_vectors_is_pinned_from_pypi() -> None:
    assert metadata.version("agent-evidence-vectors") == "0.17.9"


def test_pinned_vector_harness_self_test_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["agent-evidence-vectors", "--self-test"])
    assert vectors_main() == 0
