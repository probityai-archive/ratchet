"""The verifier is our pinned public package, not a copy: assert the exact pins and that they run."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

import pytest
from agent_evidence_vectors.run_vectors import main as vectors_main  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
# The PyPI wheel of probity-verify 0.1.1.
PROBITY_VERIFY_WHEEL_SHA256 = "e5788644f64d4faf73d999f9b8371f761141493403f642659ba0ad36a15ef6cf"


def test_probity_verify_is_the_hash_pinned_release_from_pypi() -> None:
    dist = metadata.distribution("probity-verify")
    assert dist.version == "0.1.1"
    assert dist.read_text("direct_url.json") is None  # from the index, not a git checkout or a path
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    entry = lock[lock.index('name = "probity-verify"\nversion = "0.1.1"') :]
    entry = entry[: entry.index("\n[[package]]")]
    assert 'source = { registry = "https://pypi.org/simple" }' in entry
    assert f'hash = "sha256:{PROBITY_VERIFY_WHEEL_SHA256}"' in entry


def test_the_demo_image_build_installs_no_system_package() -> None:
    dockerfile = (ROOT / "docker/demo.Dockerfile").read_text(encoding="utf-8")
    assert "dnf " not in dockerfile and "apt-get" not in dockerfile and "microdnf" not in dockerfile


def test_agent_evidence_vectors_is_pinned_from_pypi() -> None:
    assert metadata.version("agent-evidence-vectors") == "0.19.0"


def test_pinned_vector_harness_self_test_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["agent-evidence-vectors", "--self-test"])
    assert vectors_main() == 0


def test_the_demo_image_builds_probityai_from_the_locked_commit() -> None:
    """Until probityai is on PyPI the image builds its wheel; it must be the commit uv.lock pins."""
    dockerfile = (ROOT / "docker/demo.Dockerfile").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    entry = lock[lock.index('[[package]]\nname = "probityai"') :]
    entry = entry[: entry.index("\n[[package]]", 1)]
    source = entry[entry.index("source = { git = ") :].split("\n", 1)[0]
    commit = source[source.index("#") + 1 :][:40]
    assert len(commit) == 40 and "jcs-admit?subdirectory=python" in source
    assert f"ARG PROBITYAI_COMMIT={commit}\n" in dockerfile
