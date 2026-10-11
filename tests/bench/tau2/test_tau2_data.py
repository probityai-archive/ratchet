from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path

import pytest

from ratchet.bench.tau2 import data
from ratchet.bench.tau2.data import (
    MARKER,
    TAU2_COMMIT,
    data_digest,
    default_data_dir,
    domain_files,
    fetch_data,
    installed_commit,
    is_complete,
)


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _origin(tmp: Path) -> tuple[Path, str]:
    repo = tmp / "origin"
    for rel, text in {
        "data/tau2/domains/retail/db.json": "{}",
        "data/tau2/domains/retail/tasks_voice.json": "voice",
        "data/tau2/domains/retail/task_issues/a.md": "issue",
        "data/tau2/user_simulator/simulation_guidelines.md": "be a user",
        "data/tau2/results/big.json": "not needed",
        "src/code.py": "x = 1",
    }.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "uploadpack.allowFilter", "true")
    _git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "data")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_fetch_checks_out_only_the_data_of_the_commit(tmp_path: Path) -> None:
    repo, commit = _origin(tmp_path)
    dest = tmp_path / "cache" / commit / "data"
    out = fetch_data(dest, commit=commit, repo=f"file://{repo}")
    assert out == dest and is_complete(dest, commit) and not is_complete(dest, "other")
    assert (dest / "tau2" / "domains" / "retail" / "db.json").read_text(encoding="utf-8") == "{}"
    assert (dest / "tau2" / "user_simulator" / "simulation_guidelines.md").exists()
    assert not (dest / "tau2" / "domains" / "retail" / "tasks_voice.json").exists()
    assert not (dest / "tau2" / "domains" / "retail" / "task_issues").exists()
    assert not (dest / "tau2" / "results").exists() and not (dest.parent / "src").exists()
    files = domain_files(dest, ["retail"])
    assert [f.name for f in files] == ["db.json", "simulation_guidelines.md"]
    digest = data_digest(dest, ["retail"])
    assert len(digest) == 64 and digest == data_digest(dest, ["retail"])
    assert data_digest(dest, ["airline"]) != digest
    # A second call is a no-op once the marker is written.
    assert fetch_data(dest, commit=commit, repo="file:///nowhere") == dest


def test_fetch_refuses_a_checkout_of_another_commit(tmp_path: Path) -> None:
    repo, commit = _origin(tmp_path)
    calls: list[list[str]] = []

    def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        out = "f" * 40 if cmd[1:3] == ["rev-parse", "HEAD"] else ""
        return subprocess.CompletedProcess(cmd, 0, out, "")

    dest = tmp_path / "x" / "data"
    with pytest.raises(RuntimeError, match="expected"):
        fetch_data(dest, commit=commit, repo=f"file://{repo}", run=run)
    assert not (dest / MARKER).exists()
    assert ["git", "init", "-q"] in calls


def test_default_data_dir_follows_the_cache_setting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RATCHET_TAU2_CACHE", str(tmp_path))
    assert default_data_dir() == tmp_path / TAU2_COMMIT / "data"
    monkeypatch.delenv("RATCHET_TAU2_CACHE")
    assert default_data_dir("abc").parts[-3:] == ("tau2-bench", "abc", "data")
    monkeypatch.setenv("TAU2_DATA_DIR", str(tmp_path / "d"))
    assert data.data_dir() == tmp_path / "d"


def test_installed_commit_is_the_pin() -> None:
    assert installed_commit() == TAU2_COMMIT


def test_installed_commit_without_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    assert installed_commit() == "not installed"
