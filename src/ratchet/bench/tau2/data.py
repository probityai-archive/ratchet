"""The tau2-bench task data, pinned to the same commit as the installed package.

The tau2-bench wheel ships code only; its domains (databases, tasks, policies) and the user simulator's
guidelines live in the repository's `data/` directory, which tau2-bench reads from `TAU2_DATA_DIR` at
import time. This module fetches exactly those files from the pinned commit with a sparse, blob-filtered
git checkout, so the code and the data always come from one commit, and records a digest of the files a
run reads.

Nothing here imports tau2-bench: the package `__init__` sets `TAU2_DATA_DIR` from this module before any
tau2-bench module is loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path

TAU2_COMMIT = "4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699"
TAU2_REPO = "https://github.com/sierra-research/tau2-bench"
DOMAINS = ("retail", "airline", "telecom")
# The files a text run reads: each domain's directory and the user simulator's guidelines. Voice-only
# files (audio difficulty, voice tasks) and the retail task-issue notes are not needed and not fetched.
SPARSE_PATTERNS = (
    "/data/tau2/user_simulator/",
    *(f"/data/tau2/domains/{d}/" for d in DOMAINS),
    "!/data/tau2/domains/*/tasks_voice.json",
    "!/data/tau2/domains/*/audio_difficulty.json",
    "!/data/tau2/domains/retail/task_issues/",
)
MARKER = ".ratchet-tau2-commit"

Run = Callable[..., subprocess.CompletedProcess[str]]


def default_data_dir(commit: str = TAU2_COMMIT) -> Path:
    """`$RATCHET_TAU2_CACHE/<commit>/data`, default cache `~/.cache/ratchet/tau2-bench`."""
    cache = os.environ.get("RATCHET_TAU2_CACHE") or str(Path.home() / ".cache" / "ratchet" / "tau2-bench")
    return Path(cache) / commit / "data"


def data_dir() -> Path:
    """The data directory tau2-bench reads: `TAU2_DATA_DIR`, else the pinned cache directory."""
    return Path(os.environ.get("TAU2_DATA_DIR") or default_data_dir())


def is_complete(path: Path, commit: str = TAU2_COMMIT) -> bool:
    """True when `path` holds a finished fetch of `commit` (the marker is written last)."""
    marker = path / MARKER
    return marker.is_file() and marker.read_text(encoding="utf-8").strip() == commit


def _git(run: Run, cwd: Path, *args: str) -> str:
    done = run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def fetch_data(
    dest: Path, *, commit: str = TAU2_COMMIT, repo: str = TAU2_REPO, run: Run = subprocess.run
) -> Path:
    """Fetch the data of `commit` into `dest` (the directory that will hold `tau2/`); idempotent.

    The checkout lives in `dest`'s parent; `dest` is its `data/` directory. The commit is checked against
    `git rev-parse HEAD` before the marker is written, so a half-finished or wrong fetch is never used."""
    if is_complete(dest, commit):
        return dest
    root = dest.parent
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").exists():
        _git(run, root, "init", "-q")
        _git(run, root, "remote", "add", "origin", repo)
    _git(run, root, "sparse-checkout", "set", "--no-cone", *SPARSE_PATTERNS)
    _git(run, root, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    _git(run, root, "checkout", "-q", "--force", "FETCH_HEAD")
    head = _git(run, root, "rev-parse", "HEAD")
    if head != commit:
        raise RuntimeError(f"fetched {head}, expected {commit}")
    (dest / MARKER).write_text(commit + "\n", encoding="utf-8")
    return dest


def domain_files(path: Path, domains: Sequence[str] = DOMAINS) -> list[Path]:
    """Every file a run over `domains` reads, sorted."""
    roots = [path / "tau2" / "user_simulator", *(path / "tau2" / "domains" / d for d in domains)]
    return sorted(p for r in roots if r.is_dir() for p in r.rglob("*") if p.is_file())


def data_digest(path: Path, domains: Sequence[str] = DOMAINS) -> str:
    """SHA-256 over the relative path and bytes of every file in `domain_files`."""
    h = hashlib.sha256()
    for f in domain_files(path, domains):
        h.update(str(f.relative_to(path)).encode("utf-8") + b"\x00")
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def installed_commit() -> str:
    """The tau2-bench commit installed, from the package's PEP 610 record."""
    try:
        raw = metadata.distribution("tau2").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return "not installed"
    info = json.loads(raw or "{}")
    return str(info.get("vcs_info", {}).get("commit_id", "unknown"))
