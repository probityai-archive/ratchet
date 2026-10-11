from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def private_runtime_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Session servers started by a test live in a short, private directory (a Unix socket path is limited
    to about 100 bytes) and stop when their socket is removed at teardown."""
    directory = Path(tempfile.mkdtemp(prefix="rh-", dir="/tmp"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(directory))
    yield directory
    sockets = list(directory.glob("ratchet-hook/*.sock"))
    for sock in sockets:
        sock.unlink(missing_ok=True)
    deadline = time.monotonic() + 3
    while sockets and time.monotonic() < deadline and any(p.with_suffix(".lock").exists() for p in sockets):
        time.sleep(0.1)
        if all(not _locked(p.with_suffix(".lock")) for p in sockets):
            break
    shutil.rmtree(directory, ignore_errors=True)


def _locked(lock: Path) -> bool:
    import fcntl

    try:
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    return False
