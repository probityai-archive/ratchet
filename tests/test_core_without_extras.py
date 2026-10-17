"""The core install (no `atlas`, no `telemetry` extra) imports and runs: pymongo and raindrop are optional."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

PROGRAM = textwrap.dedent(
    """
    import importlib.abc, sys

    class Refuse(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("pymongo", "bson", "gridfs", "raindrop"):
                raise ImportError(f"{name} is not installed in this test")
            return None

    sys.meta_path.insert(0, Refuse())
    import ratchet.cli, ratchet.verify, ratchet.verify.cli, ratchet.improve.loop, ratchet.improve.sessions
    import ratchet.improve.sessions_live, ratchet.store, ratchet.store.file_store, ratchet.view.live

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from ratchet.cli import CliError, LiveBackend

    root = sys.argv[1]
    seed = Ed25519PrivateKey.generate().private_bytes_raw().hex()
    store = LiveBackend({"RATCHET_STORE_DIR": root, "RATCHET_STORE_KEY": seed}).open_store()
    print("file store", type(store).__name__)
    for env, needle in (({"RATCHET_MONGODB_URI": "mongodb://x"}, "atlas extra"), ({}, "RATCHET_STORE_DIR")):
        try:
            LiveBackend(env).open_store()
        except CliError as exc:
            assert needle in str(exc), exc
    try:
        LiveBackend({}).telemetry()
    except CliError as exc:
        assert "telemetry extra" in str(exc), exc
    print("ok")
    """
)


def test_the_core_imports_and_opens_its_store_without_the_extras(tmp_path: Path) -> None:
    out = subprocess.run([sys.executable, "-c", PROGRAM, str(tmp_path / "store")], capture_output=True,
                         text=True, check=False, timeout=120)  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["file", "store", "FileStore", "ok"]
