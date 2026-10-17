"""A referee factory for the process tests: the loop tests' held-out cases, fakes, and a signed file store."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ratchet.improve.receipt import Signer
from ratchet.improve.referee import RefereeParts
from ratchet.store.file_store import FileActivator, FileStore

from .fakes import FakeRunCase, FakeSimulator, LocalRunner
from .test_loop import CASES


def parts(config: Mapping[str, Any]) -> RefereeParts:
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(config["store_key"]))
    store = FileStore(Path(config["store_dir"]), key=key)
    return RefereeParts(
        cases=[c for c in CASES if c.split == "heldout"],
        run_case=FakeRunCase(),
        model=None,
        simulator=FakeSimulator(),
        runner=LocalRunner(),
        signer=Signer.from_seed_hex(config["decision_key"]),
        store=store,
        activator=FileActivator(store),
        suite_digest="suite",
        workers=4,
    )
