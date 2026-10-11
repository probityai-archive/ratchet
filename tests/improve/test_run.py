from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ratchet.improve import run
from ratchet.improve.loop import EvolveResult, GenerationRecord
from ratchet.improve.propose import SEED
from ratchet.types import Harness

HARNESS = Harness.build(
    parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=[], created_by="seed"
)


class _Ctx:
    def __init__(self, *a: Any, **kw: Any) -> None:
        self.args = a

    def __enter__(self) -> _Ctx:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __getitem__(self, name: str) -> Any:
        return name

    def setup(self, wait_search: bool = True) -> None:
        self.ready = wait_search

    def cost(self) -> Any:
        return type("C", (), {"__dict__": {"model_calls": 3}})()


def test_main_refuses_without_a_store_or_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RATCHET_MONGODB_URI", raising=False)
    monkeypatch.delenv("RATCHET_API_KEY", raising=False)
    assert run.main(["--out", str(tmp_path / "x.json")]) == 2


def test_main_wires_the_lanes_and_writes_a_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake_evolve(store: Any, model: Any, generations: int, *, lab: Any, seed: Harness) -> EvolveResult:
        seen.update(generations=generations, cases=len(lab.cases), config=lab.model_config)
        return EvolveResult(HARNESS, SEED, [GenerationRecord(1, seed.version, note="n")])

    for name in ("Store", "ModelClient", "GuardRuntime", "MongoClient"):
        monkeypatch.setattr(run, name, _Ctx)
    monkeypatch.setattr(run, "evolve", fake_evolve)
    monkeypatch.setenv("RATCHET_API_KEY", "secret-value")
    out = tmp_path / "live.json"
    rc = run.main(
        ["--uri", "mongodb://x", "--generations", "2", "--family", "refund_lost_reply", "--out", str(out)]
    )
    assert rc == 0 and seen["generations"] == 2 and seen["cases"] == 6
    text = out.read_text()
    assert "secret-value" not in text
    body = json.loads(text)
    assert body["final_harness"] == HARNESS.version and body["generations"][0]["note"] == "n"
    assert body["config"]["model"] == seen["config"]["model"] and len(body["signer_public_key_hex"]) == 64
