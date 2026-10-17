from __future__ import annotations

import base64
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from tau2_testkit import STOP, case, config, fresh_factory

from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.bench.tau.store import JsonlStore
from ratchet.bench.tau2.evolve import RecordingRunner, frozen_record, signer_from_env
from ratchet.bench.tau2.results import RunWriter
from ratchet.bench.tau2.runner import Tau2Runner, seed_harness
from ratchet.bench.tau2.tasks import family_of
from ratchet.improve.receipt import ReceiptError, Signer


def test_recording_runner_keeps_trace_and_simulation(tmp_path: Path) -> None:
    logs: list[str] = []
    store = JsonlStore(tmp_path)
    runner = Tau2Runner(config("airline"), factory=fresh_factory([STOP]))
    rec = RecordingRunner(runner, store, RunWriter(tmp_path), logs.append)
    trace = rec(
        seed_harness("airline"), case("airline", "0"), ScriptedModel([scripted_turn("Hi")]), None, run_id="e1"
    )
    assert trace.outcome.passed and len(store.traces()) == 1
    assert (tmp_path / "simulations" / "e1.json").exists() and "PASS" in logs[0]

    def boom(*a: Any) -> Any:
        raise ConnectionError("down")

    rec.runner = Tau2Runner(config("airline"), factory=boom)
    with pytest.raises(ConnectionError):
        rec(seed_harness("airline"), case("airline", "0"), ScriptedModel([]), None, run_id="e2")
    assert "ERROR ConnectionError" in logs[1]


def test_frozen_record_names_the_admitting_decision(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    seed = seed_harness("airline")
    assert frozen_record(store, store, None, seed)["decision"] is None
    child = seed.child(created_by="improver")
    store.put_harness(seed)
    store.put_harness(child)
    assert frozen_record(store, store, seed.version, seed)["harness"]["version"] == seed.version
    envelope = base64.b64encode(b'{"payload": "x"}').decode("ascii")
    store.put_decision({"candidate": child.version, "admitted": False}, decision_id="d0")
    store.put_decision({"candidate": child.version, "admitted": True, "envelope": envelope}, decision_id="d1")
    out = frozen_record(store, store, child.version, seed)
    assert out["harness"] == asdict(child) and out["decision_id"] == "d1"
    store.put_harness(seed.child(created_by="human", procedures=[]))
    lone = frozen_record(store, store, store.read("harnesses")[-1]["version"], seed)
    assert lone["decision"] is None


def test_signer_is_the_pinned_decision_key(monkeypatch: pytest.MonkeyPatch, decision_key: Signer) -> None:
    assert signer_from_env().keyid == decision_key.keyid
    monkeypatch.delenv("RATCHET_SIGNING_KEY")
    with pytest.raises(ReceiptError, match="no decision signing key"):
        signer_from_env()


def test_case_families() -> None:
    assert family_of("retail", "12") == "tau2-retail"
    assert family_of("telecom", "[mobile_data_issue]x[PERSONA:None]") == "tau2-telecom-mobile_data_issue"
