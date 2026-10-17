from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from tau_testkit import ScriptedUser, gold_turns, make_env

import ratchet.bench.tau.__main__ as cli
from ratchet.agent.model import ScriptedModel
from ratchet.bench.tau.runner import TauRunner, seed_harness
from ratchet.bench.tau.store import JsonlStore
from ratchet.bench.tau.tasks import TauCase


class _FakeRunner:
    """Stands in for `TauRunner`: every case runs task 0's gold actions against a scripted customer."""

    def __init__(self, user_model: str) -> None:
        self.inner = TauRunner(user_model=user_model, env_factory=lambda c, _m: make_env(0, ScriptedUser()))

    def __call__(self, harness: Any, case: Any, model: Any, store: Any, *, run_id: str) -> Any:
        if case.index == 1:
            raise RuntimeError("user simulator down")
        return self.inner(harness, case, ScriptedModel(gold_turns(0)), store, run_id=run_id)


def _cases(*_a: Any, **_k: Any) -> list[TauCase]:
    return [
        TauCase("tau-retail-train-005", "train", "train", 5),
        TauCase("tau-retail-test-000", "heldout", "test", 0),
        TauCase("tau-retail-test-001", "heldout", "test", 1),
    ]


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, inprocess_referee: Callable[..., Any]) -> None:
    monkeypatch.setattr(cli, "ProcessReferee", inprocess_referee)
    monkeypatch.setenv("RATCHET_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("RATCHET_API_KEY", "k")
    monkeypatch.setattr(cli, "TauRunner", _FakeRunner)
    monkeypatch.setattr(cli, "default_cases", _cases)
    monkeypatch.setattr(cli, "configure_user_endpoint", lambda *_a, **_k: None)


def test_trial_records_traces_and_an_eval_then_summary_reads_them(
    patched: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--out", str(tmp_path), "trial", "--split", "all", "--trial", "1"]) == 0
    run_dir = next((tmp_path / "runs" / "retail").glob("*/all-trial-1"))
    store = JsonlStore(run_dir)
    assert len(store.read("traces")) == 2
    config = json.loads((run_dir / "config.json").read_text())
    assert config["agent_model"] == "gpt-5.6-luna" and "dummy" not in json.dumps(config)
    assert config["tau_bench_commit"].startswith("59a200c")
    out = capsys.readouterr().out
    assert "ERROR RuntimeError" in out and "run errors 1" in out
    with pytest.raises(SystemExit):
        cli.main(["--out", str(tmp_path), "trial", "--split", "all", "--trial", "1"])
    capsys.readouterr()
    assert cli.main(["--out", str(tmp_path), "summary", "--max-k", "2"]) == 0
    rows = {(r["split"]): r for r in json.loads(capsys.readouterr().out)}
    assert (
        rows["heldout"]["tasks"] == 2 and rows["heldout"]["successes"] == 1 and rows["heldout"]["errors"] == 1
    )
    assert rows["heldout"]["pass^1"] == 0.5 and "pass^2" not in rows["heldout"]
    assert rows["train"]["domain"] == "retail"


def test_trial_with_limit_and_recorded_harness(patched: None, tmp_path: Path) -> None:
    h = seed_harness().child(system_prompt="changed", created_by="human")
    JsonlStore(tmp_path / "somewhere").put_harness(h)
    assert cli.load_harness(h.version, "retail", tmp_path) == h
    with pytest.raises(SystemExit):
        cli.load_harness("nope", "retail", tmp_path)
    args = ["--out", str(tmp_path), "trial", "--harness", h.version, "--split", "heldout", "--limit", "1"]
    assert cli.main(args) == 0
    run_dir = tmp_path / "runs" / "retail" / h.version / "heldout-trial-1"
    assert len(JsonlStore(run_dir).read("traces")) == 1


def test_harness_from_rejects_a_tampered_record() -> None:
    raw = asdict(seed_harness())
    assert cli.harness_from(raw) == seed_harness()
    raw["system_prompt"] = "tampered"
    with pytest.raises(ValueError):
        cli.harness_from(raw)


def test_endpoint_is_required(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RATCHET_BASE_URL", raising=False)
    monkeypatch.delenv("RATCHET_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["--out", str(tmp_path), "trial"])


def test_evolve_wires_the_lab_and_writes_the_result(
    patched: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    class _Result:
        harness = seed_harness()

        class instructions:
            version = "ins1"

        generations: tuple[Any, ...] = ()

    def fake_evolve(store: Any, model: Any, generations: int, *, lab: Any, seed: Any) -> Any:
        seen.update(lab=lab, generations=generations, seed=seed, model=model)
        lab.log("hello")
        trace = lab.run_case(seed, _cases()[0], model, None, run_id="x")
        store.put_trace(trace)
        return _Result()

    monkeypatch.setattr(cli, "evolve", fake_evolve)
    args = ["--out", str(tmp_path), "evolve", "--name", "t", "--generations", "2", "--max-evals", "2"]
    assert cli.main(args) == 0
    root = tmp_path / "evolve" / "retail" / "t"
    result = json.loads((root / "result.json").read_text())
    assert result["final_harness"] == seed_harness().version and result["config"]["max_evals"] == 2
    assert seen["generations"] == 2 and seen["lab"].max_evals == 2
    assert seen["lab"].suite_digest == result["config"]["split_record"]["digest"]
    assert "hello" in (root / "log.txt").read_text()
    assert result["active"] == [None, 0]
    with pytest.raises(SystemExit):
        cli.main(args)
