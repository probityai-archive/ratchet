from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from tau2_testkit import fresh_factory

import ratchet.bench.tau2.__main__ as cli
from ratchet.agent.model import Message, ModelTurn, ToolSpec, scripted_turn
from ratchet.bench.tau2.report import check_runs, render_report
from ratchet.bench.tau2.results import (
    by_task,
    load_results,
    pass_hat_k,
    read_jsonl,
    summarize_run,
)
from ratchet.bench.tau2.runner import Tau2Runner, seed_harness
from ratchet.improve.evaluate import INFRA_RETRIES
from ratchet.types import Guard, Procedure


class SayDone:
    """An agent model that answers every turn with the same text; `fail_first` raises on the first calls."""

    def __init__(self, *, fail_first: int = 0) -> None:
        self.fail_first = fail_first
        self.calls = 0

    @property
    def name(self) -> str:
        return "say-done"

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("endpoint down")
        return scripted_turn("Done.")

    def close(self) -> None:
        return None


class FakeMongo:
    docs: list[dict[str, Any]] = []  # noqa: RUF012  # shared across instances on purpose

    def __init__(self, uri: str, **kw: Any) -> None:
        self.uri = uri

    def __getitem__(self, name: str) -> Any:
        return {"tau2_traces": self}

    def insert_one(self, doc: dict[str, Any]) -> None:
        FakeMongo.docs.append(doc)

    def replace_one(self, query: dict[str, Any], doc: dict[str, Any], upsert: bool = False) -> None:
        FakeMongo.docs = [d for d in FakeMongo.docs if d["_id"] != query["_id"]] + [doc]

    def close(self) -> None:
        return None


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> SayDone:
    model = SayDone()
    monkeypatch.setattr(cli, "ModelClient", lambda config: model)
    monkeypatch.setattr(cli, "Tau2Runner", lambda cfg: Tau2Runner(cfg, factory=fresh_factory()))
    monkeypatch.setenv("RATCHET_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("RATCHET_API_KEY", "k")
    monkeypatch.delenv("RATCHET_MONGODB_URI", raising=False)
    return model


def _run(out: Path, *extra: str) -> Path:
    args = [
        "--out",
        str(out),
        "run",
        "--domain",
        "airline",
        "--tasks",
        "0,1",
        "--trials",
        "2",
        "--workers",
        "1",
    ]
    assert cli.main([*args, *extra]) == 0
    return next(out.glob("runs/airline/*/run-1"))


def test_run_writes_tau2_bench_results_and_ratchet_records(
    scripted: SayDone, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run(tmp_path)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert config["tasks"] == 2 and config["trials"] == 2 and config["judge_model"] == "gpt-5.6-luna"
    assert "k" not in json.dumps(config).replace('"', " ").split()  # the key is never recorded
    results = load_results(root)
    assert len(results.simulations) == 4 and {s.trial for s in results.simulations} == {0, 1}
    assert results.info.agent_info.implementation == f"probity-ratchet:{config['harness']['version']}"
    rows = read_jsonl(root / "runs.jsonl")
    assert len(rows) == 4 and len(read_jsonl(root / "traces.jsonl")) == 4
    summary = summarize_run(root)
    assert summary["units_planned"] == 4 and summary["simulations"] == 4
    assert summary["tau2"]["pass_hat_k"].keys() == {1, 2}
    assert summary["pass_hat_k_lost_as_fail"][1] == summary["tau2"]["pass_hat_k"][1]
    with pytest.raises(SystemExit, match="--resume"):
        _run(tmp_path)
    capsys.readouterr()
    assert cli.main(["--out", str(tmp_path), "summary"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [r["run"] for r in listed] == [str(root)]


def test_resume_finishes_only_missing_units(scripted: SayDone, tmp_path: Path) -> None:
    root = _run(tmp_path)
    rows = read_jsonl(root / "runs.jsonl")
    (root / "runs.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    calls = scripted.calls
    _run(tmp_path, "--resume")
    assert len(read_jsonl(root / "runs.jsonl")) == 4 and scripted.calls == calls + 3


def test_lost_units_are_retried_and_recorded(
    monkeypatch: pytest.MonkeyPatch, scripted: SayDone, tmp_path: Path
) -> None:
    scripted.fail_first = 100
    root = _run(tmp_path, "--name", "run-1")
    rows = read_jsonl(root / "runs.jsonl")
    assert all(r["infra_failed"] and not r["passed"] for r in rows)
    assert all(r["attempts"] == INFRA_RETRIES + 1 for r in rows)
    attempts = read_jsonl(root / "attempts.jsonl")
    assert len(attempts) == 4 * (INFRA_RETRIES + 1)
    summary = summarize_run(root)
    assert summary["lost_to_endpoint"] == 4 and summary["simulations"] == 0
    assert summary["pass_hat_k_lost_as_fail"] == {1: 0.0, 2: 0.0}


def test_runner_exceptions_count_as_lost(
    monkeypatch: pytest.MonkeyPatch, scripted: SayDone, tmp_path: Path
) -> None:
    def broken(cfg: Any) -> Tau2Runner:
        def boom(*a: Any) -> Any:
            raise ConnectionError("user simulator down")

        return Tau2Runner(cfg, factory=boom)

    monkeypatch.setattr(cli, "Tau2Runner", broken)
    root = _run(tmp_path)
    rows = read_jsonl(root / "runs.jsonl")
    assert all(r["infra_failed"] and r["run_id"] is None for r in rows)
    assert rows[0]["reasons"] == ["run error: ConnectionError: user simulator down"]


def test_traces_go_to_the_evaluator_only_database(
    monkeypatch: pytest.MonkeyPatch, scripted: SayDone, tmp_path: Path
) -> None:
    import pymongo

    FakeMongo.docs = []
    monkeypatch.setattr(pymongo, "MongoClient", FakeMongo)
    monkeypatch.setenv("RATCHET_MONGODB_URI", "mongodb://fake")
    _run(tmp_path)
    assert len(FakeMongo.docs) == 4
    assert all(d["_id"].startswith("airline/") and "row" in d for d in FakeMongo.docs)


def test_run_needs_an_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RATCHET_BASE_URL", raising=False)
    monkeypatch.delenv("RATCHET_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="RATCHET_BASE_URL"):
        cli.main(["--out", str(tmp_path), "run"])


def test_finalize_rebuilds_results(scripted: SayDone, tmp_path: Path) -> None:
    root = _run(tmp_path)
    (root / "results.json").unlink()
    assert cli.main(["--out", str(tmp_path), "finalize", str(root)]) == 0
    assert len(load_results(root).simulations) == 4


def test_report_lists_every_run_with_denominators(scripted: SayDone, tmp_path: Path) -> None:
    assert render_report(tmp_path).endswith("No runs recorded yet.\n")
    root = _run(tmp_path)
    page = tmp_path / "page.md"
    assert cli.main(["--out", str(tmp_path), "report", "--write", str(page)]) == 0
    text = page.read_text(encoding="utf-8")
    assert "conversations planned 4, recorded 4, completed 4" in text
    assert "| `0` |" in text and "| `1` |" in text and "pass^2" in text
    (root / "runs.jsonl").write_text("", encoding="utf-8")
    assert "not computed" in render_report(tmp_path)


def test_package_refuses_a_run_that_does_not_qualify_and_drafts_it(
    scripted: SayDone, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run(tmp_path)
    problems = check_runs([root])
    assert any("2 of 50 base tasks" in p for p in problems)
    assert any("2 trials" in p for p in problems)
    base = ["package", str(root), "--dest", str(tmp_path / "sub"), "--organization", "Org"]
    base += [
        "--model-organization",
        "OpenAI",
        "--email",
        "a@example.com",
        "--reference",
        "Code=https://example.com",
    ]
    with pytest.raises(SystemExit, match="not a qualifying submission"):
        cli.main(base)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    assert cli.main([*base, "--draft", "--receipt", str(receipt)]) == 0
    sub = json.loads((tmp_path / "sub" / "submission.json").read_text(encoding="utf-8"))
    assert sub["submission_type"] == "custom" and sub["model_name"] == "gpt-5.6-luna"
    notes = sub["methodology"]["notes"]
    assert "DRAFT" in notes and "receipt.json sha256" in notes and "Guards (0): none" in notes
    assert sub["methodology"]["verification"]["omitted_questions"] is True
    files = list((tmp_path / "sub" / "trajectories").glob("*.json"))
    assert len(files) == 1 and sub["trajectory_files"]["airline"] == files[0].name
    pct = sub["results"]["airline"]["pass_1"]
    assert pct == pytest.approx(100 * summarize_run(root)["tau2"]["pass_hat_k"][1], abs=0.01)


def test_check_runs_flags_duplicates_and_mixed_configurations(scripted: SayDone, tmp_path: Path) -> None:
    a = _run(tmp_path)
    args = ["--out", str(tmp_path), "run", "--domain", "airline", "--tasks", "0", "--trials", "1"]
    assert cli.main([*args, "--name", "run-2", "--agent-model", "other"]) == 0
    b = next(tmp_path.glob("runs/airline/*/run-2"))
    problems = check_runs([a, b])
    assert any("appears 2 times" in p for p in problems)
    assert any("model configurations differ" in p for p in problems)


def test_harness_files_load_verify_and_rebase(tmp_path: Path) -> None:
    seed = seed_harness("airline")
    guard = Guard("g1", "no cancel", "def check(call, view, history):\n    return None\n")
    proc = Procedure("p1", "refunds", "a refund is asked", ["check the policy"])
    other = seed_harness("retail").child(guards=[guard], procedures=[proc], created_by="improver")
    path = tmp_path / "h.json"
    path.write_text(json.dumps({"harness": asdict(other)}), encoding="utf-8")
    assert cli.load_harness(str(path), "airline") == other
    moved = cli.load_harness(str(path), "airline", onto_seed=True)
    assert moved.parent == seed.version and moved.system_prompt == seed.system_prompt
    assert [g.id for g in moved.guards] == ["g1"] and [p.id for p in moved.procedures] == ["p1"]
    assert cli.rebase(moved, moved).guards == moved.guards
    assert cli.load_harness("seed", "airline") == seed
    bad = asdict(other)
    bad["system_prompt"] = "tampered"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        cli.load_harness(str(path), "airline")
    record = cli.harness_record(moved)
    assert record["guards"] == ["g1"] and record["procedures"] == ["p1"]


def test_pass_hat_k_and_grouping() -> None:
    rows = [
        {"task_id": "a", "passed": True},
        {"task_id": "a", "passed": False},
        {"task_id": "b", "passed": True},
    ]
    grouped = by_task(rows)
    assert grouped == {"a": [True, False], "b": [True]}
    assert pass_hat_k({"a": [True, False], "b": [True, True]}, 1) == 0.75
    assert pass_hat_k({"a": [True, False], "b": [True, True]}, 2) == 0.5
    with pytest.raises(ValueError):
        pass_hat_k({}, 1)
    with pytest.raises(ValueError):
        pass_hat_k({"a": [True]}, 0)
    with pytest.raises(ValueError, match="fewer than"):
        pass_hat_k({"a": [True]}, 2)


def test_fetch_data_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "fetch_data", lambda path: path)
    assert cli.main(["fetch-data"]) == 0
    assert len(capsys.readouterr().out.split()[-1]) == 64


def test_git_commit_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*a: Any, **k: Any) -> Any:
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", fail)
    assert cli._git_commit() == "unknown"


def test_evolve_with_nothing_to_fix_freezes_the_seed(
    monkeypatch: pytest.MonkeyPatch, scripted: SayDone, tmp_path: Path
) -> None:
    monkeypatch.delenv("RATCHET_SIGNING_KEY", raising=False)
    args = ["--out", str(tmp_path), "evolve", "--domain", "airline", "--tasks", "0,13", "--workers", "2"]
    assert cli.main(args) == 0
    root = tmp_path / "evolve" / "airline" / "evolve-1"
    frozen = json.loads((root / "frozen.json").read_text(encoding="utf-8"))
    assert frozen["decision"] is None and frozen["harness"]["version"] == seed_harness("airline").version
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    assert result["active"] == frozen["harness"]["version"] and result["frozen_decision"] is None
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert (config["train"], config["heldout"]) == (1, 1)
    assert (root / "signer.pub").read_bytes().startswith(b"-----BEGIN PUBLIC KEY-----")
    assert len(list((root / "simulations").glob("*.json"))) >= 2
    assert cli.load_harness(str(root / "frozen.json"), "airline").version == frozen["harness"]["version"]
    with pytest.raises(SystemExit, match="already has a run"):
        cli.main(args)


def test_upload_stores_a_box_run_once(
    monkeypatch: pytest.MonkeyPatch, scripted: SayDone, tmp_path: Path
) -> None:
    import pymongo

    root = _run(tmp_path)
    monkeypatch.setattr(pymongo, "MongoClient", FakeMongo)
    with pytest.raises(SystemExit, match="RATCHET_MONGODB_URI"):
        cli.main(["upload", str(root)])
    monkeypatch.setenv("RATCHET_MONGODB_URI", "mongodb://fake")
    FakeMongo.docs = []
    assert cli.main(["upload", str(root)]) == 0
    assert cli.main(["upload", str(root)]) == 0
    assert len(FakeMongo.docs) == 4 and all("row" in d for d in FakeMongo.docs)


def test_a_run_over_several_endpoints_records_each_and_finishes(
    monkeypatch: pytest.MonkeyPatch, scripted: SayDone, tmp_path: Path
) -> None:
    users: list[Any] = []

    def factory(cfg: Any, t: Any, seed: int | None, sim_id: str) -> Any:
        orch = fresh_factory()(cfg, t, seed, sim_id)
        orch.user.llm_args = {}
        users.append(orch.user)
        return orch

    monkeypatch.setattr(cli, "Tau2Runner", lambda cfg, **kw: Tau2Runner(cfg, factory=factory, **kw))
    root = _run(tmp_path, "--base-url", "http://a/v1,http://b/v1")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert config["agent_base_url"] == config["user_base_url"] == "http://a/v1,http://b/v1"
    assert len(read_jsonl(root / "runs.jsonl")) == 4 and summarize_run(root)["simulations"] == 4
    assert {u.llm_args["api_base"] for u in users} <= {"http://a/v1", "http://b/v1"} and len(users) == 4
    users.clear()
    _run(tmp_path / "own", "--base-url", "http://a/v1,http://b/v1", "--user-base-url", "http://u/v1")
    assert len(users) == 4 and all(u.llm_args == {} for u in users)


def test_a_stopped_run_shows_its_reason_until_it_is_resumed(scripted: SayDone, tmp_path: Path) -> None:
    root = _run(tmp_path)
    rows = read_jsonl(root / "runs.jsonl")
    (root / "runs.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    page = render_report(tmp_path)
    assert "Incomplete: 1 of 4 conversations recorded; no stop reason" in page
    assert "| 1 / 4 | incomplete, no reason recorded |" in page
    assert cli.main(["--out", str(tmp_path), "stop", str(root), "--reason", "endpoint cancelled"]) == 0
    page = render_report(tmp_path)
    assert "Stopped after 1 of 4 conversations: endpoint cancelled" in page and "| stopped |" in page
    _run(tmp_path, "--resume")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert "stopped" not in config and config["stops"][0]["reason"] == "endpoint cancelled"
    page = render_report(tmp_path)
    assert "Stopped once after 1 conversations and resumed: endpoint cancelled" in page
    assert "| 4 / 4 | complete |" in page
