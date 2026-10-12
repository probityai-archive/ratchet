from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from dojo_fakes import GroundTruthModel, JsonImprover

from ratchet.agent.model import ModelClient
from ratchet.bench.agentdojo import __main__ as cli
from ratchet.bench.agentdojo.cases import DojoSuite
from ratchet.bench.agentdojo.experiment import (
    Settings,
    _git_head,
    _uncommitted,
    admitted_changes,
    baseline,
    decisions,
    evaluation_rows,
    evolve_from,
    harness_from,
    lineage,
    provenance,
    suite_digest,
)
from ratchet.bench.agentdojo.run import DojoRunner, FirstToolCallOnly, JsonStore, seed_harness

PLAN = {"refund GB29NWBK60161331926819": [("get_most_recent_transactions", {"n": 100}),
        ("send_money", {"recipient": "GB29NWBK60161331926819", "amount": 10.0, "subject": "Refund",
                        "date": "2022-04-01"})]}  # fmt: skip


def _settings(tmp_path: Path, **kw: object) -> Settings:
    return Settings(
        suite="banking", out=tmp_path / "run", base_url="http://x/v1/", api_key="secret-xyz", **kw
    )  # type: ignore[arg-type]


def test_settings_build_clients_and_refuse_claude(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    client = s.client("gpt-5.6-luna")
    assert isinstance(client, ModelClient) and client.name == "gpt-5.6-luna"
    assert client._config.max_backoff_total_s == 900.0
    client.close()
    with pytest.raises(ValueError, match="Claude"):
        s.client("claude-sonnet-4-6")
    assert "secret-xyz" not in repr(s)


def test_agent_and_improver_clients_follow_their_settings(tmp_path: Path) -> None:
    s = _settings(tmp_path, improver_base_url="http://improver/v1", single_tool_call=True)
    agent, improver = s.agent_client(), s.improver_client()
    assert isinstance(agent, FirstToolCallOnly) and agent.name == "gpt-5.6-luna"
    assert isinstance(improver, ModelClient) and improver._config.base_url == "http://improver/v1"
    plain = _settings(tmp_path).agent_client()
    assert isinstance(plain, ModelClient) and plain._config.base_url == "http://x/v1"
    for c in (agent.inner, improver, plain):
        assert isinstance(c, ModelClient)
        c.close()


def test_provenance_and_digest(tmp_path: Path, banking: DojoSuite) -> None:
    s = _settings(tmp_path)
    p = provenance(s, banking, None)
    assert p["model"] == "gpt-5.6-luna" and p["seed"] == 0 and p["suite"] == "banking"
    assert p["agentdojo_package_version"] == "0.1.35" and p["agentdojo_reference_clone_commit"] is None
    assert p["suite_digest"] == suite_digest(banking) and len(p["suite_digest"]) == 64
    assert _git_head(tmp_path) is None and _uncommitted(tmp_path) is None
    assert isinstance(p["ratchet_uncommitted_source"], list) and p["ratchet_commit"]
    assert _git_head(tmp_path / "missing" / "dir") is None


def test_baseline_records_everything_and_metrics_read_it(tmp_path: Path, banking: DojoSuite) -> None:
    out = tmp_path / "run"
    store, runner = JsonStore(out), DojoRunner(banking)
    v0, report = baseline(banking, runner, GroundTruthModel(PLAN), store, workers=4)
    assert report.count() == len(banking.cases)
    table = evaluation_rows(banking, out)
    assert [(r["version"], r["role"]) for r in table] == [(v0.version, "baseline")]
    train = table[0]["train"]
    # user_task_4 is the only task the plan solves: its clean case and its attacked ones pass.
    assert train["utility"]["hits"] == 1 and train["targeted_attack_success"]["hits"] == 0
    assert admitted_changes(out, [v0.version])[v0.version]["guards"] == []
    assert lineage(v0, None) == [v0.version]
    assert evaluation_rows(banking, tmp_path / "empty") == []
    assert decisions(out) == []
    with (out / "evals.jsonl").open("a") as f:
        gen = {"type": "generation", "candidates": [{"harness_version": "c1", "kind": "guard"}]}
        f.write(json.dumps(gen) + "\n")
    checks = [{"name": "replay", "passed": True}, {"name": "heldout", "passed": False}]
    for did, cand in (("p->c1", "c1"), ("p->c2", "c2")):
        record = {"decision_id": did, "candidate": cand, "admitted": False, "checks": checks}
        with (out / "decisions.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
    assert decisions(out) == [
        {
            "decision_id": "p->c1",
            "candidate": "c1",
            "kind": "guard",
            "admitted": False,
            "failed": ["heldout"],
        },
        {
            "decision_id": "p->c2",
            "candidate": "c2",
            "kind": "composite",
            "admitted": False,
            "failed": ["heldout"],
        },
    ]


def test_evolve_runs_one_generation_with_a_silent_improver(tmp_path: Path, banking: DojoSuite) -> None:
    out = tmp_path / "run"
    s = _settings(tmp_path, generations=1, workers=4)
    store, runner = JsonStore(out), DojoRunner(banking)
    model = GroundTruthModel(PLAN)
    v0, _ = baseline(banking, runner, model, store, workers=4)
    result = evolve_from(banking, runner, model, JsonImprover(), store, v0, s, log=lambda _m: None)
    assert result.harness.version == v0.version
    assert len(result.generations) == 1 and result.generations[0].promoted is None
    assert json.loads((out / "signer.json").read_text())["keyid"]


def test_cli_parses_and_requires_the_endpoint(tmp_path: Path) -> None:
    args = cli.parse(["baseline", "--out", str(tmp_path), "--max-users", "1"])
    assert args.suite == "banking" and args.workers == 8 and args.max_users == 1
    with pytest.raises(SystemExit, match="RATCHET_BASE_URL"):
        cli.settings_from(args, {})
    s = cli.settings_from(args, {"RATCHET_BASE_URL": "http://x/v1", "RATCHET_API_KEY": "k"})
    assert s.generations == 0 and s.model == "gpt-5.6-luna"


def test_cli_run_writes_results_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Settings, "client", lambda self, model, base_url=None: GroundTruthModel(PLAN))
    monkeypatch.setattr(cli, "REFERENCE_CLONE", tmp_path / "no-clone")
    s = _settings(tmp_path, max_users=1, workers=4)
    results = cli.run("baseline", s, log=lambda _m: None)
    assert results["versions"] == [seed_harness_version(results)]
    assert (s.out / "results.json").is_file() and (s.out / "summary.md").read_text().startswith("### banking")
    monkeypatch.setattr(cli, "run", lambda command, settings: results)
    monkeypatch.setenv("RATCHET_BASE_URL", "http://x/v1")
    monkeypatch.setenv("RATCHET_API_KEY", "k")
    assert cli.main(["baseline", "--out", str(tmp_path / "other")]) == 0


def test_summarize_rebuilds_results_from_the_run_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Settings, "client", lambda self, model, base_url=None: GroundTruthModel(PLAN))
    monkeypatch.setattr(cli, "REFERENCE_CLONE", tmp_path / "no-clone")
    s = _settings(tmp_path, max_users=1, workers=4)
    first = cli.run("baseline", s, log=lambda _m: None)
    assert first["status"] == "complete"
    with (s.out / "evals.jsonl").open("a") as f:
        f.write(json.dumps({"type": "generation", "level1": {"promoted": "v-next"}}) + "\n")
        f.write(json.dumps({"type": "generation", "level1": {}}) + "\n")
    again = cli.main(["summarize", "--out", str(s.out)])
    assert again == 0
    results = json.loads((s.out / "results.json").read_text())
    assert results["versions"] == [first["versions"][0], "v-next"] and results["status"] == "complete"
    assert results["metrics"] == first["metrics"] and len(results["generations"]) == 2
    (s.out / cli.COMPLETE).unlink()
    assert cli.summarize(s.out)["status"] == "in progress"
    (s.out / cli.STOPPED).write_text("no headroom\n")
    assert cli.summarize(s.out)["status"] == "stopped: no headroom"
    (s.out / cli.COMPLETE).write_text("done\n")
    assert cli.summarize(s.out)["status"] == "complete"
    prov = json.loads((s.out / "provenance.json").read_text())
    (s.out / "provenance.json").write_text(json.dumps({**prov, "seed": 7}))
    with pytest.raises(SystemExit, match="suite digest"):
        cli.summarize(s.out)


def test_report_writes_one_document_over_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Settings, "client", lambda self, model, base_url=None: GroundTruthModel(PLAN))
    monkeypatch.setattr(cli, "REFERENCE_CLONE", tmp_path / "no-clone")
    s = _settings(tmp_path, max_users=1, workers=4)
    cli.run("baseline", s, log=lambda _m: None)
    doc = tmp_path / "doc.md"
    assert cli.main(["report", "--out", str(doc), "--runs", str(s.out)]) == 0
    text = doc.read_text()
    assert text.startswith("# AgentDojo results\n\nAgentDojo is a set of tasks where tool outputs carry")
    assert "AgentDojo v1.2.2 (package 0.1.35), run as a Ratchet world" in text
    assert "et al" not in text and "NeurIPS" not in text
    assert "| banking | `run` | `gpt-5.6-luna` | complete, 0 gen | " in text and "none promoted" in text
    assert "### banking (v1.2.2), run `run`, complete" in text
    notes = tmp_path / "notes.md"
    notes.write_text("## Notes on the runs\n\nhand notes\n")
    assert cli.main(["report", "--out", str(doc), "--notes", str(notes), "--runs", str(s.out)]) == 0
    assert "\n## Notes on the runs\n\nhand notes\n\n## Overview" in doc.read_text()
    with pytest.raises(SystemExit):
        cli.parse(["report"])


def test_run_labels_keep_same_named_runs_in_two_batches_apart(tmp_path: Path) -> None:
    first, rerun = tmp_path / "box-1" / "banking-hpg", tmp_path / "box-2" / "banking-hpg"
    alone = tmp_path / "banking-evolve"
    labels = cli.run_labels([alone, first, rerun])
    assert labels == ["banking-evolve", "box-1/banking-hpg", "box-2/banking-hpg"]
    assert cli.run_labels([first]) == ["banking-hpg"]
    assert cli.run_labels([first, first]) == ["banking-hpg", "banking-hpg"]
    assert cli.run_labels([]) == []


def test_transfer_reruns_every_version_under_another_attack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Settings, "client", lambda self, model, base_url=None: GroundTruthModel(PLAN))
    monkeypatch.setattr(cli, "REFERENCE_CLONE", tmp_path / "no-clone")
    source = _settings(tmp_path, max_users=1, workers=4)
    first = cli.run("baseline", source, log=lambda _m: None)
    out = tmp_path / "transfer"
    monkeypatch.setenv("RATCHET_BASE_URL", "http://x/v1")
    monkeypatch.setenv("RATCHET_API_KEY", "k")
    argv = ["transfer", "--out", str(out), "--source", str(source.out), "--attack", "ignore_previous"]
    assert cli.main(argv) == 0
    results = json.loads((out / "results.json").read_text())
    p = results["provenance"]
    assert p["attack"] == "ignore_previous" and p["trained_attack"] == first["provenance"]["attack"]
    assert p["transfer_from"] == "run" and results["versions"] == first["versions"]
    (row,) = results["metrics"]
    assert row["role"] == "transfer:ignore_previous" and "train" not in row
    with pytest.raises(SystemExit, match="must differ"):
        cli.main(["transfer", "--out", str(tmp_path / "t2"), "--source", str(source.out)])
    with pytest.raises(SystemExit, match="--source"):
        cli.main(["transfer", "--out", str(tmp_path / "t3")])


def test_stored_harness_must_match_its_digest(banking: DojoSuite) -> None:
    record = dataclasses.asdict(seed_harness(banking))
    assert harness_from(record) == seed_harness(banking)
    with pytest.raises(ValueError, match="digest"):
        harness_from({**record, "system_prompt": "changed"})


def seed_harness_version(results: dict[str, object]) -> str:
    versions = results["versions"]
    assert isinstance(versions, list)
    return str(versions[0])


def test_seed_version_is_stable(banking: DojoSuite) -> None:
    assert seed_harness(banking).version == seed_harness(banking).version
