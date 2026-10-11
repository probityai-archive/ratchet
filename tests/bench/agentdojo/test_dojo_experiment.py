from __future__ import annotations

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
    admitted_changes,
    baseline,
    evolve_from,
    lineage,
    provenance,
    suite_digest,
    version_metrics,
)
from ratchet.bench.agentdojo.run import DojoRunner, JsonStore, seed_harness

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
    client.close()
    with pytest.raises(ValueError, match="Claude"):
        s.client("claude-sonnet-4-6")
    assert "secret-xyz" not in repr(s)


def test_provenance_and_digest(tmp_path: Path, banking: DojoSuite) -> None:
    s = _settings(tmp_path)
    p = provenance(s, banking, None)
    assert p["model"] == "gpt-5.6-luna" and p["seed"] == 0 and p["suite"] == "banking"
    assert p["agentdojo_package_version"] == "0.1.35" and p["agentdojo_reference_clone_commit"] is None
    assert p["suite_digest"] == suite_digest(banking) and len(p["suite_digest"]) == 64
    assert _git_head(tmp_path) is None
    assert _git_head(tmp_path / "missing" / "dir") is None


def test_baseline_records_everything_and_metrics_read_it(tmp_path: Path, banking: DojoSuite) -> None:
    out = tmp_path / "run"
    store, runner = JsonStore(out), DojoRunner(banking)
    v0, report = baseline(banking, runner, GroundTruthModel(PLAN), store, workers=4)
    assert report.count() == len(banking.cases)
    table = version_metrics(banking, out, [v0.version, "absent"])
    assert [(r["version"], r["role"]) for r in table] == [(v0.version, "baseline")]
    train = table[0]["train"]
    # user_task_4 is the only task the plan solves: its clean case and its attacked ones pass.
    assert train["utility"]["hits"] == 1 and train["targeted_attack_success"]["hits"] == 0
    assert admitted_changes(out, [v0.version])[v0.version]["guards"] == []
    assert lineage(v0, None) == [v0.version]
    assert version_metrics(banking, tmp_path / "empty", [v0.version]) == []
    assert version_metrics(banking, out, ["absent"]) == []


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
    monkeypatch.setattr(Settings, "client", lambda self, model: GroundTruthModel(PLAN))
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
    monkeypatch.setattr(Settings, "client", lambda self, model: GroundTruthModel(PLAN))
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
    (s.out / "results.json").unlink()
    assert cli.summarize(s.out)["status"] == "in progress"
    prov = json.loads((s.out / "provenance.json").read_text())
    (s.out / "provenance.json").write_text(json.dumps({**prov, "seed": 7}))
    with pytest.raises(SystemExit, match="suite digest"):
        cli.summarize(s.out)


def test_report_writes_one_document_over_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Settings, "client", lambda self, model: GroundTruthModel(PLAN))
    monkeypatch.setattr(cli, "REFERENCE_CLONE", tmp_path / "no-clone")
    s = _settings(tmp_path, max_users=1, workers=4)
    cli.run("baseline", s, log=lambda _m: None)
    pre = tmp_path / "pre.md"
    pre.write_text("Intro text.")
    doc = tmp_path / "doc.md"
    assert cli.main(["report", "--out", str(doc), "--runs", str(s.out), "--preamble", str(pre)]) == 0
    text = doc.read_text()
    assert text.startswith("# AgentDojo results\n\nIntro text.")
    assert "| banking | `gpt-5.6-luna` | complete, 0 gen | " in text and "none promoted" in text
    assert "### banking (v1.2.2)" in text
    with pytest.raises(SystemExit):
        cli.parse(["report"])


def seed_harness_version(results: dict[str, object]) -> str:
    versions = results["versions"]
    assert isinstance(versions, list)
    return str(versions[0])


def test_seed_version_is_stable(banking: DojoSuite) -> None:
    assert seed_harness(banking).version == seed_harness(banking).version
