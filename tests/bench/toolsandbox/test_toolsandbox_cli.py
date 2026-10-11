from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest
from toolsandbox_testkit import Steady, user_factory

from ratchet.bench.toolsandbox import __main__ as cli
from ratchet.bench.toolsandbox.cases import ToolSandboxCase
from ratchet.bench.toolsandbox.replay import END
from ratchet.bench.toolsandbox.runner import seed_harness
from ratchet.types import to_jsonable


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, all_cases: list[ToolSandboxCase]) -> None:
    monkeypatch.setenv("RATCHET_BASE_URL", "http://endpoint.test/v1")
    monkeypatch.setenv("RATCHET_API_KEY", "k")
    monkeypatch.setattr(cli, "keyring_key", lambda: None)
    monkeypatch.setattr(cli, "build_cases", lambda **_kw: all_cases)
    monkeypatch.setattr(cli, "_client", lambda *_a, **_k: Steady())
    monkeypatch.setattr(cli, "endpoint_user", lambda *_a, **_k: user_factory([END]))


def test_select_takes_a_split_or_a_seeded_sample_of_runnable_cases(all_cases: list[ToolSandboxCase]) -> None:
    heldout = cli.select(all_cases, "heldout", 0, 1)
    assert heldout and all(c.split == "heldout" for c in heldout)
    assert len(cli.select(all_cases, "all", 0, 1)) == 1032
    sample = cli.select(all_cases, "all", 10, 1)
    assert len(sample) == 10 and all(not c.not_run for c in sample)
    assert sample == cli.select(all_cases, "all", 10, 1)
    assert sample != cli.select(all_cases, "all", 10, 2)
    assert [c.id for c in sample] == sorted(c.id for c in sample)


def test_select_never_asks_for_more_than_exist(all_cases: list[ToolSandboxCase]) -> None:
    runnable = [c for c in all_cases if not c.not_run]
    assert len(cli.select(all_cases, "all", 5000, 1)) == len(runnable)


def test_the_seed_harness_loads_by_name_and_a_recorded_one_by_version(tmp_path: Path) -> None:
    seed = cli.load_harness("seed", tmp_path)
    assert seed.version == seed_harness().version
    folder = tmp_path / "runs" / seed.version / "x"
    folder.mkdir(parents=True)
    (folder / "harnesses.jsonl").write_text(json.dumps(to_jsonable(seed)) + "\n", encoding="utf-8")
    assert cli.load_harness(seed.version, tmp_path).digest == seed.digest
    exported = tmp_path / "candidate.json"
    exported.write_text(
        json.dumps({"harness": to_jsonable(seed), "decision": {"admitted": True}}), encoding="utf-8"
    )
    assert cli.load_harness(f"@{exported}", tmp_path).digest == seed.digest
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(to_jsonable(seed)), encoding="utf-8")
    assert cli.load_harness(f"@{bare}", tmp_path).version == seed.version
    with pytest.raises(SystemExit, match="no harness 'nope'"):
        cli.load_harness("nope", tmp_path)


def test_a_missing_endpoint_is_a_clear_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RATCHET_BASE_URL", raising=False)
    monkeypatch.delenv("RATCHET_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="RATCHET_BASE_URL"):
        cli.main(["trial"])
    monkeypatch.setenv("RATCHET_API_KEY", "k")
    assert cli._endpoint(cli.parser().parse_args(["trial", "--base-url", "http://x/v1"])) == (
        "http://x/v1",
        "k",
    )


def test_the_parser_defaults_name_the_split_the_trial_and_the_seed_harness() -> None:
    args = cli.parser().parse_args(["trial"])
    assert (args.harness, args.split, args.trial, args.limit, args.n_train) == ("seed", "all", 1, 0, 26)
    assert args.temperature == 0.0 and args.workers == 8


def test_a_trial_records_traces_grades_and_an_evaluation(
    offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    assert (
        cli.main(["--out", str(out), "trial", "--limit", "2", "--workers", "2", "--agent-model", "agent-x"])
        == 0
    )
    folder = next((out / "runs").glob("*/all-trial-1"))
    for name in (
        "config.json",
        "cases.json",
        "harnesses.jsonl",
        "traces.jsonl",
        "evals.jsonl",
        "scenarios.jsonl",
    ):
        assert (folder / name).exists(), name
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert config["agent_model"] == "agent-x" and config["cases"] == 2 and config["rapid_api_key"] == "absent"
    assert config["tool_sandbox_commit"] == "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1"
    assert "api_key" not in json.dumps(config).replace("rapid_api_key", "")
    grades = [
        json.loads(line) for line in (folder / "scenarios.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(grades) == 2 and all("similarity" in g["graded"] for g in grades)
    shown = capsys.readouterr().out
    assert "[2/2]" in shown and "run errors 0" in shown


def test_a_second_trial_into_the_same_directory_is_refused(offline: None, tmp_path: Path) -> None:
    args = ["--out", str(tmp_path), "trial", "--limit", "1", "--workers", "1"]
    assert cli.main(args) == 0
    with pytest.raises(SystemExit, match="already has traces"):
        cli.main(args)
    assert cli.main([*args[:-2], "--workers", "1", "--trial", "2"]) == 0


def test_a_keyring_key_is_passed_to_the_tools_not_to_the_record(
    offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "keyring_key", lambda: "SECRET-VALUE")
    monkeypatch.delenv("RAPID_API_KEY", raising=False)
    try:
        assert cli.main(["--out", str(tmp_path), "trial", "--limit", "1", "--workers", "1"]) == 0
        import os

        assert os.environ["RAPID_API_KEY"] == "SECRET-VALUE"
    finally:
        import os

        os.environ.pop("RAPID_API_KEY", None)
    text = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json*"))
    assert "SECRET-VALUE" not in text and '"rapid_api_key": "present"' in text


def test_summary_and_report_read_the_recorded_run(
    offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--out", str(tmp_path), "trial", "--limit", "2", "--workers", "2"]) == 0
    capsys.readouterr()
    assert cli.main(["--out", str(tmp_path), "summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary[0]["categories"]["ALL_CATEGORIES"]["n"] == 2
    md, csv_path = tmp_path / "r.md", tmp_path / "r.csv"
    assert cli.main(["--out", str(tmp_path), "report", "--write", str(md), "--csv", str(csv_path)]) == 0
    assert md.read_text(encoding="utf-8").startswith("# ToolSandbox results")
    assert "[r.csv](r.csv)" in md.read_text(encoding="utf-8")
    assert len(list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))) == 2
    assert cli.main(["--out", str(tmp_path), "report"]) == 0
    assert capsys.readouterr().out.startswith("# ToolSandbox results")


def test_a_trial_prints_an_error_line_and_raises_for_a_run_that_crashes(
    offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Down(Steady):
        def complete(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError("endpoint exploded")

    monkeypatch.setattr(cli, "_client", lambda *_a, **_k: Down())
    assert cli.main(["--out", str(tmp_path), "trial", "--limit", "1", "--workers", "1"]) == 0
    shown = capsys.readouterr().out
    assert "ERROR RuntimeError: endpoint exploded" in shown
    assert "run errors 1" in shown


@pytest.mark.parametrize("text", ["x", "1", "1/0", "2/2", "-1/3", "a/b"])
def test_a_bad_shard_is_a_clear_exit(text: str) -> None:
    with pytest.raises(SystemExit, match="--shard"):
        cli.parse_shard(text)


def test_shards_are_disjoint_and_together_every_case(all_cases: list[ToolSandboxCase]) -> None:
    assert cli.parse_shard("1/3") == (1, 3)
    parts = [cli.shard_of(all_cases, i, 3) for i in range(3)]
    ids = [c.id for part in parts for c in part]
    assert sorted(ids) == sorted(c.id for c in all_cases) and len(set(ids)) == len(all_cases)
    assert cli.shard_of(all_cases, 0, 1) == all_cases


def test_two_shards_of_a_trial_are_one_run_in_the_report(
    offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--out", str(tmp_path), "trial", "--limit", "4", "--workers", "2"]
    assert cli.main([*base, "--shard", "0/2"]) == 0
    assert cli.main([*base, "--shard", "1/2"]) == 0
    capsys.readouterr()
    assert sorted(p.name for p in (tmp_path / "runs").glob("*/*")) == [
        "all-trial-1-shard-0-of-2",
        "all-trial-1-shard-1-of-2",
    ]
    assert cli.main(["--out", str(tmp_path), "summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert len(summary) == 1
    assert summary[0]["categories"]["ALL_CATEGORIES"]["n"] == 4
    assert summary[0]["categories"]["ALL_CATEGORIES"]["lost"] == 0
    assert "(shards 0,1 of 2)" in summary[0]["run"]


def test_a_shard_that_was_not_recorded_stays_in_the_count_as_lost(
    offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--out", str(tmp_path), "trial", "--limit", "4", "--workers", "2"]
    assert cli.main([*base, "--shard", "1/2"]) == 0
    capsys.readouterr()
    assert cli.main(["--out", str(tmp_path), "summary"]) == 0
    all_categories = json.loads(capsys.readouterr().out)[0]["categories"]["ALL_CATEGORIES"]
    assert (all_categories["n"], all_categories["ran"], all_categories["lost"]) == (4, 2, 2)
    csv_path = tmp_path / "r.csv"
    report = ["--out", str(tmp_path), "report", "--write", str(tmp_path / "r.md"), "--csv", str(csv_path)]
    assert cli.main(report) == 0
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    lost = [r for r in rows if r["status"] == "lost"]
    assert len(lost) == 2
    assert all("its shard was not recorded (shards [1] of 2 are)" in r["detail"] for r in lost)


def test_evolve_cases_are_whole_bases_from_both_sides_and_never_a_blocked_one(
    all_cases: list[ToolSandboxCase],
) -> None:
    cases = cli.evolve_cases(all_cases, 3, 5, 1)
    assert {c.base for c in cases if c.split == "train"} and len(
        {c.base for c in cases if c.split == "train"}
    ) == 3
    assert len({c.base for c in cases if c.split == "heldout"}) == 5
    assert len(cases) == 8 * 8
    assert all(c.not_run is None for c in cases)
    assert not {c.base for c in cases if c.split == "train"} & {c.base for c in cases if c.split == "heldout"}
    assert cases == cli.evolve_cases(all_cases, 3, 5, 1)
    assert len(cli.evolve_cases(all_cases, 10_000, 0, 1)) == sum(
        1 for c in all_cases if c.split == "train" and not c.not_run
    )


def test_evolve_wires_the_lab_and_writes_the_result(
    offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    class _Result:
        harness = seed_harness()

        class instructions:
            version = "ins1"

        generations: tuple[Any, ...] = ()

    def fake_evolve(store: Any, model: Any, generations: int, *, lab: Any, seed: Any) -> Any:
        seen.update(lab=lab, generations=generations, seed=seed)
        lab.log("hello")
        case = next(c for c in lab.cases if c.split == "train")
        store.put_trace(lab.run_case(seed, case, model, None, run_id="x"))
        return _Result()

    monkeypatch.setattr(cli, "evolve", fake_evolve)
    args = [
        "--out", str(tmp_path), "evolve", "--name", "t", "--generations", "2", "--max-evals", "2",
        "--train-bases", "1", "--heldout-bases", "1",
    ]  # fmt: skip
    assert cli.main(args) == 0
    root = tmp_path / "evolve" / "t"
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    assert result["final_harness"] == seed_harness().version and result["config"]["max_evals"] == 2
    assert result["config"]["cases"] == 16 and len(seen["lab"].cases) == 16
    assert seen["generations"] == 2 and seen["lab"].max_evals == 2
    assert seen["lab"].suite_digest == result["config"]["split_record"]["digest"]
    assert "hello" in (root / "log.txt").read_text(encoding="utf-8")
    assert result["active"] == [None, 0]
    assert (root / "scenarios.jsonl").exists() and (root / "cases.json").exists()
    with pytest.raises(SystemExit, match="already has a run"):
        cli.main(args)


def test_the_user_simulator_can_use_its_own_endpoint(
    offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def user(url: str, *_a: Any, **_k: Any) -> Any:
        asked.append(url)
        return user_factory([END])

    monkeypatch.setattr(cli, "endpoint_user", user)
    base = ["--out", str(tmp_path), "trial", "--limit", "1", "--workers", "1"]
    assert cli.main([*base, "--user-base-url", "http://users.test/v1"]) == 0
    assert cli.main([*base, "--trial", "2"]) == 0
    assert asked == ["http://users.test/v1", "http://endpoint.test/v1"]
    config = json.loads(next(tmp_path.rglob("all-trial-1/config.json")).read_text(encoding="utf-8"))
    assert config["agent_endpoint"] == "http://endpoint.test/v1"
    assert config["user_endpoint"] == "http://users.test/v1"


def test_the_run_records_the_commit_it_came_from(offline: None, tmp_path: Path) -> None:
    assert cli.main(["--out", str(tmp_path), "trial", "--limit", "1", "--workers", "1"]) == 0
    config = json.loads(next(tmp_path.rglob("all-trial-1/config.json")).read_text(encoding="utf-8"))
    assert config["ratchet_commit"] == cli.ratchet_commit()
    assert config["ratchet_commit"] and len(config["ratchet_commit"]) == 40


def test_the_commit_is_none_when_git_does_not_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def fails(*_a: Any, **_k: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", fails)
    assert cli.ratchet_commit() is None
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], 128, stdout="", stderr="no repo")
    )
    assert cli.ratchet_commit() is None


def test_an_admitted_harness_can_be_carried_onto_the_benchmarks_seed(tmp_path: Path) -> None:
    from ratchet.types import Guard, Harness

    guard = Guard("g1", "Block x.", "def check(call, view, history):\n    return None\n", ["f1"])
    other = Harness.build(
        parent="p",
        system_prompt="another world's prompt",
        procedures=[],
        guards=[guard],
        tool_allowlist=["payments_refund"],
        created_by="improver",
    )
    path = tmp_path / "admitted.json"
    path.write_text(json.dumps({"harness": to_jsonable(other)}), encoding="utf-8")
    seed = seed_harness()
    as_is = cli.load_harness(f"@{path}", tmp_path)
    assert as_is.digest == other.digest and as_is.tool_allowlist == ["payments_refund"]
    carried = cli.load_harness(f"@{path}", tmp_path, onto_seed=True)
    assert carried.parent == seed.version and carried.system_prompt == seed.system_prompt
    assert carried.tool_allowlist == seed.tool_allowlist
    assert [g.id for g in carried.guards] == ["g1"] and carried.created_by == "improver"
    again = cli.load_harness(f"@{path}", tmp_path, onto_seed=True)
    assert again.digest == carried.digest
    assert cli.load_harness("seed", tmp_path, onto_seed=True).digest == seed.digest


def test_a_trial_records_where_its_harness_came_from(offline: None, tmp_path: Path) -> None:
    from ratchet.types import Guard, Harness

    guard = Guard("g1", "Block x.", "def check(call, view, history):\n    return None\n", ["f1"])
    other = Harness.build(
        parent="p",
        system_prompt="s",
        procedures=[],
        guards=[guard],
        tool_allowlist=["t"],
        created_by="improver",
    )
    path = tmp_path / "admitted.json"
    path.write_text(json.dumps({"harness": to_jsonable(other)}), encoding="utf-8")
    out = tmp_path / "out"
    args = [
        "--out",
        str(out),
        "trial",
        "--limit",
        "1",
        "--workers",
        "1",
        "--harness",
        f"@{path}",
        "--onto-seed",
    ]
    assert cli.main(args) == 0
    config = json.loads(next(out.rglob("config.json")).read_text(encoding="utf-8"))
    assert config["harness_source"] == f"@{path}" and config["onto_seed"] is True
