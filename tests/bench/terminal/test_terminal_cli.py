from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
from ratchet_cases.types import Cost, Outcome, Trace
from terminal_testkit import FakeContainer, make_task

from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.bench.terminal import __main__ as M
from ratchet.bench.terminal import report
from ratchet.bench.terminal import tasks as T
from ratchet.bench.terminal.container import Container
from ratchet.bench.terminal.runner import seed_harness
from ratchet.types import Harness


def suite(root: Path) -> Path:
    for name in T.PHASE_1:
        make_task(root, name, timeout=900.0)
    for name in T.PHASE_2:
        make_task(root, name, timeout=1800.0)
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("abc123\n", encoding="utf-8")
    return root


class StubRunner:
    """A `TerminalRunner` stand-in: passes every case whose name is in `passing`."""

    passing: ClassVar[set[str]] = set()

    def __init__(self, **kw: Any) -> None:
        self.kw = kw

    def __call__(self, harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
        if case.task.name == "boom":
            raise RuntimeError("container died")
        ok = case.task.name in self.passing
        reasons = [] if ok else ["test test_out failed"]
        return Trace(run_id, case.id, harness.version, "m", [], Outcome(ok, reasons, "d", Cost()))


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, inprocess_referee: Callable[..., Any]) -> None:
    monkeypatch.setattr(M, "ProcessReferee", inprocess_referee)
    monkeypatch.setenv("RATCHET_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("RATCHET_API_KEY", "k")
    monkeypatch.setattr(M, "TerminalRunner", StubRunner)
    monkeypatch.setattr(M, "_client", lambda *a, **k: ScriptedModel([scripted_turn("x")] * 50))


def test_split_prints_and_writes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "split.json"
    assert M.main(["split", "--suite", str(suite(tmp_path / "s")), "--write", str(out)]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["suite_check"] == {"1": 12, "2": 10}
    assert record["digest"] == T.committed_record()["digest"]
    assert '"seed": 20261011' in capsys.readouterr().out


def test_split_with_a_wider_held_out_set(tmp_path: Path) -> None:
    out = tmp_path / "wide.json"
    assert M.main(["split", "--heldout", "6,5", "--write", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == T.committed_record(T.WIDE_SPLIT_FILE)


def test_trial_and_evolve_take_the_wider_split(env: None, tmp_path: Path) -> None:
    root = suite(tmp_path / "s")
    out = tmp_path / "out"
    args = ["--out", str(out), "trial", "--suite", str(root), "--set", "phase1-heldout", "--heldout", "6,5"]
    assert M.main(args) == 0
    run_dir = next((out / "runs" / seed_harness().version).glob("phase1-heldout-trial-1"))
    assert len((run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()) == 6
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["split_record"]["heldout_per_phase"] == {"1": 6, "2": 5}


def test_endpoint_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RATCHET_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="RATCHET_BASE_URL"):
        M._endpoint()


def test_client_reads_the_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATCHET_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("RATCHET_API_KEY", "k")
    assert M._client("gpt-x").name == "gpt-x"


def test_trial_records_every_trace_and_the_report(env: None, tmp_path: Path) -> None:
    root = suite(tmp_path / "s")
    StubRunner.passing = {"cancel-async-tasks"}
    out = tmp_path / "out"
    args = [
        "--out",
        str(out),
        "trial",
        "--suite",
        str(root),
        "--tasks",
        "cancel-async-tasks,sparql-university",
    ]
    assert M.main(args) == 0
    version = seed_harness().version
    run_dir = out / "runs" / version / "tasks-trial-1"
    assert len((run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["suite_commit"] == "abc123" and config["package"] == "terminal-bench==0.2.18"
    with pytest.raises(SystemExit, match="already has traces"):
        M.main(args)
    rows = report.collect(out)
    assert [r["passed"] for r in rows[0]["results"]] == [True, False]


def test_trial_sets(env: None, tmp_path: Path) -> None:
    root = suite(tmp_path / "s")
    out = tmp_path / "out"
    for i, (s, n) in enumerate([("phase1", 12), ("phase2", 10), ("phase1-heldout", 4), ("phase2-audit", 4)]):
        assert M.main(["--out", str(out), "trial", "--suite", str(root), "--set", s, "--trial", str(i)]) == 0
        run_dir = next((out / "runs" / seed_harness().version).glob(f"{s}-trial-{i}"))
        assert len((run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()) == n


def test_union_trial_and_rates(env: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = suite(tmp_path / "s")
    out = tmp_path / "out"
    StubRunner.passing = set(T.PHASE_1[:6])
    assert M.main(["--out", str(out), "trial", "--suite", str(root)]) == 0
    capsys.readouterr()
    version = seed_harness().version
    assert M.main(["--out", str(out), "rates", "--harness", version]) == 0
    rates = json.loads(capsys.readouterr().out)[version]
    assert rates["phase 1 (12 tasks)"]["text"] == "6/12 runs = 50.0% over 12 tasks"
    assert rates["phase 2 (10 tasks)"]["text"] == "0/10 runs = 0.0% over 10 tasks"
    assert rates["union (22 tasks)"]["text"] == "6/22 runs = 27.3% over 22 tasks"
    assert M.main(["--out", str(out), "rates", "--harness", version, "--markdown"]) == 0
    text = capsys.readouterr().out
    assert f"| `{version}` phase 1 (12 tasks) | 6/12 | 50.0% | 12 | 0 |" in text
    assert "| term-cancel-async-tasks | 1/1 |" in text


def test_load_harness(tmp_path: Path) -> None:
    assert M.load_harness("seed", tmp_path).version == seed_harness().version
    child = seed_harness().child(system_prompt="more", created_by="improver")
    store = M.TerminalStore(tmp_path / "x")
    store.put_harness(child)
    assert M.load_harness(child.version, tmp_path) == child
    assert store.get_harness(child.version) == child and store.get_harness("nope") is None
    with pytest.raises(SystemExit, match="no harness"):
        M.load_harness("missing", tmp_path)
    raw = json.loads((tmp_path / "x" / "harnesses.jsonl").read_text(encoding="utf-8"))
    raw["system_prompt"] = "tampered"
    with pytest.raises(ValueError, match="does not match its digest"):
        M.harness_from(raw)


def test_evolve_phase_one_stops_when_nothing_fails(env: None, tmp_path: Path) -> None:
    root = suite(tmp_path / "s")
    StubRunner.passing = set(T.PHASE_1)
    out = tmp_path / "out"
    assert M.main(["--out", str(out), "evolve", "--suite", str(root), "--phase", "1", "--name", "r"]) == 0
    run = out / "evolve" / "phase-1" / "r"
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert result["final_harness"] == seed_harness().version
    assert result["generations"][0]["note"] == "no training failures left to diagnose"
    assert result["config"]["sides"]["term-cancel-async-tasks"] == "train"
    assert "gen 1" in (run / "log.txt").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="already has a run"):
        M.main(["--out", str(out), "evolve", "--suite", str(root), "--phase", "1", "--name", "r"])


def test_evolve_phase_two_needs_an_evolved_parent(env: None, tmp_path: Path) -> None:
    root = suite(tmp_path / "s")
    with pytest.raises(SystemExit, match="phase 2 starts from"):
        M.main(["--out", str(tmp_path / "o"), "evolve", "--suite", str(root), "--phase", "2"])


def test_recording_runner_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case = T.task_cases(T.load_suite(suite(tmp_path / "s")), ["cancel-async-tasks"])[0]
    boom = T.TerminalCase(
        "term-boom", "train", T.TaskSpec("boom", str(tmp_path), "", "i", "hard", "", 1.0, 1.0), 1
    )
    rec = M.RecordingRunner(StubRunner(), M.TerminalStore(tmp_path / "st"))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="container died"):
        rec(seed_harness(), boom, None, None, run_id="r1")
    rec(seed_harness(), case, None, None, run_id="r2")
    lines = capsys.readouterr().out.splitlines()
    assert "ERROR RuntimeError: container died" in lines[0] and "PASS" not in lines[0]


def oracle_factory(fc: FakeContainer) -> Callable[[T.TaskSpec], Container]:
    return lambda _task: fc


def test_oracle_runs_the_reference_solution(tmp_path: Path) -> None:
    task = T.load_task(make_task(tmp_path))
    fc = FakeContainer(reward_txt="1")
    row = M.oracle_one(task, oracle_factory(fc))
    assert row["passed"] and row["reward"] == 1.0 and row["solution_exit_code"] == 0
    assert (task.path / "solution", "/solution") in fc.dirs and fc.closed
    assert fc.files["/tmp/.ratchet/cmd"] == b"bash /solution/solve.sh"


def test_oracle_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = suite(tmp_path / "s")
    calls: list[str] = []

    def fake(task: T.TaskSpec, factory: object = None) -> dict[str, Any]:
        calls.append(task.name)
        if task.name == "sparql-university":
            raise RuntimeError("pull failed")
        return {"task": task.name, "passed": True, "reward": 1.0}

    monkeypatch.setattr(M, "oracle_one", fake)
    out = tmp_path / "out"
    args = [
        "--out",
        str(out),
        "oracle",
        "--suite",
        str(root),
        "--tasks",
        "cancel-async-tasks,sparql-university",
    ]
    assert M.main(args) == 0
    assert sorted(calls) == ["cancel-async-tasks", "sparql-university"]
    assert "oracle: 1/2 reference solutions pass" in capsys.readouterr().out
    rows = [json.loads(x) for x in next(out.glob("oracle/*/oracle.jsonl")).read_text().splitlines()]
    assert {r["task"]: r["passed"] for r in rows} == {"cancel-async-tasks": True, "sparql-university": False}
