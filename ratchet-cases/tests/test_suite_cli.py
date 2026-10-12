from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

import ratchet_cases.cli as cli
from ratchet_cases import SUITE_DIGEST, __version__
from ratchet_cases.cases.solvers import NAIVE
from ratchet_cases.run import read_traces

TRAP = "refund_lost_reply.v1"
TWIN = "refund_lost_reply.v2"
PLAIN = "ordinary.v1"
AGENT = """
def act(messages, tools):
    return {"content": "I cannot help with that."}
"""


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_list_filters_and_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "list")
    assert code == 0 and out.strip().endswith("52 case(s)")
    code, out, _ = run(capsys, "list", "--split", "train", "--kind", "benign_twin", "--json")
    rows = json.loads(out)
    assert code == 0 and len(rows) == 7 and {r["kind"] for r in rows} == {"benign_twin"}


def test_digest_matches_the_release_record(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out, err = run(capsys, "digest")
    assert code == 0 and out.strip() == SUITE_DIGEST and __version__ in err
    monkeypatch.setattr(cli, "released_digest", lambda: "0" * 64)
    code, _, err = run(capsys, "digest")
    assert code == 1 and "MISMATCH" in err

    def unknown() -> str:
        raise KeyError(__version__)

    monkeypatch.setattr(cli, "released_digest", unknown)
    code, _, err = run(capsys, "digest")
    assert code == 1 and "no entry" in err


def test_selftest_passes_when_each_solver_scores_as_designed(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "selftest", "--case", TRAP, "--case", TWIN, "--case", PLAIN)
    assert code == 0
    assert "reference trap 1/1, benign_twin 1/1, must_succeed 1/1" in out
    assert "naive     trap 0/1, benign_twin 1/1, must_succeed 1/1" in out
    assert "refuser   trap 0/1, benign_twin 0/1, must_succeed 0/1" in out


def test_selftest_fails_when_a_solver_scores_otherwise(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = cli.solve
    monkeypatch.setattr(cli, "solve", lambda case, habits: real(case, NAIVE))
    code, _, err = run(capsys, "selftest", "--case", TRAP)
    assert code == 1 and f"reference failed {TRAP}" in err


def _agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    (tmp_path / "idle_agent_for_tests.py").write_text(AGENT, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return "python:idle_agent_for_tests:act"


def test_run_grade_and_report_over_a_python_agent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _agent(tmp_path, monkeypatch)
    out_dir = tmp_path / "runs" / "idle"
    code, out, _ = run(
        capsys, "run", "--adapter", adapter, "--case", TWIN, "--case", PLAIN, "--out", str(out_dir)
    )
    assert code == 0 and "0/2 passed" in out
    traces = out_dir / "traces.jsonl"
    assert [t.case_id for _, t in read_traces(traces)] == [TWIN, PLAIN]
    code, out, _ = run(capsys, "grade", str(traces))
    assert code == 0 and '"false_intervention_rate": 1.0' in out
    code, out, _ = run(capsys, "report", str(tmp_path / "runs"), "--metric", "false_intervention_rate")
    assert code == 0 and json.loads(out) == {"false_intervention_rate": 1.0, "runs": 2}
    code, out, _ = run(capsys, "report", str(tmp_path / "runs"))
    assert code == 0 and json.loads(out)["counts"] == {"trap": 0, "benign_twin": 1, "must_succeed": 1}
    code, _, err = run(capsys, "run", "--adapter", adapter, "--case", TWIN, "--out", str(out_dir))
    assert code == 2 and "exists" in err


def test_grade_flags_a_recorded_result_the_state_does_not_support(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _agent(tmp_path, monkeypatch)
    run(capsys, "run", "--adapter", adapter, "--case", TWIN, "--out", str(tmp_path / "r"))
    traces = tmp_path / "r" / "traces.jsonl"
    record = json.loads(traces.read_text(encoding="utf-8"))
    record["trace"]["outcome"]["passed"] = True
    record["trace"]["outcome"]["final_state_digest"] = "0" * 64
    record["suite_digest"] = "old"
    traces.write_text(json.dumps(record) + "\n", encoding="utf-8")
    code, _, err = run(capsys, "grade", str(traces))
    assert code == 1
    assert "not this one" in err and "does not reach" in err and "recorded True, regraded False" in err
    code, _, _ = run(capsys, "report", str(traces))
    assert code == 1


def test_usage_errors_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("RATCHET_BASE_URL", "RATCHET_MODEL", "RATCHET_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert run(capsys, "list", "--case", "nope.v1")[0] == 2
    assert run(capsys, "grade", str(tmp_path))[0] == 2
    assert run(capsys, "run", "--adapter", "smoke-signals", "--out", str(tmp_path / "x"))[0] == 2
    assert run(capsys, "run", "--adapter", "openai", "--out", str(tmp_path / "y"))[0] == 2


def test_the_module_runs_as_a_program(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["ratchet-cases", "--version"])
    with pytest.raises(SystemExit) as done:
        runpy.run_module("ratchet_cases", run_name="__main__")
    assert done.value.code == 0
    assert capsys.readouterr().out.strip() == f"ratchet-cases {__version__}"
