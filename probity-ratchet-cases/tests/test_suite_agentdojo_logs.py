"""`agentdojo-report`: the false intervention rate from AgentDojo episode logs, with no AgentDojo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ratchet_cases.agentdojo_logs import LogError, report
from ratchet_cases.cli import main


def _episode(root: Path, pipeline: str, suite: str, task: str, utility: object, attack: str = "none") -> None:
    path = root / pipeline / suite / task / attack / "none.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"utility": utility, "security": True}), encoding="utf-8")


def test_the_rate_is_twins_failed_over_twins_run_per_pipeline_and_suite(tmp_path: Path) -> None:
    _episode(tmp_path, "gpt", "banking", "user_task_0", True)
    _episode(tmp_path, "gpt", "banking", "user_task_1", False)
    _episode(tmp_path, "gpt", "slack", "user_task_1", True)
    _episode(tmp_path, "gpt-tool_filter", "slack", "user_task_1", False)
    table = report(tmp_path)
    assert table["gpt"]["twins"] == 3 and table["gpt"]["carried_out"] == 2
    assert table["gpt"]["false_intervention_rate"] == pytest.approx(1 / 3)
    assert table["gpt"]["suites"]["banking"]["false_intervention_rate"] == pytest.approx(0.5)
    assert table["gpt-tool_filter"]["false_intervention_rate"] == 1.0


def test_attacked_episodes_and_other_suites_are_not_twins(tmp_path: Path) -> None:
    _episode(tmp_path, "gpt", "banking", "user_task_0", True)
    _episode(tmp_path, "gpt", "banking", "user_task_0", False, attack="important_instructions")
    _episode(tmp_path, "gpt", "other", "user_task_0", False)
    assert report(tmp_path)["gpt"]["twins"] == 1


def test_no_episodes_and_unreadable_episodes_are_errors_not_zero(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        report(tmp_path)
    _episode(tmp_path, "gpt", "travel", "user_task_0", "yes")
    with pytest.raises(LogError):
        report(tmp_path)


def test_the_cli_prints_the_metric_and_exits_2_on_a_bad_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _episode(tmp_path, "gpt", "travel", "user_task_0", False)
    assert main(["agentdojo-report", str(tmp_path), "--metric", "false_intervention_rate"]) == 0
    assert json.loads(capsys.readouterr().out) == {"gpt": {"false_intervention_rate": 1.0, "twins": 1}}
    _episode(tmp_path, "gpt", "travel", "user_task_1", None)
    assert main(["agentdojo-report", str(tmp_path)]) == 2
