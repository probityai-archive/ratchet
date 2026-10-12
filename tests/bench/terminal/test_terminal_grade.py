from __future__ import annotations

import json
from pathlib import Path

from ratchet_cases.types import Cost
from terminal_testkit import Clock, FakeContainer, make_task

from ratchet.bench.terminal import grade as G
from ratchet.bench.terminal.tasks import load_task
from ratchet.bench.terminal.world import TerminalWorld

PYTEST_OUT = """============================= test session starts ==============================
collected 3 items
=========================== short test summary info ============================
PASSED ../tests/test_outputs.py::test_exists
FAILED ../tests/test_outputs.py::test_content - AssertionError: wrong
ERROR ../tests/test_outputs.py::test_setup
"""


def ctrf(*tests: dict[str, str]) -> str:
    return json.dumps({"results": {"tests": list(tests)}})


def test_sections_split_the_collect_output() -> None:
    got = G.sections("noise\n@@reward_txt\n1\n@@ctrf\n{}\n@@stdout\nline one\nline two\n")
    assert got == {"reward_txt": "1", "ctrf": "{}", "stdout": "line one\nline two"}


def test_reward_of_every_form() -> None:
    assert G.reward_of("", "1\n") == (1.0, None)
    assert G.reward_of('{"reward": 0.5, "other": 1}', "") == (0.5, None)
    assert G.reward_of('{"a": 1, "b": 0}', "1") == (0.0, None)
    assert G.reward_of("0.25", "") == (0.25, None)
    assert G.reward_of("{oops", "") == (None, "reward.json is not JSON")
    assert G.reward_of('{"a": "x"}', "") == (None, "reward.json holds no finite number")
    assert G.reward_of('{"a": true}', "") == (None, "reward.json holds no finite number")
    assert G.reward_of("", "pass") == (None, "reward.txt is not a number: 'pass'")
    assert G.reward_of("", "nan") == (None, "reward.txt is not finite")
    assert G.reward_of("", "") == (None, "the tests wrote no reward file")


def test_ctrf_tests() -> None:
    raw = ctrf(
        {"name": "test_a", "status": "passed"}, {"name": "test_b", "status": "failed", "message": "x" * 500}
    )
    status, messages = G.ctrf_tests(raw)
    assert status == {"test_a": "passed", "test_b": "failed"}
    assert messages == {"test_b": "x" * G.MAX_MESSAGE}
    assert G.ctrf_tests("") == ({}, {})
    assert G.ctrf_tests("not json") == ({}, {})
    assert G.ctrf_tests("[1]") == ({}, {})
    assert G.ctrf_tests(json.dumps({"results": {"tests": ["bad", {"status": "passed"}]}})) == ({}, {})


def test_pytest_tests_uses_the_package_parser() -> None:
    assert G.pytest_tests(PYTEST_OUT) == {
        "test_exists": "passed",
        "test_content": "failed",
        "test_setup": "failed",
    }
    assert G.pytest_tests("no summary here") == {}


def task(tmp_path: Path) -> object:
    return load_task(make_task(tmp_path))


def test_verify_runs_the_tests_unchanged_and_reads_the_reward(tmp_path: Path) -> None:
    t = load_task(make_task(tmp_path))
    fc = FakeContainer(reward_txt="1", ctrf=ctrf({"name": "test_a", "status": "passed"}))
    v = G.verify(fc, t)
    assert v.passed and v.reward == 1.0 and v.tests == {"test_a": "passed"} and v.error is None
    assert fc.dirs == [(t.tests_dir, "/tests")]
    assert fc.scripts == [G.RUN_TESTS, G.COLLECT]
    assert fc.envs[0] == {"MODE": "strict"}
    assert G.summary(v) == {"reward": 1.0, "passed": True, "tests": {"test_a": "passed"}, "error": None,
                            "timed_out": False}  # fmt: skip


def test_verify_falls_back_to_pytest_output_and_reports_failures(tmp_path: Path) -> None:
    t = load_task(make_task(tmp_path))
    fc = FakeContainer(reward_txt="0", stdout=PYTEST_OUT)
    v = G.verify(fc, t)
    assert not v.passed and v.reward == 0.0
    assert v.tests["test_content"] == "failed"
    reasons = G.reasons_of(v, None)
    assert reasons == ["test test_content failed", "test test_setup failed"]


def test_verify_reports_missing_rewards_and_timeouts(tmp_path: Path) -> None:
    t = load_task(make_task(tmp_path))
    v = G.verify(FakeContainer(reward_txt=""), t)
    assert v.reward is None and v.error == "the tests wrote no reward file"
    assert G.reasons_of(v, None) == ["the tests wrote no reward file"]
    late = G.verify(FakeContainer(reward_txt="", tests_timeout=True), t)
    assert late.timed_out and late.error == "the tests did not finish within 60 s"


def test_reasons_cap_the_failing_tests_and_name_the_time_limit(tmp_path: Path) -> None:
    tests = {f"test_{i:02d}": "failed" for i in range(G.MAX_REASONS + 2)}
    v = G.Verification(0.0, tests, {"test_00": "boom"})
    clock = Clock()
    world = TerminalWorld(FakeContainer(), "t", 10.0, clock=clock)
    clock.now += 11
    reasons = G.reasons_of(v, world)
    assert reasons[0] == "test test_00 failed: boom"
    assert reasons[G.MAX_REASONS] == "... and 2 more failing tests"
    assert reasons[-1] == "the agent's time limit ran out before it finished"
    assert G.reasons_of(G.Verification(0.5), None) == ["the tests gave reward 0.5"]
    assert G.reasons_of(G.Verification(1.0), world) == []


def test_outcome_of(tmp_path: Path) -> None:
    world = TerminalWorld(FakeContainer(), "t", 10.0, clock=Clock())
    o = G.outcome_of(G.Verification(1.0), world, Cost(model_calls=2))
    assert o.passed and o.reasons == [] and o.final_state_digest == world.digest() and o.cost.model_calls == 2
