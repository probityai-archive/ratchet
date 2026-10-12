"""Grading: the task's own tests, run unchanged in the agent's container after the agent finishes.

The tests directory is copied to `/tests`, `/tests/test.sh` runs from the image's working directory with the
task's verifier environment and time limit, its output goes to `/logs/verifier/test-stdout.txt`, and the
reward is what the script writes to `/logs/verifier/reward.json` or `/logs/verifier/reward.txt`. A run passes
when the reward is 1.

The reward alone decides the outcome. The reasons only locate a failure for the improver: the failing tests
from the script's CTRF report (`/logs/verifier/ctrf.json`) when it writes one, otherwise from pytest's short
test summary in the script's output, parsed by the benchmark package's own pytest parser.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ratchet_cases.types import Cost, Outcome

from ratchet.bench.terminal.container import Container
from ratchet.bench.terminal.tasks import TaskSpec
from ratchet.bench.terminal.world import TerminalWorld

TESTS_DIR = "/tests"
VERIFIER_DIR = "/logs/verifier"
STDOUT_TAIL = 20000
MAX_REASONS = 12
MAX_MESSAGE = 200

# The script is started from bash when the image has it, as the suite's own runner does: a test.sh whose
# `#!` line is not its first line then runs under bash, not under sh.
RUN_TESTS = f"""mkdir -p {VERIFIER_DIR} /logs/agent
chmod +x {TESTS_DIR}/test.sh
if command -v bash >/dev/null 2>&1; then
  bash -c '{TESTS_DIR}/test.sh > {VERIFIER_DIR}/test-stdout.txt 2>&1'
else
  {TESTS_DIR}/test.sh > {VERIFIER_DIR}/test-stdout.txt 2>&1
fi
echo "test.sh exited $?"
"""

COLLECT = f"""d={VERIFIER_DIR}
echo "@@reward_json"; cat "$d/reward.json" 2>/dev/null; echo
echo "@@reward_txt"; cat "$d/reward.txt" 2>/dev/null; echo
echo "@@ctrf"; cat "$d/ctrf.json" 2>/dev/null; echo
echo "@@stdout"; tail -c {STDOUT_TAIL} "$d/test-stdout.txt" 2>/dev/null
"""


@dataclass(frozen=True)
class Verification:
    reward: float | None
    tests: dict[str, str] = field(default_factory=dict)
    messages: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    timed_out: bool = False
    stdout_tail: str = ""

    @property
    def passed(self) -> bool:
        return self.reward is not None and self.reward >= 1.0


def sections(text: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("@@") and line[2:] in ("reward_json", "reward_txt", "ctrf", "stdout"):
            current = out.setdefault(line[2:], [])
        elif current is not None:
            current.append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def reward_of(raw_json: str, raw_txt: str) -> tuple[float | None, str | None]:
    """The reward from reward.json (its `reward` value, else the smallest value) or reward.txt."""
    if raw_json:
        try:
            value = json.loads(raw_json)
        except json.JSONDecodeError:
            return None, "reward.json is not JSON"
        values = list(value.values()) if isinstance(value, dict) else [value]
        if isinstance(value, dict) and "reward" in value:
            values = [value["reward"]]
        numbers = [float(v) for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
        if not numbers or not all(math.isfinite(n) for n in numbers):
            return None, "reward.json holds no finite number"
        return min(numbers), None
    if raw_txt:
        try:
            number = float(raw_txt.split()[0])
        except (ValueError, IndexError):
            return None, f"reward.txt is not a number: {raw_txt[:40]!r}"
        return (number, None) if math.isfinite(number) else (None, "reward.txt is not finite")
    return None, "the tests wrote no reward file"


def ctrf_tests(raw: str) -> tuple[dict[str, str], dict[str, str]]:
    """Per-test status and failure message from a CTRF report; empty when there is none."""
    if not raw:
        return {}, {}
    try:
        report = json.loads(raw)
    except json.JSONDecodeError:
        return {}, {}
    tests = report.get("results", {}).get("tests", []) if isinstance(report, dict) else []
    status: dict[str, str] = {}
    messages: dict[str, str] = {}
    for t in tests:
        if not isinstance(t, dict) or "name" not in t:
            continue
        name = str(t["name"])
        status[name] = str(t.get("status", "unknown"))
        if t.get("message"):
            messages[name] = str(t["message"])[:MAX_MESSAGE]
    return status, messages


def pytest_tests(stdout: str) -> dict[str, str]:
    """Per-test status from pytest's short test summary, through the benchmark package's parser."""
    with warnings.catch_warnings():
        # Importing the package runs its `db` module, which calls SQLAlchemy's `declarative_base()` and warns
        # that the function moved. The parser does not use the database; the warning is the package's.
        warnings.filterwarnings("ignore", message=r".*declarative_base", module=r"terminal_bench\.db")
        from terminal_bench.parsers.pytest_parser import (  # type: ignore[import-untyped]  # no type info
            PytestParser,
        )

    try:
        parsed = PytestParser().parse(stdout)
    except ValueError:
        return {}
    return {name: status.value for name, status in parsed.items()}


def verify(container: Container, task: TaskSpec) -> Verification:
    container.put_dir(task.tests_dir, TESTS_DIR)
    run = container.exec(RUN_TESTS, task.verifier_timeout_s + 60.0, env=dict(task.verifier_env))
    got = sections(container.exec(COLLECT, 300.0).output)
    reward, problem = reward_of(got.get("reward_json", ""), got.get("reward_txt", ""))
    stdout = got.get("stdout", "")
    tests, messages = ctrf_tests(got.get("ctrf", ""))
    if not tests:
        tests = pytest_tests(stdout)
    error = problem if reward is None else None
    if run.timed_out:
        error = f"the tests did not finish within {task.verifier_timeout_s:g} s"
    return Verification(reward, tests, messages, error, run.timed_out, stdout[-4000:])


def reasons_of(v: Verification, world: TerminalWorld | None) -> list[str]:
    if v.passed:
        return []
    out: list[str] = []
    if v.error:
        out.append(v.error)
    failed = [n for n, s in sorted(v.tests.items()) if s not in ("passed", "skipped", "xfail")]
    for name in failed[:MAX_REASONS]:
        note = v.messages.get(name)
        out.append(f"test {name} failed" + (f": {note}" if note else ""))
    if len(failed) > MAX_REASONS:
        out.append(f"... and {len(failed) - MAX_REASONS} more failing tests")
    if world is not None and world.expired:
        out.append("the agent's time limit ran out before it finished")
    if not out:
        out.append(f"the tests gave reward {v.reward}")
    return out


def outcome_of(v: Verification, world: TerminalWorld, cost: Cost) -> Outcome:
    return Outcome(v.passed, reasons_of(v, world), world.digest(), cost)


def summary(v: Verification) -> Mapping[str, Any]:
    """The verification as JSON, for run records."""
    return {
        "reward": v.reward,
        "passed": v.passed,
        "tests": dict(v.tests),
        "error": v.error,
        "timed_out": v.timed_out,
    }
