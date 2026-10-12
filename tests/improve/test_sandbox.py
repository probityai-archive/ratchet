from __future__ import annotations

import time

import pytest

from ratchet.agent import guards
from ratchet.agent.guards import GuardRuntime
from ratchet.improve.diagnose import diagnose
from ratchet.improve.replay import WalkCache, replay
from ratchet.improve.sandbox import SandboxRunner
from ratchet.types import Guard

from .fakes import (
    CRASHING_GUARD,
    NO_DUP_GUARD,
    FakeSimulator,
    LocalRunner,
    bad_timeout,
    good_plain,
    good_timeout,
)


def test_replay_through_the_sandbox_matches_the_local_runner() -> None:
    failing = [bad_timeout(i) for i in range(4)]
    passing = [good_timeout(10 + i) for i in range(20)] + [good_plain(40 + i) for i in range(20)]
    cluster = diagnose(failing, FakeSimulator())[0]
    walks = WalkCache(FakeSimulator())
    guard = Guard("g", "no dup", NO_DUP_GUARD)
    local = replay(guard, cluster.members, cluster.traces, passing, walks, LocalRunner())
    with GuardRuntime(workers=2) as rt:
        started = time.perf_counter()
        boxed = replay(guard, cluster.members, cluster.traces, passing, walks, SandboxRunner(rt))
        assert time.perf_counter() - started < 30
        crash = replay(Guard("c", "", CRASHING_GUARD), cluster.members, cluster.traces, passing, walks,
                       SandboxRunner(rt))  # fmt: skip
    assert boxed.passed and local.passed and boxed.targets == local.targets and boxed.checks == local.checks
    assert not crash.passed and crash.errors


def test_a_sandbox_failure_makes_replay_incomplete_not_a_fire() -> None:
    from collections.abc import Sequence

    from ratchet.agent.guards import GuardInfrastructureError, GuardItem

    class Broken(GuardRuntime):
        def batch(self, items: Sequence[GuardItem]) -> list:  # type: ignore[type-arg]
            raise GuardInfrastructureError("child died before the guard started")

    failing = [bad_timeout(i) for i in range(2)]
    cluster = diagnose(failing, FakeSimulator())[0]
    with Broken(workers=1) as rt:
        rep = replay(Guard("g", "", NO_DUP_GUARD), cluster.members, cluster.traces, [good_plain(5)],
                     WalkCache(FakeSimulator()), SandboxRunner(rt))  # fmt: skip
    assert rep.verdict == "INCOMPLETE" and not rep.false_positives and not any(t["hit"] for t in rep.targets)
    assert rep.errors[0]["error"].startswith("guard infrastructure error")


def test_replay_reports_the_sandbox_reason_for_every_errored_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """A child outside the protocol makes replay INCOMPLETE, and every error names what the sandbox sent."""
    child = (
        "import sys\nsys.stdin.buffer.readline()\n"
        "print('{\"protocol\": 99}', flush=True)\nsys.stdin.buffer.read()\n"
    )
    monkeypatch.setattr(guards, "CHILD_SOURCE", child)
    failing = [bad_timeout(i) for i in range(2)]
    cluster = diagnose(failing, FakeSimulator())[0]
    with GuardRuntime(workers=1) as rt:
        rep = replay(Guard("g", "", NO_DUP_GUARD), cluster.members, cluster.traces, [good_plain(5)],
                     WalkCache(FakeSimulator()), SandboxRunner(rt))  # fmt: skip
    assert rep.verdict == "INCOMPLETE" and rep.errors and len(rep.errors) == rep.checks
    for e in rep.errors:
        assert e["error"].startswith("guard infrastructure error: ")
        assert 'out-of-protocol startup reply: {"protocol": 99}' in e["error"]
