from __future__ import annotations

import time

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
