from __future__ import annotations

from collections.abc import Sequence

import pytest

from ratchet.improve.diagnose import diagnose
from ratchet.improve.replay import GuardInput, Verdict, WalkCache, replay
from ratchet.types import Guard

from .fakes import (
    BLOCK_ALL_REFUNDS,
    CRASHING_GUARD,
    NO_DUP_GUARD,
    FakeSimulator,
    LocalRunner,
    bad_timeout,
    good_plain,
    good_timeout,
    without_views,
)


def _setup() -> tuple[list[object], dict[str, object], list[object]]:
    failing = [bad_timeout(i) for i in range(3)]
    passing = [good_timeout(10), good_plain(11), good_plain(12)]
    cluster = diagnose(failing, FakeSimulator())[0]
    return list(cluster.members), cluster.traces, passing  # type: ignore[return-value]


def test_precise_guard_passes_replay() -> None:
    targets, failing, passing = _setup()
    sim, runner = FakeSimulator(), LocalRunner()
    rep = replay(Guard("g", "no dup", NO_DUP_GUARD), targets, failing, passing, WalkCache(sim), runner)  # type: ignore[arg-type]
    assert rep.passed and runner.batches == 2 and rep.coverage == "6/6 traces checked"
    assert all(t["hit"] and t["first_block_seq"] == 5 for t in rep.targets)
    assert rep.traces == 6 and rep.checks > 0 and rep.checks_per_s > 0
    assert rep.to_payload()["targets_hit"] == 3


def test_overbroad_guard_fails_on_passing_traces() -> None:
    targets, failing, passing = _setup()
    rep = replay(
        Guard("g", "", BLOCK_ALL_REFUNDS),
        targets,
        failing,
        passing,
        WalkCache(FakeSimulator()),
        LocalRunner(),
    )  # type: ignore[arg-type]
    assert not rep.passed
    assert {fp["run_id"] for fp in rep.false_positives} == {"good-10", "plain-11", "plain-12"}
    # It does block the targets, and early: at the first refund, before the failing step.
    assert all(t["hit"] and t["first_block_seq"] == 3 for t in rep.targets)


def test_guard_that_never_fires_misses_targets() -> None:
    targets, failing, passing = _setup()
    src = "def check(call, view, history):\n    return None\n"
    rep = replay(Guard("g", "", src), targets, failing, passing, WalkCache(FakeSimulator()), LocalRunner())  # type: ignore[arg-type]
    assert not rep.passed and not any(t["hit"] for t in rep.targets)


def test_guard_that_fires_after_the_failing_step_is_a_miss() -> None:
    targets, failing, passing = _setup()
    src = "def check(call, view, history):\n    return 'late' if len(history) > 6 else None\n"
    rep = replay(Guard("g", "", src), targets, failing, passing, WalkCache(FakeSimulator()), LocalRunner())  # type: ignore[arg-type]
    assert not rep.passed and rep.targets[0]["first_block_seq"] is None


def test_crashing_guard_fails_with_errors() -> None:
    targets, failing, passing = _setup()
    rep = replay(
        Guard("g", "", CRASHING_GUARD), targets, failing, passing, WalkCache(FakeSimulator()), LocalRunner()
    )  # type: ignore[arg-type]
    assert not rep.passed and rep.errors and "KeyError" in rep.errors[0]["error"]


def test_walks_are_cached_across_guards_and_missing_targets_noted() -> None:
    targets, failing, passing = _setup()
    sim = FakeSimulator()
    cache = WalkCache(sim)
    for src in (NO_DUP_GUARD, BLOCK_ALL_REFUNDS):
        replay(Guard("g", "", src), targets, failing, passing, cache, LocalRunner())  # type: ignore[arg-type]
    assert len(sim.walked) == 6
    rep = replay(Guard("g", "", NO_DUP_GUARD), targets, {}, [], cache, LocalRunner())  # type: ignore[arg-type]
    assert not rep.passed and len(rep.notes) == 3 and rep.checks == 0


def test_unfaithful_walks_are_reported() -> None:
    from ratchet.improve.sim import RecordedViews

    targets, failing, passing = _setup()
    rep = replay(
        Guard("g", "", NO_DUP_GUARD),
        targets,
        failing,
        [without_views(passing[0])],
        WalkCache(RecordedViews()),
        LocalRunner(),
    )  # type: ignore[arg-type]
    assert "good-10" in rep.unfaithful


def test_runtime_returning_wrong_count_is_an_error() -> None:
    class Short:
        def check_batch(self, source: str, inputs: Sequence[GuardInput]) -> list[Verdict]:
            return []

    targets, failing, passing = _setup()
    with pytest.raises(RuntimeError, match="verdicts"):
        replay(Guard("g", "", NO_DUP_GUARD), targets, failing, passing, WalkCache(FakeSimulator()), Short())  # type: ignore[arg-type]


def test_replay_always_completes_and_records_every_false_fire() -> None:
    targets, failing, passing = _setup()
    g = Guard("g", "", BLOCK_ALL_REFUNDS)
    rep = replay(g, targets, failing, passing, WalkCache(FakeSimulator()), LocalRunner(), chunk=1)  # type: ignore[arg-type]
    assert (
        rep.verdict == "REJECTED" and rep.coverage == "6/6 traces checked" and len(rep.false_positives) == 3
    )
    assert rep.to_payload()["exhaustive"]


def test_runtime_errors_make_replay_incomplete_and_summaries_carry_no_ids() -> None:
    targets, failing, passing = _setup()
    rep = replay(
        Guard("c", "", CRASHING_GUARD), targets, failing, passing, WalkCache(FakeSimulator()), LocalRunner()
    )  # type: ignore[arg-type]
    assert rep.verdict == "INCOMPLETE" and not rep.passed
    summary = rep.sanitized()
    assert summary["verdict"] == "INCOMPLETE" and summary["errors"] > 0
    assert "bad-" not in str(summary) and "KeyError" not in str(summary)


def test_composed_guards_fire_when_any_member_fires() -> None:
    targets, failing, passing = _setup()
    quiet = Guard("q", "", "def check(call, view, history):\n    return None\n")
    walks = WalkCache(FakeSimulator())
    together = replay([quiet, Guard("n", "", NO_DUP_GUARD)], targets, failing, passing, walks, LocalRunner())  # type: ignore[arg-type]
    assert together.passed and together.guard_id == "composed:q+n"
    broad = replay(
        [quiet, Guard("b", "", BLOCK_ALL_REFUNDS)], targets, failing, passing, walks, LocalRunner()
    )  # type: ignore[arg-type]
    assert broad.verdict == "REJECTED" and len(broad.false_positives) == 3


def test_similarity_order_puts_target_like_traces_first() -> None:
    from ratchet.improve.replay import similarity_order

    order = similarity_order([good_plain(1), good_timeout(2)], [bad_timeout(3)], "payments_refund")
    assert [t.run_id for t in order] == ["plain-1", "good-2"]
