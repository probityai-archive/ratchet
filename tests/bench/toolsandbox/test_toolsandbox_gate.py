from __future__ import annotations

import threading

import pytest
from tool_sandbox.common.execution_context import (
    ExecutionContext,
    get_current_context,
)

from ratchet.bench.toolsandbox.gate import ContextGate


def test_held_points_the_global_context_at_the_given_one() -> None:
    gate = ContextGate()
    ctx = ExecutionContext()
    with gate.held(ctx):
        assert get_current_context() is ctx


def test_two_threads_never_hold_the_gate_together() -> None:
    gate = ContextGate()
    inside = 0
    peak = 0
    guard = threading.Lock()

    def work() -> None:
        nonlocal inside, peak
        with gate.held(ExecutionContext()):
            with guard:
                inside += 1
                peak = max(peak, inside)
            threading.Event().wait(0.01)
            with guard:
                inside -= 1

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1


def test_released_lets_another_thread_in_and_restores_the_context() -> None:
    gate = ContextGate()
    mine, theirs = ExecutionContext(), ExecutionContext()
    seen: list[object] = []

    def other() -> None:
        with gate.held(theirs):
            seen.append(get_current_context())

    with gate.held(mine):
        with gate.released():
            t = threading.Thread(target=other)
            t.start()
            t.join(timeout=10)
        assert get_current_context() is mine
    assert seen == [theirs]


def test_released_outside_a_held_block_is_refused() -> None:
    gate = ContextGate()
    with pytest.raises(RuntimeError, match="not held"), gate.released():
        pass


def test_released_from_a_thread_that_does_not_hold_the_gate_is_refused() -> None:
    gate = ContextGate()
    errors: list[BaseException] = []

    def intruder() -> None:
        try:
            with gate.released():
                pass
        except RuntimeError as exc:
            errors.append(exc)

    with gate.held(ExecutionContext()):
        t = threading.Thread(target=intruder)
        t.start()
        t.join(timeout=10)
    assert len(errors) == 1


def test_the_gate_is_free_again_after_a_block_raises() -> None:
    gate = ContextGate()
    with pytest.raises(ValueError, match="boom"), gate.held(ExecutionContext()):
        raise ValueError("boom")
    with gate.held(ExecutionContext()):
        pass
