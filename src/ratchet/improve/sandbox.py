"""Replay through the agent lane's sandboxed guard runtime: the same isolation as a live run."""

from __future__ import annotations

from collections.abc import Sequence

from ratchet.agent.guards import GuardItem, GuardRuntime
from ratchet.improve.replay import GuardInput, Verdict
from ratchet.types import Guard


class SandboxRunner:
    """A `GuardRunner` over `GuardRuntime.batch`. A fail-closed error decision becomes a `Verdict` error,
    so replay reports a crashing guard as broken rather than as a guard that blocks precisely."""

    def __init__(self, runtime: GuardRuntime) -> None:
        self.runtime = runtime

    def check_batch(self, guard: Guard, inputs: Sequence[GuardInput]) -> list[Verdict]:
        items = [GuardItem(guard, i.call, i.view, i.history) for i in inputs]
        return [
            Verdict(None, d.error) if d.error is not None else Verdict(d.reason if d.blocked else None)
            for d in self.runtime.batch(items)
        ]
