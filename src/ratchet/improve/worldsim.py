"""The production simulator: rebuild the case's world and re-apply each executed call, grading as it goes.

The world is deterministic (the clock moves one minute per agent call, faults fire at declared points), so
re-applying the executed calls in order reproduces every intermediate state. The guard sees the view the
agent recorded on the `tool_call` event when there is one; the rebuilt view is compared against it, and any
difference marks the walk unfaithful in the replay report instead of being silently trusted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ratchet_cases.cases.grading import grade_state
from ratchet_cases.cases.model import Case
from ratchet_cases.types import Trace, canonical_json
from ratchet_cases.world.core import World

from ratchet.improve.sim import Step, Walk, result_ids, tool_call_of


class WorldSimulator:
    def __init__(self, cases: Iterable[Case] | Mapping[str, Case]) -> None:
        self.cases = dict(cases) if isinstance(cases, Mapping) else {c.id: c for c in cases}

    def walk(self, trace: Trace) -> Walk:
        case = self.cases[trace.case_id]
        world = World.from_case(case)
        done = result_ids(trace.events)
        initial = tuple(grade_state(case, world.view()))
        steps: list[Step] = []
        mismatches = 0
        for i, e in enumerate(trace.events):
            if e.kind != "tool_call":
                continue
            call = tool_call_of(e)
            rebuilt = world.view().to_json()
            recorded = e.payload.get("view")
            view = rebuilt
            if isinstance(recorded, dict):
                view = recorded
                mismatches += canonical_json(recorded) != canonical_json(rebuilt)
            executed = call.call_id in done
            if executed:
                world.call(call)
            steps.append(
                Step(e.seq, call, view, trace.events[:i], executed, tuple(grade_state(case, world.view())))
            )
        note = f"{mismatches} recorded view(s) differ from the rebuilt world" if mismatches else ""
        return Walk(trace.run_id, steps, initial, mismatches == 0, note)
