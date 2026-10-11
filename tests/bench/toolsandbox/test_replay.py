from __future__ import annotations

from typing import Any

from toolsandbox_testkit import make_world, user_factory

from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.bench.toolsandbox.cases import ToolSandboxCase
from ratchet.bench.toolsandbox.replay import (
    END,
    ReplayUser,
    ToolSandboxSimulator,
    executed_ids,
    user_replies,
)
from ratchet.bench.toolsandbox.runner import ToolSandboxRunner, seed_harness
from ratchet.improve.diagnose import locate
from ratchet.types import Event, Trace


def _trace(
    cases: list[ToolSandboxCase], turns: list[Any], replies: list[str], name: str = "cellular_off"
) -> Trace:
    case = next(c for c in cases if c.name == name)
    runner = ToolSandboxRunner(user=user_factory(replies))
    return runner(seed_harness(), case, ScriptedModel(turns), None, run_id="run-r")


def _on_off_on() -> list[Any]:
    return [
        scripted_turn(tool_calls=[("set_cellular_service_status", {"on": False})]),
        scripted_turn(tool_calls=[("set_cellular_service_status", {"on": True})]),
        scripted_turn("Cellular service is turned off"),
    ]


def test_replay_user_speaks_then_ends_the_conversation() -> None:
    world = make_world("cellular_off", ["first", END])
    from ratchet.bench.toolsandbox.world import RESPOND
    from ratchet.types import ToolCall

    assert world.call(ToolCall(RESPOND, {"content": "a"}, "1")).data == {
        "observation": "first",
        "done": False,
    }
    assert world.call(ToolCall(RESPOND, {"content": "b"}, "2")).data == {"observation": "", "done": True}


def test_replay_user_ends_when_the_recording_runs_out() -> None:
    world = make_world("cellular_off", [])
    from ratchet.bench.toolsandbox.world import RESPOND
    from ratchet.types import ToolCall

    assert world.call(ToolCall(RESPOND, {"content": "a"}, "1")).data == {"observation": "", "done": True}


def test_executed_ids_and_user_replies_read_the_recording() -> None:
    events = [
        Event(0, "tool_call", {"call": {"name": "respond", "args": {}, "call_id": "r1"}}, 0.0),
        Event(
            1,
            "tool_result",
            {"result": {"call_id": "r1", "ok": True, "data": {"observation": "hi", "done": False}}},
            0.0,
        ),
        Event(2, "tool_call", {"call": {"name": "get_x", "args": {}, "call_id": "g1"}}, 0.0),
        Event(
            3,
            "tool_result",
            {"result": {"call_id": "g1", "ok": True, "data": {"observation": "7", "done": False}}},
            0.0,
        ),
        Event(4, "tool_call", {"call": {"name": "respond", "args": {}, "call_id": "r2"}}, 0.0),
        Event(5, "guard_block", {"call_id": "r2", "source": "guard"}, 0.0),
        Event(6, "tool_call", {"call": {"name": "respond", "args": {}, "call_id": "r3"}}, 0.0),
        Event(
            7,
            "tool_result",
            {"result": {"call_id": "r3", "ok": True, "data": {"observation": "", "done": True}}},
            0.0,
        ),
    ]
    assert executed_ids(events) == {"r1", "g1", "r3"}
    assert user_replies(events) == ["hi", ""]


def test_a_walk_grades_after_every_call_and_matches_the_recorded_state(
    all_cases: list[ToolSandboxCase],
) -> None:
    trace = _trace(all_cases, [
        scripted_turn(tool_calls=[("set_cellular_service_status", {"on": False})]),
        scripted_turn("Cellular service is turned off"),
    ], [END])  # fmt: skip
    walk = ToolSandboxSimulator().walk(trace)
    assert walk.faithful and walk.note == ""
    assert walk.initial_reasons is not None and len(walk.initial_reasons) == 2
    assert [s.call.name for s in walk.steps] == ["set_cellular_service_status", "respond"]
    assert [s.executed for s in walk.steps] == [True, True]
    assert walk.steps[0].view["setting"][0]["cellular"] is True  # the state the guard saw, before the call
    assert walk.steps[0].reasons_after is not None and len(walk.steps[0].reasons_after) == 1
    assert walk.steps[1].reasons_after == ()
    assert [e.kind for e in walk.steps[1].history][-2:] == ["tool_result", "model"]


def test_diagnose_locates_the_call_that_reached_a_minefield(all_cases: list[ToolSandboxCase]) -> None:
    name = "find_days_till_holiday_insufficient_information"
    turns = [
        scripted_turn(tool_calls=[("timestamp_diff", {"timestamp_0": 0.0, "timestamp_1": 86400.0})]),
        scripted_turn("Which year do you mean?"),
    ]
    trace = _trace(all_cases, turns, [END], name)
    walk = ToolSandboxSimulator().walk(trace)
    assert walk.faithful
    reason = next(r for r in trace.outcome.reasons if r.startswith("minefield 1 reached:"))
    seq, mode = locate(trace, walk, reason)
    call = next(s for s in walk.steps if s.call.name == "timestamp_diff")
    assert (seq, mode) == (call.seq, "commission")
    assert reason not in (walk.initial_reasons or ())


def test_a_state_that_was_reached_and_left_still_counts_as_a_milestone_met(
    all_cases: list[ToolSandboxCase],
) -> None:
    trace = _trace(all_cases, _on_off_on(), [END])
    assert trace.outcome.passed and trace.outcome.reasons == []
    walk = ToolSandboxSimulator().walk(trace)
    assert walk.faithful and walk.steps[-1].reasons_after == ()


def test_diagnose_calls_a_milestone_that_was_never_met_an_omission(all_cases: list[ToolSandboxCase]) -> None:
    trace = _trace(all_cases, [scripted_turn("I cannot do that.")], [END])
    walk = ToolSandboxSimulator().walk(trace)
    reason = 'milestone 1 not met: SETTING has {"cellular": false}'
    seq, mode = locate(trace, walk, reason)
    assert mode == "omission" and seq == trace.events[-1].seq


def test_a_blocked_call_is_listed_but_not_replayed(all_cases: list[ToolSandboxCase]) -> None:
    trace = _trace(all_cases, [scripted_turn(tool_calls=[("not_a_tool", {})]), scripted_turn("done")], [END])
    walk = ToolSandboxSimulator().walk(trace)
    blocked = [s for s in walk.steps if s.call.name == "not_a_tool"]
    assert len(blocked) == 1 and blocked[0].executed is False
    assert walk.faithful


def test_an_unknown_scenario_is_an_unfaithful_empty_walk(all_cases: list[ToolSandboxCase]) -> None:
    trace = _trace(all_cases, [scripted_turn("x")], [END])
    other = Trace(
        trace.run_id, "toolsandbox-nonesuch", trace.harness_version, trace.model, trace.events, trace.outcome
    )
    walk = ToolSandboxSimulator().walk(other)
    assert not walk.faithful and walk.steps == [] and "nonesuch" in walk.note


def test_a_replay_that_ends_elsewhere_says_so(all_cases: list[ToolSandboxCase]) -> None:
    trace = _trace(all_cases, _on_off_on(), [END])
    from dataclasses import replace

    forged = replace(trace, outcome=replace(trace.outcome, final_state_digest="0" * 64))
    walk = ToolSandboxSimulator().walk(forged)
    assert not walk.faithful and "different state" in walk.note
    assert ReplayUser().replies == [END]
