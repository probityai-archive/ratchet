from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from scripted import answer_at_once, replaying, solver_calls

from ratchet_cases import __version__
from ratchet_cases.cases import get
from ratchet_cases.cases.solvers import NAIVE, REFERENCE
from ratchet_cases.run import (
    TraceFileError,
    read_traces,
    regrade,
    replay,
    run_case,
    trace_from_json,
    trace_line,
)
from ratchet_cases.types import Event, to_jsonable

TRAP = "refund_lost_reply.v1"
TWIN = "refund_lost_reply.v2"


def _ids() -> tuple[str, str]:
    trap = get(TRAP)
    twin = get(TWIN)
    assert trap.kind == "trap" and twin.kind == "benign_twin"
    return trap.id, twin.id


def test_replaying_the_reference_solver_passes_and_regrades_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get(_ids()[0])
    trace = run_case(case, replaying(solver_calls(case, REFERENCE, monkeypatch)), model="scripted")
    assert trace.outcome.passed, trace.outcome.reasons
    assert trace.events[-1].kind == "final" and trace.events[-1].payload["reason"] == "answered"
    assert trace.outcome.cost.model_calls == sum(e.kind == "model" for e in trace.events)
    again = regrade(trace_from_json(json.loads(trace_line(trace, __version__))["trace"]))
    assert again.outcome.passed and again.recorded_passed and again.digest_matches


def test_replaying_the_naive_solver_fails_the_trap(monkeypatch: pytest.MonkeyPatch) -> None:
    case = get(_ids()[0])
    trace = run_case(case, replaying(solver_calls(case, NAIVE, monkeypatch)))
    assert not trace.outcome.passed
    assert not regrade(trace).outcome.passed


def test_an_agent_that_never_acts_fails_a_benign_twin() -> None:
    trace = run_case(get(_ids()[1]), answer_at_once)
    assert not trace.outcome.passed and trace.outcome.reasons


def test_the_step_budget_ends_a_run_that_never_answers() -> None:
    def forever(messages: list[dict[str, object]], tools: list[dict[str, object]]) -> dict[str, object]:
        return {"tool_calls": [{"id": "", "function": {"name": "tickets_list", "arguments": "{}"}}]}

    trace = run_case(get(_ids()[1]), forever, max_steps=3)
    assert trace.events[-1].payload["reason"] == "step budget"
    assert sum(e.kind == "tool_call" for e in trace.events) == 3


def test_unparseable_arguments_reach_the_model_and_never_the_world() -> None:
    sent: list[list[dict[str, object]]] = []

    def bad_then_done(messages: list[dict[str, object]], tools: list[dict[str, object]]) -> dict[str, object]:
        sent.append(list(messages))
        if len(sent) == 1:
            return {
                "tool_calls": [
                    {"id": "a", "function": {"name": "tickets_get", "arguments": "{nope"}},
                    {"id": "b", "function": {"name": "tickets_get", "arguments": "[1]"}},
                ]
            }
        return {"content": "stop"}

    trace = run_case(get(_ids()[1]), bad_then_done)
    assert not any(e.kind == "tool_call" for e in trace.events)
    errors = [json.loads(str(m["content"]))["error"] for m in sent[1] if m["role"] == "tool"]
    assert errors[0].startswith("invalid_arguments: not JSON") and errors[1].endswith("not a JSON object")


def test_tool_calls_must_be_a_list_of_objects() -> None:
    with pytest.raises(TypeError):
        run_case(get(_ids()[1]), lambda m, t: {"tool_calls": "tickets_get"})


def test_replay_skips_calls_that_never_reached_the_world(monkeypatch: pytest.MonkeyPatch) -> None:
    case = get(_ids()[0])
    trace = run_case(case, replaying(solver_calls(case, REFERENCE, monkeypatch)))
    blocked = Event(
        seq=len(trace.events),
        kind="tool_call",
        payload={"call": {"name": "payments_refund", "args": {}, "call_id": "blocked-1"}},
        ts=0.0,
    )
    guard = Event(seq=blocked.seq + 1, kind="guard_block", payload={"call_id": "blocked-1"}, ts=0.0)
    world = replay(case, [*trace.events, blocked, guard])
    assert world.digest() == trace.outcome.final_state_digest


def test_trace_files_round_trip_and_refuse_what_is_not_a_trace(tmp_path: Path) -> None:
    trace = run_case(get(_ids()[1]), answer_at_once)
    good = tmp_path / "traces.jsonl"
    good.write_text(trace_line(trace, __version__) + "\n\n", encoding="utf-8")
    [(record, back)] = list(read_traces(good))
    assert record["ratchet_cases"] == __version__ and to_jsonable(back) == to_jsonable(trace)
    for text, message in (
        ("{nope\n", "not JSON"),
        ('{"x": 1}\n', "no trace object"),
        ('{"trace": {"run_id": "r"}}\n', "not a trace"),
    ):
        bad = tmp_path / "bad.jsonl"
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(TraceFileError, match=message):
            list(read_traces(bad))


def test_regrade_reports_a_final_state_the_replay_does_not_reach() -> None:
    trace = run_case(get(_ids()[1]), answer_at_once)
    tampered = replace(trace, outcome=replace(trace.outcome, final_state_digest="0" * 64))
    assert not regrade(tampered).digest_matches
