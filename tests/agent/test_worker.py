from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any, cast

import pytest
from ratchet_cases.cases.model import Case, Expect
from ratchet_cases.types import Outcome, ToolCall, ToolResult, Trace
from ratchet_cases.world import World

from ratchet.agent.guards import GUARD_ERROR_REASON, GuardInfrastructureError, GuardLimits, GuardRuntime
from ratchet.agent.model import ModelError, ModelTurn, ScriptedModel, scripted_turn
from ratchet.agent.worker import (
    READ_PROCEDURE,
    ResumeError,
    RunLimits,
    content_words,
    default_run_id,
    keyword_matches,
    run_case,
    system_prompt,
)
from ratchet.types import Guard, Harness, Procedure


class FakeView:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = json.loads(json.dumps(data))

    def to_json(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(json.dumps(self._data)))


class FakeWorld:
    """Two tools over a balance. `refund` moves money; `lookup` reads."""

    def __init__(self) -> None:
        self.balance = 100
        self.calls: list[ToolCall] = []

    def tools(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object"}}}
            for n in ("lookup", "refund", "wipe")
        ]

    def tool_names(self) -> list[str]:
        return ["lookup", "refund", "wipe"]

    def call(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        if call.name == "refund":
            self.balance -= int(call.args.get("amount", 0))
            return ToolResult(call.call_id, True, {"balance": self.balance})
        if call.name == "lookup":
            return ToolResult(call.call_id, True, {"balance": self.balance})
        return ToolResult(call.call_id, False, None, "unknown tool")

    def view(self) -> FakeView:
        return FakeView({"balance": self.balance, "calls": len(self.calls)})

    def digest(self) -> str:
        return f"balance={self.balance}"


class FakeStore:
    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {}
        self.writes = 0

    def checkpoint(self, run_id: str, state: Mapping[str, Any]) -> int:
        self.states[run_id] = json.loads(json.dumps(dict(state)))
        self.writes += 1
        return self.writes

    def resume(self, run_id: str) -> Mapping[str, Any] | None:
        return self.states.get(run_id)


class Killed(ModelError):
    pass


def _case(message: str = "Please refund 30 dollars on my order") -> Case:
    return Case(
        id="case-1",
        family="refunds",
        variant="v1",
        split="train",
        must_succeed=True,
        customer_message=message,
        initial_state={},
        faults=[],
        grader=Expect(),
        ticket_id="T-1",
        kind="must_succeed",
    )


def _harness(
    guards: list[Guard] | None = None,
    procedures: list[Procedure] | None = None,
    allow: tuple[str, ...] = ("lookup", "refund"),
) -> Harness:
    return Harness.build(
        parent=None,
        system_prompt="You are support.",
        procedures=procedures or [],
        guards=guards or [],
        tool_allowlist=list(allow),
        created_by="seed",
    )


def _grader(case: Case, world: World, trace: Trace) -> Outcome:
    fake = cast(FakeWorld, world)
    reasons = [] if fake.balance == 70 else [f"balance is {fake.balance}"]
    return Outcome(passed=not reasons, reasons=reasons, final_state_digest=fake.digest())


def _run(harness: Harness, model: Any, world: FakeWorld | None = None, **kw: Any) -> tuple[Trace, FakeWorld]:
    w = world or FakeWorld()
    trace = run_case(
        harness,
        _case(kw.pop("message", "Please refund 30 dollars on my order")),
        model,
        world_factory=lambda c: cast(World, w),
        grader=_grader,
        **kw,
    )
    return trace, w


@pytest.fixture(scope="module")
def runtime() -> Iterator[GuardRuntime]:
    with GuardRuntime() as rt:
        yield rt


def test_happy_path_records_every_event_and_passes(runtime: GuardRuntime) -> None:
    model = ScriptedModel(
        [
            scripted_turn(tool_calls=[("lookup", {})]),
            scripted_turn(tool_calls=[("refund", {"amount": 30})], id_prefix="r"),
            scripted_turn("Refunded 30."),
        ]
    )
    trace, world = _run(_harness(), model, runtime=runtime)
    assert trace.outcome.passed and world.balance == 70
    kinds = [e.kind for e in trace.events]
    assert kinds == [
        "model",
        "tool_call",
        "tool_result",
        "model",
        "tool_call",
        "tool_result",
        "model",
        "final",
    ]
    assert [e.seq for e in trace.events] == list(range(len(kinds)))
    call_event = trace.events[4]
    assert call_event.payload["call"] == {"name": "refund", "args": {"amount": 30}, "call_id": "r0"}
    assert call_event.payload["view"] == {"balance": 100, "calls": 1}
    assert trace.events[0].payload["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert trace.events[-1].payload == {"reason": "answered", "content": "Refunded 30.", "complete": True}
    assert trace.model == "scripted" and trace.harness_version == _harness().version
    first_request = model.requests[0]
    assert first_request[0][1]["content"].startswith("Ticket T-1")
    assert [t["function"]["name"] for t in first_request[1]] == ["lookup", "refund"]
    tool_msg = model.requests[1][0][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["data"] == {"balance": 100}


def test_allowlist_blocks_before_guards(runtime: GuardRuntime) -> None:
    model = ScriptedModel([scripted_turn(tool_calls=[("wipe", {})]), scripted_turn("done")])
    trace, world = _run(_harness(), model, runtime=runtime)
    block = next(e for e in trace.events if e.kind == "guard_block")
    assert block.payload["source"] == "allowlist" and world.calls == []
    assert "not available" in json.loads(model.requests[1][0][-1]["content"])["error"]


def test_guard_blocks_with_reason_and_sees_history_and_view(runtime: GuardRuntime) -> None:
    guard = Guard(
        id="lookup-first",
        description="refund only after a lookup",
        source=(
            "def check(call, view, history):\n"
            "    if call.name != 'refund':\n"
            "        return None\n"
            "    looked = any(\n"
            "        e.kind == 'tool_call' and e.payload['call']['name'] == 'lookup' for e in history\n"
            "    )\n"
            "    if not looked:\n"
            "        return 'look up the balance first'\n"
            "    if call.args['amount'] > view['balance']:\n"
            "        return 'more than the balance'\n"
            "    return None\n"
        ),
    )
    model = ScriptedModel(
        [
            scripted_turn(tool_calls=[("refund", {"amount": 30})]),
            scripted_turn(tool_calls=[("lookup", {})], id_prefix="l"),
            scripted_turn(tool_calls=[("refund", {"amount": 30})], id_prefix="r"),
            scripted_turn("ok"),
        ]
    )
    trace, world = _run(_harness([guard]), model, runtime=runtime)
    assert trace.outcome.passed and [c.name for c in world.calls] == ["lookup", "refund"]
    block = next(e for e in trace.events if e.kind == "guard_block")
    assert block.payload["source"] == "guard"
    assert block.payload["message"] == "blocked by guard lookup-first: look up the balance first"
    assert block.payload["decisions"][0]["blocked"] is True
    result = [e for e in trace.events if e.kind == "tool_result"][-1]
    assert result.payload["decisions"] == [
        {
            "guard_id": "lookup-first",
            "blocked": False,
            "reason": None,
            "error": None,
            "elapsed_s": result.payload["decisions"][0]["elapsed_s"],
            "kind": "verdict",
        }
    ]
    assert "look up the balance first" in model.requests[1][0][-1]["content"]


def test_crashing_guard_fails_closed(runtime: GuardRuntime) -> None:
    guard = Guard(id="boom", description="crashes", source="def check(call, view, history):\n    1 / 0\n")
    model = ScriptedModel([scripted_turn(tool_calls=[("refund", {"amount": 30})]), scripted_turn("stop")])
    trace, world = _run(_harness([guard]), model, runtime=runtime)
    block = next(e for e in trace.events if e.kind == "guard_block")
    decision = block.payload["decisions"][0]
    assert decision["reason"] == GUARD_ERROR_REASON and "ZeroDivisionError" in decision["error"]
    assert world.calls == [] and not trace.outcome.passed


def test_invalid_arguments_are_refused(runtime: GuardRuntime) -> None:
    bad = scripted_turn(tool_calls=[("refund", {})])
    bad = ModelTurn(**{**bad.__dict__, "arg_errors": {"c0": "arguments are not valid JSON"}})
    model = ScriptedModel([bad, scripted_turn("stop")])
    trace, world = _run(_harness(), model, runtime=runtime)
    block = next(e for e in trace.events if e.kind == "guard_block")
    assert block.payload["source"] == "arguments" and world.calls == []


def test_step_budget_marks_incomplete(runtime: GuardRuntime) -> None:
    model = ScriptedModel([scripted_turn(tool_calls=[("lookup", {})], id_prefix=f"s{i}") for i in range(5)])
    trace, _ = _run(_harness(), model, runtime=runtime, limits=RunLimits(max_steps=2))
    assert trace.events[-1].payload["reason"] == "step budget"
    assert not trace.outcome.passed and trace.outcome.reasons[-1] == "incomplete: step budget"


def test_token_budget_marks_incomplete(runtime: GuardRuntime) -> None:
    model = ScriptedModel([scripted_turn(tool_calls=[("lookup", {})], input_tokens=500)] * 3)
    trace, _ = _run(_harness(), model, runtime=runtime, limits=RunLimits(max_tokens=400))
    assert trace.events[-1].payload["reason"] == "token budget" and len(model.requests) == 1


def test_model_error_marks_incomplete(runtime: GuardRuntime) -> None:
    trace, _ = _run(_harness(), ScriptedModel([]), runtime=runtime)
    assert trace.events[-1].payload["reason"].startswith("model error")
    assert trace.outcome.reasons[-1].startswith("incomplete: model error")


REFUND_PROC = Procedure(
    id="p1", name="refund-check", when="refund requested on an order", steps=["Look up the order.", "Refund."]
)
MAIL_PROC = Procedure(id="p2", name="mail-change", when="email address change", steps=["Verify identity."])


def test_procedures_keyword_preload_and_read_procedure(runtime: GuardRuntime) -> None:
    model = ScriptedModel(
        [
            scripted_turn(
                tool_calls=[(READ_PROCEDURE, {"name": "mail-change"}), (READ_PROCEDURE, {"name": "x"})]
            ),
            scripted_turn(tool_calls=[("refund", {"amount": 30})], id_prefix="r"),
            scripted_turn("done"),
        ]
    )
    trace, world = _run(_harness(procedures=[REFUND_PROC, MAIL_PROC]), model, runtime=runtime)
    system = model.requests[0][0][0]["content"]
    assert "- refund-check: refund requested on an order" in system and "- mail-change:" in system
    assert "1. Look up the order." in system and "Verify identity." not in system
    assert READ_PROCEDURE in [t["function"]["name"] for t in model.requests[0][1]]
    procs = [e.payload for e in trace.events if e.kind == "procedure"]
    assert procs[0] == {
        "name": "refund-check",
        "how": "keyword",
        "matched": ["order", "refund"],
        "found": True,
    }
    assert procs[1]["how"] == "selected" and procs[1]["found"] and not procs[2]["found"]
    replies = [m for m in model.requests[1][0] if m["role"] == "tool"]
    assert "Verify identity." in replies[0]["content"] and "no procedure" in replies[1]["content"]
    assert [e.kind for e in trace.events].count("tool_call") == 1 and world.balance == 70


def test_keyword_rules() -> None:
    assert content_words("The customer asks for a REFUND of order #12") == {"refund", "order"}
    one = Procedure(id="p", name="n", when="chargeback", steps=[])
    assert keyword_matches([one], "about a chargeback") and not keyword_matches([one], "about a refund")
    assert not keyword_matches([REFUND_PROC], "where is my order")
    empty = Procedure(id="e", name="e", when="when it is", steps=[])
    assert not keyword_matches([empty], "when it is")
    assert system_prompt(_harness(), []) == "You are support."


def test_world_tool_named_like_read_procedure_is_rejected(runtime: GuardRuntime) -> None:
    class Clash(FakeWorld):
        def tool_names(self) -> list[str]:
            return [READ_PROCEDURE]

    with pytest.raises(ValueError, match="read_procedure"):
        _run(_harness(procedures=[REFUND_PROC]), ScriptedModel([]), world=Clash(), runtime=runtime)


class KillAfter(ScriptedModel):
    """Raises a non-model error (a crash) on the n-th request, like a killed process."""

    def __init__(self, turns: list[ModelTurn], crash_at: int) -> None:
        super().__init__(turns)
        self.crash_at = crash_at

    def complete(self, messages: Any, tools: Any) -> ModelTurn:
        if len(self.requests) == self.crash_at:
            self.requests.append(([], []))
            raise KeyboardInterrupt("killed")
        return super().complete(messages, tools)


def _script() -> list[ModelTurn]:
    return [
        scripted_turn(tool_calls=[("lookup", {}), ("refund", {"amount": 10})]),
        scripted_turn(tool_calls=[("refund", {"amount": 20})], id_prefix="r"),
        scripted_turn("done"),
    ]


def test_killed_run_resumes_from_checkpoint(runtime: GuardRuntime) -> None:
    store = FakeStore()
    with pytest.raises(KeyboardInterrupt):
        _run(_harness(), KillAfter(_script(), crash_at=1), store=store, runtime=runtime, run_id="r1")
    saved = store.states["r1"]
    assert saved["status"] == "running" and len(saved["executed"]) == 2 and saved["pending"] == []
    resumed_model = ScriptedModel(_script()[1:])
    trace, world = _run(_harness(), resumed_model, store=store, runtime=runtime, run_id="r1")
    assert trace.outcome.passed and world.balance == 70 and len(world.calls) == 3
    assert len(resumed_model.requests) == 2
    assert resumed_model.requests[0][0][-1]["role"] == "tool"
    assert store.states["r1"]["status"] == "final"
    assert [e.kind for e in trace.events].count("checkpoint") == 3
    # A finished run is not resumed: the same id starts fresh.
    again, _ = _run(_harness(), ScriptedModel(_script()), store=store, runtime=runtime, run_id="r1")
    assert again.outcome.passed


def test_resume_mid_turn_finishes_pending_calls(runtime: GuardRuntime) -> None:
    store = FakeStore()

    class OneCheckpointThenDie(FakeStore):
        def checkpoint(self, run_id: str, state: Mapping[str, Any]) -> int:
            n = store.checkpoint(run_id, state)
            if n == 1:
                raise KeyboardInterrupt("killed after first tool")
            return n

        def resume(self, run_id: str) -> Mapping[str, Any] | None:
            return store.resume(run_id)

    with pytest.raises(KeyboardInterrupt):
        _run(_harness(), ScriptedModel(_script()), store=OneCheckpointThenDie(), runtime=runtime, run_id="r2")
    assert len(store.states["r2"]["pending"]) == 1
    trace, world = _run(_harness(), ScriptedModel(_script()[1:]), store=store, runtime=runtime, run_id="r2")
    assert trace.outcome.passed and [c.name for c in world.calls] == ["lookup", "refund", "refund"]


def test_resume_refuses_foreign_or_diverged_checkpoint(runtime: GuardRuntime) -> None:
    store = FakeStore()
    with pytest.raises(KeyboardInterrupt):
        _run(_harness(), KillAfter(_script(), crash_at=1), store=store, runtime=runtime, run_id="r3")
    store.states["r3"]["world_digest"] = "tampered"
    with pytest.raises(ResumeError, match="does not match"):
        _run(_harness(), ScriptedModel(_script()), store=store, runtime=runtime, run_id="r3")
    store.states["r3"]["case_id"] = "other"
    with pytest.raises(ResumeError, match="case_id"):
        _run(_harness(), ScriptedModel(_script()), store=store, runtime=runtime, run_id="r3")


def test_default_run_id_is_stable_and_owns_its_runtime() -> None:
    h = _harness()
    assert (
        default_run_id(h, _case(), "m") == default_run_id(h, _case(), "m") != default_run_id(h, _case(), "n")
    )
    trace, _ = _run(h, ScriptedModel([scripted_turn("nothing to do")]))
    assert trace.run_id == default_run_id(h, _case(), "scripted") and not trace.outcome.passed


def test_sandbox_failure_stops_the_run_instead_of_blocking() -> None:
    guard = Guard(id="g", description="allow", source="def check(call, view, history):\n    return None\n")
    model = ScriptedModel([scripted_turn(tool_calls=[("refund", {"amount": 30})]), scripted_turn("done")])
    store = FakeStore()
    with (
        GuardRuntime(GuardLimits(startup_timeout_s=0.001)) as broken,
        pytest.raises(GuardInfrastructureError),
    ):
        _run(_harness([guard]), model, runtime=broken, store=store, run_id="infra")
    assert store.states == {} or store.states["infra"]["status"] == "running"
    trace, world = _run(_harness([guard]), ScriptedModel(model._turns), store=store, run_id="infra")
    assert trace.outcome.passed and world.balance == 70
