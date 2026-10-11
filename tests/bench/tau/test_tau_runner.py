from __future__ import annotations

import random

import litellm
import pytest
from tau_testkit import ScriptedUser, case, gold_turns, make_env

from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.bench.tau.conversation import ConversationModel
from ratchet.bench.tau.runner import (
    TauRunner,
    agent_cost,
    configure_user_endpoint,
    patient,
    seed_harness,
    tau_bench_commit,
    tau_tool_names,
    transient_errors,
)
from ratchet.bench.tau.tasks import TauCase
from ratchet.bench.tau.world import RESPOND
from ratchet.types import Guard


def _runner(user: ScriptedUser, max_actions: int = 30) -> TauRunner:
    return TauRunner(
        user_model="u", env_factory=lambda c, _m: make_env(c.index, user), max_actions=max_actions
    )


def test_seed_harness_is_the_tau_bench_policy_with_every_tool() -> None:
    h = seed_harness()
    assert h.verify_digest() and h.created_by == "seed" and h.parent is None
    assert h.system_prompt.startswith("# Retail agent policy")
    assert h.tool_allowlist == sorted([*tau_tool_names(), RESPOND])
    assert not h.guards and not h.procedures


def test_gold_run_through_the_worker_loop_passes() -> None:
    user = ScriptedUser(["###STOP###"])
    trace = _runner(user)(seed_harness(), case(0), ScriptedModel(gold_turns(0)), None, run_id="r1")
    assert trace.outcome.passed, trace.outcome.reasons
    assert trace.case_id == "tau-retail-test-000" and trace.run_id == "r1"
    assert user.heard == ["All done."]
    kinds = [e.kind for e in trace.events]
    assert kinds[-1] == "final" and trace.events[-1].payload["reason"] == "answered"
    assert trace.outcome.cost.model_calls == len(gold_turns(0))
    views = [e.payload["view"] for e in trace.events if e.kind == "tool_call"]
    assert "#W2378156" in views[-1]["orders"]


def test_a_guard_blocks_a_tool_call_in_a_tau_run() -> None:
    guard = Guard(
        "no-exchange",
        "block exchanges",
        "def check(call, view, history):\n"
        "    if call.name == 'exchange_delivered_order_items':\n"
        "        return 'ask the customer first'\n"
        "    return None\n",
    )
    harness = seed_harness().child(guards=[guard], created_by="human")
    trace = _runner(ScriptedUser())(harness, case(0), ScriptedModel(gold_turns(0)), None, run_id="r2")
    blocks = [e for e in trace.events if e.kind == "guard_block"]
    assert len(blocks) == 1 and "ask the customer first" in blocks[0].payload["message"]
    assert not trace.outcome.passed


def test_action_budget_ends_the_run_unfinished() -> None:
    turns = [scripted_turn("still working")] * 3
    trace = _runner(ScriptedUser(["ok"]), max_actions=2)(
        seed_harness(), case(0), ConversationModel(ScriptedModel(turns), 2), None, run_id="r3"
    )
    assert not trace.outcome.passed
    assert trace.outcome.reasons[0] == "the conversation did not end within the agent's action budget"
    assert trace.outcome.cost.model_calls == 2


def test_agent_cost_skips_closing_turns() -> None:
    trace = _runner(ScriptedUser())(seed_harness(), case(0), ScriptedModel(gold_turns(0)), None, run_id="r4")
    cost = agent_cost(trace.events)
    assert cost.model_calls == len(gold_turns(0))
    assert cost.tool_calls == len(gold_turns(0))
    assert agent_cost([]).wall_s == 0.0


def test_endpoint_configuration_installs_one_patient_retry() -> None:
    import tau_bench.envs.user as user_module  # type: ignore[import-untyped]  # no type info

    saved = (litellm.api_base, litellm.api_key, litellm.num_retries, litellm.suppress_debug_info)
    original = user_module.completion
    try:
        configure_user_endpoint("http://127.0.0.1:1/v1", "k")
        configure_user_endpoint("http://127.0.0.1:1/v1", "k")
        assert (litellm.api_base, litellm.api_key, litellm.num_retries) == ("http://127.0.0.1:1/v1", "k", 0)
        assert user_module.completion.patient is True
        assert user_module.completion is not original
    finally:
        litellm.api_base, litellm.api_key, litellm.num_retries, litellm.suppress_debug_info = saved
        user_module.completion = original
    assert tau_bench_commit() == "59a200c6d575d595120f1cb70fea53cef0632f6b"
    assert litellm.RateLimitError in transient_errors()


class _Busy(Exception):
    pass


def test_patient_retries_transient_errors_then_succeeds() -> None:
    calls: list[int] = []
    slept: list[float] = []

    def flaky(x: int) -> int:
        calls.append(x)
        if len(calls) < 3:
            raise _Busy
        return x * 2

    fn = patient(flaky, (_Busy,), sleep=slept.append, rng=random.Random(0))
    assert fn(4) == 8 and len(calls) == 3 and len(slept) == 2


def test_patient_gives_up_after_attempts_or_budget_and_passes_other_errors() -> None:
    def busy() -> None:
        raise _Busy

    with pytest.raises(_Busy):
        patient(busy, (_Busy,), attempts=3, sleep=lambda _s: None)()
    with pytest.raises(_Busy):
        patient(busy, (_Busy,), budget_s=0.0, sleep=lambda _s: None)()

    def broken() -> None:
        raise ValueError("bad request")

    slept: list[float] = []
    with pytest.raises(ValueError):
        patient(broken, (_Busy,), sleep=slept.append)()
    assert slept == []


def test_case_object_is_built_from_the_tau_case() -> None:
    tc = TauCase("tau-retail-train-007", "train", "train", 7)
    seen: list[TauCase] = []

    def factory(c: TauCase, model: str) -> tuple[object, str]:
        seen.append(c)
        return make_env(0, ScriptedUser())

    runner = TauRunner(user_model="u", env_factory=factory)
    trace = runner(seed_harness(), tc, ScriptedModel(gold_turns(0)), None, run_id="r5")
    assert seen == [tc] and trace.case_id == "tau-retail-train-007"
