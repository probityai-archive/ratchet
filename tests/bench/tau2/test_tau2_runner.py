from __future__ import annotations

import litellm
from tau2_testkit import STOP, ScriptedUser, case, config, factory, gold_turns

from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.bench.tau2.conversation import ConversationModel
from ratchet.bench.tau2.runner import (
    Tau2Runner,
    agent_cost,
    agent_system_prompt,
    attach_usage,
    configure_user_endpoint,
    litellm_model,
    run_config,
    seed_harness,
    tool_names,
)
from ratchet.bench.tau2.world import RESPOND
from ratchet.types import Guard

TELECOM = "[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]"


def _runner(domain: str, user: ScriptedUser) -> Tau2Runner:
    return Tau2Runner(config(domain), factory=factory(user))  # type: ignore[arg-type]


def test_seed_harness_is_tau2_bench_agent_prompt_with_every_tool() -> None:
    h = seed_harness("retail")
    assert h.verify_digest() and h.created_by == "seed" and h.parent is None
    assert h.system_prompt == agent_system_prompt("retail")
    assert h.system_prompt.startswith("<instructions>") and "<policy>" in h.system_prompt
    assert h.tool_allowlist == sorted([*tool_names("retail"), RESPOND])
    assert not h.guards and not h.procedures


def test_retail_gold_run_through_the_worker_loop_passes_with_tau2_bench_reward() -> None:
    user = ScriptedUser([STOP])
    runner = _runner("retail", user)
    trace = runner(
        seed_harness("retail"),
        case("retail", "0"),
        ScriptedModel(gold_turns("retail", "0")),
        None,
        run_id="r1",
    )
    assert trace.outcome.passed, trace.outcome.reasons
    assert trace.case_id == "tau2-retail-0" and trace.run_id == "r1"
    assert trace.events[-1].payload["reason"] == "answered"
    sim = runner.simulations["r1"]
    assert sim.id == "r1" and sim.reward_info.reward == 1.0 and sim.trial == 0
    assert sim.termination_reason.value == "user_stop"
    calls = [m for m in sim.messages if m.role == "assistant" and m.tool_calls]
    assert len(calls) == 5 and all(m.usage == {"prompt_tokens": 10, "completion_tokens": 5} for m in calls)
    views = [e.payload["view"] for e in trace.events if e.kind == "tool_call"]
    assert "#W2378156" in views[-1]["orders"]


def test_airline_and_telecom_gold_runs_pass() -> None:
    airline = _runner("airline", ScriptedUser([STOP]))
    trace = airline(
        seed_harness("airline"),
        case("airline", "0"),
        ScriptedModel([scripted_turn("Hello.")]),
        None,
        run_id="a1",
    )
    assert trace.outcome.passed, trace.outcome.reasons
    user = ScriptedUser([[("toggle_airplane_mode", {}), ("toggle_roaming", {})], "Done.", STOP])
    telecom = _runner("telecom", user)
    script = [scripted_turn("Turn off airplane mode and turn on roaming."), scripted_turn("Great.")]
    trace = telecom(
        seed_harness("telecom"), case("telecom", TELECOM), ScriptedModel(script), None, run_id="t1"
    )
    assert trace.outcome.passed, trace.outcome.reasons
    results = [e.payload["result"] for e in trace.events if e.kind == "tool_result"]
    assert [a["name"] for a in results[0]["data"]["user_actions"]] == [
        "toggle_airplane_mode",
        "toggle_roaming",
    ]
    views = [e.payload["view"] for e in trace.events if e.kind == "tool_call"]
    assert views[0]["user"] != views[1]["user"]


def test_a_guard_blocks_a_tool_call_before_the_environment() -> None:
    guard = Guard(
        "no-exchange",
        "block exchanges",
        "def check(call, view, history):\n"
        "    if call.name == 'exchange_delivered_order_items':\n"
        "        return 'ask the customer first'\n"
        "    return None\n",
    )
    harness = seed_harness("retail").child(guards=[guard], created_by="human")
    runner = _runner("retail", ScriptedUser([STOP]))
    trace = runner(harness, case("retail", "0"), ScriptedModel(gold_turns("retail", "0")), None, run_id="r2")
    blocks = [e for e in trace.events if e.kind == "guard_block"]
    assert len(blocks) == 1 and "ask the customer first" in blocks[0].payload["message"]
    assert not trace.outcome.passed
    sim = runner.simulations["r2"]
    assert not any(
        tc.name == "exchange_delivered_order_items"
        for m in sim.messages
        if m.role == "assistant"
        for tc in m.tool_calls or []
    )


def test_a_model_failure_ends_the_run_as_an_agent_error() -> None:
    runner = _runner("retail", ScriptedUser(["ok"]))
    trace = runner(
        seed_harness("retail"), case("retail", "0"), ScriptedModel([scripted_turn("Hi")]), None, run_id="r3"
    )
    assert not trace.outcome.passed and trace.outcome.reasons[-1].startswith("incomplete: model error")
    assert runner.simulations["r3"].termination_reason.value == "agent_error"


def test_wrapped_model_is_used_as_is_and_cost_skips_closing_turns() -> None:
    runner = _runner("retail", ScriptedUser([STOP]))
    model = ConversationModel(ScriptedModel(gold_turns("retail", "0")))
    trace = runner(seed_harness("retail"), case("retail", "0"), model, None, run_id="r4")
    cost = agent_cost(trace.events)
    assert cost.model_calls == len(gold_turns("retail", "0")) and cost.tool_calls == cost.model_calls
    assert agent_cost([]).wall_s == 0.0
    assert runner.task("0").id == "0"


def test_attach_usage_skips_unknown_messages() -> None:
    runner = _runner("retail", ScriptedUser([STOP]))
    trace = runner(
        seed_harness("retail"),
        case("retail", "0"),
        ScriptedModel(gold_turns("retail", "0")),
        None,
        run_id="r5",
    )
    sim = runner.simulations["r5"]
    for m in sim.messages:
        if m.role == "assistant":
            m.usage = None
            m.raw_data = None
    sim.messages[0].tool_calls = None
    attach_usage(sim, trace)
    assert sim.messages[0].usage is None


def test_run_config_and_endpoints() -> None:
    cfg = run_config("airline", agent_model="a", user_model="u", temperature=None, trials=4, seed=5)
    assert cfg.llm_user == "openai/u" and cfg.llm_args_agent == {} and cfg.num_trials == 4 and cfg.seed == 5
    assert cfg.max_steps == 200 and cfg.max_errors == 10
    assert litellm_model("hosted_vllm/x") == "hosted_vllm/x"
    saved = (litellm.api_base, litellm.api_key, litellm.suppress_debug_info)
    try:
        configure_user_endpoint("http://127.0.0.1:1/v1", "k")
        assert (litellm.api_base, litellm.api_key) == ("http://127.0.0.1:1/v1", "k")
    finally:
        litellm.api_base, litellm.api_key, litellm.suppress_debug_info = saved


def test_a_customer_who_stops_at_once_closes_the_run_after_one_action() -> None:
    runner = _runner("airline", ScriptedUser(opening=STOP))
    model = ScriptedModel([scripted_turn("Hello?")] * 3)
    trace = runner(seed_harness("airline"), case("airline", "0"), model, None, run_id="s1")
    assert len(model.requests) == 1 and trace.events[-1].payload["reason"] == "answered"
    assert runner.simulations["s1"].termination_reason.value == "user_stop"
