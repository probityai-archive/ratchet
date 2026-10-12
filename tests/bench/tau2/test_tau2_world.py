from __future__ import annotations

from typing import Any

import pytest
from ratchet_cases.types import Cost, ToolCall
from tau2_testkit import STOP, ScriptedUser, make_world, task

from ratchet.bench.tau2.world import (
    RESPOND,
    PuppetAgent,
    Tau2World,
    ViewSpec,
    gold_databases,
    outcome_of,
    record_diffs,
    reward_reasons,
)


def test_start_runs_until_the_customer_opens() -> None:
    world = make_world("retail", "0", ScriptedUser(opening="I want to exchange two items."))
    assert world.opening == "I want to exchange two items."
    assert not world.done and world.steps == 0
    names = world.tool_names()
    assert RESPOND in names and "exchange_delivered_order_items" in names
    assert len(world.tools()) == len(names)


def test_a_tool_call_runs_on_the_environment_and_touches_records() -> None:
    world = make_world("retail", "0")
    result = world.call(ToolCall("get_order_details", {"order_id": "#W2378156"}, "c1"))
    assert result.ok and result.data is not None and "#W2378156" in result.data["observation"]
    view = world.view().to_json()
    assert "#W2378156" in view["orders"] and view["step"] == 1 and view["done"] is False
    user_id = view["orders"]["#W2378156"]["user_id"]
    assert user_id in view["users"]


def test_a_failing_tool_call_is_an_error_result() -> None:
    world = make_world("retail", "0")
    result = world.call(ToolCall("get_order_details", {"order_id": "#nope"}, "c1"))
    assert not result.ok and result.error is not None and result.error.startswith("Error")


def test_respond_returns_the_reply_and_stop_ends_with_a_reward() -> None:
    user = ScriptedUser(["Thanks, bye " + STOP])
    world = make_world("airline", "0", user)
    result = world.call(ToolCall(RESPOND, {"content": "How can I help?"}, "r1"))
    assert result.ok and result.data is not None and STOP in result.data["observation"]
    assert result.data["done"] is True and world.done and user.heard[-1] == "How can I help?"
    assert world.ending is not None and world.ending.by == RESPOND
    sim = world.ending.simulation
    assert sim.termination_reason.value == "user_stop" and sim.reward_info is not None
    after = world.call(ToolCall(RESPOND, {"content": "again"}, "r2"))
    assert not after.ok and after.error == "the conversation has ended"
    assert after.data == {"observation": "", "done": True}


def test_empty_respond_is_refused_without_stepping() -> None:
    world = make_world("retail", "0")
    before = len(world.orchestrator.trajectory)
    result = world.call(ToolCall(RESPOND, {"content": "  "}, "r1"))
    assert not result.ok and "non-empty" in (result.error or "")
    assert len(world.orchestrator.trajectory) == before


def test_telecom_user_actions_are_returned_and_visible_in_the_view() -> None:
    tid = task(
        "telecom", "[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]"
    ).id
    user = ScriptedUser([[("toggle_airplane_mode", {})], "Done it."])
    world = make_world("telecom", str(tid), user)
    before = world.view().to_json()["user"]
    result = world.call(ToolCall(RESPOND, {"content": "Please turn off airplane mode."}, "r1"))
    assert result.ok and result.data is not None
    actions = result.data["user_actions"]
    assert [a["name"] for a in actions] == ["toggle_airplane_mode"] and not actions[0]["error"]
    assert result.data["observation"] == "Done it."
    after = world.view().to_json()
    assert after["user"] != before and "customers" in after and "lines" in after


def test_abort_ends_with_tau2_bench_zero_reward() -> None:
    world = make_world("retail", "0")
    world.abort("step budget")
    assert world.ending is not None and world.ending.reward == 0.0
    assert world.ending.simulation.termination_reason.value == "max_steps"
    world.abort("again")  # a second abort changes nothing
    out = outcome_of(world, Cost())
    assert not out.passed and out.reasons[0].startswith("the conversation ended by max_steps")


def test_model_failure_aborts_as_agent_error() -> None:
    world = make_world("retail", "0")
    out_before = world.digest()
    world.abort("model error: boom")
    assert world.ending is not None and world.ending.simulation.termination_reason.value == "agent_error"
    assert world.digest() == out_before


def test_outcome_of_an_unfinished_world_aborts_it() -> None:
    world = make_world("retail", "0")
    out = outcome_of(world, Cost())
    assert not out.passed and world.done


def test_wrong_write_gives_database_reasons() -> None:
    user = ScriptedUser([STOP])
    world = make_world("retail", "0", user)
    world.call(
        ToolCall("cancel_pending_order", {"order_id": "#W2378156", "reason": "no longer needed"}, "c1")
    )
    world.call(ToolCall(RESPOND, {"content": "Cancelled."}, "r1"))
    out = outcome_of(world, Cost())
    assert world.done and not out.passed
    assert any("#W2378156" in r or "orders" in r for r in out.reasons), out.reasons


def test_gold_databases_apply_the_gold_actions() -> None:
    gold, user = gold_databases(task("retail", "0"), "retail")
    assert user is None
    items = gold["orders"]["#W2378156"]
    assert items["status"] == "exchange requested"


def test_record_diffs_names_fields_lists_missing_and_limits() -> None:
    final = {"t": {"a": {"x": 1, "l": [1, 2]}, "b": {"x": 1}}, "s": 1, "u": {"k": 1}}
    gold = {"t": {"a": {"x": 2, "l": [1, 3]}, "c": {"x": 1}}, "s": 2, "u": {"k": [1]}}
    diffs = record_diffs(final, gold)
    assert "t['a'].x is 1, the task needs 2" in diffs
    assert "t['a'].l[1] is 2, the task needs 3" in diffs
    assert "t['b'] should not exist in the final database" in diffs
    assert "t['c'] is missing in the final database" in diffs
    assert "s is 1, the task needs 2" in diffs
    assert "u['k'] is 1, the task needs [1]" in diffs
    short = {"t": {"a": {"l": [1]}}}
    assert record_diffs(short, {"t": {"a": {"l": [1, 2]}}}) == [
        "t['a'].l has 1 entries, the task needs 2 (extra or missing at the end)"
    ]
    many = record_diffs({"t": {str(i): {"x": 0} for i in range(20)}}, {"t": {}}, limit=3)
    assert len(many) == 4 and many[-1] == "... and 17 more differences"
    long = record_diffs({"t": {"a": {"x": "y" * 300}}}, {"t": {"a": {"x": "z"}}})
    assert "..." in long[0]


def test_reward_reasons_cover_every_check_kind() -> None:
    world = make_world("retail", "0")
    world.abort("step budget")
    assert world.ending is not None
    info = world.ending.simulation.reward_info
    action = task("retail", "0").evaluation_criteria.actions[0]

    class Obj:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {"func_name": "f"}

    info.action_checks = [Obj(action_match=False, action=action)]
    info.communicate_checks = [Obj(met=False, info="the refund amount")]
    info.nl_assertions = [Obj(met=False, nl_assertion="agent asks first", justification="it did not")]
    info.env_assertions = [Obj(met=False, env_assertion=Obj())]
    info.reward_basis = ["ACTION", "COMMUNICATE", "NL_ASSERTION"]
    reasons = reward_reasons(world)
    assert reasons[0] == "the conversation ended by max_steps, not by the customer"
    assert reasons[1].startswith("the agent never made the call find_user_id_by_name_zip")
    assert reasons[2] == "the agent never told the user 'the refund amount'"
    assert reasons[3] == "not met: agent asks first (judge: it did not)"
    assert reasons[4].startswith("(not scored) environment assertion failed")
    world.ending.simulation.reward_info = None
    assert reward_reasons(world) == ["the conversation ended by max_steps, not by the customer"]


def test_a_task_with_a_message_history_is_refused() -> None:
    class State:
        message_history = ("x",)

    class Task:
        id = "t"
        initial_state = State()

    class Orch:
        task = Task()

    with pytest.raises(ValueError, match="message history"):
        Tau2World.start(Orch(), ViewSpec())


def test_puppet_agent_contract() -> None:
    agent = PuppetAgent([], "policy")
    with pytest.raises(RuntimeError):
        agent.generate_next_message("m", [])
    agent.queued = "reply"
    out, state = agent.generate_next_message("m", [])
    assert out == "reply" and state == ["m", "reply"] and agent.queued is None
    agent.set_seed(3)
    agent.stop()
    assert agent.seed == 3 and not PuppetAgent.is_stop("x")
    assert agent.get_init_state(None) == [] and agent.get_init_state(["a"]) == ["a"]
