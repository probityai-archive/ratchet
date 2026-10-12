from __future__ import annotations

import copy
import threading
from typing import Any

import pytest
from ratchet_cases.types import Cost, ToolCall
from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
)
from toolsandbox_testkit import make_world, user_factory

from ratchet.bench.toolsandbox.cases import scenarios
from ratchet.bench.toolsandbox.gate import GATE
from ratchet.bench.toolsandbox.replay import END, ReplayUser
from ratchet.bench.toolsandbox.world import ENDED, RESPOND, ToolSandboxWorld


def _call(world: ToolSandboxWorld, name: str, args: dict[str, Any], cid: str = "c") -> Any:
    return world.call(ToolCall(name, args, cid))


def test_the_world_opens_with_the_users_message_and_the_benchmarks_prompt() -> None:
    world = make_world("cellular_off")
    assert world.opening == "Turn off cellular"
    assert world.agent_prompt.startswith("Don't make assumptions about what values to plug into functions.")
    assert not world.done and world.steps == 0


def test_the_tools_are_the_scenarios_tools_plus_respond() -> None:
    world = make_world("cellular_off")
    assert world.tool_names() == [RESPOND, "set_cellular_service_status"]
    assert [t["function"]["name"] for t in world.tools()] == ["set_cellular_service_status", RESPOND]
    world.tools()[0]["function"]["name"] = "changed"
    assert world.tool_names() == [RESPOND, "set_cellular_service_status"]


def test_a_scrambled_scenario_shows_scrambled_names_and_runs_the_real_tool() -> None:
    name = "cellular_off_3_distraction_tools_tool_name_scrambled"
    world = make_world(name)
    shown = set(world.tool_names()) - {RESPOND}
    assert shown and all(n.split("_")[-1].isdigit() for n in shown)
    scrambled = world._context.get_agent_facing_tool_name("set_cellular_service_status")
    assert scrambled in shown
    result = _call(world, scrambled, {"on": False})
    assert result.ok
    assert world.view().to_json()["setting"][0]["cellular"] is False


def test_a_tool_call_changes_the_state_the_view_and_the_digest_show() -> None:
    world = make_world("cellular_off")
    before = world.view().to_json()
    digest = world.digest()
    result = _call(world, "set_cellular_service_status", {"on": False})
    assert (result.ok, result.error) == (True, None)
    assert result.data == {"observation": "None", "done": False}
    after = world.view().to_json()
    assert before["setting"][0]["cellular"] is True and after["setting"][0]["cellular"] is False
    assert after["step"] == 1 and after["message_index"] > before["message_index"]
    assert world.digest() != digest


def test_the_view_holds_the_four_databases_and_is_json() -> None:
    import json

    view = make_world("cellular_off").view().to_json()
    assert [k for k in view if k not in {"step", "done", "message_index", "conversation_active"}] == [
        "setting",
        "contact",
        "messaging",
        "reminder",
    ]
    assert (len(view["setting"]), len(view["contact"]), len(view["messaging"]), len(view["reminder"])) == (
        1,
        4,
        5,
        3,
    )
    assert view["conversation_active"] is True
    assert json.loads(json.dumps(view)) == view


def test_the_view_is_a_copy() -> None:
    world = make_world("cellular_off")
    view = world.view()
    view.to_json()["setting"][0]["cellular"] = "x"
    assert world.view().to_json()["setting"][0]["cellular"] is True


def test_a_failing_tool_returns_the_exception_and_leaves_the_state() -> None:
    world = make_world("cellular_off")
    digest = world.digest()
    result = _call(world, "set_cellular_service_status", {"on": "not a bool"})
    assert result.ok is False and result.error
    assert "TypeError" in result.error or "ValueError" in result.error
    assert result.data is not None and result.data["done"] is False
    assert world.digest() == digest


def test_a_tool_the_scenario_does_not_offer_is_an_error_that_changes_nothing() -> None:
    world = make_world("cellular_off")
    digest = world.digest()
    result = _call(world, "send_message_with_phone_number", {})
    assert (result.ok, result.error) == (False, "Unknown tool send_message_with_phone_number")
    assert world.digest() == digest and world.steps == 1 and not world.done


def test_respond_returns_the_users_reply() -> None:
    world = make_world("cellular_off", ["Thanks, now check wifi."])
    result = _call(world, RESPOND, {"content": "Cellular is off."})
    assert result.ok
    assert result.data == {"observation": "Thanks, now check wifi.", "done": False}
    assert not world.done


def test_the_user_ending_the_conversation_ends_the_world() -> None:
    world = make_world("cellular_off", [END])
    result = _call(world, RESPOND, {"content": "Cellular is off."})
    assert result.ok and result.data == {"observation": "", "done": True}
    assert world.done
    again = _call(world, "set_cellular_service_status", {"on": False}, "c2")
    assert (again.ok, again.error, again.data) == (False, ENDED, None)
    assert _call(world, RESPOND, {"content": "hi"}, "c3").error == ENDED


def test_respond_needs_text() -> None:
    world = make_world("cellular_off")
    result = _call(world, RESPOND, {"content": 7})
    assert (result.ok, result.error) == (False, "respond needs a string 'content'")
    assert not world.done


def test_the_users_simulator_hears_what_the_agent_said() -> None:
    user = ReplayUser(["ok"])
    world = ToolSandboxWorld(scenarios()["cellular_off"], "cellular_off", lambda: user)
    _call(world, RESPOND, {"content": "Cellular service is turned off"})
    assert user.heard == ["Cellular service is turned off"]


def test_the_message_budget_ends_the_conversation_as_the_benchmarks_loop_does() -> None:
    scenario = copy.copy(scenarios()["cellular_off"])
    scenario.max_messages = 2
    world = ToolSandboxWorld(scenario, "cellular_off", user_factory(["more"]))
    result = _call(world, "set_cellular_service_status", {"on": False})
    assert result.ok and result.data is not None and result.data["done"] is True
    assert world.done
    assert _call(world, RESPOND, {"content": "x"}, "c2").error == ENDED


def test_a_call_that_the_budget_stops_before_it_runs_is_reported_as_ended() -> None:
    scenario = copy.copy(scenarios()["cellular_off"])
    scenario.max_messages = 1
    world = ToolSandboxWorld(scenario, "cellular_off", user_factory())
    digest = world.digest()
    result = _call(world, "set_cellular_service_status", {"on": False})
    assert (result.ok, result.error, result.data) == (False, ENDED, None)
    assert world.done and world.digest() == digest


def test_a_reply_that_uses_the_last_message_of_the_budget_ends_the_conversation() -> None:
    scenario = copy.copy(scenarios()["cellular_off"])
    scenario.max_messages = 1
    world = ToolSandboxWorld(scenario, "cellular_off", user_factory(["never heard"]))
    result = _call(world, RESPOND, {"content": "hi"})
    assert result.ok and result.data == {"observation": "", "done": True}


def test_a_scenario_that_does_not_open_with_the_user_is_refused() -> None:
    scenario = copy.deepcopy(scenarios()["cellular_off"])
    scenario.starting_context.add_to_database(
        DatabaseNamespace.SANDBOX, [{"sender": RoleType.SYSTEM, "recipient": RoleType.USER, "content": "x"}]
    )
    with pytest.raises(ValueError, match="does not open with a user message"):
        ToolSandboxWorld(scenario, "odd", user_factory())


def test_close_releases_every_role() -> None:
    closed: list[str] = []

    class Watched(ReplayUser):
        def teardown(self) -> None:
            closed.append("user")

    world = ToolSandboxWorld(scenarios()["cellular_off"], "cellular_off", lambda: Watched())
    world.close()
    assert closed == ["user"]


def test_the_outcome_passes_a_conversation_that_meets_every_milestone() -> None:
    world = make_world("cellular_off", [END])
    _call(world, "set_cellular_service_status", {"on": False}, "c1")
    _call(world, RESPOND, {"content": "Cellular service is turned off"}, "c2")
    cost = Cost(2, 10, 5, 1, 0.5)
    outcome = world.outcome(cost)
    assert outcome.passed and outcome.reasons == [] and outcome.cost is cost
    assert outcome.final_state_digest == world.digest()
    assert world.graded() is world.graded()
    assert world.graded().exact


def test_the_outcome_fails_with_a_reason_per_unmet_milestone() -> None:
    world = make_world("cellular_off", [END])
    _call(world, RESPOND, {"content": "I cannot do that."}, "c1")
    outcome = world.outcome(Cost())
    assert not outcome.passed
    assert outcome.reasons[0] == 'milestone 1 not met: SETTING has {"cellular": false}'
    assert outcome.reasons[1].startswith("milestone 2 not met: SANDBOX has")


def test_a_minefield_reached_is_a_reason_and_zeroes_the_score() -> None:
    world = make_world("find_days_till_holiday_insufficient_information", [END])
    _call(world, "timestamp_diff", {"timestamp_0": 0.0, "timestamp_1": 86400.0})
    _call(world, RESPOND, {"content": "Which Christmas Day do you mean?"}, "c2")
    graded = world.graded()
    assert graded.minefields[0].similarity > 0.0
    assert graded.similarity == 0.0 and graded.milestone_similarity > 0.0
    outcome = world.outcome(Cost())
    assert not outcome.passed
    assert any(r.startswith("minefield 1 reached:") for r in outcome.reasons)


def test_asking_instead_of_guessing_passes_an_insufficient_information_scenario() -> None:
    world = make_world("find_days_till_holiday_insufficient_information", [END])
    _call(world, RESPOND, {"content": "Which year do you mean? I need the holiday's date."}, "c1")
    graded = world.graded()
    assert graded.minefields[0].similarity == 0.0
    assert graded.similarity == graded.milestone_similarity


def test_two_worlds_in_two_threads_do_not_see_each_other() -> None:
    started, proceed = threading.Event(), threading.Event()
    results: dict[str, Any] = {}

    class Slow(ReplayUser):
        def respond(self, ending_index: int | None = None) -> None:
            with GATE.released():
                started.set()
                assert proceed.wait(30)
            super().respond(ending_index)

    slow = ToolSandboxWorld(scenarios()["cellular_off"], "cellular_off", lambda: Slow(["fine"]))
    quick = make_world("cellular_off")

    def talk() -> None:
        results["slow"] = _call(slow, RESPOND, {"content": "hello"})

    t = threading.Thread(target=talk)
    t.start()
    assert started.wait(30)
    results["quick"] = _call(quick, "set_cellular_service_status", {"on": False})
    proceed.set()
    t.join(timeout=60)
    assert results["quick"].ok
    assert results["slow"].data == {"observation": "fine", "done": False}
    assert quick.view().to_json()["setting"][0]["cellular"] is False
    assert slow.view().to_json()["setting"][0]["cellular"] is True


def test_an_error_inside_the_user_simulator_propagates() -> None:
    class Broken(ReplayUser):
        def respond(self, ending_index: int | None = None) -> None:
            raise RuntimeError("endpoint down")

    world = ToolSandboxWorld(scenarios()["cellular_off"], "cellular_off", lambda: Broken())
    with pytest.raises(RuntimeError, match="endpoint down"):
        _call(world, RESPOND, {"content": "hi"})
    # the gate is free again for the next caller
    assert make_world("cellular_off").view().to_json()["step"] == 0


def test_a_scenario_with_no_message_budget_ends_on_the_first_call() -> None:
    scenario = copy.copy(scenarios()["cellular_off"])
    scenario.max_messages = 0
    world = ToolSandboxWorld(scenario, "cellular_off", user_factory())
    assert not world.done
    assert _call(world, "set_cellular_service_status", {"on": False}).error == ENDED
    assert world.done
    again = ToolSandboxWorld(scenario, "cellular_off", user_factory())
    assert _call(again, RESPOND, {"content": "hi"}).error == ENDED and again.done
