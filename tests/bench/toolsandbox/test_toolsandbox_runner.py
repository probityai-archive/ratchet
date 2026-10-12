from __future__ import annotations

import json
from importlib import metadata
from typing import Any

import pytest
from ratchet_cases.types import Trace
from toolsandbox_testkit import Steady, user_factory

from ratchet.agent.guards import GuardRuntime
from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.bench.toolsandbox import runner as runner_module
from ratchet.bench.toolsandbox.cases import ToolSandboxCase, scenarios
from ratchet.bench.toolsandbox.grading import Graded
from ratchet.bench.toolsandbox.replay import END
from ratchet.bench.toolsandbox.runner import (
    NOT_RUN,
    ToolSandboxRunner,
    agent_prompt,
    agent_tool_names,
    endpoint_user,
    is_not_run,
    seed_harness,
    tool_sandbox_commit,
)
from ratchet.bench.toolsandbox.user import EndpointUser
from ratchet.bench.toolsandbox.world import RESPOND, ToolSandboxWorld
from ratchet.types import Guard, Harness

PINNED = "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1"


def _case(cases: list[ToolSandboxCase], name: str) -> ToolSandboxCase:
    return next(c for c in cases if c.name == name)


def _passing_model() -> ScriptedModel:
    return ScriptedModel(
        [
            scripted_turn(tool_calls=[("set_cellular_service_status", {"on": False})]),
            scripted_turn("Cellular service is turned off"),
        ]
    )


def _run(
    cases: list[ToolSandboxCase],
    model: Any,
    harness: Harness | None = None,
    name: str = "cellular_off",
    **kw: Any,
) -> Trace:
    runner = ToolSandboxRunner(user=user_factory([END]), **kw)
    return runner(harness or seed_harness(), _case(cases, name), model, None, run_id="run-1")


def test_the_seed_harness_is_the_benchmarks_prompt_and_every_tool_the_agent_can_see() -> None:
    harness = seed_harness()
    assert harness.created_by == "seed" and harness.parent is None
    assert harness.system_prompt == agent_prompt()
    assert harness.procedures == [] and harness.guards == []
    assert RESPOND in harness.tool_allowlist
    names = agent_tool_names()
    assert "set_cellular_service_status" in names and "setting_5" in names
    assert "end_conversation" not in harness.tool_allowlist
    assert set(names) <= set(harness.tool_allowlist)
    assert harness.verify_digest()


def test_every_scenarios_tools_are_on_the_seed_allowlist() -> None:
    allowed = set(seed_harness().tool_allowlist)
    for name in (
        "cellular_off",
        "cellular_off_all_tools",
        "cellular_off_3_distraction_tools_tool_name_scrambled",
    ):
        world = ToolSandboxWorld(scenarios()[name], name, user_factory())
        assert set(world.tool_names()) <= allowed


def test_a_run_that_meets_every_milestone_passes_with_the_benchmarks_score(
    all_cases: list[ToolSandboxCase],
) -> None:
    seen: list[tuple[str, Graded]] = []
    model = _passing_model()
    trace = _run(all_cases, model, on_graded=lambda case, _trace, graded: seen.append((case.id, graded)))
    assert trace.outcome.passed and trace.outcome.reasons == []
    assert trace.case_id == "toolsandbox-cellular_off" and trace.run_id == "run-1"
    assert [(i, g.similarity) for i, g in seen] == [("toolsandbox-cellular_off", 1.0)]
    assert trace.outcome.cost.model_calls == 2 and trace.outcome.cost.tool_calls == 2
    kinds = [e.kind for e in trace.events]
    assert kinds.count("tool_call") == 2 and kinds[-1] == "final"


def test_the_agent_model_sees_a_plain_transcript_without_the_respond_tool(
    all_cases: list[ToolSandboxCase],
) -> None:
    model = _passing_model()
    _run(all_cases, model)
    messages, tools = model.requests[1]
    assert [t["function"]["name"] for t in tools] == ["set_cellular_service_status"]
    assert messages[0]["role"] == "system" and messages[0]["content"] == agent_prompt()
    assert messages[1] == {"role": "user", "content": "Turn off cellular"}
    assert messages[-1]["role"] == "tool" and messages[-1]["content"] == "None"


def test_a_run_that_misses_the_milestones_fails_with_reasons(all_cases: list[ToolSandboxCase]) -> None:
    trace = _run(all_cases, Steady())
    assert not trace.outcome.passed
    assert trace.outcome.reasons[0] == 'milestone 1 not met: SETTING has {"cellular": false}'


def test_a_scenario_that_needs_a_missing_key_is_not_run(all_cases: list[ToolSandboxCase]) -> None:
    case = next(c for c in all_cases if c.not_run)
    model = Steady()
    trace = ToolSandboxRunner(user=user_factory())(seed_harness(), case, model, None, run_id="run-2")
    assert model.requests == 0
    assert is_not_run(trace)
    assert trace.outcome.passed is False
    assert trace.outcome.reasons == [f"{NOT_RUN}: {case.not_run}"]
    assert trace.events[0].kind == "final" and trace.events[0].payload["complete"] is False
    assert trace.outcome.cost.model_calls == 0


def test_a_run_is_not_a_not_run(all_cases: list[ToolSandboxCase]) -> None:
    assert not is_not_run(_run(all_cases, Steady()))


def test_a_guard_sees_the_state_view_and_can_stop_a_call(all_cases: list[ToolSandboxCase]) -> None:
    guard = Guard(
        "keep-cellular",
        "Block turning cellular off while it is on.",
        "def check(call, view, history):\n"
        "    if call.name == 'set_cellular_service_status' and view['setting'][0]['cellular']:\n"
        "        return 'cellular is on; ask the user first'\n"
        "    return None\n",
        ["f1"],
    )
    base = seed_harness()
    harness = Harness.build(
        parent=base.version,
        system_prompt=base.system_prompt,
        procedures=[],
        guards=[guard],
        tool_allowlist=base.tool_allowlist,
        created_by="improver",
    )
    with GuardRuntime() as runtime:
        trace = _run(all_cases, _passing_model(), harness, runtime=runtime)
    blocks = [e for e in trace.events if e.kind == "guard_block"]
    assert len(blocks) == 1 and blocks[0].payload["source"] == "guard"
    assert "cellular is on" in blocks[0].payload["message"]
    assert not trace.outcome.passed


def test_the_world_is_released_after_a_run(
    all_cases: list[ToolSandboxCase], monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[int] = []

    class Watched(ToolSandboxWorld):
        def close(self) -> None:
            closed.append(1)
            super().close()

    monkeypatch.setattr(runner_module, "ToolSandboxWorld", Watched)
    _run(all_cases, Steady())
    assert closed == [1]


def test_endpoint_user_makes_the_benchmarks_simulated_user() -> None:
    factory = endpoint_user("http://endpoint.test/v1", "k", "the-model", 0.0)
    user = factory()
    try:
        assert isinstance(user, EndpointUser)
        assert user.model_name == "the-model" and user.temperature == 0.0
    finally:
        user.teardown()


def test_the_installed_commit_is_the_pinned_one() -> None:
    assert tool_sandbox_commit() == PINNED


def test_commit_lookup_reports_a_missing_package_and_a_missing_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> Any:
        raise metadata.PackageNotFoundError("tool-sandbox")

    monkeypatch.setattr(metadata, "distribution", missing)
    assert tool_sandbox_commit() == "not installed"

    class Dist:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def read_text(self, _name: str) -> str | None:
            return self.text

    monkeypatch.setattr(metadata, "distribution", lambda _n: Dist(None))
    assert tool_sandbox_commit() == "unknown"
    monkeypatch.setattr(
        metadata, "distribution", lambda _n: Dist(json.dumps({"vcs_info": {"commit_id": "abc"}}))
    )
    assert tool_sandbox_commit() == "abc"


def test_scenarios_that_disagree_on_the_agents_prompt_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = iter(["a", "b"])
    monkeypatch.setattr(runner_module, "world_prompt", lambda _ctx: next(prompts))
    monkeypatch.setattr(
        runner_module,
        "scenarios",
        lambda: {"x": scenarios()["cellular_off"], "y": scenarios()["get_cellular"]},
    )
    with pytest.raises(ValueError, match="disagree"):
        agent_prompt()
