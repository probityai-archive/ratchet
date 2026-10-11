"""Run one benchmark scenario under a Ratchet harness, through the agent lane's worker loop.

`ToolSandboxRunner` is a `RunCase`: the improver's evaluator and the trial command call it like the native
case runner. Each call builds a fresh `ToolSandboxWorld` for the scenario (a copy of its starting context,
with the benchmark's user simulator), wraps the agent's chat model so a plain reply is a message to the
user, hands the worker loop both, and grades the finished conversation with the benchmark's milestone and
minefield grader.
Guards run in the agent lane's sandboxed guard runtime, as in every Ratchet run.

A scenario marked `not_run` (it needs a third-party key that is absent) is not played: its trace is one
`final` event saying so, and its outcome is a failure whose single reason starts with `NOT RUN`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any, cast

from tool_sandbox.common.execution_context import (
    TOOL_AUGMENTATION_TYPE,
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    ScenarioCategories,
)
from tool_sandbox.roles.base_role import BaseRole

from ratchet.agent.guards import GuardRuntime
from ratchet.agent.model import ChatModel
from ratchet.agent.worker import RunLimits, run_case
from ratchet.bench.tau.conversation import ConversationModel
from ratchet.bench.tau.runner import agent_cost
from ratchet.bench.toolsandbox.cases import ToolSandboxCase, scenarios
from ratchet.bench.toolsandbox.gate import GATE
from ratchet.bench.toolsandbox.grading import Graded
from ratchet.bench.toolsandbox.user import EndpointUser
from ratchet.bench.toolsandbox.world import RESPOND, ToolSandboxWorld, UserFactory
from ratchet.cases.model import Case, Expect
from ratchet.types import Cost, Event, Harness, Outcome, Trace
from ratchet.world import World

NOT_RUN = "NOT RUN"
OnGraded = Callable[[ToolSandboxCase, Trace, Graded], None]


def world_prompt(context: ExecutionContext) -> str:
    """The system messages addressed to the agent in a scenario's starting context."""
    sandbox = context.get_database(DatabaseNamespace.SANDBOX)
    rows = sandbox.filter((sandbox["sender"] == RoleType.SYSTEM) & (sandbox["recipient"] == RoleType.AGENT))
    return "\n".join(str(r["content"]) for r in rows.to_dicts())


def agent_prompt() -> str:
    """The benchmark's system message to the agent, which every scenario shares."""
    prompts = {world_prompt(s.starting_context) for s in scenarios().values()}
    if len(prompts) != 1:
        raise ValueError(f"scenarios disagree on the agent's system message: {sorted(prompts)}")
    return prompts.pop()


def agent_tool_names() -> list[str]:
    """Every tool name an agent can be shown in any scenario: the benchmark's names and its scrambled ones."""

    class _Agent(BaseRole):
        role_type = RoleType.AGENT

    names: set[str] = set()
    plain: list[TOOL_AUGMENTATION_TYPE] = []
    scrambled: list[TOOL_AUGMENTATION_TYPE] = [ScenarioCategories.TOOL_NAME_SCRAMBLED]
    for augmentation in (plain, scrambled):
        with GATE.held(ExecutionContext(tool_augmentation_list=augmentation)):
            names |= set(_Agent.get_available_tools())
    return sorted(names)


def seed_harness() -> Harness:
    """Harness v0: the benchmark's own agent prompt, no procedures, no guards, every tool the agent can see,
    plus `respond`."""
    return Harness.build(
        parent=None,
        system_prompt=agent_prompt(),
        procedures=[],
        guards=[],
        tool_allowlist=[*agent_tool_names(), RESPOND],
        created_by="seed",
    )


def rebase(source: Harness, target: Harness) -> Harness:
    """`source`'s guards and procedures appended to `target` (this benchmark's seed harness).

    What an improver learned travels, as executable and instructional changes; the benchmark's own prompt and
    the tools its scenarios offer stay, since the source harness was admitted for another world's tools.
    """
    have_g = {g.id for g in target.guards}
    have_p = {p.id for p in target.procedures}
    return target.child(
        guards=[*target.guards, *(g for g in source.guards if g.id not in have_g)],
        procedures=[*target.procedures, *(p for p in source.procedures if p.id not in have_p)],
        created_by=source.created_by,
    )


def endpoint_user(base_url: str, api_key: str, model: str, temperature: float | None = None) -> UserFactory:
    """A factory for the benchmark's simulated user, answering from `model` at `base_url`."""
    return lambda: cast(BaseRole, EndpointUser(base_url, api_key, model, temperature))


def tool_sandbox_commit() -> str:
    """The tool-sandbox commit installed, from the package's PEP 610 record."""
    try:
        raw = metadata.distribution("tool-sandbox").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return "not installed"
    info = json.loads(raw or "{}")
    return str(info.get("vcs_info", {}).get("commit_id", "unknown"))


def not_run_trace(harness: Harness, case: ToolSandboxCase, model_name: str, run_id: str) -> Trace:
    reason = f"{NOT_RUN}: {case.not_run}"
    now = time.time()
    final = Event(0, "final", {"reason": "not run", "content": case.not_run, "complete": False}, now)
    return Trace(run_id, case.id, harness.version, model_name, [final], Outcome(False, [reason], "", Cost()))


def is_not_run(trace: Trace) -> bool:
    return any(r.startswith(NOT_RUN) for r in trace.outcome.reasons)


@dataclass
class ToolSandboxRunner:
    """`RunCase` for the benchmark: `model` is the agent's chat model, `user` makes the simulated user."""

    user: UserFactory
    max_actions: int = 30  # the benchmark's own bound on messages after the setup
    runtime: GuardRuntime | None = None
    on_graded: OnGraded | None = None

    def __call__(self, harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
        tcase = cast(ToolSandboxCase, case)
        if tcase.not_run:
            return not_run_trace(harness, tcase, str(model.name), run_id)
        world = ToolSandboxWorld(scenarios()[tcase.name], tcase.name, self.user)
        ratchet_case = Case(
            id=tcase.id,
            family=tcase.family,
            variant=tcase.base,
            split=tcase.split,
            must_succeed=tcase.must_succeed,
            customer_message=world.opening,
            initial_state={},
            faults=[],
            grader=Expect(),
            ticket_id=tcase.id,
            kind="must_succeed",
        )
        agent = (
            model
            if isinstance(model, ConversationModel)
            else ConversationModel(cast(ChatModel, model), self.max_actions)
        )

        def grader(_case: Case, _world: World, trace: Trace) -> Outcome:
            return world.outcome(agent_cost(trace.events))

        factory = cast(Callable[[Case], World], lambda _case: world)
        try:
            trace = run_case(
                harness,
                ratchet_case,
                agent,
                None,
                run_id=run_id,
                limits=RunLimits(max_steps=self.max_actions + 1, max_tokens=10**9),
                runtime=self.runtime,
                world_factory=factory,
                grader=grader,
            )
        finally:
            world.close()
        if self.on_graded is not None:
            self.on_graded(tcase, trace, world.graded())
        return trace
