"""Run one tau-bench retail task under a Ratchet harness, through the agent lane's worker loop.

`TauRunner` is a `RunCase`: the improver's evaluator and `evolve` call it like the native case runner. Each
call builds a fresh tau-bench environment for the task, lets tau-bench's user simulator open the
conversation, and hands the worker loop a `TauWorld`, the `ConversationModel` wrapper and tau-bench's reward
as the grader. Guards run in the agent lane's sandboxed guard runtime, as in every Ratchet run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any, cast

from ratchet.agent.guards import GuardRuntime
from ratchet.agent.model import ChatModel
from ratchet.agent.worker import RunLimits, run_case
from ratchet.bench.tau.conversation import CLOSED, OUT_OF_ACTIONS, ConversationModel
from ratchet.bench.tau.tasks import VIEW_TABLES, Domain, TauCase
from ratchet.bench.tau.world import RESPOND, TauWorld, outcome_of
from ratchet.cases.model import Case, Expect
from ratchet.types import Cost, Harness, Trace
from ratchet.world import World

MAX_ACTIONS = 30  # tau-bench's `max_num_steps` for the tool-calling agent
USER_PROVIDER = "openai"


def configure_user_endpoint(base_url: str, api_key: str, *, retries: int = 3) -> None:
    """Point tau-bench's user simulator (litellm, provider `openai`) at an OpenAI-compatible endpoint."""
    import litellm

    litellm.api_base = base_url
    litellm.api_key = api_key
    litellm.num_retries = retries
    litellm.suppress_debug_info = True


def _env_class(domain: Domain) -> Any:
    if domain == "airline":
        from tau_bench.envs.airline import MockAirlineDomainEnv  # type: ignore[import-untyped]  # no type info

        return MockAirlineDomainEnv
    from tau_bench.envs.retail import MockRetailDomainEnv  # type: ignore[import-untyped]  # no type info

    return MockRetailDomainEnv


def make_env(case: TauCase, user_model: str) -> tuple[Any, str]:
    """A tau-bench environment reset to the case's task, and the user's opening message."""
    env = _env_class(case.domain)(
        user_strategy="llm",
        user_model=user_model,
        user_provider=USER_PROVIDER,
        task_split=case.tau_split,
        task_index=case.index,
    )
    reset = env.reset(task_index=case.index)
    return env, str(reset.observation)


EnvFactory = Callable[[TauCase, str], tuple[Any, str]]


def tau_wiki(domain: Domain = "retail") -> str:
    if domain == "airline":
        from tau_bench.envs.airline.wiki import WIKI as AIRLINE  # type: ignore[import-untyped]  # no type info

        return str(AIRLINE)
    from tau_bench.envs.retail.wiki import WIKI  # type: ignore[import-untyped]  # no type info

    return str(WIKI)


def tau_tool_names(domain: Domain = "retail") -> list[str]:
    if domain == "airline":
        from tau_bench.envs.airline.tools import ALL_TOOLS as AIRLINE  # type: ignore[import-untyped]  # no type info

        return sorted(str(t.get_info()["function"]["name"]) for t in AIRLINE)
    from tau_bench.envs.retail.tools import ALL_TOOLS  # type: ignore[import-untyped]  # no type info

    return sorted(str(t.get_info()["function"]["name"]) for t in ALL_TOOLS)


def seed_harness(domain: Domain = "retail") -> Harness:
    """Harness v0 for a tau-bench domain: tau-bench's own agent prompt (the domain policy), no procedures,
    no guards, every tau-bench tool of the domain plus `respond`."""
    return Harness.build(
        parent=None,
        system_prompt=tau_wiki(domain),
        procedures=[],
        guards=[],
        tool_allowlist=[*tau_tool_names(domain), RESPOND],
        created_by="seed",
    )


def agent_cost(trace_events: list[Any]) -> Cost:
    """Cost of the agent's real model calls (the wrapper's closing turns are not model calls)."""
    models = [
        e
        for e in trace_events
        if e.kind == "model" and e.payload.get("finish_reason") not in (CLOSED, OUT_OF_ACTIONS)
    ]
    usage = [e.payload.get("usage", {}) for e in models]
    return Cost(
        model_calls=len(models),
        input_tokens=sum(int(u.get("input_tokens", 0)) for u in usage),
        output_tokens=sum(int(u.get("output_tokens", 0)) for u in usage),
        tool_calls=sum(1 for e in trace_events if e.kind == "tool_call"),
        wall_s=round(trace_events[-1].ts - trace_events[0].ts, 3) if trace_events else 0.0,
    )


def tau_bench_commit() -> str:
    """The tau-bench commit installed, from the package's PEP 610 record."""
    try:
        raw = metadata.distribution("tau-bench").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return "not installed"
    info = json.loads(raw or "{}")
    return str(info.get("vcs_info", {}).get("commit_id", "unknown"))


@dataclass
class TauRunner:
    """`RunCase` for tau-bench; the case names its domain. `model` is the agent's plain chat model."""

    user_model: str
    env_factory: EnvFactory = make_env
    max_actions: int = MAX_ACTIONS
    runtime: GuardRuntime | None = None

    def __call__(self, harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
        tau_case = cast(TauCase, case)
        env, opening = self.env_factory(tau_case, self.user_model)
        world = TauWorld(env, opening, VIEW_TABLES[tau_case.domain])
        ratchet_case = Case(
            id=tau_case.id,
            family=f"tau-{tau_case.domain}",
            variant=tau_case.tau_split,
            split=tau_case.split,
            must_succeed=tau_case.must_succeed,
            customer_message=opening,
            initial_state={},
            faults=[],
            grader=Expect(),
            ticket_id=tau_case.id,
            kind="must_succeed",
        )
        agent = (
            model
            if isinstance(model, ConversationModel)
            else ConversationModel(cast(ChatModel, model), self.max_actions)
        )

        def grader(_case: Case, _world: World, trace: Trace) -> Any:
            return outcome_of(world, agent_cost(trace.events))

        # The worker builds its world through `world_factory`; a `TauWorld` provides every method the
        # worker calls on a world (tools, tool_names, call, view().to_json(), digest).
        factory = cast(Callable[[Case], World], lambda _case: world)
        return run_case(
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
