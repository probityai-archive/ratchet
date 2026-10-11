"""Run one tau2-bench task under a Ratchet harness, through the agent lane's worker loop.

`Tau2Runner` is a `RunCase`: the improver's evaluator and `evolve` call it like the native case runner. Each
call builds tau2-bench's own text orchestrator for the task (environment, user simulator, step and error
limits, seed) from a `TextRunConfig`, puts a `PuppetAgent` in the agent's seat, and hands the worker loop
a `Tau2World`, the `ConversationModel` wrapper and tau2-bench's reward as the grader. Guards run in the
agent lane's sandboxed guard runtime, as in every Ratchet run.

The finished `SimulationRun` (tau2-bench's trajectory format, scored by tau2-bench) is kept on the runner
by run id, so a caller can write it next to the Ratchet trace.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import ratchet.bench.tau2  # noqa: F401  # sets TAU2_DATA_DIR before tau2-bench is imported
from ratchet.agent.guards import GuardRuntime
from ratchet.agent.model import ChatModel
from ratchet.agent.worker import RunLimits, run_case
from ratchet.bench.tau2.conversation import CLOSED, ConversationModel
from ratchet.bench.tau2.tasks import DEFAULT_SEED, Domain, Tau2Case, load_tasks
from ratchet.bench.tau2.world import RESPOND, VIEWS, PuppetAgent, Tau2World, outcome_of
from ratchet.cases.model import Case, Expect
from ratchet.types import Cost, Harness, Trace
from ratchet.world import World

MAX_STEPS = 200  # tau2-bench's DEFAULT_MAX_STEPS: orchestrator steps (agent, user and tool messages)
MAX_ERRORS = 10  # tau2-bench's DEFAULT_MAX_ERRORS
USER_PROVIDER = "openai"


def configure_user_endpoint(base_url: str, api_key: str) -> None:
    """Point tau2-bench's litellm calls (user simulator, assertion judge) at an OpenAI-compatible endpoint.
    The key is set on litellm only, never in a run config, so it is never written to a record."""
    import litellm

    litellm.api_base = base_url
    litellm.api_key = api_key
    litellm.suppress_debug_info = True


def litellm_model(model: str) -> str:
    """A model name as litellm routes it to an OpenAI-compatible endpoint."""
    return model if "/" in model else f"{USER_PROVIDER}/{model}"


def run_config(
    domain: Domain,
    *,
    agent_model: str,
    user_model: str,
    temperature: float | None = 0.0,
    user_temperature: float = 0.0,
    trials: int = 1,
    seed: int = DEFAULT_SEED,
) -> Any:
    """tau2-bench's `TextRunConfig` for a run: its default user simulator, step and error limits."""
    from tau2.data_model.simulation import TextRunConfig  # type: ignore[import-untyped]  # no type info

    return TextRunConfig(
        domain=domain,
        agent="llm_agent",
        llm_agent=agent_model,
        llm_args_agent={} if temperature is None else {"temperature": temperature},
        user="user_simulator",
        llm_user=litellm_model(user_model),
        llm_args_user={"temperature": user_temperature},
        num_trials=trials,
        seed=seed,
        max_steps=MAX_STEPS,
        max_errors=MAX_ERRORS,
        task_split_name="base",
    )


def agent_system_prompt(domain: Domain) -> str:
    """The system prompt tau2-bench's half-duplex LLM agent uses for `domain`."""
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT  # type: ignore[import-untyped]
    from tau2.runner.build import build_environment  # type: ignore[import-untyped]  # no type info

    policy = build_environment(domain).get_policy()
    return str(SYSTEM_PROMPT.format(domain_policy=policy, agent_instruction=AGENT_INSTRUCTION))


def tool_names(domain: Domain) -> list[str]:
    from tau2.runner.build import build_environment

    return sorted(str(t.name) for t in build_environment(domain).get_tools())


def seed_harness(domain: Domain = "retail") -> Harness:
    """Harness v0 for a tau2-bench domain: tau2-bench's own agent prompt (instructions and the domain
    policy), no procedures, no guards, every tool of the domain plus `respond`."""
    return Harness.build(
        parent=None,
        system_prompt=agent_system_prompt(domain),
        procedures=[],
        guards=[],
        tool_allowlist=[*tool_names(domain), RESPOND],
        created_by="seed",
    )


def agent_cost(trace_events: list[Any]) -> Cost:
    """Cost of the agent's real model calls (the wrapper's closing turn is not a model call)."""
    models = [e for e in trace_events if e.kind == "model" and e.payload.get("finish_reason") != CLOSED]
    usage = [e.payload.get("usage", {}) for e in models]
    return Cost(
        model_calls=len(models),
        input_tokens=sum(int(u.get("input_tokens", 0)) for u in usage),
        output_tokens=sum(int(u.get("output_tokens", 0)) for u in usage),
        tool_calls=sum(1 for e in trace_events if e.kind == "tool_call"),
        wall_s=round(trace_events[-1].ts - trace_events[0].ts, 3) if trace_events else 0.0,
    )


def attach_usage(simulation: Any, trace: Trace) -> None:
    """Copy each agent model call's token usage onto the assistant message it produced, keyed by call id.

    A turn that called several tools carries its usage on the first; `cost` stays unset because the
    endpoint reports tokens, not prices."""
    usage: dict[str, dict[str, int]] = {}
    for e in trace.events:
        if e.kind == "model" and e.payload.get("tool_calls"):
            u = e.payload.get("usage", {})
            first = str(e.payload["tool_calls"][0]["call_id"])
            usage[first] = {
                "prompt_tokens": int(u.get("input_tokens", 0)),
                "completion_tokens": int(u.get("output_tokens", 0)),
            }
    for m in simulation.messages or []:
        if getattr(m, "role", None) != "assistant":
            continue
        key = (m.raw_data or {}).get("ratchet_call_id") or (m.tool_calls[0].id if m.tool_calls else None)
        if key in usage:
            m.usage = usage[key]


def build_orchestrator(config: Any, task: Any, seed: int | None, simulation_id: str) -> Any:
    """tau2-bench's own text orchestrator for `task`, with a `PuppetAgent` in the agent's seat."""
    from tau2.runner.build import build_text_orchestrator

    orch = build_text_orchestrator(config, task, seed=seed, simulation_id=simulation_id)
    env = orch.environment
    orch.agent = PuppetAgent(env.get_tools(), env.get_policy())
    return orch


OrchestratorFactory = Callable[[Any, Any, "int | None", str], Any]


@dataclass
class Tau2Runner:
    """`RunCase` for tau2-bench. `config` is the run's `TextRunConfig`; `model` is the agent's chat model."""

    config: Any
    runtime: GuardRuntime | None = None
    factory: OrchestratorFactory = build_orchestrator
    user_api_base: str | None = None
    simulations: dict[str, Any] = field(default_factory=dict)
    _tasks: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def pop_simulation(self, run_id: str) -> Any:
        with self._lock:
            return self.simulations.pop(run_id, None)

    def task(self, task_id: str) -> Any:
        with self._lock:
            if not self._tasks:
                self._tasks = {str(t.id): t for t in load_tasks(self.config.domain)}
        return self._tasks[task_id]

    def __call__(self, harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
        tcase = cast(Tau2Case, case)
        orch = self.factory(self.config, self.task(tcase.task_id), tcase.seed, run_id)
        if self.user_api_base is not None:
            # This conversation's user simulator calls the same endpoint as its agent; the run's recorded
            # user configuration (model and sampling arguments) is unchanged.
            orch.user.llm_args["api_base"] = self.user_api_base
        world = Tau2World.start(orch, VIEWS[tcase.domain])
        ratchet_case = Case(
            id=tcase.id,
            family=f"tau2-{tcase.domain}",
            variant=tcase.task_id,
            split=tcase.split,
            must_succeed=tcase.must_succeed,
            customer_message=world.opening,
            initial_state={},
            faults=[],
            grader=Expect(),
            ticket_id=tcase.id,
            kind="must_succeed",
        )
        agent = model if isinstance(model, ConversationModel) else ConversationModel(cast(ChatModel, model))

        def grader(_case: Case, _world: World, trace: Trace) -> Any:
            final = trace.events[-1].payload if trace.events else {}
            if world.ending is None:
                world.abort(str(final.get("reason", "the worker stopped first")))
            return outcome_of(world, agent_cost(trace.events))

        factory = cast(Callable[[Case], World], lambda _case: world)
        trace = run_case(
            harness,
            ratchet_case,
            agent,
            None,
            run_id=run_id,
            limits=RunLimits(max_steps=MAX_STEPS, max_tokens=10**10),
            runtime=self.runtime,
            world_factory=factory,
            grader=grader,
        )
        assert world.ending is not None
        simulation = world.ending.simulation
        simulation.trial = tcase.trial
        attach_usage(simulation, trace)
        with self._lock:
            self.simulations[run_id] = simulation
        return trace


@dataclass
class EndpointPool:
    """A `RunCase` that spreads conversations over endpoints serving the same model. Each conversation goes
    whole (agent and user simulator) to the endpoint with the fewest conversations in flight; the `model`
    argument is ignored in favour of that endpoint's client."""

    runners: list[Tau2Runner]
    models: list[Any]
    inflight: list[int] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if not self.runners or len(self.runners) != len(self.models):
            raise ValueError("one runner and one model per endpoint")
        self.inflight = [0] * len(self.runners)

    @property
    def runtime(self) -> GuardRuntime | None:
        return self.runners[0].runtime

    @runtime.setter
    def runtime(self, value: GuardRuntime | None) -> None:
        for r in self.runners:
            r.runtime = value

    def pop_simulation(self, run_id: str) -> Any:
        for r in self.runners:
            sim = r.pop_simulation(run_id)
            if sim is not None:
                return sim
        return None

    def __call__(self, harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
        with self._lock:
            i = min(range(len(self.inflight)), key=lambda j: self.inflight[j])
            self.inflight[i] += 1
        try:
            return self.runners[i](harness, case, self.models[i], store, run_id=run_id)
        finally:
            with self._lock:
                self.inflight[i] -= 1
