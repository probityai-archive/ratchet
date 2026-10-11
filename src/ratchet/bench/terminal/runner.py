"""Run one terminal task under a Ratchet harness, through the agent lane's worker loop.

`TerminalRunner` is a `RunCase`, so the improver's evaluator and `evolve` call it like the native case runner.
Each call starts a fresh container from the task's image, hands the worker loop a `TerminalWorld` over it,
grades the final container with the task's own tests, and removes the container. Guards run in the agent
lane's sandboxed guard runtime, as in every Ratchet run.

The agent's time limit is the task's agent timeout. When it runs out, `DeadlineModel` ends the agent's turn
with a plain reply instead of calling the model, so the worker records a finished run and the tests still
decide the outcome, as they would for an agent stopped at its limit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any, cast

from ratchet.agent.guards import GuardRuntime
from ratchet.agent.model import ChatModel, Message, ModelTurn, ToolSpec
from ratchet.agent.worker import RunLimits, run_case
from ratchet.bench.terminal.container import Container, DockerContainer
from ratchet.bench.terminal.grade import outcome_of, verify
from ratchet.bench.terminal.tasks import TaskSpec, TerminalCase
from ratchet.bench.terminal.world import TOOL_NAMES, TerminalWorld
from ratchet.cases.model import Case, Expect
from ratchet.types import Cost, Event, Harness, Outcome, Trace
from ratchet.world import World

AGENT_TIMEOUT = "agent_timeout"
TICKET = "terminal-task"
MAX_STEPS = 400

SEED_PROMPT = """You are an autonomous agent working on a task inside a Linux container, through a terminal.
No person will answer questions or check in on you: decide for yourself and keep working until the task
is done.

Use run_command to run shell commands and write_file to create or replace files. Look at the environment
before you change it, run what you build to check that it works, and leave your result in place: after you
finish, the task is graded by tests that run in this same container against its final state.

When the task is complete, reply with a short summary and no tool call. That reply ends your work for good."""


def seed_harness() -> Harness:
    """Harness v0: a generic terminal-agent prompt, no procedures, no guards, both tools."""
    return Harness.build(
        parent=None,
        system_prompt=SEED_PROMPT,
        procedures=[],
        guards=[],
        tool_allowlist=list(TOOL_NAMES),
        created_by="seed",
    )


def package_version() -> str:
    try:
        return metadata.version("terminal-bench")
    except metadata.PackageNotFoundError:
        return "not installed"


class DeadlineModel:
    """A `ChatModel` that stops calling the inner model once the world's time limit has passed."""

    def __init__(self, inner: ChatModel, world: TerminalWorld) -> None:
        self.inner = inner
        self.world = world

    @property
    def name(self) -> str:
        return self.inner.name

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        if self.world.expired:
            note = "The time limit for this task has passed; work stops here."
            return ModelTurn(
                message={"role": "assistant", "content": note},
                content=note,
                tool_calls=[],
                finish_reason=AGENT_TIMEOUT,
                input_tokens=0,
                output_tokens=0,
                latency_s=0.0,
            )
        return self.inner.complete(messages, tools)


def agent_cost(events: Sequence[Event]) -> Cost:
    """Cost of the agent's real model calls (a deadline reply is not a model call)."""
    models = [e for e in events if e.kind == "model" and e.payload.get("finish_reason") != AGENT_TIMEOUT]
    usage = [e.payload.get("usage", {}) for e in models]
    return Cost(
        model_calls=len(models),
        input_tokens=sum(int(u.get("input_tokens", 0)) for u in usage),
        output_tokens=sum(int(u.get("output_tokens", 0)) for u in usage),
        tool_calls=sum(1 for e in events if e.kind == "tool_call"),
        wall_s=round(events[-1].ts - events[0].ts, 3) if events else 0.0,
    )


def start_container(task: TaskSpec) -> Container:
    return DockerContainer.start(task.image, cpus=task.cpus, memory_mb=task.memory_mb)


def ratchet_case(case: TerminalCase) -> Case:
    task = case.task
    return Case(
        id=case.id,
        family="terminal",
        variant=task.category or "task",
        split=case.split,
        must_succeed=True,
        customer_message=task.instruction,
        initial_state={},
        faults=[],
        grader=Expect(),
        ticket_id=TICKET,
        kind="must_succeed",
    )


@dataclass
class TerminalRunner:
    """`RunCase` for terminal tasks. `model` is the agent's plain tool-calling chat model."""

    container_factory: Callable[[TaskSpec], Container] = start_container
    runtime: GuardRuntime | None = None
    max_steps: int = MAX_STEPS
    clock: Callable[[], float] | None = None

    def __call__(self, harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
        tcase = cast(TerminalCase, case)
        task = tcase.task
        container = self.container_factory(task)
        try:
            if self.clock is None:
                world = TerminalWorld(container, task.name, task.agent_timeout_s)
            else:
                world = TerminalWorld(container, task.name, task.agent_timeout_s, clock=self.clock)
            agent = DeadlineModel(cast(ChatModel, model), world)

            def grader(_case: Case, _world: World, trace: Trace) -> Outcome:
                return outcome_of(verify(container, task), world, agent_cost(trace.events))

            # The worker builds its world through `world_factory`; a `TerminalWorld` provides every method the
            # worker calls on a world (tools, tool_names, call, view().to_json(), digest).
            factory = cast(Callable[[Case], World], lambda _case: world)
            return run_case(
                harness,
                ratchet_case(tcase),
                agent,
                None,
                run_id=run_id,
                limits=RunLimits(max_steps=self.max_steps, max_tokens=10**9),
                runtime=self.runtime,
                world_factory=factory,
                grader=grader,
            )
        finally:
            container.close()
