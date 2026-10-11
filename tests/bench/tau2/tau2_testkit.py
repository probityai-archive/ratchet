"""Shared helpers for the tau2-bench adapter tests: real tau2-bench orchestrators with a scripted customer."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Any

from loguru import logger
from tau2.data_model.message import ToolCall as Tau2ToolCall  # type: ignore[import-untyped]  # no type info
from tau2.data_model.message import UserMessage

from ratchet.agent.model import ModelTurn, scripted_turn
from ratchet.bench.tau2.runner import build_orchestrator, run_config
from ratchet.bench.tau2.tasks import Domain, Tau2Case, case_id, load_tasks
from ratchet.bench.tau2.world import VIEWS, Tau2World

logger.remove()
STOP = "###STOP###"

Reply = str | list[tuple[str, dict[str, Any]]]


class ScriptedUser:
    """A tau2-bench half-duplex user that answers from a list (the last answer repeats). An answer is a
    text, or a list of (tool, arguments) the user runs on their own device before the next answer."""

    def __init__(self, replies: Sequence[Reply] = (STOP,), opening: str = "Hi, I need help.") -> None:
        self.replies = list(replies)
        self.opening = opening
        self.heard: list[str] = []
        self.seed: int | None = None
        self.started = False

    def get_init_state(self, message_history: Any = None) -> list[Any]:
        return []

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        return bool(message.content and STOP in message.content)

    def stop(self, message: Any = None, state: Any = None) -> None:
        return None

    def generate_next_message(self, message: Any, state: list[Any]) -> tuple[Any, list[Any]]:
        if getattr(message, "role", None) == "assistant" and message.content:
            self.heard.append(str(message.content))
        if not self.started:
            self.started = True
            return UserMessage(role="user", content=self.opening), state
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, str):
            return UserMessage(role="user", content=reply), state
        calls = [
            Tau2ToolCall(id=f"u{len(self.heard)}_{i}", name=n, arguments=a, requestor="user")
            for i, (n, a) in enumerate(reply)
        ]
        return UserMessage(role="user", content=None, tool_calls=calls), state


@cache
def _tasks(domain: Domain) -> dict[str, Any]:
    return {str(t.id): t for t in load_tasks(domain)}


def task(domain: Domain, task_id: str) -> Any:
    return _tasks(domain)[task_id]


def config(domain: Domain) -> Any:
    return run_config(domain, agent_model="scripted", user_model="scripted-user")


def factory(user: ScriptedUser) -> Any:
    def make(cfg: Any, t: Any, seed: int | None, sim_id: str) -> Any:
        orch = build_orchestrator(cfg, t, seed, sim_id)
        orch.user = user
        return orch

    return make


def make_world(domain: Domain, task_id: str, user: ScriptedUser | None = None) -> Tau2World:
    orch = factory(user or ScriptedUser())(config(domain), task(domain, task_id), 7, "sim-test")
    return Tau2World.start(orch, VIEWS[domain])


def case(domain: Domain, task_id: str, trial: int = 0, seed: int | None = 11) -> Tau2Case:
    return Tau2Case(case_id(domain, task_id), task_id, domain, "train", trial=trial, seed=seed)


def gold_actions(domain: Domain, task_id: str) -> list[tuple[str, dict[str, Any]]]:
    acts = task(domain, task_id).evaluation_criteria.actions or []
    return [(a.name, dict(a.arguments)) for a in acts if a.requestor == "assistant"]


def gold_turns(domain: Domain, task_id: str, closing: str = "All done.") -> list[ModelTurn]:
    """An agent script that performs the task's gold agent actions one per turn, then tells the customer."""
    turns = [
        scripted_turn(None, [action], id_prefix=f"g{i}_")
        for i, action in enumerate(gold_actions(domain, task_id))
    ]
    return [*turns, scripted_turn(closing)]
