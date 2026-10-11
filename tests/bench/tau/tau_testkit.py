"""Shared helpers for the tau-bench adapter tests: real tau-bench environments with a scripted customer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tau_bench.envs.retail import MockRetailDomainEnv  # type: ignore[import-untyped]  # no type info
from tau_bench.envs.retail.tasks_test import TASKS_TEST  # type: ignore[import-untyped]  # no type info

from ratchet.agent.model import ModelTurn, scripted_turn
from ratchet.bench.tau.tasks import TauCase
from ratchet.bench.tau.world import STOP, TauWorld


class ScriptedUser:
    """A tau-bench user simulator that answers from a list (the last answer repeats)."""

    def __init__(
        self, replies: Sequence[str] = (STOP,), opening: str = "Hi, I need help with an order."
    ) -> None:
        self.replies = list(replies)
        self.opening = opening
        self.heard: list[str] = []

    def reset(self, instruction: str | None = None) -> str:
        return self.opening

    def step(self, content: str) -> str:
        self.heard.append(content)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def get_total_cost(self) -> float:
        return 0.0


def make_env(index: int, user: ScriptedUser | None = None, split: str = "test") -> tuple[Any, str]:
    env = MockRetailDomainEnv(user_strategy="human", task_split=split, task_index=index)
    env.user = user or ScriptedUser()
    reset = env.reset(task_index=index)
    return env, str(reset.observation)


def make_world(index: int, user: ScriptedUser | None = None) -> TauWorld:
    env, opening = make_env(index, user)
    return TauWorld(env, opening)


def gold_actions(index: int) -> list[tuple[str, dict[str, Any]]]:
    return [(a.name, dict(a.kwargs)) for a in TASKS_TEST[index].actions]


def gold_turns(index: int, closing: str = "All done.") -> list[ModelTurn]:
    """An agent script that performs the task's gold actions one per turn, then tells the customer."""
    turns = [
        scripted_turn(None, [action], id_prefix=f"g{i}_") for i, action in enumerate(gold_actions(index))
    ]
    return [*turns, scripted_turn(closing)]


def case(index: int) -> TauCase:
    return TauCase(f"tau-retail-test-{index:03d}", "heldout", "test", index)
