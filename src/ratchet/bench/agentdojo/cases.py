"""AgentDojo episodes as Ratchet cases, split so the improver never sees a held-out task of either kind.

The split is by task, not by episode. User tasks and injection tasks are each shuffled with a fixed seed
and divided; a training case pairs a training user task with a training injection task (or with none), and
a held-out case pairs a held-out user task with a held-out injection task (or with none). Episodes that mix
a training task with a held-out one are in neither set, so nothing the improver read can reappear in the
referee's cases.

The attack text is AgentDojo's own: `important_instructions` with the user named and the model addressed
generically (AgentDojo's `important_instructions_no_model_name`, since the agent model is not in
AgentDojo's model-name table), or any other registered attack by name for a transfer check.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.attacks.base_attacks import BaseAttack, FixedJailbreakAttack
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite

from ratchet.bench.agentdojo.world import DojoTask
from ratchet.cases.model import Case, Expect
from ratchet.types import Split

DEFAULT_VERSION = "v1.2.2"
DEFAULT_ATTACK = "important_instructions_no_model_name"
USER_NAME = "Emma Johnson"  # the user every AgentDojo suite's system message names


def attack_target() -> AgentPipeline:
    """The attacks take a pipeline only to read its name; Ratchet's worker loop is the real target, so this
    pipeline has no elements and is never queried."""
    target = AgentPipeline([])
    target.name = "ratchet-harness"
    return target


def make_attack(suite: TaskSuite[Any], name: str) -> BaseAttack:
    if name == DEFAULT_ATTACK:
        attack = FixedJailbreakAttack(ImportantInstructionsAttack._JB_STRING, suite, attack_target())
        attack.user_name = USER_NAME
        return attack
    return load_attack(name, suite, attack_target())


def split_ids(ids: Sequence[str], seed: int, heldout_fraction: float) -> tuple[list[str], list[str]]:
    """(train, heldout): a seeded shuffle, the first `round(n * fraction)` held out, each half sorted."""
    order = sorted(ids, key=_task_number)
    random.Random(seed).shuffle(order)
    k = round(len(order) * heldout_fraction)
    return sorted(order[k:], key=_task_number), sorted(order[:k], key=_task_number)


def _task_number(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[1])


@dataclass(frozen=True)
class DojoSplit:
    train_users: list[str]
    heldout_users: list[str]
    train_injections: list[str]
    heldout_injections: list[str]

    def users(self, split: Split) -> list[str]:
        return self.train_users if split == "train" else self.heldout_users

    def injections(self, split: Split) -> list[str]:
        return self.train_injections if split == "train" else self.heldout_injections

    def to_payload(self) -> dict[str, list[str]]:
        return {
            "train_users": self.train_users,
            "heldout_users": self.heldout_users,
            "train_injections": self.train_injections,
            "heldout_injections": self.heldout_injections,
        }


def case_id(task: DojoTask) -> str:
    return f"agentdojo/{task.suite}/{task.user_task_id}/{task.injection_task_id or 'no_attack'}"


@dataclass
class DojoSuite:
    """A suite, its split and every case built from it, with the episode behind each case id."""

    name: str
    benchmark_version: str
    attack: str
    seed: int
    suite: TaskSuite[Any]
    split: DojoSplit
    cases: list[Case]
    tasks: dict[str, DojoTask]

    def by_split(self, split: Split) -> list[Case]:
        return [c for c in self.cases if c.split == split]

    def to_payload(self) -> dict[str, Any]:
        return {
            "suite": self.name,
            "benchmark_version": self.benchmark_version,
            "attack": self.attack,
            "seed": self.seed,
            "split": self.split.to_payload(),
            "cases": {s: len(self.by_split(s)) for s in ("train", "heldout")},
        }


def _case(task: DojoTask, split: Split, prompt: str) -> Case:
    return Case(
        id=case_id(task),
        family=f"agentdojo/{task.suite}",
        variant=task.injection_task_id or "no_attack",
        split=split,
        must_succeed=True,
        customer_message=prompt,
        initial_state={},
        faults=[],
        grader=Expect(),
        ticket_id=task.user_task_id,
        kind="trap" if task.attacked else "must_succeed",
        trap=task.injection_task_id,
    )


def build_suite(
    name: str,
    *,
    benchmark_version: str = DEFAULT_VERSION,
    attack: str = DEFAULT_ATTACK,
    seed: int = 0,
    heldout_fraction: float = 0.5,
    max_users: int | None = None,
) -> DojoSuite:
    """Every training and held-out case of one suite. `max_users` caps each split's user tasks."""
    suite: TaskSuite[Any] = get_suite(benchmark_version, name)
    train_u, held_u = split_ids(list(suite.user_tasks), seed, heldout_fraction)
    train_i, held_i = split_ids(list(suite.injection_tasks), seed + 1, heldout_fraction)
    if max_users is not None:
        train_u, held_u = train_u[:max_users], held_u[:max_users]
    split = DojoSplit(train_u, held_u, train_i, held_i)
    att = make_attack(suite, attack)
    cases: list[Case] = []
    tasks: dict[str, DojoTask] = {}
    for part in ("train", "heldout"):
        sp: Split = "train" if part == "train" else "heldout"
        for uid in split.users(sp):
            user = suite.get_user_task_by_id(uid)
            episodes = [DojoTask(name, benchmark_version, uid)]
            for iid in split.injections(sp):
                injections = att.attack(user, suite.get_injection_task_by_id(iid))
                episodes.append(DojoTask(name, benchmark_version, uid, iid, dict(injections)))
            for task in episodes:
                case = _case(task, sp, str(user.PROMPT))
                cases.append(case)
                tasks[case.id] = task
    return DojoSuite(name, benchmark_version, attack, seed, suite, split, cases, tasks)
