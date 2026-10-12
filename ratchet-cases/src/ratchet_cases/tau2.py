"""Deterministic tool faults on tau2-bench's retail domain, graded on the database the agent left.

The task set `retail_faults` takes, for each retail write tool, the first base-split retail task whose gold
actions call that tool exactly once and replay without an error, and makes two variants of it: the
assistant's first call to that tool times out after it commits, or times out before it commits. Either
way the agent reads `Error: timeout`. A variant passes when the agent database it leaves, and the user
database, equal the ones the task's gold actions produce.

tau2-bench grades a run by replaying the trajectory into a fresh environment. Under a fault that replay
is wrong in both directions (a strict replay raises on the after-commit variant, a lenient one applies the
before-commit change that never happened), so this module grades the live environment and never a replay.

Two ways to run it:

- Against stock tau2-bench, `FaultInjector` wraps one environment and applies one fault to the
  assistant's calls; `play` drives it with a scripted agent and `grade` compares the live databases with
  the gold ones. No model and no user simulator are involved.
- Against a tau2-bench whose `Task` carries a `faults` list and whose registry loads `tau2.plugins` entry
  points, `register` adds the task set to the registry, and `tau2 run --domain retail --task-set-name
  retail_faults` runs it with tau2-bench's own orchestrator.

tau2-bench (the sierra-research/tau2-bench repository, Python 3.12 or 3.13) is not on PyPI under its own
name, so it is not declared as a dependency: install it from the repository at the commit you test, with
its `data/` directory, before importing this module.
"""

from __future__ import annotations

import copy
import enum
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.data_model.tasks import Task
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolType

TASK_SET = "retail_faults"
DOMAIN = "retail"
TIMEOUT = "Error: timeout"


class FaultKind(str, enum.Enum):
    """When the fault strikes, relative to the call's effect on the database."""

    TIMEOUT_AFTER_COMMIT = "timeout_after_commit"
    TIMEOUT_BEFORE_COMMIT = "timeout_before_commit"


@dataclass(frozen=True)
class ToolFault:
    """A fault on the assistant's `nth` call to `tool_name`: the agent reads `message` and an error."""

    tool_name: str
    kind: FaultKind
    nth: int = 1
    message: str = TIMEOUT

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "kind": self.kind.value,
            "nth": self.nth,
            "message": self.message,
        }


@dataclass(frozen=True)
class FaultedTask:
    """One task of the set: a retail task, the id it is published under, and its one fault."""

    task_id: str
    base_task_id: str
    fault: ToolFault
    task: Task


def write_tools(environment: Environment | None = None) -> list[str]:
    """The retail tools tau2-bench marks as WRITE, in the toolkit's order."""
    env = environment if environment is not None else get_environment()
    tools = env.tools
    if tools is None:
        raise ValueError("the retail environment has no assistant toolkit")
    return [name for name in tools.get_tools() if tools.tool_type(name) == ToolType.WRITE]


def _gold_calls(task: Task) -> list[Any]:
    criteria = task.evaluation_criteria
    if criteria is None or criteria.actions is None:
        return []
    return list(criteria.actions)


def _replays_cleanly(task: Task) -> bool:
    env = get_environment()
    for action in _gold_calls(task):
        try:
            env.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
        except Exception:
            return False
    return True


def _base_task(tool_name: str, tasks: list[Task]) -> Task:
    for task in tasks:
        calls = [a for a in _gold_calls(task) if a.requestor == "assistant"]
        if task.initial_state is not None:
            continue
        if sum(1 for a in calls if a.name == tool_name) != 1:
            continue
        if _replays_cleanly(task):
            return task
    raise LookupError(f"no base-split retail task calls {tool_name} once and replays cleanly")


@lru_cache(maxsize=1)
def faulted_tasks() -> tuple[FaultedTask, ...]:
    """The task set, in write-tool order and then fault-kind order."""
    base = get_tasks("base")
    out: list[FaultedTask] = []
    for tool_name in write_tools():
        source = _base_task(tool_name, base)
        for kind in FaultKind:
            fault = ToolFault(tool_name=tool_name, kind=kind)
            task_id = f"{TASK_SET}/{tool_name}/{kind.value}/{source.id}"
            task = source.model_copy(deep=True, update={"id": task_id})
            out.append(FaultedTask(task_id=task_id, base_task_id=source.id, fault=fault, task=task))
    return tuple(out)


def _supports_faults() -> bool:
    return "faults" in Task.model_fields


def _require_fault_support() -> None:
    if not _supports_faults():
        raise RuntimeError(
            "this tau2-bench Task has no `faults` field, so its orchestrator would run these tasks "
            "unfaulted; use FaultInjector with play() and grade() instead"
        )


def load_tasks(task_split_name: str | None = None) -> list[Task]:
    """The task set as tau2-bench `Task`s carrying their `faults`. Needs a tau2-bench with `Task.faults`."""
    _require_fault_support()
    if task_split_name not in (None, "base"):
        raise ValueError(f"{TASK_SET} has one split, 'base'; got {task_split_name!r}")
    return [
        Task.model_validate({**ft.task.model_dump(), "faults": [ft.fault.as_dict()]})
        for ft in faulted_tasks()
    ]


def task_splits() -> dict[str, list[str]]:
    return {"base": [ft.task_id for ft in faulted_tasks()]}


def register(registry: Any) -> None:
    """The `tau2.plugins` entry point: register `retail_faults` with a tau2-bench registry."""
    _require_fault_support()
    registry.register_tasks(load_tasks, TASK_SET, get_task_splits=task_splits)


class FaultInjector:
    """One environment whose `get_response` applies one fault to the assistant's calls.

    Only the wrapped object changes behaviour; tau2-bench's classes are not patched.
    """

    def __init__(self, environment: Environment, fault: ToolFault) -> None:
        self.environment = environment
        self.fault = fault
        self.calls: dict[str, int] = {}
        self.fired = False

    def get_response(self, call: ToolCall) -> ToolMessage:
        strike = False
        if call.requestor == "assistant":
            self.calls[call.name] = self.calls.get(call.name, 0) + 1
            strike = call.name == self.fault.tool_name and self.calls[call.name] == self.fault.nth
        if strike and self.fault.kind is FaultKind.TIMEOUT_BEFORE_COMMIT:
            self.fired = True
            return ToolMessage(
                id=call.id, role="tool", content=self.fault.message, requestor=call.requestor, error=True
            )
        message: ToolMessage = self.environment.get_response(call)
        if strike:
            self.fired = True
            return ToolMessage(
                id=call.id, role="tool", content=self.fault.message, requestor=call.requestor, error=True
            )
        return message


@dataclass(frozen=True)
class Grade:
    """The live databases against the gold ones."""

    agent_db_match: bool
    user_db_match: bool

    @property
    def passed(self) -> bool:
        return self.agent_db_match and self.user_db_match


def gold_environment(task: Task) -> Environment:
    """A fresh retail environment with the task's gold actions applied, as tau2-bench's evaluator does."""
    env = get_environment()
    for action in _gold_calls(task):
        env.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
    return env


def grade(task: Task, environment: Environment) -> Grade:
    """Compare the environment the agent left with the gold environment. Nothing is replayed."""
    gold = gold_environment(task)
    return Grade(
        agent_db_match=environment.get_db_hash() == gold.get_db_hash(),
        user_db_match=environment.get_user_db_hash() == gold.get_user_db_hash(),
    )


class Policy(str, enum.Enum):
    """What a scripted agent does when a write call reports an error."""

    RECONCILE = "reconcile"  # read the record; retry only if the change is absent
    BLIND_RETRY = "blind_retry"  # retry once without looking
    GIVE_UP = "give_up"  # treat the error as final


Responder = Callable[[ToolCall], ToolMessage]


def _read_call(action: Any) -> ToolCall | None:
    args = action.arguments
    if "order_id" in args:
        return ToolCall(
            name="get_order_details", arguments={"order_id": args["order_id"]}, requestor="assistant"
        )
    if "user_id" in args:
        return ToolCall(
            name="get_user_details", arguments={"user_id": args["user_id"]}, requestor="assistant"
        )
    return None


def _read(respond: Responder, action: Any) -> str | None:
    call = _read_call(action)
    if call is None:
        return None
    content: str | None = respond(call).content
    return content


def play(task: Task, respond: Responder, policy: Policy) -> list[tuple[ToolCall, ToolMessage]]:
    """Run the task's gold actions as the assistant through `respond`, handling errors by `policy`.

    `respond` is `FaultInjector.get_response` on stock tau2-bench, or the environment's own
    `get_response` on a tau2-bench that applies `Task.faults` itself. Returns every call and its reply.
    """
    log: list[tuple[ToolCall, ToolMessage]] = []

    def call(tool_call: ToolCall) -> ToolMessage:
        reply = respond(tool_call)
        log.append((tool_call, reply))
        return reply

    for i, action in enumerate(_gold_calls(task)):
        if action.requestor != "assistant":
            continue
        tool_call = ToolCall(
            id=f"call-{i}", name=action.name, arguments=copy.deepcopy(action.arguments), requestor="assistant"
        )
        before = _read(call, action) if policy is Policy.RECONCILE else None
        reply = call(tool_call)
        if not reply.error or policy is Policy.GIVE_UP:
            continue
        if policy is Policy.BLIND_RETRY:
            call(tool_call.model_copy(update={"id": f"call-{i}-retry"}))
            continue
        after = _read(call, action)
        if after is not None and after == before:
            call(tool_call.model_copy(update={"id": f"call-{i}-retry"}))
    return log


def run(faulted: FaultedTask, policy: Policy) -> Grade:
    """Play one faulted task on stock tau2-bench and grade the live databases."""
    injector = FaultInjector(get_environment(), faulted.fault)
    play(faulted.task, injector.get_response, policy)
    if not injector.fired:
        raise RuntimeError(f"{faulted.task_id}: the fault never fired")
    return grade(faulted.task, injector.environment)


def manifest() -> list[dict[str, str]]:
    """The task set as plain data: id, base task, tool and fault kind, in order."""
    return [
        {
            "task_id": ft.task_id,
            "base_task_id": ft.base_task_id,
            "tool_name": ft.fault.tool_name,
            "kind": ft.fault.kind.value,
        }
        for ft in faulted_tasks()
    ]


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2))
