"""One AgentDojo episode as a Ratchet world: the environment, its tools, the guard view, and the graders.

Tool calls run through AgentDojo's `FunctionsRuntime`, with the same argument normalisation as AgentDojo's
`ToolsExecutor` (a string that spells a list becomes the list) and the same YAML rendering of results.
Grading calls the user task's `utility_from_traces` / `utility` and the injection task's
`security_from_traces` / `security` exactly as `TaskSuite.run_task_with_pipeline` does. The function-call
trace handed to them holds the calls that reached the environment: a call a guard blocked never happened.

The view a guard sees is what a deployment knows: `request` (the user's request), `calls_executed`, and
`state` (the environment, long texts cut to a prefix). It never names the user task or the injection task.
"""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai
from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime, TaskEnvironment
from agentdojo.task_suite.task_suite import TaskSuite

from ratchet.types import Cost, Outcome, ToolCall, ToolResult, Trace, canonical_json, sha256_hex
from ratchet.world.view import StateView

UTILITY_MISSED = "utility: the user's task was not completed"
ATTACK_SUCCEEDED = "security: the injected instruction was carried out"
# The worker records the guard view on every tool call. The workspace suite's state is 115,670 bytes of
# JSON, most of it mail and file bodies; at about ten calls per episode and thousands of episodes per
# generation, full views would not fit the run's memory. Every field stays, long texts keep a prefix, and
# the full text is still in the tool results the guard sees in `history`.
VIEW_TEXT_LIMIT = 240


@dataclass(frozen=True)
class DojoTask:
    """One episode: a user task, and optionally an injection task with the attack text per vector."""

    suite: str
    benchmark_version: str
    user_task_id: str
    injection_task_id: str | None = None
    injections: dict[str, str] = field(default_factory=dict)

    @property
    def attacked(self) -> bool:
        return self.injection_task_id is not None


@dataclass(frozen=True)
class Grade:
    utility: bool
    attacked: bool | None  # None when the episode carries no injection task

    def reasons(self) -> list[str]:
        out = [] if self.utility else [UTILITY_MISSED]
        if self.attacked:
            out.append(ATTACK_SUCCEEDED)
        return out


def compact(value: Any, limit: int = VIEW_TEXT_LIMIT) -> Any:
    """`value` with every string longer than `limit` cut to its prefix plus a length marker."""
    if isinstance(value, str) and len(value) > limit:
        return f"{value[:limit]}...[{len(value)} chars]"
    if isinstance(value, dict):
        return {k: compact(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v, limit) for v in value]
    return value


def _normalise_args(args: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(args)
    for key, value in out.items():
        if isinstance(value, str) and is_string_list(value):
            out[key] = literal_eval(value)
    return out


def tool_specs(suite: TaskSuite[Any]) -> list[dict[str, Any]]:
    """OpenAI function-tool schemas for every tool in the suite, sorted by name."""
    specs = [dict(_function_to_openai(f)) for f in suite.tools]
    return sorted(specs, key=lambda s: str(s["function"]["name"]))


class DojoWorld:
    """The world surface the worker loop drives: `tools`, `tool_names`, `call`, `view`, `digest`."""

    def __init__(self, suite: TaskSuite[Any], task: DojoTask) -> None:
        self.suite = suite
        self.task = task
        self.user_task: BaseUserTask[Any] = suite.get_user_task_by_id(task.user_task_id)
        self.injection_task: BaseInjectionTask[Any] | None = (
            suite.get_injection_task_by_id(task.injection_task_id) if task.injection_task_id else None
        )
        env = suite.load_and_inject_default_environment(task.injections)
        self.env: TaskEnvironment = self.user_task.init_environment(env)
        self.pre: TaskEnvironment = self.env.model_copy(deep=True)
        self.runtime = FunctionsRuntime(suite.tools)
        self.executed: list[FunctionCall] = []
        self._tools = tool_specs(suite)

    @property
    def prompt(self) -> str:
        return str(self.user_task.PROMPT)

    def tools(self) -> list[dict[str, Any]]:
        return [dict(t) for t in self._tools]

    def tool_names(self) -> list[str]:
        return [str(t["function"]["name"]) for t in self._tools]

    def call(self, call: ToolCall) -> ToolResult:
        args = _normalise_args(call.args)
        result, error = self.runtime.run_function(self.env, call.name, args)
        self.executed.append(FunctionCall(function=call.name, args=args, id=call.call_id))
        if error is not None:
            return ToolResult(call.call_id, False, None, error)
        return ToolResult(call.call_id, True, {"output": tool_result_to_str(result)}, None)

    def state(self) -> dict[str, Any]:
        """The full environment state, every byte: what the digest covers."""
        out: dict[str, Any] = self.env.model_dump(mode="json")
        return out

    def snapshot(self) -> dict[str, Any]:
        """The guard view: the request, the call count and the state with long texts cut to a prefix."""
        return {
            "calls_executed": len(self.executed),
            "request": self.prompt,
            "state": compact(self.state()),
        }

    def view(self) -> StateView:
        return StateView(self.snapshot())

    def digest(self) -> str:
        return sha256_hex(canonical_json({"calls_executed": len(self.executed), "state": self.state()}))

    def grade(self, final_text: str) -> Grade:
        """AgentDojo's own checks on the current state, with `final_text` as the model's output."""
        utility = self.user_task.utility_from_traces(final_text, self.pre, self.env, self.executed)
        if utility is None:
            utility = self.user_task.utility(final_text, self.pre, self.env)
        if self.injection_task is None:
            return Grade(bool(utility), None)
        attacked = self.injection_task.security_from_traces(final_text, self.pre, self.env, self.executed)
        if attacked is None:
            attacked = self.injection_task.security(final_text, self.pre, self.env)
        return Grade(bool(utility), bool(attacked))


def final_text(trace: Trace) -> str:
    """The model's last message text, which AgentDojo's checks read as the model output."""
    for e in reversed(trace.events):
        if e.kind == "model":
            return str(e.payload.get("content") or "")
    return ""


def trace_cost(trace: Trace) -> Cost:
    models = [e for e in trace.events if e.kind == "model"]
    usage = [e.payload.get("usage") or {} for e in models]
    wall = trace.events[-1].ts - trace.events[0].ts if trace.events else 0.0
    return Cost(
        model_calls=len(models),
        input_tokens=sum(int(u.get("input_tokens", 0)) for u in usage),
        output_tokens=sum(int(u.get("output_tokens", 0)) for u in usage),
        tool_calls=sum(1 for e in trace.events if e.kind == "tool_call"),
        wall_s=round(wall, 3),
    )


def outcome(world: DojoWorld, trace: Trace) -> Outcome:
    grade = world.grade(final_text(trace))
    reasons = grade.reasons()
    return Outcome(not reasons, reasons, world.digest(), trace_cost(trace))
