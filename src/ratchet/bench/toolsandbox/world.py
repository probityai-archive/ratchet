"""One benchmark scenario as a Ratchet world: the benchmark's execution context, tools and user simulator.

`ToolSandboxWorld` gives the worker loop the surface it needs (`tools`, `tool_names`, `call`, `view`,
`digest`) over one scenario's `ExecutionContext`. The conversation is written into the context exactly as the
benchmark's own runner writes it, message by message, so the milestone and minefield grader reads the same
record it reads in a benchmark run:

- a tool call from the agent becomes the message the benchmark's agent role would send to the execution
  environment, the stock `ExecutionEnvironment` role runs it in the scenario's Python console, and its answer
  (the printed result, or the exception) goes back to the agent;
- talking to the user is the tool `respond`: the agent's text becomes a message to the user, the benchmark's
  user simulator answers it (or ends the conversation with its `end_conversation` tool), and the answer is
  the tool's result;
- the conversation ends when the user ends it or when the scenario's message budget (`max_messages`) is used.

The state view a guard sees is the current snapshot of the scenario's databases: settings, contacts,
messages and reminders. They hold a handful of rows each (settings 1, contacts 4, messages 5 and reminders 3
at the start of every scenario), so the view is whole rather than a slice.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from ratchet_cases.types import Cost, Outcome, ToolCall, ToolResult
from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    get_current_context,
)
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
)
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_conversion import (
    convert_to_openai_tools,
)
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.execution_environment import (
    ExecutionEnvironment,
)

from ratchet.bench.toolsandbox.gate import GATE
from ratchet.bench.toolsandbox.grading import Graded, grade, reasons_for

RESPOND = "respond"
RESPOND_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": RESPOND,
        "description": "Send a message to the user. The result is the user's reply.",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string", "description": "The message to the user."}},
            "required": ["content"],
        },
    },
}
VIEW_DATABASES = (
    DatabaseNamespace.SETTING,
    DatabaseNamespace.CONTACT,
    DatabaseNamespace.MESSAGING,
    DatabaseNamespace.REMINDER,
)
ENDED = "the conversation has ended"

UserFactory = Callable[[], BaseRole]


class _AgentRole(BaseRole):
    """Stands for the agent only to ask the benchmark which tools the agent is allowed to see."""

    role_type = RoleType.AGENT


@dataclass(frozen=True)
class ToolSandboxView:
    """The JSON state view: step count, whether the conversation ended, and the databases' current rows."""

    data: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def _jsonable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = json.loads(json.dumps(rows, default=str))
    return out


class ToolSandboxWorld:
    """One scenario in progress. `user` makes the benchmark role that plays the simulated user."""

    def __init__(self, scenario: Scenario, name: str, user: UserFactory) -> None:
        self.scenario, self.name = scenario, name
        self.steps = 0
        self._ended = False
        self._context: ExecutionContext = copy.deepcopy(scenario.starting_context)
        self._graded: Graded | None = None
        with GATE.held(self._context):
            self._roles: dict[RoleType, BaseRole] = {
                RoleType.USER: user(),
                RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
            }
            self._boot()
            self._limit = self._context.max_sandbox_message_index + scenario.max_messages
            self._specs = convert_to_openai_tools(_AgentRole.get_available_tools())
            messages = BaseRole.get_messages()
        self._names = sorted(s["function"]["name"] for s in self._specs)
        first = messages[-1]
        if (first.sender, first.recipient) != (RoleType.USER, RoleType.AGENT):
            raise ValueError(f"scenario {name} does not open with a user message to the agent")
        self.opening: str = first.content
        self.agent_prompt: str = next(
            (m.content for m in messages if (m.sender, m.recipient) == (RoleType.SYSTEM, RoleType.AGENT)), ""
        )

    # -- the surface the worker loop uses --------------------------------------------------------------

    def tools(self) -> list[dict[str, Any]]:
        return [*copy.deepcopy(self._specs), copy.deepcopy(RESPOND_SPEC)]

    def tool_names(self) -> list[str]:
        return sorted([*self._names, RESPOND])

    @property
    def done(self) -> bool:
        return self._ended

    def call(self, call: ToolCall) -> ToolResult:
        self.steps += 1
        if self._ended:
            return ToolResult(call.call_id, False, None, ENDED)
        if call.name == RESPOND:
            return self._respond(call)
        if call.name not in self._names:
            error = f"Unknown tool {call.name}"
            return ToolResult(call.call_id, False, {"observation": error, "done": False}, error)
        return self._act(call)

    def view(self) -> ToolSandboxView:
        with GATE.held(self._context):
            data: dict[str, Any] = {
                "step": self.steps,
                "done": self._ended,
                "message_index": self._context.max_sandbox_message_index,
            }
            data.update(self._rows())
        return ToolSandboxView(data)

    def digest(self) -> str:
        with GATE.held(self._context):
            return _digest(self._rows())

    def close(self) -> None:
        """Release what the roles hold (the simulated user's connection)."""
        for role in self._roles.values():
            role.teardown()

    # -- grading ----------------------------------------------------------------------------------------

    def grade(self) -> Graded:
        """The benchmark's evaluation of the conversation so far (callable at any step)."""
        with GATE.held(self._context):
            return grade(self.scenario.evaluation, self._context, self.scenario.max_messages)

    def outcome(self, cost: Cost) -> Outcome:
        """The benchmark's grade as an `Outcome`: passed at `PASS_SIMILARITY`, reasons for what was missed."""
        graded = self.graded()
        passed = graded.passed
        return Outcome(
            passed, [] if passed else reasons_for(graded) or ["similarity below 1"], self.digest(), cost
        )

    def graded(self) -> Graded:
        """The final grade, computed once after the conversation."""
        if self._graded is None:
            self._graded = self.grade()
        return self._graded

    # -- internals ----------------------------------------------------------------------------------------

    def _boot(self) -> None:
        """Run the scenario's system messages for the execution environment (its imports), as `play` does."""
        sandbox = self._context.get_database(
            DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False, get_all_history_snapshots=True
        )
        last = self._context.max_sandbox_message_index
        for index in range(last + 1):
            if (
                sandbox["recipient"][index] == RoleType.EXECUTION_ENVIRONMENT
                and sandbox["sender"][index] == RoleType.SYSTEM
            ):
                self._roles[RoleType.EXECUTION_ENVIRONMENT].respond(ending_index=index)
        if self._context.max_sandbox_message_index != last:
            raise ValueError("the scenario's system messages produced messages of their own")

    def _rows(self) -> dict[str, Any]:
        rows: dict[str, Any] = {
            str(ns).lower(): _jsonable(self._context.get_database(ns).to_dicts()) for ns in VIEW_DATABASES
        }
        rows["conversation_active"] = self._active()
        return rows

    def _active(self) -> bool:
        sandbox = self._context.get_database(DatabaseNamespace.SANDBOX)
        return bool(sandbox["conversation_active"][-1])

    def _open(self) -> bool:
        """True while the benchmark's own loop would still let a role respond."""
        return self._active() and self._context.max_sandbox_message_index < self._limit

    def _act(self, call: ToolCall) -> ToolResult:
        with GATE.held(self._context):
            if not self._open():
                self._ended = True
                return ToolResult(call.call_id, False, None, ENDED)
            tool_id = f"call_{self.steps}"
            raw = ChatCompletionMessageToolCall(
                id=tool_id,
                type="function",
                function=Function(name=call.name, arguments=json.dumps(call.args)),
            )
            code = openai_tool_call_to_python_code(
                raw,
                set(self._names),
                self._context.get_execution_facing_tool_name(call.name),
            )
            BaseRole.add_messages(
                [
                    Message(
                        sender=RoleType.AGENT,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=code,
                        openai_tool_call_id=tool_id,
                        openai_function_name=call.name,
                    )
                ]
            )
            self._dispatch()
            reply = BaseRole.get_messages()[-1]
            self._ended = not self._open()
        if (reply.sender, reply.recipient) != (RoleType.EXECUTION_ENVIRONMENT, RoleType.AGENT):
            # The message budget ended the conversation before the call could run, as in the benchmark's loop.
            return ToolResult(call.call_id, False, None, ENDED)
        error = reply.tool_call_exception
        data = {"observation": reply.content, "done": self._ended}
        return ToolResult(call.call_id, error is None, data, error)

    def _respond(self, call: ToolCall) -> ToolResult:
        content = call.args.get("content")
        if not isinstance(content, str):
            return ToolResult(call.call_id, False, None, "respond needs a string 'content'")
        with GATE.held(self._context):
            if not self._open():
                self._ended = True
                return ToolResult(call.call_id, False, None, ENDED)
            BaseRole.add_messages([Message(sender=RoleType.AGENT, recipient=RoleType.USER, content=content)])
            self._dispatch()
            reply = BaseRole.get_messages()[-1]
            self._ended = not self._open()
        spoke = (reply.sender, reply.recipient) == (RoleType.USER, RoleType.AGENT)
        return ToolResult(
            call.call_id, True, {"observation": reply.content if spoke else "", "done": self._ended}
        )

    def _dispatch(self) -> None:
        """Let the other roles answer until a message is addressed to the agent or the conversation is over.

        The loop is `Scenario.play`'s: the recipient of the newest message responds while the conversation is
        active and the message budget is not used. A role may replace the global context (the execution
        environment does when it runs parallel calls), so the context is read back after each response.
        """
        while self._open():
            recipient = BaseRole.get_messages()[-1].recipient
            if recipient == RoleType.AGENT:
                break
            self._roles[recipient].respond()
            self._context = get_current_context()


def _digest(rows: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode("utf-8")).hexdigest()
