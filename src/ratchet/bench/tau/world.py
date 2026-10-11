"""tau-bench retail as a Ratchet world: tau-bench's own environment, tools, user simulator and reward.

`TauWorld` gives the worker loop the surface it needs (`tools`, `tool_names`, `call`, `view`, `digest`) over
one `tau_bench` environment. Every action goes through `Env.step`, so the database changes, the recorded
actions and the reward are exactly tau-bench's. Talking to the customer is the tool `respond`: its result
is the simulated user's reply, produced by tau-bench's user simulator. The conversation ends when the user
says `###STOP###` or the agent calls `transfer_to_human_agents`; tau-bench then computes the reward.

The reward is tau-bench's, unchanged: 1 when the final database equals the one the task's gold actions
produce and every required output was said to the user, else 0, and 0 when the conversation never ended.
The grader adds reasons (which records differ from the gold state, which outputs were never said) so the
improver can locate a failure; the reasons never change the reward.

The state view a guard sees is the slice of the database this conversation has touched: the users and
orders named by any executed call's arguments or results. The full database is 2.3 MB; a guard needs the
records in play, and the slice keeps every recorded trace small.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tau_bench.envs.base import consistent_hash, to_hashable  # type: ignore[import-untyped]  # no type info
from tau_bench.types import Action  # type: ignore[import-untyped]  # tau-bench ships no type information

from ratchet.types import Cost, Outcome, ToolCall, ToolResult

RESPOND = "respond"
STOP = "###STOP###"
TERMINATE_TOOLS = ("transfer_to_human_agents",)

RESPOND_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": RESPOND,
        "description": "Send a message to the customer. The result is the customer's reply.",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string", "description": "The message to the customer."}},
            "required": ["content"],
        },
    },
}

_TOKEN = re.compile(r"#?[A-Za-z0-9_]+")
_MAX_DIFFS = 12


@dataclass(frozen=True)
class TauView:
    """The JSON state view: step count, whether the conversation ended, and the touched records."""

    data: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


@dataclass
class _Ending:
    """Set once the conversation ends: tau-bench's reward and the agent's final database."""

    reward: float
    reward_info: dict[str, Any]
    final_data: dict[str, Any]
    by: str


@dataclass
class TauWorld:
    """One tau-bench task in progress. `env` is a `tau_bench` `Env` already reset to its task."""

    env: Any
    first_message: str
    tables: tuple[str, ...] = ("users", "orders")
    touched: dict[str, set[str]] = field(default_factory=dict)
    steps: int = 0
    ending: _Ending | None = None

    # -- the surface the worker loop uses --------------------------------------------------------------

    def tools(self) -> list[dict[str, Any]]:
        return [*copy.deepcopy(self.env.tools_info), copy.deepcopy(RESPOND_SPEC)]

    def tool_names(self) -> list[str]:
        return sorted(t["function"]["name"] for t in self.tools())

    @property
    def done(self) -> bool:
        return self.ending is not None

    def call(self, call: ToolCall) -> ToolResult:
        self.steps += 1
        if self.ending is not None:
            return ToolResult(call.call_id, False, None, "the conversation has ended")
        if call.name == RESPOND:
            return self._respond(call)
        if call.name not in self.env.tools_map:
            observation = f"Unknown action {call.name}"
            return ToolResult(call.call_id, False, {"observation": observation, "done": False}, observation)
        before = self.env.data
        response = self.env.step(Action(name=call.name, kwargs=dict(call.args)))
        observation = str(response.observation)
        self._touch(call.args, observation)
        if response.done:
            self._end(response, before, call.name)
        error = observation if observation.startswith("Error") else None
        data = {"observation": observation, "done": bool(response.done)}
        return ToolResult(call.call_id, error is None, data, error)

    def view(self) -> TauView:
        data = self.ending.final_data if self.ending is not None else self.env.data
        view: dict[str, Any] = {"step": self.steps, "done": self.done}
        for table in self.tables:
            ids = sorted(self.touched.get(table, set()))
            view[table] = {k: data[table][k] for k in ids if k in data[table]}
        return TauView(view)

    def digest(self) -> str:
        data = self.ending.final_data if self.ending is not None else self.env.data
        digest: str = consistent_hash(to_hashable(data))
        return digest

    # -- internals ----------------------------------------------------------------------------------------

    def _respond(self, call: ToolCall) -> ToolResult:
        content = call.args.get("content")
        if not isinstance(content, str):
            return ToolResult(call.call_id, False, None, "respond needs a string 'content'")
        before = self.env.data
        response = self.env.step(Action(name=RESPOND, kwargs={"content": content}))
        if response.done:
            self._end(response, before, RESPOND)
        return ToolResult(
            call.call_id, True, {"observation": str(response.observation), "done": response.done}
        )

    def _end(self, response: Any, before: dict[str, Any], by: str) -> None:
        # tau-bench's reward computation replaces `env.data` with the gold database; `before` is the
        # object the agent's calls changed, which no later step touches.
        info = response.info.reward_info
        self.ending = _Ending(
            reward=float(response.reward),
            reward_info=info.info.model_dump() if info is not None else {},
            final_data=before,
            by=by,
        )

    def _touch(self, args: Mapping[str, Any], observation: str) -> None:
        text = json.dumps(args, sort_keys=True) + "\n" + observation
        tokens = set(_TOKEN.findall(text))
        for table in self.tables:
            self.touched.setdefault(table, set()).update(tokens & self.env.data[table].keys())


# -- grading ------------------------------------------------------------------------------------------------


def _short(value: Any) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= 160 else text[:157] + "..."


def _field_diff(path: str, mine: Any, want: Any) -> str:
    """One field's difference; for two lists, the first element that differs, so the reason shows it."""
    if isinstance(mine, list) and isinstance(want, list):
        for i, (x, y) in enumerate(zip(mine, want, strict=False)):
            if x != y:
                return f"{path}[{i}] is {_short(x)}, the task needs {_short(y)}"
        return f"{path} has {len(mine)} entries, the task needs {len(want)} (extra or missing at the end)"
    return f"{path} is {_short(mine)}, the task needs {_short(want)}"


def record_diffs(final: Mapping[str, Any], gold: Mapping[str, Any], limit: int = _MAX_DIFFS) -> list[str]:
    """Every record whose final value differs from the gold database, field by field, at most `limit`."""
    out: list[str] = []
    for table in sorted(set(final) | set(gold)):
        mine, want = final.get(table, {}), gold.get(table, {})
        for key in sorted(set(mine) | set(want)):
            a, b = mine.get(key), want.get(key)
            if a == b:
                continue
            if a is None or b is None:
                state = "is missing" if a is None else "should not exist"
                out.append(f"{table}['{key}'] {state} in the final database")
                continue
            for name in sorted(set(a) | set(b)):
                if a.get(name) != b.get(name):
                    out.append(_field_diff(f"{table}['{key}'].{name}", a.get(name), b.get(name)))
    if len(out) > limit:
        out = [*out[:limit], f"... and {len(out) - limit} more differences"]
    return out


def outcome_of(world: TauWorld, cost: Cost) -> Outcome:
    """tau-bench's reward as an `Outcome`: passed iff reward is 1, with reasons that locate a failure."""
    reasons: list[str] = []
    if world.ending is None:
        # tau-bench gives no reward to a conversation that never ended. The gold state is still computed so
        # the reasons say what was left undone.
        final = world.env.data
        world.env.calculate_reward()
        gold = world.env.data
        world.env.data = final
        reasons.append("the conversation did not end within the agent's action budget")
        reasons += record_diffs(final, gold)
        return Outcome(False, reasons, consistent_hash(to_hashable(final)), cost)
    ending = world.ending
    gold = world.env.data
    info = ending.reward_info
    if "r_actions" in info and not info["r_actions"]:
        reasons += record_diffs(ending.final_data, gold) or ["the final database differs from the gold one"]
    if "outputs" in info:
        if not _actions_match(ending.final_data, gold):
            reasons += record_diffs(ending.final_data, gold)
        reasons += [f"the agent never told the user {o!r}" for o, said in info["outputs"].items() if not said]
    passed = ending.reward == 1.0
    if not passed and not reasons:
        reasons.append("tau-bench reward is 0")
    return Outcome(passed, reasons, consistent_hash(to_hashable(ending.final_data)), cost)


def _actions_match(final: Mapping[str, Any], gold: Mapping[str, Any]) -> bool:
    return bool(consistent_hash(to_hashable(dict(final))) == consistent_hash(to_hashable(dict(gold))))
