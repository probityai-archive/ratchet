"""tau2-bench as a Ratchet world: tau2-bench's own orchestrator, environment, user simulator and evaluator.

`Tau2World` gives the worker loop the surface it needs (`tools`, `tool_names`, `call`, `view`, `digest`)
over one tau2-bench `Orchestrator` built by tau2-bench's own builder. The orchestrator keeps its role: it
routes every message, runs the user simulator, executes tool calls (the agent's and, in telecom, the
user's), counts steps and errors and decides when the conversation ends. Only its agent is replaced, by a
`PuppetAgent` whose next message is the worker's action:

- an agent tool call becomes an assistant message with that one tool call, and the orchestrator runs it
  on the environment;
- the tool `respond` becomes an assistant text message to the customer; the orchestrator lets the user
  simulator answer, executing the user's own tool calls on the user's device in telecom, and the world
  returns the customer's reply with the user's actions.

The world steps the orchestrator exactly as `Orchestrator.run` does (step, then the termination check)
until the orchestrator next needs the agent, so the trajectory, the termination reason and the reward are
tau2-bench's. When the conversation ends the world finalizes the run into tau2-bench's `SimulationRun` and
scores it with tau2-bench's `evaluate_simulation`, unchanged.

The grader adds reasons (which records differ from the gold database, which actions, statements or
assertions were missed) so the improver can locate a failure; the reasons never change the reward.

The state view a guard sees is the slice of the agent database this conversation has touched (retail
users and orders, airline users and reservations, named by any executed call's arguments or results)
and, in telecom, the whole small agent database plus the user's device and surroundings, so a guard sees
every change the user made on their side.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import ratchet.bench.tau2  # noqa: F401  # sets TAU2_DATA_DIR before tau2-bench is imported
from ratchet.types import Cost, Outcome, ToolCall, ToolResult

RESPOND = "respond"

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


@dataclass(frozen=True)
class ViewSpec:
    """Which agent tables a guard sees: `sliced` ones keep only touched records, `whole` ones are complete.
    `user_db` adds the user's own database (telecom: the device and the surroundings)."""

    sliced: tuple[str, ...] = ()
    whole: tuple[str, ...] = ()
    user_db: bool = False


VIEWS: dict[str, ViewSpec] = {
    "retail": ViewSpec(sliced=("users", "orders")),
    "airline": ViewSpec(sliced=("users", "reservations")),
    "telecom": ViewSpec(whole=("customers", "lines", "bills", "devices", "plans"), user_db=True),
}

_TOKEN = re.compile(r"#?[A-Za-z0-9_\-]+")
_MAX_DIFFS = 12


@dataclass(frozen=True)
class Tau2View:
    """The JSON state view: step count, whether the conversation ended, and the records in play."""

    data: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


class PuppetAgent:
    """A tau2-bench half-duplex agent whose next message is the one the world queued before the step."""

    def __init__(self, tools: list[Any], domain_policy: str) -> None:
        self.tools = tools
        self.domain_policy = domain_policy
        self.queued: Any = None
        self.seed: int | None = None

    def get_init_state(self, message_history: list[Any] | None = None) -> list[Any]:
        return list(message_history or [])

    def generate_next_message(self, message: Any, state: list[Any]) -> tuple[Any, list[Any]]:
        if self.queued is None:
            raise RuntimeError("the orchestrator asked the agent for a message the worker has not sent")
        out, self.queued = self.queued, None
        state.extend([message, out])
        return out, state

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        # tau2-bench's half-duplex LLM agent never stops a conversation itself; neither does this one.
        return False

    def stop(self, message: Any = None, state: Any = None) -> None:
        return None


@dataclass
class Ending:
    """Set once the conversation ends: tau2-bench's finalized and scored `SimulationRun`."""

    simulation: Any
    by: str

    @property
    def reward(self) -> float:
        info = self.simulation.reward_info
        return float(info.reward) if info is not None else 0.0


def _role(name: str) -> Any:
    from tau2.orchestrator.orchestrator import Role  # type: ignore[import-untyped]  # no type info

    return Role(name)


@dataclass
class Tau2World:
    """One tau2-bench task in progress, driven through `orchestrator` (agent: a `PuppetAgent`)."""

    orchestrator: Any
    view_spec: ViewSpec = field(default_factory=ViewSpec)
    touched: dict[str, set[str]] = field(default_factory=dict)
    steps: int = 0
    ending: Ending | None = None

    @classmethod
    def start(cls, orchestrator: Any, view_spec: ViewSpec) -> Tau2World:
        """Initialize the orchestrator and run it until the customer's opening message reaches the agent."""
        from tau2.utils.utils import get_now  # type: ignore[import-untyped]  # no type info

        state = orchestrator.task.initial_state
        if state is not None and state.message_history:
            raise ValueError(f"task {orchestrator.task.id} starts from a message history; not supported")
        world = cls(orchestrator, view_spec)
        orchestrator._run_start_time = get_now()
        orchestrator._run_start_perf = time.perf_counter()
        orchestrator.initialize()
        world._advance()
        if orchestrator.done:
            world._end("start")
        return world

    # -- the surface the worker loop uses --------------------------------------------------------------

    @property
    def env(self) -> Any:
        return self.orchestrator.environment

    @property
    def opening(self) -> str:
        """The customer's first message to the agent."""
        return next((str(m.content) for m in self._messages("user") if not m.is_tool_call()), "")

    def tools(self) -> list[dict[str, Any]]:
        return [*(copy.deepcopy(t.openai_schema) for t in self.env.get_tools()), copy.deepcopy(RESPOND_SPEC)]

    def tool_names(self) -> list[str]:
        return sorted(t["function"]["name"] for t in self.tools())

    @property
    def done(self) -> bool:
        return self.ending is not None

    def call(self, call: ToolCall) -> ToolResult:
        from tau2.data_model.message import AssistantMessage  # type: ignore[import-untyped]  # no types
        from tau2.data_model.message import ToolCall as Tau2ToolCall

        self.steps += 1
        if self.ending is not None:
            ended = {"observation": "", "done": True}
            return ToolResult(call.call_id, False, ended, "the conversation has ended")
        if call.name == RESPOND:
            content = call.args.get("content")
            if not isinstance(content, str) or not content.strip():
                return ToolResult(call.call_id, False, None, "respond needs a non-empty string 'content'")
            message = AssistantMessage(
                role="assistant", content=content, raw_data={"ratchet_call_id": call.call_id}
            )
        else:
            tc = Tau2ToolCall(
                id=call.call_id, name=call.name, arguments=dict(call.args), requestor="assistant"
            )
            message = AssistantMessage(role="assistant", content=None, tool_calls=[tc])
        mark = len(self.orchestrator.trajectory)
        self.orchestrator.agent.queued = message
        self.orchestrator.step()
        self.orchestrator._check_termination()
        self._advance()
        new = self.orchestrator.trajectory[mark + 1 :]
        if self.orchestrator.done:
            self._end(call.name)
        if call.name == RESPOND:
            return self._reply(call, new)
        return self._tool_result(call, new)

    def view(self) -> Tau2View:
        db = self.env.tools.db.model_dump(mode="json")
        view: dict[str, Any] = {"step": self.steps, "done": self.done}
        for table in self.view_spec.sliced:
            ids = sorted(self.touched.get(table, set()))
            view[table] = {k: db[table][k] for k in ids if k in db[table]}
        for table in self.view_spec.whole:
            view[table] = db.get(table)
        if self.view_spec.user_db and self.env.user_tools is not None:
            view["user"] = self.env.user_tools.db.model_dump(mode="json")
        return Tau2View(view)

    def digest(self) -> str:
        both = {"db": self.env.get_db_hash(), "user_db": self.env.get_user_db_hash()}
        return hashlib.sha256(json.dumps(both, sort_keys=True).encode("utf-8")).hexdigest()

    def abort(self, reason: str) -> None:
        """End a conversation the worker stopped before tau2-bench did (budget or model failure)."""
        if self.ending is not None:
            return
        from tau2.data_model.simulation import TerminationReason  # type: ignore[import-untyped]

        self.orchestrator.done = True
        budget = "budget" in reason
        self.orchestrator.termination_reason = (
            TerminationReason.MAX_STEPS if budget else TerminationReason.AGENT_ERROR
        )
        self._end(f"worker: {reason}")

    # -- internals ----------------------------------------------------------------------------------------

    def _messages(self, role: str) -> list[Any]:
        return [m for m in self.orchestrator.trajectory if getattr(m, "role", None) == role]

    def _advance(self) -> None:
        """Step until the orchestrator needs the agent's next message or the conversation is over."""
        agent = _role("agent")
        while not self.orchestrator.done and self.orchestrator.to_role != agent:
            self.orchestrator.step()
            self.orchestrator._check_termination()

    def _end(self, by: str) -> None:
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation  # type: ignore[import-untyped]
        from tau2.orchestrator.modes import CommunicationMode  # type: ignore[import-untyped]  # no types

        simulation = self.orchestrator._finalize()
        simulation.policy = self.env.get_policy()
        simulation.reward_info = evaluate_simulation(
            simulation=simulation,
            task=self.orchestrator.task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=self.env.get_domain_name(),
            mode=CommunicationMode.HALF_DUPLEX,
        )
        self.ending = Ending(simulation, by)

    def _tool_result(self, call: ToolCall, new: list[Any]) -> ToolResult:
        result = next((m for m in new if getattr(m, "role", None) == "tool" and m.id == call.call_id), None)
        observation = str(result.content) if result is not None else ""
        self._touch(call.args, observation)
        data = {"observation": observation, "done": self.done}
        if result is not None and result.error:
            return ToolResult(call.call_id, False, data, observation)
        return ToolResult(call.call_id, True, data, None)

    def _reply(self, call: ToolCall, new: list[Any]) -> ToolResult:
        """The customer's answer, and every tool call the user made on their own device before it."""
        results = {m.id: m for m in new if getattr(m, "role", None) == "tool"}
        actions: list[dict[str, Any]] = []
        reply = ""
        for m in new:
            if getattr(m, "role", None) != "user":
                continue
            for tc in m.tool_calls or []:
                res = results.get(tc.id)
                actions.append(
                    {
                        "name": tc.name,
                        "arguments": dict(tc.arguments),
                        "result": str(res.content) if res is not None else None,
                        "error": bool(res.error) if res is not None else False,
                    }
                )
            if not m.is_tool_call() and m.content is not None:
                reply = str(m.content)
        self._touch({}, reply + "\n" + json.dumps(actions, default=str))
        data: dict[str, Any] = {"observation": reply, "done": self.done}
        if actions:
            data["user_actions"] = actions
        return ToolResult(call.call_id, True, data, None)

    def _touch(self, args: Mapping[str, Any], observation: str) -> None:
        if not self.view_spec.sliced:
            return
        db = self.env.tools.db.model_dump(mode="json")
        tokens = set(_TOKEN.findall(json.dumps(args, sort_keys=True) + "\n" + observation))
        for table in self.view_spec.sliced:
            self.touched.setdefault(table, set()).update(tokens & db[table].keys())


# -- grading ------------------------------------------------------------------------------------------------


def _short(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= 160 else text[:157] + "..."


def _field_diff(path: str, mine: Any, want: Any) -> str:
    """One field's difference; for two lists, the first element that differs."""
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
        mine, want = final.get(table), gold.get(table)
        if not isinstance(mine, dict) or not isinstance(want, dict):
            if mine != want:
                out.append(_field_diff(table, mine, want))
            continue
        for key in sorted(set(mine) | set(want)):
            a, b = mine.get(key), want.get(key)
            if a == b:
                continue
            if a is None or b is None:
                state = "is missing" if a is None else "should not exist"
                out.append(f"{table}['{key}'] {state} in the final database")
            elif isinstance(a, dict) and isinstance(b, dict):
                out += [
                    _field_diff(f"{table}['{key}'].{n}", a.get(n), b.get(n))
                    for n in sorted(set(a) | set(b))
                    if a.get(n) != b.get(n)
                ]
            else:
                out.append(_field_diff(f"{table}['{key}']", a, b))
    if len(out) > limit:
        out = [*out[:limit], f"... and {len(out) - limit} more differences"]
    return out


def gold_databases(task: Any, domain: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The agent and user databases the task's gold actions produce, built as tau2-bench's evaluator does."""
    from tau2.registry import registry  # type: ignore[import-untyped]  # no type info

    env = registry.get_env_constructor(domain)()
    state = task.initial_state
    env.set_state(
        initialization_data=state.initialization_data if state is not None else None,
        initialization_actions=state.initialization_actions if state is not None else None,
        message_history=list(state.message_history or []) if state is not None else [],
    )
    for action in task.evaluation_criteria.actions or []:
        try:
            env.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
        except Exception:  # tau2-bench's evaluator logs and skips a failing gold action too
            continue
    user = env.user_tools.db.model_dump(mode="json") if env.user_tools is not None else None
    return env.tools.db.model_dump(mode="json"), user


def reward_reasons(world: Tau2World) -> list[str]:
    """Why tau2-bench's reward is below 1, from its own reward breakdown."""
    assert world.ending is not None
    sim = world.ending.simulation
    info = sim.reward_info
    reasons: list[str] = []
    stop = {"user_stop", "agent_stop"}
    if str(sim.termination_reason.value) not in stop:
        reasons.append(f"the conversation ended by {sim.termination_reason.value}, not by the customer")
    if info is None:
        return reasons
    by_kind: dict[str, list[str]] = {}
    if info.db_check is not None and not info.db_check.db_match:
        gold, gold_user = gold_databases(world.orchestrator.task, world.env.get_domain_name())
        diffs = record_diffs(world.env.tools.db.model_dump(mode="json"), gold)
        if gold_user is not None and world.env.user_tools is not None:
            diffs += record_diffs(
                {"user": world.env.user_tools.db.model_dump(mode="json")}, {"user": gold_user}
            )
        by_kind["DB"] = diffs or ["the final database differs from the gold one"]
    by_kind["ACTION"] = [
        f"the agent never made the call {a.action.name}({_short(a.action.arguments)})"
        for a in info.action_checks or []
        if not a.action_match
    ]
    by_kind["COMMUNICATE"] = [
        f"the agent never told the user {c.info!r}" for c in info.communicate_checks or [] if not c.met
    ]
    by_kind["NL_ASSERTION"] = [
        f"not met: {n.nl_assertion} (judge: {n.justification})" for n in info.nl_assertions or [] if not n.met
    ]
    by_kind["ENV_ASSERTION"] = [
        f"environment assertion failed: {_short(e.env_assertion.model_dump(mode='json'))}"
        for e in info.env_assertions or []
        if not e.met
    ]
    # Checks in the task's reward basis decide the reward and come first; tau2-bench also runs checks
    # outside the basis, which are reported but marked, since they did not cost the reward.
    basis = {str(getattr(b, "value", b)) for b in info.reward_basis or []}
    for kind, found in by_kind.items():
        if kind in basis:
            reasons += found
    for kind, found in by_kind.items():
        if kind not in basis:
            reasons += [f"(not scored) {r}" for r in found]
    return reasons


def outcome_of(world: Tau2World, cost: Cost) -> Outcome:
    """tau2-bench's reward as an `Outcome`: passed iff tau2-bench counts it a success."""
    from tau2.metrics.agent_metrics import is_successful  # type: ignore[import-untyped]  # no type info

    if world.ending is None:
        world.abort("the worker stopped first")
    assert world.ending is not None
    passed = bool(is_successful(world.ending.reward))
    reasons = [] if passed else reward_reasons(world)
    if not passed and not reasons:
        reasons.append(f"tau2-bench reward is {world.ending.reward}")
    return Outcome(passed, reasons, world.digest(), cost)
