"""The worker loop: run one case under one harness and record everything as a trace.

Order of checks for every tool call the model makes:
  1. the call's arguments must have parsed as a JSON object,
  2. the tool must be on the harness allowlist,
  3. every guard runs in the sandbox (fail closed) on the call, the state view and the history so far.
Any failure returns an error to the model as the tool's result and records a `guard_block` event; the world
is not touched. Otherwise the world executes the call and a `tool_result` event records the result.

Events and what they carry (the improver and the store read these keys):
  model        model, content, tool_calls, finish_reason, usage{input_tokens, output_tokens}, latency_s
  tool_call    call{name, args, call_id}, view (the JSON state view the guards saw)
  guard_block  call_id, source (arguments | allowlist | guard), message, decisions (every guard's decision)
  tool_result  result{call_id, ok, data, error}, decisions
  procedure    name, how (keyword | selected), matched (keyword) or call_id (selected), found
  checkpoint   executed (world calls so far); written before the state is stored
  final        reason (answered | step budget | token budget | model error), content, complete
The `history` a guard sees for a call is every event recorded before that call's `tool_call` event, which is
exactly what replay reconstructs from a stored trace.

Procedure selection. The system prompt always lists every procedure's name and `when` line, so the model
can see the whole playbook. A procedure's full steps are put in the prompt up front when its `when` line
shares at least two content words with the customer message (or its only content word, for a one-word
line); the model can load any other procedure's steps itself by calling `read_procedure`. Both paths are
recorded as `procedure` events. The keyword rule is explicit and deterministic, so the improver can predict
from a trace whether a procedure was in front of the model; the `read_procedure` tool covers paraphrases
the keywords miss, and the event tells a diagnosis apart "never loaded" from "loaded and not followed".
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

from ratchet.agent.guards import GuardDecision, GuardRuntime
from ratchet.agent.model import ChatModel, Message, ModelError, ModelTurn, ToolSpec
from ratchet.cases.grade import grade
from ratchet.cases.model import Case
from ratchet.types import Event, EventKind, Harness, Outcome, Procedure, ToolCall, ToolResult, Trace
from ratchet.world import World

READ_PROCEDURE = "read_procedure"

_STOPWORDS = frozenset(
    {
        "been",
        "from",
        "have",
        "into",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "were",
        "when",
        "where",
        "which",
        "while",
        "will",
        "with",
        "without",
        "about",
        "after",
        "again",
        "asks",
        "before",
        "customer",
        "customers",
        "does",
        "more",
        "most",
        "other",
        "over",
        "some",
        "such",
        "than",
        "very",
        "what",
        "your",
    }
)


class CheckpointStore(Protocol):
    def checkpoint(self, run_id: str, state: Mapping[str, Any]) -> int: ...

    def resume(self, run_id: str) -> Mapping[str, Any] | None: ...


class ResumeError(RuntimeError):
    """A checkpoint could not be resumed faithfully (it belongs to another run, or the world diverged)."""


@dataclass(frozen=True)
class RunLimits:
    max_steps: int = 30
    max_tokens: int = 400_000


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4 and w not in _STOPWORDS}


def keyword_matches(procedures: list[Procedure], message: str) -> list[tuple[Procedure, list[str]]]:
    """Procedures whose `when` line shares enough content words with the message, with the words matched."""
    words = content_words(message)
    out: list[tuple[Procedure, list[str]]] = []
    for proc in procedures:
        wanted = content_words(proc.when)
        hit = sorted(wanted & words)
        if wanted and len(hit) >= min(2, len(wanted)):
            out.append((proc, hit))
    return out


def render_steps(proc: Procedure) -> str:
    lines = [f"Procedure: {proc.name}", f"When: {proc.when}"]
    lines += [f"{i}. {step}" for i, step in enumerate(proc.steps, start=1)]
    return "\n".join(lines)


def system_prompt(harness: Harness, preloaded: list[Procedure]) -> str:
    if not harness.procedures:
        return harness.system_prompt
    index = "\n".join(f"- {p.name}: {p.when}" for p in harness.procedures)
    parts = [
        harness.system_prompt,
        "## Procedures\nThese are the company's procedures. When your situation matches a procedure's "
        f"condition, follow its steps. Call {READ_PROCEDURE} with a procedure's name to read steps not "
        f"shown below.\n{index}",
    ]
    parts += [render_steps(p) for p in preloaded]
    return "\n\n".join(parts)


def read_procedure_spec(harness: Harness) -> ToolSpec:
    return {
        "type": "function",
        "function": {
            "name": READ_PROCEDURE,
            "description": "Read the full steps of one of the company's procedures by name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": [p.name for p in harness.procedures]}},
                "required": ["name"],
            },
        },
    }


def default_run_id(harness: Harness, case: Case, model_name: str) -> str:
    key = f"{case.id}\x00{harness.version}\x00{model_name}"
    return "run-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _call_json(call: ToolCall) -> dict[str, Any]:
    return {"name": call.name, "args": call.args, "call_id": call.call_id}


def _call_from(raw: Mapping[str, Any]) -> ToolCall:
    return ToolCall(name=str(raw["name"]), args=dict(raw["args"]), call_id=str(raw["call_id"]))


def _event_from(raw: Mapping[str, Any]) -> Event:
    return Event(seq=int(raw["seq"]), kind=raw["kind"], payload=dict(raw["payload"]), ts=float(raw["ts"]))


class _Run:
    """Mutable state of one run. Everything in `state()` is JSON, so a checkpoint can restore it."""

    def __init__(
        self,
        harness: Harness,
        case: Case,
        model: ChatModel,
        run_id: str,
        world: World,
        runtime: GuardRuntime,
        limits: RunLimits,
        store: CheckpointStore | None,
    ) -> None:
        self.harness, self.case, self.model, self.run_id = harness, case, model, run_id
        self.world, self.runtime, self.limits, self.store = world, runtime, limits, store
        self.events: list[Event] = []
        self.messages: list[Message] = []
        self.executed: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.arg_errors: dict[str, str] = {}
        self.steps = 0
        self.tokens = 0
        self.procedures = {p.name: p for p in harness.procedures}
        self.tools = [t for t in world.tools() if t["function"]["name"] in set(harness.tool_allowlist)]
        if harness.procedures:
            if READ_PROCEDURE in world.tool_names():
                raise ValueError(f"the world defines a tool named {READ_PROCEDURE!r}")
            self.tools.append(read_procedure_spec(harness))

    # -- recording ----------------------------------------------------------------------------------------

    def record(self, kind: EventKind, payload: dict[str, Any]) -> Event:
        event = Event(seq=len(self.events), kind=kind, payload=payload, ts=time.time())
        self.events.append(event)
        return event

    def state(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "run_id": self.run_id,
            "case_id": self.case.id,
            "harness_version": self.harness.version,
            "model": self.model.name,
            "events": [asdict(e) for e in self.events],
            "messages": self.messages,
            "executed": self.executed,
            "pending": self.pending,
            "arg_errors": self.arg_errors,
            "steps": self.steps,
            "tokens": self.tokens,
            "world_digest": self.world.digest(),
        }

    def checkpoint(self) -> None:
        if self.store is None:
            return
        # Recorded before the write so the stored state contains its own checkpoint event.
        self.record("checkpoint", {"executed": len(self.executed)})
        self.store.checkpoint(self.run_id, self.state("running"))

    # -- start and resume ---------------------------------------------------------------------------------

    def start(self) -> None:
        preloaded = keyword_matches(self.harness.procedures, self.case.customer_message)
        for proc, words in preloaded:
            self.record("procedure", {"name": proc.name, "how": "keyword", "matched": words, "found": True})
        self.messages = [
            {"role": "system", "content": system_prompt(self.harness, [p for p, _ in preloaded])},
            {"role": "user", "content": f"Ticket {self.case.ticket_id}\n\n{self.case.customer_message}"},
        ]

    def restore(self, saved: Mapping[str, Any]) -> None:
        for key, want in (
            ("case_id", self.case.id),
            ("harness_version", self.harness.version),
            ("model", self.model.name),
        ):
            if saved.get(key) != want:
                raise ResumeError(f"checkpoint {key} is {saved.get(key)!r}, this run has {want!r}")
        for raw in saved["executed"]:
            self.world.call(_call_from(raw))
        if self.world.digest() != saved["world_digest"]:
            raise ResumeError(
                "world state after re-applying the executed calls does not match the checkpoint"
            )
        self.events = [_event_from(e) for e in saved["events"]]
        self.messages = list(saved["messages"])
        self.executed = list(saved["executed"])
        self.pending = list(saved["pending"])
        self.arg_errors = dict(saved["arg_errors"])
        self.steps = int(saved["steps"])
        self.tokens = int(saved["tokens"])

    # -- the loop -----------------------------------------------------------------------------------------

    def loop(self) -> tuple[str, str | None]:
        """Run until the model answers without tool calls or a budget runs out. Returns (reason, content)."""
        self.drain_pending()
        while True:
            if self.steps >= self.limits.max_steps:
                return "step budget", None
            if self.tokens >= self.limits.max_tokens:
                return "token budget", None
            try:
                turn = self.model.complete(self.messages, self.tools)
            except ModelError as exc:
                return f"model error: {exc}", None
            self.steps += 1
            self.tokens += turn.input_tokens + turn.output_tokens
            self.record_turn(turn)
            if not turn.tool_calls:
                return "answered", turn.content
            self.pending = [_call_json(c) for c in turn.tool_calls]
            self.arg_errors = dict(turn.arg_errors)
            self.drain_pending()

    def record_turn(self, turn: ModelTurn) -> None:
        self.messages.append(turn.message)
        self.record(
            "model",
            {
                "model": self.model.name,
                "content": turn.content,
                "tool_calls": [_call_json(c) for c in turn.tool_calls],
                "finish_reason": turn.finish_reason,
                "usage": {"input_tokens": turn.input_tokens, "output_tokens": turn.output_tokens},
                "latency_s": turn.latency_s,
                "arg_errors": turn.arg_errors,
            },
        )

    def drain_pending(self) -> None:
        while self.pending:
            call = _call_from(self.pending[0])
            content = self.handle(call)
            self.messages.append({"role": "tool", "tool_call_id": call.call_id, "content": content})
            self.pending = self.pending[1:]
            self.checkpoint()

    def handle(self, call: ToolCall) -> str:
        if call.name == READ_PROCEDURE:
            return self.read_procedure(call)
        view = self.world.view().to_json()
        history = list(self.events)
        self.record("tool_call", {"call": _call_json(call), "view": view})
        if call.call_id in self.arg_errors:
            return self.block(call, "arguments", f"invalid arguments: {self.arg_errors[call.call_id]}", [])
        if call.name not in self.harness.tool_allowlist:
            return self.block(call, "allowlist", f"tool {call.name!r} is not available to this agent", [])
        decisions = self.runtime.check_all(self.harness.guards, call, view, history)
        blocks = [d for d in decisions if d.blocked]
        if blocks:
            message = "; ".join(f"blocked by guard {d.guard_id}: {d.reason}" for d in blocks)
            return self.block(call, "guard", message, decisions)
        result = self.world.call(call)
        self.executed.append(_call_json(call))
        payload = {"result": asdict(result), "decisions": [d.to_payload() for d in decisions]}
        self.record("tool_result", payload)
        return _result_content(result)

    def block(self, call: ToolCall, source: str, message: str, decisions: list[GuardDecision]) -> str:
        self.record(
            "guard_block",
            {
                "call_id": call.call_id,
                "source": source,
                "message": message,
                "decisions": [d.to_payload() for d in decisions],
            },
        )
        return json.dumps({"ok": False, "error": message})

    def read_procedure(self, call: ToolCall) -> str:
        name = str(call.args.get("name", ""))
        proc = self.procedures.get(name)
        self.record(
            "procedure", {"name": name, "how": "selected", "call_id": call.call_id, "found": proc is not None}
        )
        if proc is None:
            return json.dumps({"ok": False, "error": f"no procedure named {name!r}"})
        return json.dumps({"ok": True, "data": {"steps": render_steps(proc)}})


def _result_content(result: ToolResult) -> str:
    return json.dumps({"ok": result.ok, "data": result.data, "error": result.error})


Grader = Callable[[Case, World, Trace], Outcome]


def run_case(
    harness: Harness,
    case: Case,
    model: ChatModel,
    store: CheckpointStore | None = None,
    *,
    run_id: str | None = None,
    limits: RunLimits | None = None,
    runtime: GuardRuntime | None = None,
    world_factory: Callable[[Case], World] = World.from_case,
    grader: Grader = grade,
) -> Trace:
    """Run one case. With a `store`, the run checkpoints after every tool call and a checkpoint left by a
    killed run with the same `run_id` is resumed. A run that does not end with the model's answer (a budget
    ran out or the model failed) is graded and then marked failed with an `incomplete` reason."""
    rid = run_id or default_run_id(harness, case, model.name)
    own_runtime = runtime is None
    guards = runtime or GuardRuntime()
    try:
        run = _Run(harness, case, model, rid, world_factory(case), guards, limits or RunLimits(), store)
        saved = store.resume(rid) if store is not None else None
        if saved is not None and saved.get("status") == "running":
            run.restore(saved)
        else:
            run.start()
        reason, content = run.loop()
    finally:
        if own_runtime:
            guards.close()
    complete = reason == "answered"
    run.record("final", {"reason": reason, "content": content, "complete": complete})
    draft = Trace(rid, case.id, harness.version, model.name, list(run.events), Outcome(False, [], ""))
    outcome = grader(case, run.world, draft)
    if not complete:
        outcome = replace(outcome, passed=False, reasons=[*outcome.reasons, f"incomplete: {reason}"])
    if store is not None:
        store.checkpoint(rid, run.state("final"))
    return replace(draft, outcome=outcome)
