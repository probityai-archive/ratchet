"""Candidate synthesis: from one failure cluster, ask the model for guards, procedures and prompt edits.

The model sees the cluster's failing steps in context, the state view at the failing step (so a guard is
written against real field names), and what passing runs looked like. It never sees a held-out trace: the
caller hands this module training traces only.

The proposal instructions are themselves a versioned artifact (`Instructions`): the loop scores them by what
their candidates achieved and lets the model revise them.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ratchet_cases.types import Trace, canonical_json, sha256_hex

from ratchet.improve.diagnose import FailureCluster, FailurePoint, excerpt
from ratchet.improve.llm import Chat, Meter, ask_json
from ratchet.improve.replay import ReplayReport, WalkCache
from ratchet.improve.sim import call_of
from ratchet.types import Guard, Harness, Procedure

Kind = Literal["guard", "procedure", "prompt_edit"]
KINDS: tuple[Kind, ...] = ("guard", "procedure", "prompt_edit")

GUARD_API = """\
Guard API. A guard is Python source defining `def check(call, view, history)`, run before every tool call.
  call     has .name (str), .args (dict) and .call_id
  view     the current world state as a plain JSON dict (an example from the failing step is below)
  history  the list of earlier events, each with .seq, .kind ("model", "tool_call", "tool_result",
           "guard_block", ...) and .payload (a read-only mapping)
Return a short reason string to block the call (the agent reads it and can choose another action), or
None to allow it. The function must be pure and deterministic; only these modules may be imported: re,
math, decimal, fractions, datetime, collections, itertools, functools, json. A guard that raises blocks
every call, so guard every lookup (use .get)."""

SEED_INSTRUCTIONS = """\
You improve the harness of a customer-operations AI agent so that one cluster of recorded failures cannot
happen again, without blocking anything that already worked. You may propose three kinds of change.

1. guard: Python source (see the Guard API). Prefer a guard whenever the failure is a checkable property of
   a tool call and the current state, for example "never issue a refund for a charge that already has a
   succeeded refund at the provider". Key the check on facts in `view` and `history`, never on the wording
   of the agent's messages. Make it narrow: it must fire at or before the failing step in every failing
   trace and on no call in any passing trace. Write the block reason as an instruction: say what is wrong
   and what to do instead.
2. procedure: {"name", "when", "steps"}: a named sequence the agent must follow when `when` holds. Prefer
   a procedure when the failure is a missing sequence ("on a timeout, query the status before retrying")
   or something the agent never did.
3. prompt_edit: {"text"}: a short paragraph appended to the system prompt, only for a fact or policy the
   agent did not know.

Give several distinct candidates: at least two guards when the failure is checkable at a tool call (vary
the property checked), and at least one procedure."""

_SYSTEM = (
    "You are the improver in a self-improving agent harness. You write precise, minimal changes that fix a "
    "failure mechanism. Reply with one JSON object only."
)

_REPLY_SHAPE = """\
Reply with JSON:
{"candidates": [
  {"kind": "guard", "description": "<one line>", "source": "def check(call, view, history):\\n    ...",
   "rationale": "<why this fires on the failures and nowhere else>"},
  {"kind": "procedure", "name": "<name>", "when": "<trigger>", "steps": ["<step>", "..."],
   "rationale": "..."},
  {"kind": "prompt_edit", "text": "<paragraph>", "rationale": "..."}
]}"""


@dataclass(frozen=True)
class Instructions:
    """The improver's own proposal instructions: a content-addressed, versioned artifact."""

    version: str
    parent: str | None
    text: str

    @classmethod
    def build(cls, text: str, parent: str | None = None) -> Instructions:
        return cls(sha256_hex(text.encode("utf-8"))[:12], parent, text)


SEED = Instructions.build(SEED_INSTRUCTIONS)


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: Kind
    cluster_id: str
    instructions_version: str
    harness: Harness
    change: dict[str, Any]
    rationale: str
    targets: list[FailurePoint] = field(default_factory=list)

    def new_guards(self, parent: Harness) -> list[Guard]:
        old = {g.id for g in parent.guards}
        return [g for g in self.harness.guards if g.id not in old]

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "cluster_id": self.cluster_id,
            "instructions_version": self.instructions_version,
            "harness_version": self.harness.version,
            "change": self.change,
            "rationale": self.rationale,
            "targets": [t.id for t in self.targets],
        }


@dataclass
class Proposal:
    candidates: list[Candidate] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    duplicates: int = 0


def check_guard_source(source: str) -> str | None:
    """A reason the source cannot be a guard, or None. Compiling never executes it."""
    if not re.search(r"^def check\(\s*call\s*,\s*view\s*,\s*history\s*\)", source, re.MULTILINE):
        return "source does not define check(call, view, history) at top level"
    try:
        compile(source, "<guard>", "exec")
    except (SyntaxError, ValueError) as exc:
        return f"does not compile: {exc}"
    return None


def passing_shapes(passing: Sequence[Trace], limit: int = 8) -> list[str]:
    """The most common tool-call sequences among passing traces."""
    shapes = Counter(
        " -> ".join(call_of(e)["name"] for e in t.events if e.kind == "tool_call") or "(no tool calls)"
        for t in passing
    )
    return [f"{n}x  {shape}" for shape, n in shapes.most_common(limit)]


def _passing_contrast(passing: Sequence[Trace], tool: str) -> str:
    for t in passing:
        for e in t.events:
            if e.kind == "tool_call" and call_of(e)["name"] == tool:
                return excerpt(t, e.seq, before=4, after=1).replace(
                    "<== FAILING STEP", "<== same tool, passing"
                )
    return "(no passing trace calls this tool)"


def _view_at(walks: WalkCache | None, trace: Trace, seq: int, limit: int = 3000) -> str:
    if walks is None:
        return "(not available)"
    steps = walks.walk(trace).steps
    at = [s for s in steps if s.seq <= seq]
    if not at:
        return "(no tool call before the failing step)"
    text = json.dumps(at[-1].view, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def _harness_text(h: Harness) -> str:
    guards = "\n".join(f"  - {g.id}: {g.description}" for g in h.guards) or "  (none)"
    procs = (
        "\n".join(f"  - {p.name} (when {p.when}): {'; '.join(p.steps)}" for p in h.procedures) or "  (none)"
    )
    return (
        f"Tools allowed: {', '.join(h.tool_allowlist)}\nExisting guards:\n{guards}\n"
        f"Existing procedures:\n{procs}\nSystem prompt:\n{h.system_prompt[:2000]}"
    )


def build_prompt(
    cluster: FailureCluster,
    harness: Harness,
    instructions: Instructions,
    passing: Sequence[Trace],
    walks: WalkCache | None,
    n: int,
    inspiration: Harness | None = None,
) -> str:
    shown = cluster.members[:3]
    failing = "\n\n".join(
        f"Failure {m.id} (case family member, reason: {m.reason}; failing step {m.seq}, {m.mode}):\n"
        + excerpt(cluster.traces[m.run_id], m.seq)
        for m in shown
    )
    first = shown[0]
    view = _view_at(walks, cluster.traces[first.run_id], first.seq)
    return "\n\n".join(
        [
            instructions.text,
            GUARD_API,
            f"Current harness:\n{_harness_text(harness)}",
            *(
                [
                    "An earlier admitted harness, for ideas only (read-only: every change you propose "
                    f"applies to the current harness above):\n{_harness_text(inspiration)}"
                ]
                if inspiration is not None and inspiration.version != harness.version
                else []
            ),
            f"Failure cluster {cluster.id} ({len(cluster.members)} failures): {cluster.summary}\n"
            f"Mechanism key: {json.dumps(cluster.key)}",
            failing,
            f"State view just before failing step {first.seq} of {first.run_id}:\n{view}",
            "Passing runs (tool sequences):\n" + ("\n".join(passing_shapes(passing)) or "(none)"),
            "A passing run calling the same tool:\n" + _passing_contrast(passing, first.tool),
            f"Propose {n} candidates.\n{_REPLY_SHAPE}",
        ]
    )


def _short(*parts: str) -> str:
    return sha256_hex(canonical_json(list(parts)))[:8]


def make_candidate(
    raw: dict[str, Any], cluster: FailureCluster, parent: Harness, instructions: Instructions
) -> Candidate | str:
    """A candidate harness from one model item, or the reason it was rejected."""
    kind = raw.get("kind")
    rationale = str(raw.get("rationale", ""))
    origin = [m.id for m in cluster.members]
    if kind == "guard":
        source = str(raw.get("source", ""))
        problem = check_guard_source(source)
        if problem:
            return f"guard rejected: {problem}"
        gid = f"g-{cluster.id}-{_short(source)}"
        guard = Guard(gid, str(raw.get("description", ""))[:300], source, origin)
        child = parent.child(guards=[*parent.guards, guard], created_by="improver")
        return Candidate(f"c-{gid}", "guard", cluster.id, instructions.version, child, asdict(guard),
                         rationale, list(cluster.members))  # fmt: skip
    if kind == "procedure":
        steps = [str(s) for s in raw.get("steps") or [] if str(s).strip()]
        name, when = str(raw.get("name", "")).strip(), str(raw.get("when", "")).strip()
        if not (steps and name and when):
            return "procedure rejected: needs name, when and steps"
        pid = f"p-{cluster.id}-{_short(name, when, *steps)}"
        proc = Procedure(pid, name, when, steps, origin)
        child = parent.child(procedures=[*parent.procedures, proc], created_by="improver")
        return Candidate(f"c-{pid}", "procedure", cluster.id, instructions.version, child, asdict(proc),
                         rationale, list(cluster.members))  # fmt: skip
    if kind == "prompt_edit":
        text = str(raw.get("text", "")).strip()
        if not text:
            return "prompt_edit rejected: empty text"
        eid = f"e-{cluster.id}-{_short(text)}"
        child = parent.child(system_prompt=f"{parent.system_prompt}\n\n{text}", created_by="improver")
        return Candidate(f"c-{eid}", "prompt_edit", cluster.id, instructions.version, child, {"append": text},
                         rationale, list(cluster.members))  # fmt: skip
    return f"unknown kind {kind!r}"


def propose(
    cluster: FailureCluster,
    harness: Harness,
    model: Chat,
    *,
    instructions: Instructions = SEED,
    passing: Sequence[Trace] = (),
    walks: WalkCache | None = None,
    meter: Meter | None = None,
    n: int = 4,
    inspiration: Harness | None = None,
) -> Proposal:
    """Several candidates for one cluster. Unusable items are kept in `rejected` with the reason."""
    out = Proposal()
    prompt = build_prompt(cluster, harness, instructions, passing, walks, n, inspiration)
    try:
        reply = ask_json(model, _SYSTEM, prompt, meter or Meter())
    except Exception as exc:
        out.rejected.append({"cluster_id": cluster.id, "reason": f"model call failed: {exc}"})
        return out
    seen: set[str] = set()
    for raw in reply.get("candidates") or []:
        made = (
            make_candidate(raw, cluster, harness, instructions) if isinstance(raw, dict) else "not an object"
        )
        if isinstance(made, str):
            out.rejected.append({"cluster_id": cluster.id, "reason": made})
        elif made.id in seen or made.harness.version == harness.version:
            out.duplicates += 1
        else:
            seen.add(made.id)
            out.candidates.append(made)
    return out


def _replay_feedback(report: ReplayReport, cluster: FailureCluster, walks: WalkCache | None) -> str:
    lines = []
    for t in report.targets:
        if not t["hit"]:
            lines.append(f"MISSED {t['failure_id']}: failing step {t['failing_seq']}, first block "
                         f"{t['first_block_seq']} (must block at or before the failing step)")  # fmt: skip
    for fp in report.false_positives[:5]:
        lines.append(f"FALSE POSITIVE in passing run {fp['run_id']} at step {fp['seq']}: {fp['reason']}")
    for err in report.errors[:3]:
        lines.append(f"ERROR in run {err['run_id']} at step {err['seq']}: {err['error']}")
    if report.false_positives and walks is not None:
        fp = report.false_positives[0]
        for s in walks.walk_by_id(fp["run_id"]).steps if walks.has(fp["run_id"]) else []:
            if s.seq == fp["seq"]:
                view = json.dumps(s.view, sort_keys=True, default=str)[:2000]
                lines.append(f"Call blocked wrongly: {s.call.name} {json.dumps(s.call.args)}\nView: {view}")
    return "\n".join(lines)


def revise_guard(
    candidate: Candidate,
    report: ReplayReport,
    cluster: FailureCluster,
    parent: Harness,
    model: Chat,
    *,
    instructions: Instructions = SEED,
    walks: WalkCache | None = None,
    meter: Meter | None = None,
) -> Candidate | str:
    """One repair round: show the model exactly where replay failed and ask for a corrected guard."""
    source = str(candidate.change.get("source", ""))
    prompt = "\n\n".join(
        [
            GUARD_API,
            f"This guard was proposed for failure cluster {cluster.id} ({cluster.summary}):\n{source}",
            "Counterfactual replay over the recorded traces found:\n"
            + _replay_feedback(report, cluster, walks),
            "Failing step in context:\n"
            + excerpt(cluster.traces[cluster.members[0].run_id], cluster.members[0].seq),
            'Return the corrected guard as JSON: {"description": "...", "source": "...", "rationale": "..."}',
        ]
    )
    try:
        reply = ask_json(model, _SYSTEM, prompt, meter or Meter())
    except Exception as exc:
        return f"revision failed: {exc}"
    return make_candidate({**reply, "kind": "guard"}, cluster, parent, instructions)
