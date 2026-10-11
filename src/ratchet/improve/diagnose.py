"""Group failed traces by failure mechanism and name the step in each trace that caused the failure.

Grouping is deterministic: the cluster key is built from the grader reason (with ids and numbers
normalised away), whether the failure is something the agent did or something it never did, the tool
called at the failing step, the error the world returned just before it, and whether the call repeats an
earlier one. The model only writes the one-line mechanism summary for each cluster; it never decides
membership.

The failing step comes from grading intermediate states: the earliest tool call after which the reason
holds and keeps holding. A reason that holds before the agent acts and is never cleared is an omission,
located at the trace's last event.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ratchet.improve.evaluate import is_infra_failure
from ratchet.improve.llm import Chat, Meter, ask_json
from ratchet.improve.sim import Simulator, Walk, call_of, last_seq
from ratchet.types import Event, Trace, canonical_json, sha256_hex

Mode = Literal["commission", "omission", "unlocated"]

_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_IDLIKE = re.compile(r"\b[\w.-]*\d[\w.-]*\b")
# Record ids, emails and slugs: any token joining word characters with "@", ".", "-" or "_".
_COMPOUND = re.compile(r"[\w]+(?:[@.\-_][\w]+)+")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """A reason with its specifics removed, so the same mechanism on different records matches."""
    out = _QUOTED.sub("<s>", text.lower())
    out = _COMPOUND.sub("<id>", out)
    out = _IDLIKE.sub("<id>", out)
    return _SPACE.sub(" ", out).strip()


@dataclass(frozen=True)
class FailurePoint:
    run_id: str
    case_id: str
    seq: int
    reason: str
    mode: Mode
    tool: str

    @property
    def id(self) -> str:
        return f"{self.run_id}@{self.seq}"


@dataclass(frozen=True)
class FailureCluster:
    id: str
    key: dict[str, str]
    members: list[FailurePoint]
    summary: str = ""
    traces: dict[str, Trace] = field(default_factory=dict, compare=False, repr=False)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "summary": self.summary,
            "members": [asdict(m) | {"id": m.id} for m in self.members],
        }


def _norms(reasons: Sequence[str] | None) -> set[str]:
    return {normalize(r) for r in reasons or ()}


def locate(trace: Trace, walk: Walk | None, reason: str) -> tuple[int, Mode]:
    """The seq of the event that caused `reason`, and whether it was an act or an omission."""
    end = last_seq(trace)
    if walk is None or walk.initial_reasons is None or any(s.reasons_after is None for s in walk.steps):
        return _locate_by_mention(trace, reason)
    n = normalize(reason)
    holds = [n in _norms(s.reasons_after) for s in walk.steps]
    i = len(holds)
    while i > 0 and holds[i - 1]:
        i -= 1
    if i == len(holds) or (i == 0 and n in _norms(walk.initial_reasons)):
        return end, "omission"
    return walk.steps[i].seq, "commission"


def _locate_by_mention(trace: Trace, reason: str) -> tuple[int, Mode]:
    """Without intermediate grading: the last tool call whose arguments name an id the reason names."""
    ids = set(_IDLIKE.findall(reason))
    hit: int | None = None
    for e in trace.events:
        if e.kind == "tool_call" and ids & set(_IDLIKE.findall(json.dumps(call_of(e)["args"]))):
            hit = e.seq
    if hit is not None:
        return hit, "commission"
    return last_seq(trace), "unlocated"


def _features(trace: Trace, seq: int) -> dict[str, str]:
    before = [e for e in trace.events if e.seq < seq]
    at = next((e for e in trace.events if e.seq == seq), None)
    tool = call_of(at)["name"] if at is not None and at.kind == "tool_call" else "<end>"
    prior_error = ""
    for e in reversed(before):
        if e.kind == "tool_result":
            p = e.payload.get("result", e.payload)
            if p.get("ok") is False:
                prior_error = normalize(str(p.get("error") or "error"))
            break
    repeat = ""
    if at is not None and at.kind == "tool_call":
        c = call_of(at)
        earlier = [call_of(e) for e in before if e.kind == "tool_call"]
        if any(x["name"] == c["name"] and x["args"] == c["args"] for x in earlier):
            repeat = "same_args"
        elif any(x["name"] == c["name"] for x in earlier):
            repeat = "same_tool"
    return {"tool": tool, "prior_error": prior_error, "repeat": repeat}


def failure_points(trace: Trace, walk: Walk | None) -> list[FailurePoint]:
    points: list[FailurePoint] = []
    for reason in trace.outcome.reasons or ["failed without a reason"]:
        seq, mode = locate(trace, walk, reason)
        tool = _features(trace, seq)["tool"]
        points.append(FailurePoint(trace.run_id, trace.case_id, seq, reason, mode, tool))
    return points


def _key(trace: Trace, point: FailurePoint) -> dict[str, str]:
    return {"reason": normalize(point.reason), "mode": point.mode, **_features(trace, point.seq)}


def render_event(e: Event, limit: int = 400) -> str:
    body = json.dumps(e.payload, sort_keys=True, default=str)
    if len(body) > limit:
        body = body[:limit] + "...(truncated)"
    return f"[{e.seq}] {e.kind}: {body}"


def excerpt(trace: Trace, seq: int, before: int = 8, after: int = 2) -> str:
    """The events around `seq`, the failing step marked, for a model to read."""
    idx = next((i for i, e in enumerate(trace.events) if e.seq >= seq), len(trace.events) - 1)
    lo, hi = max(0, idx - before), min(len(trace.events), idx + after + 1)
    lines = [render_event(e) + ("    <== FAILING STEP" if e.seq == seq else "") for e in trace.events[lo:hi]]
    return "\n".join(lines)


def _default_summary(key: dict[str, str], n: int) -> str:
    what = "did" if key["mode"] == "commission" else "never did"
    bits = [f"{n} failure(s): the agent {what} something that led to '{key['reason']}'"]
    if key["tool"] != "<end>":
        bits.append(f"at a {key['tool']} call")
    if key["prior_error"]:
        bits.append(f"right after the world returned '{key['prior_error']}'")
    if key["repeat"]:
        bits.append(f"repeating an earlier call ({key['repeat']})")
    return ", ".join(bits) + "."


_SUMMARY_SYSTEM = (
    "You read agent failure traces and name the failure mechanism in one sentence: what the agent did "
    "(or failed to do), in which state, and which property of the state or call it ignored. "
    'Reply with JSON: {"mechanism": "<one sentence>"}.'
)


def diagnose(
    traces: Sequence[Trace],
    simulator: Simulator | None = None,
    model: Chat | None = None,
    meter: Meter | None = None,
    on_walk_error: Callable[[Trace, Exception], None] | None = None,
) -> list[FailureCluster]:
    """Cluster the failed traces in `traces` (passing ones are ignored), largest cluster first."""
    groups: dict[str, tuple[dict[str, str], list[FailurePoint]]] = {}
    by_run: dict[str, Trace] = {}
    for t in traces:
        if t.outcome.passed or is_infra_failure(t):  # an endpoint failure says nothing about the harness
            continue
        by_run[t.run_id] = t
        walk: Walk | None = None
        if simulator is not None:
            try:
                walk = simulator.walk(t)
            except Exception as exc:  # one unreplayable trace must not sink the diagnosis
                if on_walk_error is not None:
                    on_walk_error(t, exc)
        for p in failure_points(t, walk):
            key = _key(t, p)
            cid = sha256_hex(canonical_json(key))[:10]
            groups.setdefault(cid, (key, []))[1].append(p)
    clusters: list[FailureCluster] = []
    for cid, (key, members) in sorted(groups.items(), key=lambda kv: (-len(kv[1][1]), kv[0])):
        summary = _default_summary(key, len(members))
        if model is not None:
            shown = "\n\n".join(excerpt(by_run[m.run_id], m.seq) for m in members[:2])
            user = (
                f"Cluster key: {json.dumps(key)}\nReasons: {sorted({m.reason for m in members})}\n\n{shown}"
            )
            try:
                summary = (
                    str(ask_json(model, _SUMMARY_SYSTEM, user, meter or Meter())["mechanism"]) or summary
                )
            except Exception as exc:  # the deterministic summary stands, and says why
                summary += f" (model summary unavailable: {type(exc).__name__})"
        traces_of = {m.run_id: by_run[m.run_id] for m in members}
        clusters.append(FailureCluster(cid, key, members, summary, traces_of))
    return clusters
