"""What the views read from Atlas, as plain typed values.

Every view takes these values and never touches the store, so a view can be tested with a handful of
dictionaries and a script can render the same data to a terminal, a test or a static page.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ratchet.types import Guard, Harness, Trace

Doc = dict[str, Any]
Status = Literal["seed", "promoted", "admitted", "rejected"]


class ViewStore(Protocol):
    """The slice of `ratchet.store.Store` the views read. The real store satisfies it structurally."""

    def lineage(self, root: str | None = None) -> Sequence[Any]: ...
    def get_harness(self, version: str) -> Harness | None: ...
    def evals(self, filter: Mapping[str, Any] | None = None) -> list[Doc]: ...
    def decisions(self, filter: Mapping[str, Any] | None = None) -> list[Doc]: ...
    def traces(
        self, filter: Mapping[str, Any] | None = None, *, limit: int = 0, newest_first: bool = False
    ) -> list[Trace]: ...
    def get_trace(self, run_id: str, case_id: str) -> Trace | None: ...
    def watch(
        self, collection: str, *, timeout_s: float | None = None, resume_after: Doc | None = None
    ) -> Iterator[Any]: ...
    def close(self) -> None: ...


class StoreFactory(Protocol):
    def __call__(self) -> AbstractContextManager[ViewStore]: ...


@dataclass(frozen=True)
class VersionStats:
    """Results of one harness version over the whole case suite."""

    train_passed: int = 0
    train_total: int = 0
    heldout_passed: int = 0
    heldout_total: int = 0
    interventions: int = 0
    false_interventions: int = 0
    model_calls: int = 0


def stats_from_eval(doc: Mapping[str, Any]) -> VersionStats:
    """Counts from a stored eval report. A false intervention is a harness block on a case that needed none:
    anything but a trap, where a block is the harness doing its job."""
    results = doc.get("results") or []
    train, heldout = doc.get("train") or {}, doc.get("heldout") or {}
    return VersionStats(
        train_passed=int(train.get("passed", 0)),
        train_total=int(train.get("total", 0)),
        heldout_passed=int(heldout.get("passed", 0)),
        heldout_total=int(heldout.get("total", 0)),
        interventions=sum(int(r.get("interventions", 0)) for r in results),
        false_interventions=sum(int(r.get("interventions", 0)) for r in results if r.get("kind") != "trap"),
        model_calls=int((doc.get("cost") or {}).get("model_calls", 0)),
    )


def receipt_id(envelope_b64: str | None) -> str | None:
    """A short, stable name for a signed receipt: the SHA-256 of the envelope bytes."""
    if not envelope_b64:
        return None
    try:
        raw = base64.b64decode(envelope_b64, validate=True)
    except ValueError:
        return None
    return "rcpt:" + hashlib.sha256(raw).hexdigest()[:10]


def check_reasons(check: Mapping[str, Any]) -> list[str]:
    """Why one admission check failed, in the words of the demo ("blocks 4 past successes")."""
    if check.get("passed"):
        return []
    name, ev = str(check.get("name")), check.get("evidence") or {}
    if name == "replay":
        return _replay_reasons(ev)
    if name == "heldout":
        return _heldout_reasons(ev)
    if name == "no_regression":
        n = len(ev.get("regressed") or [])
        return [f"regresses {n} must-succeed case{'s' if n != 1 else ''}"]
    if name == "benign_twins":
        n = len(ev.get("intervened") or [])
        return [f"intervenes on {n} held-out benign twin{'s' if n != 1 else ''}"]
    if name == "overfit_lint":
        hits = ev.get("hits") or []
        shown = ", ".join(str(h.get("literal")) for h in hits[:3])
        return [f"copies {len(hits)} literal(s) from training traces ({shown})"]
    if name == "authority":
        return _authority_reasons(ev)
    return [f"{name} failed"]


def _heldout_reasons(ev: Mapping[str, Any]) -> list[str]:
    if not ev.get("same_case_set", True):
        return ["held-out case set differs from the parent's"]
    got, need = ev.get("candidate_fresh_passed"), ev.get("required")
    if not ev.get("initial_ok", False):
        best = max(ev.get("parent_fresh_passed") or [0])
        return [f"held-out fresh passes {got}, needs {need} (parent best {best})"]
    if ev.get("confirm_pending"):
        return ["awaiting the confirming held-out run"]
    return [f"confirming run passed {ev.get('confirm_fresh_passed')}, needs {need}"]


def _replay_reasons(ev: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for rep in ev.get("reports") or []:
        if rep.get("missing"):
            out.append(f"no replay recorded for {rep.get('guard_id')}")
            continue
        twins = int(rep.get("benign_twin_fires", 0))
        fps = len(rep.get("false_positives") or []) - twins
        if fps > 0:
            out.append(f"blocks {fps} past success{'es' if fps != 1 else ''}")
        if twins:
            out.append(f"fires on {twins} benign twin{'s' if twins != 1 else ''}")
        miss = int(rep.get("targets_total", 0)) - int(rep.get("targets_hit", 0))
        if miss > 0:
            out.append(f"misses {miss} of {rep.get('targets_total')} target failures")
        if rep.get("errors"):
            out.append(f"{rep['errors']} guard error(s)")
    return out or ["replay failed"]


def _authority_reasons(ev: Mapping[str, Any]) -> list[str]:
    out = []
    if ev.get("tools_granted"):
        out.append("grants tools: " + ", ".join(ev["tools_granted"]))
    if ev.get("guards_removed"):
        out.append("removes guards: " + ", ".join(ev["guards_removed"]))
    if ev.get("guards_altered"):
        out.append("alters guards: " + ", ".join(ev["guards_altered"]))
    if not ev.get("digest_valid", True):
        out.append("digest does not match content")
    return out or ["authority grew"]


@dataclass(frozen=True)
class DecisionInfo:
    """One admission decision with its receipt."""

    id: str
    parent: str
    candidate: str
    admitted: bool
    checks: list[Doc]
    context: Doc
    receipt: str | None
    keyid: str | None
    decided_at: str

    @property
    def failed(self) -> list[str]:
        return [str(c["name"]) for c in self.checks if not c.get("passed")]

    def reasons(self, *, skip: Sequence[str] = ()) -> list[str]:
        """Plain-language reasons for every failed gate (optionally leaving some gates out)."""
        out: list[str] = []
        for c in self.checks:
            if c.get("name") not in skip:
                out += check_reasons(c)
        return out

    @property
    def kind(self) -> str:
        return str(self.context.get("kind") or ("composite" if self.context.get("composite_of") else "?"))

    @property
    def generation(self) -> int:
        return int(self.context.get("generation") or 0)

    @property
    def what(self) -> str:
        """The change in a few words: the guard's description, the procedure's name, or 'prompt edit'."""
        change = self.context.get("change") or {}
        if self.kind == "guard":
            return str(change.get("description") or change.get("id") or "guard")
        if self.kind == "procedure":
            return str(change.get("name") or "procedure")
        if self.kind == "prompt_edit":
            return "prompt edit"
        n = len(self.context.get("composite_of") or [])
        return f"composite of {n} changes" if n else "change"


def decision_info(doc: Mapping[str, Any]) -> DecisionInfo:
    return DecisionInfo(
        id=str(doc.get("_id", f"{doc.get('parent')}->{doc.get('candidate')}")),
        parent=str(doc.get("parent", "")),
        candidate=str(doc.get("candidate", "")),
        admitted=bool(doc.get("admitted")),
        checks=[dict(c) for c in doc.get("checks") or []],
        context=dict(doc.get("context") or {}),
        receipt=receipt_id(doc.get("envelope")),
        keyid=doc.get("keyid"),
        decided_at=str(doc.get("decided_at", "")),
    )


@dataclass
class VersionNode:
    """One harness version in the tree, with how it did and how it came to be."""

    version: str
    parent: str | None
    created_by: str
    status: Status
    stored: bool = True
    stats: VersionStats | None = None
    decision: DecisionInfo | None = None
    children: list[VersionNode] = field(default_factory=list)

    def walk(self) -> Iterator[VersionNode]:
        yield self
        for c in self.children:
            yield from c.walk()


def latest_stats(evals: Sequence[Mapping[str, Any]]) -> dict[str, VersionStats]:
    """The newest full evaluation of each harness version (docs arrive oldest first)."""
    out: dict[str, VersionStats] = {}
    for doc in evals:
        if doc.get("type") == "eval" and doc.get("harness_version") and doc.get("results") is not None:
            out[str(doc["harness_version"])] = stats_from_eval(doc)
    return out


def all_decisions(store: ViewStore) -> list[DecisionInfo]:
    return [decision_info(d) for d in store.decisions() if d.get("type", "decision") == "decision"]


def build_lineage(store: ViewStore) -> list[VersionNode]:
    """The version tree: stored versions from `$graphLookup`, plus every candidate a decision names.

    Only the seed and the promoted versions are stored as harness versions; a rejected candidate exists
    only as a decision, so the tree merges the two to show every child the improver tried.
    """
    nodes: dict[str, VersionNode] = {}
    order: list[VersionNode] = []

    def add(n: VersionNode) -> None:
        nodes[n.version] = n
        order.append(n)

    for root in store.lineage():
        for ln in root.walk():
            add(VersionNode(ln.version, ln.parent, ln.created_by, "promoted"))
    decisions = sorted(all_decisions(store), key=lambda d: d.decided_at)
    by_candidate: dict[str, DecisionInfo] = {}
    for d in decisions:
        if d.candidate not in by_candidate or d.admitted or not by_candidate[d.candidate].admitted:
            by_candidate[d.candidate] = d
        if d.candidate not in nodes:
            add(VersionNode(d.candidate, d.parent, "improver", "rejected", stored=False))
    stats = latest_stats(store.evals())
    roots: list[VersionNode] = []
    for n in order:
        n.stats = stats.get(n.version)
        n.decision = by_candidate.get(n.version)
        parent = nodes.get(n.parent) if n.parent else None
        n.status = _status(n, parent is None)
        if parent is None:
            roots.append(n)
        else:
            parent.children.append(n)
    return roots


def _status(n: VersionNode, is_root: bool) -> Status:
    if n.decision is None:
        return "seed" if is_root else "promoted"
    if not n.decision.admitted:
        return "rejected"
    return "promoted" if n.stored else "admitted"


@dataclass(frozen=True)
class CandidateRef:
    """A candidate's new guards, ready to replay, with the decision already taken on it (if any)."""

    candidate: str
    parent: str
    guards: list[Guard]
    decision: DecisionInfo | None = None
    note: str = ""


def _guard_from_change(change: Mapping[str, Any]) -> Guard | None:
    try:
        return Guard(
            id=str(change["id"]),
            description=str(change.get("description", "")),
            source=str(change["source"]),
            origin_failure_ids=[str(x) for x in change.get("origin_failure_ids") or []],
        )
    except KeyError:
        return None


def _guards_of(store: ViewStore, d: DecisionInfo, decisions: Sequence[DecisionInfo]) -> list[Guard]:
    harness, parent = store.get_harness(d.candidate), store.get_harness(d.parent)
    if harness is not None and parent is not None:
        old = {g.id for g in parent.guards}
        return [g for g in harness.guards if g.id not in old]
    if d.kind == "guard":
        g = _guard_from_change(d.context.get("change") or {})
        return [g] if g else []
    out: list[Guard] = []
    for pick in d.context.get("composite_of") or []:
        for other in decisions:
            if other.context.get("id") == pick and other.kind == "guard":
                g = _guard_from_change(other.context.get("change") or {})
                if g and g.id not in {x.id for x in out}:
                    out.append(g)
    return out


def resolve_candidates(store: ViewStore, refs: Sequence[str]) -> list[CandidateRef]:
    """Candidates by harness version, candidate id or unique prefix; with none named, the guard candidates
    of the latest generation."""
    decisions = all_decisions(store)
    if refs:
        picked = [_match(decisions, r) for r in refs]
    else:
        gen = max((d.generation for d in decisions), default=0)
        picked = [d for d in decisions if d.generation == gen and gen > 0]
    out: list[CandidateRef] = []
    for ref, d in zip(refs or [""] * len(picked), picked, strict=True):
        if d is None:
            out.append(_bare_ref(store, ref))
            continue
        guards = _guards_of(store, d, decisions)
        note = "" if guards else f"{d.kind} candidate: no guard to replay"
        out.append(CandidateRef(d.candidate, d.parent, guards, d, note))
    return out


def _match(decisions: Sequence[DecisionInfo], ref: str) -> DecisionInfo | None:
    exact = [d for d in decisions if ref in (d.candidate, d.id, d.context.get("id"))]
    pool = exact or [d for d in decisions if d.candidate.startswith(ref)]
    if not pool:
        return None
    return max(pool, key=lambda d: (d.admitted, d.decided_at))


def _bare_ref(store: ViewStore, version: str) -> CandidateRef:
    """A harness version with no decision: its guards are those the parent does not have."""
    h = store.get_harness(version)
    if h is None or h.parent is None:
        return CandidateRef(version, "", [], None, "unknown candidate: no decision and no stored harness")
    parent = store.get_harness(h.parent)
    old = {g.id for g in parent.guards} if parent else set()
    return CandidateRef(version, h.parent, [g for g in h.guards if g.id not in old], None)


class RunStore(Protocol):
    """The write side the improvement loop needs (structurally the loop's own `ImproveStore`)."""

    def put_harness(self, harness: Harness) -> bool: ...
    def put_trace(self, trace: Trace) -> bool: ...
    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str: ...
    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str: ...


class WriteStore(ViewStore, RunStore, Protocol):
    """What `run` and `evolve` need from the store: everything the views read, and the write side."""


def find_decision_doc(store: ViewStore, ref: str) -> Doc | None:
    """A decision by id (`parent->candidate`), by candidate version, or by unique candidate prefix."""
    docs = [d for d in store.decisions() if d.get("type", "decision") == "decision"]
    pool = [d for d in docs if d.get("_id") == ref] or [d for d in docs if d.get("candidate") == ref]
    pool = pool or [d for d in docs if str(d.get("candidate", "")).startswith(ref)]
    if not pool or not ref:
        return None
    return max(pool, key=lambda d: (bool(d.get("admitted")), str(d.get("decided_at", ""))))
