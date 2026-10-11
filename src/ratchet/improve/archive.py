"""The archive of every harness version, and parent selection by clade score.

The loop climbs one tip: each generation's parent is the last promoted version. The store already keeps the
whole tree, rejected candidates included (a decision names its candidate even when no harness was stored).
This module reads that tree as an archive: every version stays a node, and a later generation may draw its
ideas from any admitted version, not only the tip.

The clade score. A version's clade is itself plus every version below it. Its score is the pooled pass rate
of every FRESH case result on the archive's split across the clade, so a version counts once per
evaluation it received. Parent selection draws theta ~ Beta(tau * (1 + passes), tau * (1 + failures)) for
each qualifying clade and picks the largest (Thompson sampling, seeded so a run repeats). The qualifying
clades are those of admitted, stored versions with at least `min_descendants` attempted descendants.

Selection is a search heuristic only. CONTRACT "Admission rule (the ratchet)", ruling 2: search scores and
descendant productivity never authorize admission or activation. The chosen ancestor is the SOURCE of ideas;
the child it inspires is rebased onto the current active harness (`rebase`) and evaluated as that exact
bundle, and `admit()` compares it with that active harness. A rebased child's tree parent is therefore the
active harness, so the ancestor that inspired it is recorded as `context["archive"]["source"]` in the
decision, and the clade follows the SOURCE edge when there is one (`attribution_parent`).

The reporting metric next to the clade score is the "descendant admission yield": admitted descendants over
attempted descendants, rejected ones counted as attempts. Its formula, horizon (the whole stored history),
attribution (source edge, else comparison parent) and denominator (attempted descendants) are versioned in
`FORMULA`. The gain of an admitted version is its pooled pass rate minus its comparison parent's.

Held-out aggregates are evaluator-side. They choose which ancestor the proposer reads; they are never shown
to the proposer (CONTRACT "Evidence access"). `Archive.load(..., split="train")` is available for a
selector whose scores the proposer may see.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from ratchet.improve.admit import report_digest
from ratchet.store import LineageNode
from ratchet.types import Harness

FORMULA = "clade-pass-rate/1"
METRIC_LABEL = "descendant admission yield"
Split = Literal["train", "heldout"]
Status = Literal["seed", "admitted", "rejected", "unjudged", "unknown"]

# The keys `EvalReport.to_payload()` writes. The loop adds context keys (generation, role, harness, candidate)
# and stores add their own; only these keys enter the digest a decision names.
EVAL_PAYLOAD_KEYS = (
    "type",
    "harness_version",
    "train",
    "heldout",
    "errors",
    "fresh_calls",
    "cached_calls",
    "modes",
    "cost",
    "wall_s",
    "results",
)
_NAMED_REPORTS = ("parent_run_1", "parent_run_2", "candidate_run", "candidate_confirm_run")


class ArchiveStore(Protocol):
    """The read side of `ratchet.store.Store` that the archive needs."""

    def lineage(self, root: str | None = None) -> list[LineageNode]: ...
    def evals(self, filter: Mapping[str, Any] | None = None) -> list[dict[str, Any]]: ...
    def decisions(self, filter: Mapping[str, Any] | None = None) -> list[dict[str, Any]]: ...


class RebaseError(ValueError):
    """The child's change cannot be applied to the active harness as it stands."""


# -- evidence ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRef:
    """One stored evaluation of a version, reduced to its outcome counts on the archive's split."""

    eval_id: str
    role: str
    generation: int | None
    passes: int
    failures: int
    invalid: int
    named_by: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "role": self.role,
            "generation": self.generation,
            "passes": self.passes,
            "failures": self.failures,
            "invalid": self.invalid,
            "named_by": list(self.named_by),
        }


@dataclass(frozen=True)
class Evidence:
    """Pooled FRESH outcomes of one version. `invalid` (errors, RECORDED or CACHED results) is excluded."""

    passes: int = 0
    failures: int = 0
    invalid: int = 0
    evals: tuple[EvalRef, ...] = ()

    @property
    def n(self) -> int:
        return self.passes + self.failures

    @property
    def rate(self) -> float | None:
        return self.passes / self.n if self.n else None

    def to_payload(self) -> dict[str, Any]:
        return {
            "passes": self.passes,
            "failures": self.failures,
            "invalid": self.invalid,
            "rate": self.rate,
            "evals": [e.to_payload() for e in self.evals],
        }


@dataclass(frozen=True)
class Verdict:
    """What the stored decision says about a version: admitted or rejected, and the gate record."""

    decision_id: str
    admitted: bool
    gates: Mapping[str, bool]
    candidate_digest: str
    decided_at: str
    stable_recoveries: int | None
    parent_discordance: int | None
    decisions: int = 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "admitted": self.admitted,
            "gates": dict(self.gates),
            "candidate_digest": self.candidate_digest,
            "decided_at": self.decided_at,
            "stable_recoveries": self.stable_recoveries,
            "parent_discordance": self.parent_discordance,
            "decisions": self.decisions,
        }


@dataclass(frozen=True)
class Node:
    version: str
    parent: str | None
    source: str | None
    created_by: str
    status: Status
    stored: bool
    verdict: Verdict | None
    evidence: Evidence

    @property
    def eligible(self) -> bool:
        """May be a parent: its harness body is stored and it is the seed or was admitted."""
        return self.stored and self.status in ("seed", "admitted")


@dataclass(frozen=True)
class Clade:
    """The clade record of one version over its subtree (itself included in the pool)."""

    version: str
    descendants: int
    admitted_descendants: int
    rejected_descendants: int
    evaluated_descendants: int
    passes: int
    failures: int
    gain_n: int
    gain_sum: float | None
    gain_mean: float | None

    @property
    def rate(self) -> float | None:
        """The clade score: pooled clade passes over pooled clade outcomes."""
        n = self.passes + self.failures
        return self.passes / n if n else None

    @property
    def admission_yield(self) -> float | None:
        """The descendant admission yield: admitted over attempted descendants."""
        return self.admitted_descendants / self.descendants if self.descendants else None

    def is_thin(self, min_descendants: int) -> bool:
        return self.descendants < min_descendants

    def to_payload(self) -> dict[str, Any]:
        return {
            "formula": FORMULA,
            "descendants": self.descendants,
            "admitted_descendants": self.admitted_descendants,
            "rejected_descendants": self.rejected_descendants,
            "evaluated_descendants": self.evaluated_descendants,
            "passes": self.passes,
            "failures": self.failures,
            "clade_rate": self.rate,
            "admission_yield": self.admission_yield,
            "gain_n": self.gain_n,
            "gain_sum": self.gain_sum,
            "gain_mean": self.gain_mean,
        }


# -- the archive ------------------------------------------------------------------------------------------


class Archive:
    """Every version the store knows, as a forest: rejected candidates are nodes, admitted ones parents."""

    def __init__(self, nodes: Sequence[Node], *, split: Split = "heldout", notes: Sequence[str] = ()) -> None:
        self.split: Split = split
        self.notes: tuple[str, ...] = tuple(notes)
        self.nodes: dict[str, Node] = {n.version: n for n in nodes}
        self._children: dict[str, list[str]] = {v: [] for v in self.nodes}
        for n in nodes:
            parent = self.attribution_parent(n.version)
            if parent is not None:
                self._children[parent].append(n.version)

    # construction

    @classmethod
    def load(cls, store: ArchiveStore, *, split: Split = "heldout") -> Archive:
        """Read every harness version, evaluation and decision from the store."""
        harnesses = [n for root in store.lineage() for n in root.walk()]
        harnesses.sort(key=lambda n: (n.created_at is None, n.created_at, n.version))
        return cls.from_records(
            [{"version": h.version, "parent": h.parent, "created_by": h.created_by} for h in harnesses],
            store.evals({"type": "eval"}),
            store.decisions(),
            split=split,
        )

    @classmethod
    def from_records(
        cls,
        harnesses: Iterable[Mapping[str, Any]],
        evals: Iterable[Mapping[str, Any]],
        decisions: Iterable[Mapping[str, Any]],
        *,
        split: Split = "heldout",
    ) -> Archive:
        """Build from plain records: stored harness versions, `type == "eval"` documents, decision documents.

        Harness records are in creation order. A decision whose candidate has no harness record becomes a node
        with `stored=False`; a parent that no record describes becomes an `unknown` node.
        """
        stored = {h["version"]: h for h in harnesses}
        latest, counts = _latest_decisions(decisions)
        builders: dict[str, dict[str, Any]] = {}
        for version, h in stored.items():
            builders[version] = {"parent": h.get("parent"), "created_by": h.get("created_by", "unknown")}
        for version, d in latest.items():
            builders.setdefault(version, {"parent": d["parent"], "created_by": "improver"})
        for version in list(builders):
            parent = builders[version]["parent"]
            if parent is not None and parent not in builders:
                builders[parent] = {"parent": None, "created_by": "unknown"}
        refs = _eval_refs(evals, split, _named_digests(latest.values()))
        notes: list[str] = []
        parents = {v: b["parent"] for v, b in builders.items()}
        nodes: list[Node] = []
        for version, b in builders.items():
            decided = latest.get(version)
            source, note = _valid_source(version, decided, parents)
            if note:
                notes.append(note)
            nodes.append(
                Node(
                    version=version,
                    parent=b["parent"],
                    source=source,
                    created_by=b["created_by"],
                    status=_status(b, decided, version in stored),
                    stored=version in stored,
                    verdict=_verdict(decided, counts.get(version, 1)),
                    evidence=_evidence(refs.get(version, [])),
                )
            )
        return cls(nodes, split=split, notes=notes)

    @classmethod
    def from_run_dir(cls, path: Path, *, split: Split = "heldout") -> Archive:
        """Build from a run directory of `harnesses.jsonl`, `evals.jsonl` and `decisions.jsonl`."""
        evals = [r for r in _jsonl(path / "evals.jsonl") if r.get("type") == "eval"]
        decisions = [{"_id": r.get("decision_id"), **r} for r in _jsonl(path / "decisions.jsonl")]
        evals = [{"_id": r.get("eval_id"), **r} for r in evals]
        return cls.from_records(_jsonl(path / "harnesses.jsonl"), evals, decisions, split=split)

    # tree

    def attribution_parent(self, version: str) -> str | None:
        """The node whose clade `version` belongs to: its source ancestor if recorded, else its parent."""
        n = self.nodes[version]
        return n.source or n.parent

    @property
    def tip(self) -> str | None:
        """The last stored version that may be a parent, in creation order."""
        eligible = [v for v, n in self.nodes.items() if n.eligible]
        return eligible[-1] if eligible else None

    def clade_members(self, version: str) -> Iterator[str]:
        """`version` and every version below it, breadth first."""
        queue = deque([version])
        while queue:
            v = queue.popleft()
            yield v
            queue.extend(self._children[v])

    def clade(self, version: str) -> Clade:
        """Clade statistics of `version` (see the module docstring for the score)."""
        members = list(self.clade_members(version))
        below = [self.nodes[v] for v in members[1:]]
        gains = [g for n in below if n.status == "admitted" and (g := self.gain(n.version)) is not None]
        return Clade(
            version=version,
            descendants=len(below),
            admitted_descendants=sum(1 for n in below if n.status == "admitted"),
            rejected_descendants=sum(1 for n in below if n.status == "rejected"),
            evaluated_descendants=sum(1 for n in below if n.evidence.n),
            passes=sum(self.nodes[v].evidence.passes for v in members),
            failures=sum(self.nodes[v].evidence.failures for v in members),
            gain_n=len(gains),
            gain_sum=sum(gains) if gains else None,
            gain_mean=sum(gains) / len(gains) if gains else None,
        )

    def gain(self, version: str) -> float | None:
        """Pass-rate gain of `version` over its comparison parent on this split, or None when unmeasured."""
        n = self.nodes[version]
        if n.parent is None:
            return None
        mine, theirs = n.evidence.rate, self.nodes[n.parent].evidence.rate
        return None if mine is None or theirs is None else mine - theirs

    def to_payload(self) -> dict[str, Any]:
        """JSON for the VIEW lane: every node with its status, evidence and clade statistics."""
        return {
            "type": "archive",
            "formula": FORMULA,
            "metric": METRIC_LABEL,
            "split": self.split,
            "tip": self.tip,
            "notes": list(self.notes),
            "nodes": [self._node_payload(n) for n in self.nodes.values()],
        }

    def _node_payload(self, n: Node) -> dict[str, Any]:
        return {
            "version": n.version,
            "parent": n.parent,
            "source": n.source,
            "attribution_parent": self.attribution_parent(n.version),
            "created_by": n.created_by,
            "status": n.status,
            "stored": n.stored,
            "eligible": n.eligible,
            "verdict": n.verdict.to_payload() if n.verdict else None,
            "evidence": n.evidence.to_payload(),
            "gain": self.gain(n.version) if n.status == "admitted" else None,
            "clade": self.clade(n.version).to_payload(),
        }


# -- record parsing ---------------------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _decided_key(d: Mapping[str, Any]) -> str:
    return str(d.get("created_at") or d.get("decided_at") or "")


def _latest_decisions(
    decisions: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, int]]:
    """The last decision for each candidate version (by time, then input order) and how many there were."""
    latest: dict[str, Mapping[str, Any]] = {}
    counts: dict[str, int] = {}
    for d in decisions:
        v = d["candidate"]
        counts[v] = counts.get(v, 0) + 1
        if v not in latest or _decided_key(d) >= _decided_key(latest[v]):
            latest[v] = d
    return latest, counts


def _named_digests(decisions: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Report digest -> "decision:role" for every evaluation a decision names."""
    out: dict[str, list[str]] = {}
    for d in decisions:
        did = str(d.get("_id") or f"{d['parent']}->{d['candidate']}")
        for role in _NAMED_REPORTS:
            digest = (d.get("report_digests") or {}).get(role)
            if digest:
                out.setdefault(digest, []).append(f"{did}:{role}")
    return out


def _eval_refs(
    evals: Iterable[Mapping[str, Any]], split: Split, named: Mapping[str, list[str]]
) -> dict[str, list[EvalRef]]:
    out: dict[str, list[EvalRef]] = {}
    seen: set[str] = set()
    for doc in evals:
        if doc.get("type") != "eval" or "harness_version" not in doc:
            continue
        eid = str(doc.get("_id") or doc.get("eval_id") or report_digest(_payload_of(doc)))
        if eid in seen:
            continue
        seen.add(eid)
        passes = failures = invalid = 0
        for r in doc.get("results", []):
            if r.get("split") != split:
                continue
            if r.get("mode", "FRESH") != "FRESH" or r.get("error"):
                invalid += 1
            elif r.get("passed"):
                passes += 1
            else:
                failures += 1
        ref = EvalRef(
            eval_id=eid,
            role=str(doc.get("role", "")),
            generation=doc.get("generation"),
            passes=passes,
            failures=failures,
            invalid=invalid,
            named_by=tuple(named.get(report_digest(_payload_of(doc)), ())),
        )
        out.setdefault(doc["harness_version"], []).append(ref)
    return out


def _payload_of(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {k: doc[k] for k in EVAL_PAYLOAD_KEYS if k in doc}


def _evidence(refs: Sequence[EvalRef]) -> Evidence:
    return Evidence(
        passes=sum(r.passes for r in refs),
        failures=sum(r.failures for r in refs),
        invalid=sum(r.invalid for r in refs),
        evals=tuple(refs),
    )


def _status(b: Mapping[str, Any], d: Mapping[str, Any] | None, is_stored: bool) -> Status:
    if d is not None:
        return "admitted" if d.get("admitted") else "rejected"
    if b["parent"] is None:
        return "seed" if is_stored else "unknown"
    return "unjudged"


def _verdict(d: Mapping[str, Any] | None, count: int) -> Verdict | None:
    if d is None:
        return None
    checks = {c["name"]: bool(c["passed"]) for c in d.get("checks", [])}
    fresh = next((c for c in d.get("checks", []) if c["name"] == "fresh"), None)
    evidence = (fresh or {}).get("evidence", {})
    stable = evidence.get("stable_recoveries")
    return Verdict(
        decision_id=str(d.get("_id") or f"{d['parent']}->{d['candidate']}"),
        admitted=bool(d.get("admitted")),
        gates=checks,
        candidate_digest=str(d.get("candidate_digest", "")),
        decided_at=str(d.get("decided_at", "")),
        stable_recoveries=len(stable) if isinstance(stable, list) else None,
        parent_discordance=evidence.get("parent_discordance"),
        decisions=count,
    )


def _tree_ancestors(version: str, parents: Mapping[str, str | None]) -> list[str]:
    out: list[str] = []
    seen = {version}
    cur = parents.get(version)
    while cur is not None and cur not in seen:
        out.append(cur)
        seen.add(cur)
        cur = parents.get(cur)
    return out


def _valid_source(
    version: str, d: Mapping[str, Any] | None, parents: Mapping[str, str | None]
) -> tuple[str | None, str | None]:
    """The recorded source ancestor, kept only when it is on the node's own ancestry (clades stay trees)."""
    source = ((d or {}).get("context") or {}).get("archive", {}).get("source")
    if source is None or source == parents[version]:
        return None, None
    if source in _tree_ancestors(version, parents):
        return str(source), None
    return None, f"{version}: recorded source {source} is not an ancestor; credited to its parent"


# -- parent selection -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParentChoice:
    """Which ancestor the next generation draws its ideas from, and why. Data, not authority."""

    version: str
    tip: str
    reason: Mapping[str, Any]

    @property
    def rebase_required(self) -> bool:
        """True when the chosen ancestor is not the active harness: its child must be rebased onto the tip."""
        return self.version != self.tip

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "archive_parent_choice",
            "source": self.version,
            "comparison_parent": self.tip,
            "rebase_required": self.rebase_required,
            "reason": dict(self.reason),
        }

    def decision_context(self) -> dict[str, Any]:
        """The `context["archive"]` entry for the decision of a child drawn from this choice."""
        return {
            "archive": {"source": self.version, "comparison_parent": self.tip, "reason": dict(self.reason)}
        }


def select_parent(
    archive: Archive,
    min_descendants: int,
    *,
    tip: str | None = None,
    seed: int | str = 0,
    explore: bool = True,
    tau: float = 1.0,
) -> ParentChoice:
    """The admitted, stored version with the best clade among clades of at least `min_descendants` attempts.

    With `explore` (the default) the best is drawn by Thompson sampling from
    Beta(tau * (1 + passes), tau * (1 + failures)) over each clade's pooled outcomes, with an RNG seeded by
    `seed` so a run is reproducible; without it, the highest clade score wins. No qualifying clade
    returns the tip.
    The returned version is where ideas come from. It never changes which harness a candidate is compared to.
    """
    if min_descendants < 1:
        raise ValueError("min_descendants must be at least 1")
    if tau <= 0:
        raise ValueError("tau must be positive")
    active = tip or archive.tip
    if active is None or active not in archive.nodes:
        raise ValueError("the archive has no active tip to fall back on")
    clades = [(v, archive.clade(v)) for v, n in sorted(archive.nodes.items()) if n.eligible]
    rows = [(v, c) for v, c in clades if c.descendants >= min_descendants]
    base: dict[str, Any] = {
        "formula": FORMULA,
        "split": archive.split,
        "min_descendants": min_descendants,
        "seed": seed,
        "tau": tau,
        "candidates": [],
    }
    if not rows:
        return ParentChoice(active, active, {**base, "kind": "tip", "why": "no clade has enough descendants"})
    rng = random.Random(str(seed))
    scored = []
    for v, c in rows:
        theta = rng.betavariate(tau * (1 + c.passes), tau * (1 + c.failures))
        scored.append((theta if explore else (c.rate if c.rate is not None else -1.0), v, c, theta))
    best = max(scored, key=lambda s: (s[0], s[1]))
    base["candidates"] = [
        {
            "version": v,
            "descendants": c.descendants,
            "admitted_descendants": c.admitted_descendants,
            "passes": c.passes,
            "failures": c.failures,
            "clade_rate": c.rate,
            "admission_yield": c.admission_yield,
            "gain_mean": c.gain_mean,
            "sample": theta if explore else None,
        }
        for _, v, c, theta in scored
    ]
    kind = "thompson" if explore else "greedy"
    return ParentChoice(best[1], active, {**base, "kind": kind, "why": f"highest {kind} clade score"})


@dataclass
class ArchiveSelector:
    """What `evolve(..., select_parent=...)` calls each generation: reload the archive, then choose.

    The seed is combined with the generation so each generation draws afresh and a rerun repeats exactly.
    """

    store: ArchiveStore
    min_descendants: int
    seed: int | str = 0
    explore: bool = True
    tau: float = 1.0
    split: Split = "heldout"
    last: ParentChoice | None = field(default=None, init=False)

    def __call__(self, tip: str, generation: int) -> ParentChoice:
        archive = Archive.load(self.store, split=self.split)
        self.last = select_parent(
            archive,
            self.min_descendants,
            tip=tip,
            seed=f"{self.seed}:{generation}",
            explore=self.explore,
            tau=self.tau,
        )
        return self.last


# -- rebasing ---------------------------------------------------------------------------------------------


def rebase(child: Harness, source_parent: Harness, base: Harness) -> Harness:
    """The change `child` makes to `source_parent`, applied to `base`: the exact bundle to evaluate.

    CONTRACT ruling 2: a child of an old ancestor is rebased onto the active harness and evaluated as that
    bundle. `child` must extend `source_parent` the way an automatic candidate may: its guards and procedures
    keep the parent's in order and append, its prompt starts with the parent's prompt, its tool allowlist is a
    subset. Appended items already present in `base` unchanged are skipped; an id taken by different content
    raises. The result's parent is `base`.
    """
    guards = _appended(child.guards, source_parent.guards, "guard")
    procs = _appended(child.procedures, source_parent.procedures, "procedure")
    if not child.system_prompt.startswith(source_parent.system_prompt):
        raise RebaseError("the child's prompt does not start with its parent's prompt")
    suffix = child.system_prompt[len(source_parent.system_prompt) :]
    if not set(child.tool_allowlist) <= set(source_parent.tool_allowlist):
        raise RebaseError("the child widens the tool allowlist")
    removed = set(source_parent.tool_allowlist) - set(child.tool_allowlist)
    merged_guards = _merge(base.guards, guards, "guard")
    merged_procs = _merge(base.procedures, procs, "procedure")
    prompt = base.system_prompt if not suffix or suffix in base.system_prompt else base.system_prompt + suffix
    allow = [t for t in base.tool_allowlist if t not in removed]
    if (merged_guards, merged_procs, prompt, allow) == (
        base.guards,
        base.procedures,
        base.system_prompt,
        base.tool_allowlist,
    ):
        raise RebaseError("the change is already present in the active harness")
    return base.child(
        guards=merged_guards,
        procedures=merged_procs,
        system_prompt=prompt,
        tool_allowlist=allow,
        created_by="improver",
    )


def _appended(child_items: Sequence[Any], parent_items: Sequence[Any], what: str) -> list[Any]:
    n = len(parent_items)
    if list(child_items[:n]) != list(parent_items):
        raise RebaseError(f"the child does not keep its parent's {what}s unchanged and in order")
    return list(child_items[n:])


def _merge(existing: Sequence[Any], added: Sequence[Any], what: str) -> list[Any]:
    by_id = {x.id: x for x in existing}
    out = list(existing)
    for item in added:
        if item.id not in by_id:
            out.append(item)
            by_id[item.id] = item
        elif by_id[item.id] != item:
            raise RebaseError(f"the active harness already has a different {what} with id {item.id}")
    return out


# -- command line -----------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Write the archive of one or more run directories, or of an Atlas database, as JSON."""
    ap = argparse.ArgumentParser(prog="python -m ratchet.improve.archive", description=main.__doc__)
    ap.add_argument(
        "run_dirs", nargs="*", type=Path, help="run directories with harnesses/evals/decisions.jsonl"
    )
    ap.add_argument("--uri", help="MongoDB URI; with --db reads the archive from Atlas instead")
    ap.add_argument("--db", help="database name")
    ap.add_argument("--split", choices=("train", "heldout"), default="heldout")
    ap.add_argument("--min-descendants", type=int, default=2)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--out", type=Path, help="write here instead of standard output")
    args = ap.parse_args(argv)
    if bool(args.uri) != bool(args.db) or bool(args.uri) == bool(args.run_dirs):
        ap.error("give run directories, or --uri together with --db")
    sources: list[tuple[str, Archive]] = []
    if args.uri:
        from ratchet.store import Store

        with Store(args.uri, args.db) as store:
            sources.append((args.db, Archive.load(store, split=args.split)))
    else:
        sources = [(p.name, Archive.from_run_dir(p, split=args.split)) for p in args.run_dirs]
    out = {
        "type": "archive_export",
        "runs": [
            {
                "name": name,
                "archive": a.to_payload(),
                "selection": select_parent(a, args.min_descendants, seed=args.seed).to_payload()
                if a.tip
                else None,
            }
            for name, a in sources
        ],
    }
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
