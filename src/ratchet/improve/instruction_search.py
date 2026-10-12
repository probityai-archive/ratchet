"""Level 2 search: evolve the improver's proposal instructions on a Pareto frontier over failure clusters.

One text component, `{"instructions": <text>}`, is evolved by reflective mutation and merge. A data instance
is one training failure cluster. Evaluating an instruction text on a cluster means: the improver proposes
candidates with that text (`propose`), each candidate is screened for copied training literals
(`overfit.lint`), and each candidate guard is replayed through the guard runtime over the recorded training
traces (`replay`). The cluster's score is the share of the requested candidate slots that are guards which
clear replay and survive the screen. Replay calls no model, so the search spends only proposer and
reflection calls; no worker model call is made.

The reflection step reads the replay evidence: missed target failures, false fires on passing and
benign-twin traces with the call that was wrongly blocked, runtime errors and copied literals.

The search writes only the improver's instructions. The worker harness changes only through `admit()`, and
`compare_level2` judges every searched instruction text by what its candidates achieve under the real
admission rule, on clusters the search never saw.

The firewall: the search receives training failure clusters only. A cluster built from a held-out or audit
case, a cluster of the frozen level-2 test batch, or a clearance trace from outside the training split is
refused before any proposal is made.

    uv run --extra gepa python -m ratchet.improve.instruction_search --out private/improve/l2.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ratchet_cases.types import Trace, canonical_json, sha256_hex

from ratchet.improve.admit import GATES, Decision, admit, awaiting_confirmation
from ratchet.improve.diagnose import FailureCluster, FailurePoint, diagnose
from ratchet.improve.evaluate import CaseLike, EvalReport, RunCase, evaluate_many
from ratchet.improve.evaluator import evaluator_identity
from ratchet.improve.llm import Chat, Meter
from ratchet.improve.loop import ArmScore, propose_instructions
from ratchet.improve.overfit import Literals, lint, training_literals
from ratchet.improve.propose import SEED, Candidate, Instructions, Proposal, propose
from ratchet.improve.receipt import Signer, receipt
from ratchet.improve.replay import GuardRunner, ReplayReport, WalkCache, replay
from ratchet.improve.sim import Simulator
from ratchet.types import Guard, Harness

if TYPE_CHECKING:
    from gepa.core.adapter import EvaluationBatch, ProposalFn
    from gepa.core.result import GEPAResult

COMPONENT = "instructions"
ENGINE = "pareto-reflective"

REFLECTION_TEMPLATE = """\
You maintain the instructions an improver model follows when it proposes changes to the harness of a
customer-operations AI agent. The harness changes are guards (Python checks run before every tool call),
procedures and prompt edits, each meant to stop one cluster of recorded failures without blocking anything
that already worked.

The current instructions:
```
<curr_param>
```

Below, for several failure clusters, is what the improver proposed when following these instructions and
what counterfactual replay over the recorded training traces found. A guard counts only when it blocks every
target failure at or before its failing step, blocks no call in any passing or benign-twin trace, and copies
no record id, amount or name from the traces. Procedures and prompt edits cannot be proven by replay, so
they score nothing here.
```
<side_info>
```

Write improved instructions that raise, across all clusters, the share of requested candidates that are
guards clearing replay. Read every miss and every wrongly blocked call: say what property a guard must key
on, and what it must leave alone. Keep the three kinds (guard, procedure, prompt_edit). The guard API and the
reply format are appended separately, so do not restate them. Never quote record ids, amounts or names from
the examples. Provide the new instructions within ``` blocks."""


class HeldOutRefusedError(ValueError):
    """A held-out, audit or frozen-test item was offered to the level-2 search."""


@dataclass(frozen=True)
class Firewall:
    """What level-2 search may never read: non-training cases and the frozen level-2 test batch."""

    train_case_ids: frozenset[str]
    frozen_cluster_ids: frozenset[str] = frozenset()

    @classmethod
    def from_cases(cls, cases: Iterable[CaseLike], frozen: Iterable[FailureCluster] = ()) -> Firewall:
        return cls(frozenset(c.id for c in cases if c.split == "train"), frozenset(c.id for c in frozen))

    def check_clusters(self, clusters: Iterable[FailureCluster]) -> None:
        bad: list[str] = []
        for c in clusters:
            if c.id in self.frozen_cluster_ids:
                bad.append(f"cluster {c.id} is in the frozen level-2 test batch")
            read = {m.case_id for m in c.members} | {t.case_id for t in c.traces.values()}
            outside = sorted(read - self.train_case_ids)
            if outside:
                bad.append(f"cluster {c.id} reads non-training cases {outside}")
        if bad:
            raise HeldOutRefusedError("; ".join(bad))

    def check_traces(self, traces: Iterable[Trace]) -> None:
        outside = sorted({t.case_id for t in traces} - self.train_case_ids)
        if outside:
            raise HeldOutRefusedError(f"clearance traces from non-training cases {outside}")


@dataclass(frozen=True)
class CandidateOutcome:
    """What happened to one proposed candidate: screened, replayed, or not replayable."""

    id: str
    kind: str
    status: str  # cleared, replay_failed, replay_incomplete, screened, not_replayable
    change: dict[str, Any]
    lint_hits: list[dict[str, str]] = field(default_factory=list)
    replay: dict[str, Any] | None = None
    blocked_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cleared(self) -> bool:
        return self.status == "cleared"

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "change": self.change,
            "lint_hits": self.lint_hits,
            "replay": self.replay,
            "blocked_calls": self.blocked_calls,
        }


@dataclass(frozen=True)
class ClusterOutcome:
    """One instruction text evaluated on one cluster: the search's output and trajectory for that instance."""

    cluster_id: str
    instructions_version: str
    slots: int
    candidates: list[CandidateOutcome]
    rejected: list[str]
    duplicates: int
    proposer: dict[str, int]
    summary: str = ""

    @property
    def cleared(self) -> int:
        return sum(1 for c in self.candidates if c.cleared)

    @property
    def score(self) -> float:
        """Cleared guards over requested slots, so invalid, duplicate and missing candidates count against."""
        return self.cleared / self.slots if self.slots else 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "instructions_version": self.instructions_version,
            "slots": self.slots,
            "cleared": self.cleared,
            "score": round(self.score, 4),
            "candidates": [c.to_payload() for c in self.candidates],
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "proposer": self.proposer,
        }


class ProposeFn(Protocol):
    def __call__(
        self,
        cluster: FailureCluster,
        harness: Harness,
        model: Chat,
        *,
        instructions: Instructions,
        passing: Sequence[Trace],
        walks: WalkCache | None,
        meter: Meter | None,
        n: int,
    ) -> Proposal: ...


class ReplayFn(Protocol):
    def __call__(
        self,
        guards: Guard | Sequence[Guard],
        targets: Sequence[FailurePoint],
        failing: Mapping[str, Trace],
        passing: Iterable[Trace],
        walks: WalkCache,
        runner: GuardRunner,
        benign: Iterable[Trace] = (),
    ) -> ReplayReport: ...


def meter_payload(m: Meter) -> dict[str, int]:
    return {"model_calls": m.model_calls, "input_tokens": m.input_tokens, "output_tokens": m.output_tokens}


def _merge(into: Meter, other: Meter) -> None:
    into.model_calls += other.model_calls
    into.input_tokens += other.input_tokens
    into.output_tokens += other.output_tokens


def blocked_calls(report: ReplayReport, walks: WalkCache) -> list[dict[str, Any]]:
    """The first false fires with the call each one wrongly blocked, read back from the walked trace."""
    out: list[dict[str, Any]] = []
    for fp in report.false_positives[:5]:
        call: dict[str, Any] = {}
        if walks.has(fp["run_id"]):
            step = next((s for s in walks.walk_by_id(fp["run_id"]).steps if s.seq == fp["seq"]), None)
            if step is not None:
                call = {"name": step.call.name, "args": step.call.args}
        out.append(
            {
                "run_id": fp["run_id"],
                "seq": fp["seq"],
                "benign_twin": bool(fp.get("benign_twin", False)),
                "reason": fp["reason"],
                "call": call,
            }
        )
    return out


_STATUS = {"CLEARED": "cleared", "REJECTED": "replay_failed"}


@dataclass
class Screen:
    """The improver-side checks, on training data only: copied literals and replay over training traces."""

    parent: Harness
    passing: list[Trace]
    benign: list[Trace]
    walks: WalkCache
    runner: GuardRunner
    literals: Literals
    replay_fn: ReplayFn = replay
    lock: threading.Lock = field(default_factory=threading.Lock)

    def outcome(self, c: Candidate, cluster: FailureCluster) -> CandidateOutcome:
        hits = lint(self.parent, c.harness, self.literals)
        if hits:
            return CandidateOutcome(c.id, c.kind, "screened", c.change, hits)
        if c.kind != "guard":
            return CandidateOutcome(c.id, c.kind, "not_replayable", c.change)
        with self.lock:  # the guard runtime and the walk cache are shared between threads
            rep = self.replay_fn(
                c.harness.guards[-1],
                cluster.members,
                cluster.traces,
                self.passing,
                self.walks,
                self.runner,
                self.benign,
            )
            blocked = blocked_calls(rep, self.walks)
        status = _STATUS.get(rep.verdict, "replay_incomplete")
        return CandidateOutcome(c.id, c.kind, status, c.change, [], rep.to_payload(), blocked)


def proposal_outcome(
    cluster: FailureCluster, ins: Instructions, prop: Proposal, screen: Screen, n: int, meter: Meter
) -> ClusterOutcome:
    return ClusterOutcome(
        cluster.id,
        ins.version,
        n,
        [screen.outcome(c, cluster) for c in prop.candidates],
        [str(r["reason"]) for r in prop.rejected],
        prop.duplicates,
        meter_payload(meter),
        cluster.summary,
    )


class InstructionsAdapter:
    """The search adapter: scores an instruction text on training failure clusters by proposal and replay."""

    propose_new_texts: ProposalFn | None = None

    def __init__(
        self,
        *,
        screen: Screen,
        proposer: Chat,
        firewall: Firewall,
        seed: Instructions = SEED,
        n: int = 4,
        workers: int = 4,
        propose_fn: ProposeFn = propose,
    ) -> None:
        firewall.check_traces([*screen.passing, *screen.benign])
        self.screen = screen
        self.proposer = proposer
        self.firewall = firewall
        self.seed = seed
        self.n = n
        self.workers = max(1, workers)
        self.propose_fn = propose_fn
        self.meter = Meter()
        self.evaluations: list[dict[str, Any]] = []
        self.texts: dict[str, Instructions] = {}
        self._lock = threading.Lock()

    def instructions_for(self, text: str) -> Instructions:
        """The content-addressed version of a text; every searched text descends from the seed version."""
        ins = self.seed if text == self.seed.text else Instructions.build(text, parent=self.seed.version)
        self.texts.setdefault(ins.version, ins)
        return ins

    def run_one(self, cluster: FailureCluster, ins: Instructions) -> ClusterOutcome:
        meter = Meter()
        prop = self.propose_fn(
            cluster,
            self.screen.parent,
            self.proposer,
            instructions=ins,
            passing=self.screen.passing,
            walks=self.screen.walks,
            meter=meter,
            n=self.n,
        )
        out = proposal_outcome(cluster, ins, prop, self.screen, self.n, meter)
        with self._lock:
            _merge(self.meter, meter)
        return out

    def evaluate(
        self, batch: list[FailureCluster], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch[ClusterOutcome, ClusterOutcome]:
        from gepa.core.adapter import EvaluationBatch

        self.firewall.check_clusters(batch)
        ins = self.instructions_for(candidate[COMPONENT])
        with ThreadPoolExecutor(max_workers=min(self.workers, max(1, len(batch)))) as pool:
            outs = list(pool.map(lambda cl: self.run_one(cl, ins), batch))
        with self._lock:
            self.evaluations += [
                {
                    "instructions_version": ins.version,
                    "cluster_id": o.cluster_id,
                    "score": round(o.score, 4),
                    "cleared": o.cleared,
                    "slots": o.slots,
                    "proposer": o.proposer,
                }
                for o in outs
            ]
        return EvaluationBatch(
            outputs=outs, scores=[o.score for o in outs], trajectories=outs if capture_traces else None
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[ClusterOutcome, ClusterOutcome],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        rows = [reflection_record(o) for o in eval_batch.trajectories or eval_batch.outputs]
        return {name: rows for name in components_to_update}


def _generated(c: CandidateOutcome) -> str:
    ch = c.change
    if c.kind == "guard":
        return f"guard: {ch.get('description', '')}\n{str(ch.get('source', ''))[:1500]}"
    if c.kind == "procedure":
        return f"procedure {ch.get('name')} (when {ch.get('when')}): {'; '.join(ch.get('steps') or [])}"
    return f"prompt_edit: {str(ch.get('append', ''))[:600]}"


def _replay_lines(c: CandidateOutcome) -> list[str]:
    rep = c.replay or {}
    lines = [
        f"{c.id}: replay {rep.get('verdict')}, {rep.get('targets_hit')}/{rep.get('targets_total')} targets "
        f"blocked in time, {len(rep.get('false_positives') or [])} false fires, "
        f"{len(rep.get('errors') or [])} errors"
    ]
    for t in rep.get("targets") or []:
        if not t["hit"]:
            lines.append(
                f"  MISSED target {t['failure_id']}: failing step {t['failing_seq']}, "
                f"first block {t['first_block_seq']}"
            )
    for b in c.blocked_calls:
        where = "benign-twin" if b["benign_twin"] else "passing"
        call = json.dumps(b["call"], default=str)[:400]
        lines.append(
            f"  FALSE FIRE in {where} run {b['run_id']} at step {b['seq']}: blocked {call} "
            f"with reason {str(b['reason'])[:200]!r}"
        )
    for e in (rep.get("errors") or [])[:3]:
        lines.append(f"  ERROR in run {e['run_id']} at step {e['seq']}: {str(e['error'])[:200]}")
    return lines


def _feedback(c: CandidateOutcome) -> list[str]:
    if c.status == "screened":
        return [f"{c.id}: SCREENED OUT, copies training literals {[h['literal'] for h in c.lint_hits][:6]}"]
    if c.status == "not_replayable":
        return [f"{c.id}: {c.kind} has a natural-language trigger, so replay cannot prove it"]
    return _replay_lines(c)


def reflection_record(o: ClusterOutcome) -> dict[str, Any]:
    """One cluster's evidence as a reflection record (Inputs, Generated Outputs, Feedback)."""
    feedback = [
        f"Score {o.score:.2f}: {o.cleared} of {o.slots} requested candidates were guards that cleared "
        "replay without copied literals."
    ]
    for c in o.candidates:
        feedback += _feedback(c)
    feedback += [f"invalid item: {r}" for r in o.rejected[:5]]
    if o.duplicates:
        feedback.append(f"{o.duplicates} duplicate candidate(s) wasted a slot")
    if not o.candidates:
        feedback.append("no usable candidate was produced")
    return {
        "Inputs": {"cluster": o.cluster_id, "mechanism": o.summary, "requested candidates": o.slots},
        "Generated Outputs": [_generated(c) for c in o.candidates] or "(none)",
        "Feedback": "\n".join(feedback),
    }


class ReflectionLM:
    """The improver's chat client in the shape the search calls: prompt in, text out, real tokens counted."""

    def __init__(self, model: Chat, meter: Meter | None = None) -> None:
        self.model = model
        self.meter = meter or Meter()

    @property
    def total_cost(self) -> float:
        return 0.0  # the endpoints are subscription or self-hosted; tokens are the recorded cost

    @property
    def total_tokens_in(self) -> int:
        return self.meter.input_tokens

    @property
    def total_tokens_out(self) -> int:
        return self.meter.output_tokens

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        turn = self.model.complete(messages, [])
        self.meter.add(turn)
        return turn.content or ""


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        self.lines.append(str(message))


@dataclass
class SearchRun:
    """A finished search: the best instructions and the full record of every candidate text."""

    instructions: Instructions
    seed: Instructions
    record: dict[str, Any]

    @property
    def improved(self) -> bool:
        return self.instructions.text != self.seed.text


def clusters_digest(clusters: Sequence[FailureCluster]) -> str:
    return sha256_hex(canonical_json([c.to_payload() for c in clusters]))


def search_record(
    result: GEPAResult[ClusterOutcome, int],
    adapter: InstructionsAdapter,
    clusters: Sequence[FailureCluster],
    budget: dict[str, Any],
) -> dict[str, Any]:
    """Every candidate text with its version, parents, per-cluster scores, and the per-cluster front."""
    texts = [cand[COMPONENT] for cand in result.candidates]
    versions = [adapter.instructions_for(t).version for t in texts]
    ids = [c.id for c in clusters]
    best = versions[result.best_idx]
    candidates = [
        {
            "index": i,
            "version": versions[i],
            "text": texts[i],
            "parents": [versions[p] for p in result.parents[i] if p is not None],
            "aggregate": round(result.val_aggregate_scores[i], 4),
            "per_cluster": {ids[int(k)]: round(v, 4) for k, v in result.val_subscores[i].items()},
            "discovered_at_metric_call": result.discovery_eval_counts[i],
        }
        for i in range(len(texts))
    ]
    front = {
        ids[int(k)]: sorted({versions[i] for i in v})
        for k, v in result.per_val_instance_best_candidates.items()
    }
    return {
        "type": "instruction_search_run",
        "engine": ENGINE,
        "seed_version": adapter.seed.version,
        "best_version": best,
        "improved": best != adapter.seed.version,
        "train_clusters": ids,
        "train_digest": clusters_digest(clusters),
        "candidates": candidates,
        "pareto_front": front,
        "evaluations": adapter.evaluations,
        "metric_calls": result.total_metric_calls,
        "budget": budget,
    }


def optimize_instructions(
    clusters: Sequence[FailureCluster],
    adapter: InstructionsAdapter,
    reflection: Chat,
    *,
    max_metric_calls: int = 60,
    minibatch: int = 3,
    use_merge: bool = True,
    seed: int = 0,
    run_dir: str | None = None,
) -> SearchRun:
    """Search over training clusters (they are both the train and the validation set). Held-out data and
    the frozen test batch are refused before the first proposal."""
    from gepa.api import optimize

    adapter.firewall.check_clusters(clusters)
    if not clusters:
        raise ValueError("the search needs at least one training failure cluster")
    lm = ReflectionLM(reflection)
    log = _Log()
    started = time.perf_counter()
    result: GEPAResult[ClusterOutcome, int] = optimize(
        seed_candidate={COMPONENT: adapter.seed.text},
        trainset=list(clusters),
        valset=list(clusters),
        adapter=adapter,
        reflection_lm=lm,
        reflection_prompt_template=REFLECTION_TEMPLATE,
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        reflection_minibatch_size=min(minibatch, len(clusters)),
        use_merge=use_merge,
        max_metric_calls=max_metric_calls,
        cache_evaluation=True,
        logger=log,
        run_dir=run_dir,
        seed=seed,
        raise_on_exception=True,
    )
    budget = {
        "max_metric_calls": max_metric_calls,
        "minibatch": minibatch,
        "use_merge": use_merge,
        "candidates_per_cluster": adapter.n,
        "seed": seed,
    }
    record = search_record(result, adapter, clusters, budget)
    record |= {
        "proposer_cost": meter_payload(adapter.meter),
        "reflection_cost": meter_payload(lm.meter),
        "wall_s": round(time.perf_counter() - started, 3),
        "log": log.lines[-200:],
    }
    return SearchRun(adapter.texts[record["best_version"]], adapter.seed, record)


class EvalStore(Protocol):
    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str: ...


def record_run(
    store: EvalStore, run: SearchRun, generation: int = 0, context: Mapping[str, Any] | None = None
) -> str:
    """Write the search record and the winning instructions (the shape the loop records them in). `context`
    (an experiment name, a generation) is stored on both, so a record is attributable without a lookup."""
    ctx = dict(context or {})
    rid = store.put_eval({**run.record, **ctx})
    store.put_eval(
        {
            "type": "improver_instructions",
            "version": run.instructions.version,
            "parent": run.instructions.parent,
            "text": run.instructions.text,
            "seen_clusters": sorted(run.record["train_clusters"]),
            "generation": generation,
            "engine": ENGINE,
            "search_run": rid,
            **ctx,
        }
    )
    return rid


def search_challenger(
    store: EvalStore,
    champion: Instructions,
    clusters: Sequence[FailureCluster],
    *,
    parent: Harness,
    train: Sequence[Trace],
    benign_ids: set[str],
    walks: WalkCache,
    runner: GuardRunner,
    improver: Chat,
    firewall: Firewall,
    n: int = 4,
    max_metric_calls: int = 60,
    generation: int = 0,
) -> Instructions | None:
    """The integrator's hook: a searched challenger for the loop's level 2, or None when the search found
    nothing better than the champion. The challenger still has to beat the champion on a frozen batch."""
    screen = Screen(
        parent,
        [t for t in train if t.outcome.passed],
        [t for t in train if t.case_id in benign_ids],
        walks,
        runner,
        training_literals(train),
    )
    adapter = InstructionsAdapter(screen=screen, proposer=improver, firewall=firewall, seed=champion, n=n)
    run = optimize_instructions(clusters, adapter, improver, max_metric_calls=max_metric_calls)
    record_run(store, run, generation)
    return run.instructions if run.improved else None


# --- The three-arm comparison: seed, loop challenger and searched instructions under the real admit() ---


class Store(EvalStore, Protocol):
    def put_trace(self, trace: Trace) -> bool: ...
    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str: ...


@dataclass
class Bench:
    """Everything the comparison needs. `improver` proposes, reflects and drafts; `model` is the worker."""

    cases: Sequence[Any]
    run_case: RunCase
    simulator: Simulator
    runner: GuardRunner
    signer: Signer
    model: Any
    improver: Chat
    store: Store
    seed: Harness
    families_a: frozenset[str]
    n: int = 4
    workers: int = 16
    search_calls: int = 60
    model_config: dict[str, Any] = field(default_factory=dict)
    propose_fn: ProposeFn = propose
    log: Callable[[str], None] = lambda _m: None


@dataclass
class Arm:
    name: str
    instructions: Instructions
    meter: Meter = field(default_factory=Meter)
    outcomes: list[ClusterOutcome] = field(default_factory=list)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    decisions: dict[str, Decision] = field(default_factory=dict)


@dataclass
class Referee:
    """The frozen evaluation context shared by every candidate: parent runs, referee traces, evaluator."""

    parent_runs: tuple[EvalReport, EvalReport]
    evaluator: dict[str, Any]
    walks: WalkCache
    literals: Literals
    visible: list[str]
    passing_all: list[Trace]
    benign_all: list[Trace]


def split_clusters(
    clusters: Sequence[FailureCluster], family_of: Mapping[str, str], families_a: frozenset[str]
) -> tuple[list[FailureCluster], list[FailureCluster], list[str]]:
    """(train batch A, frozen test batch B, mixed cluster ids). A cluster whose members span both family
    sets belongs to neither, so no member of B can reach the search through A."""
    a: list[FailureCluster] = []
    b: list[FailureCluster] = []
    mixed: list[str] = []
    for c in clusters:
        sides = {family_of.get(m.case_id, "") in families_a for m in c.members}
        if sides == {True}:
            a.append(c)
        elif sides == {False}:
            b.append(c)
        else:
            mixed.append(c.id)
    return a, b, mixed


def propose_arm(arm: Arm, clusters: Sequence[FailureCluster], screen: Screen, bench: Bench) -> None:
    """One proposal call per cluster with the arm's instructions, then the training-side screen."""

    def one(cl: FailureCluster) -> tuple[ClusterOutcome, list[Candidate], Meter]:
        meter = Meter()
        prop = bench.propose_fn(
            cl,
            screen.parent,
            bench.improver,
            instructions=arm.instructions,
            passing=screen.passing,
            walks=screen.walks,
            meter=meter,
            n=bench.n,
        )
        return proposal_outcome(cl, arm.instructions, prop, screen, bench.n, meter), prop.candidates, meter

    with ThreadPoolExecutor(max_workers=max(1, min(8, len(clusters)))) as pool:
        for out, cands, meter in pool.map(one, clusters):
            _merge(arm.meter, meter)
            arm.outcomes.append(out)
            arm.candidates |= {c.id: c for c in cands}


def shortlist(arm: Arm) -> list[Candidate]:
    """Candidates the admission rule can still admit: lint-clean, and for guards CLEARED on training replay.
    A guard that failed training replay fails the referee's replay as well (same targets, a superset of
    clearance traces), so a fresh agent run could not change its decision."""
    keep = {c.id for o in arm.outcomes for c in o.candidates if c.status in ("cleared", "not_replayable")}
    return [c for cid, c in arm.candidates.items() if cid in keep]


def _referee_replays(
    cand: Candidate, cluster: FailureCluster, ref: Referee, runner: GuardRunner, parent: Harness
) -> tuple[dict[str, ReplayReport], ReplayReport | None]:
    new = cand.new_guards(parent)
    args = (cluster.members, cluster.traces, ref.passing_all, ref.walks, runner, ref.benign_all)
    reps = {g.id: replay(g, *args) for g in new}
    return reps, (replay(cand.harness.guards, *args) if new else None)


def _persist(store: Store, report: EvalReport, context: dict[str, Any]) -> None:
    for t in report.traces:
        store.put_trace(t)
    store.put_eval({**report.to_payload(), **context})


def admit_all(
    arms: Sequence[Arm], clusters: Mapping[str, FailureCluster], ref: Referee, bench: Bench, experiment: str
) -> None:
    """A fresh run C1 for every shortlisted candidate, the real admit(), C2 where it is pending, and a signed
    decision for each."""
    work = [(arm, c) for arm in arms for c in shortlist(arm)]
    if not work:
        return
    decision_cases = [c for c in bench.cases if c.split in ("train", "heldout")]
    heldout = [c for c in bench.cases if c.split == "heldout"]
    c1 = evaluate_many(
        [c.harness for _, c in work], decision_cases, bench.run_case, bench.model, workers=bench.workers
    )
    pending: list[tuple[Arm, Candidate, dict[str, Any]]] = []
    for (arm, cand), report in zip(work, c1, strict=True):
        ctx = {"experiment": experiment, "arm": arm.name, "candidate": cand.id}
        _persist(bench.store, report, {**ctx, "role": "candidate"})
        reps, composed = _referee_replays(cand, clusters[cand.cluster_id], ref, bench.runner, bench.seed)
        kw: dict[str, Any] = {
            "parent_runs": ref.parent_runs,
            "cand_eval": report,
            "replays": reps,
            "composed": composed,
            "literals": ref.literals,
            "visible_case_ids": ref.visible,
            "evaluator": ref.evaluator,
            "context": {"experiment": experiment, "arm": arm.name, **cand.to_payload()},
        }
        arm.decisions[cand.id] = d = admit(bench.seed, cand.harness, confirm_eval=None, **kw)
        if awaiting_confirmation(d):
            pending.append((arm, cand, kw))
    if pending:
        c2 = evaluate_many(
            [c.harness for _, c, _ in pending], heldout, bench.run_case, bench.model, workers=bench.workers
        )
        for (arm, cand, kw), confirm in zip(pending, c2, strict=True):
            ctx = {"experiment": experiment, "arm": arm.name, "candidate": cand.id, "role": "confirm"}
            _persist(bench.store, confirm, ctx)
            arm.decisions[cand.id] = admit(bench.seed, cand.harness, confirm_eval=confirm, **kw)
    for arm, cand in work:
        d = arm.decisions[cand.id]
        env = base64.b64encode(receipt(d, bench.signer)).decode("ascii")
        bench.store.put_decision(
            {**d.to_payload(), "envelope": env, "keyid": bench.signer.keyid},
            decision_id=f"{experiment}:{arm.name}:{d.parent}->{d.candidate}",
        )


def _gate_failures(arm: Arm) -> dict[str, int]:
    failed: dict[str, int] = {}
    for d in arm.decisions.values():
        for c in d.checks:
            if not c.passed and c.name in GATES:
                failed[c.name] = failed.get(c.name, 0) + 1
    return failed


def recovery_evidence(d: Decision) -> dict[str, Any]:
    """The evidence of the paired held-out check, found by its content rather than its name."""
    return next((c.evidence for c in d.checks if "stable_recoveries" in c.evidence), {})


def c1_flips(evidence: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Held-out cases a candidate's first fresh run flipped: (failed in both parent runs and passed in C1,
    passed in either parent run and failed in C1). Informative even when the candidate was not admitted."""
    up: list[str] = []
    down: list[str] = []
    for case, v in (evidence.get("outcomes") or {}).items():
        p1, p2, c1 = v[0], v[1], v[2]
        if p1 == p2 == "fail" and c1 == "pass":
            up.append(case)
        elif "pass" in (p1, p2) and c1 == "fail":
            down.append(case)
    return sorted(up), sorted(down)


def flips_report(evidences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """C1 flips over every freshly evaluated candidate, with the denominator."""
    per = [c1_flips(e) for e in evidences]
    return {
        "candidates": len(per),
        "c1_fail_to_pass_total": sum(len(u) for u, _ in per),
        "c1_pass_to_fail_total": sum(len(d) for _, d in per),
        "c1_fail_to_pass_max": max((len(u) for u, _ in per), default=0),
        "candidates_with_2plus_recoveries_and_no_c1_break": sum(1 for u, d in per if len(u) >= 2 and not d),
        "c1_fail_to_pass_cases": sorted({c for u, _ in per for c in u}),
    }


def arm_report(arm: Arm, slots: int) -> dict[str, Any]:
    """Per-arm numbers, every rate with its denominator."""
    outs = [c for o in arm.outcomes for c in o.candidates]
    admitted = sorted(cid for cid, d in arm.decisions.items() if d.admitted)
    recovered: set[str] = set()
    broken: set[str] = set()
    for cid in admitted:
        ev = recovery_evidence(arm.decisions[cid])
        recovered |= set(ev.get("stable_recoveries", []))
        broken |= set(ev.get("protected_broken", []))
    count = {s: sum(1 for c in outs if c.status == s) for s in ("screened", "cleared", "replay_failed")}
    cleared = sum(o.cleared for o in arm.outcomes)
    return {
        "instructions_version": arm.instructions.version,
        "instructions_parent": arm.instructions.parent,
        "slots": slots,
        "proposed": len(outs),
        "invalid": sum(len(o.rejected) for o in arm.outcomes),
        "duplicates": sum(o.duplicates for o in arm.outcomes),
        "by_kind": {k: sum(1 for c in outs if c.kind == k) for k in ("guard", "procedure", "prompt_edit")},
        "screened": count["screened"],
        "replay_cleared": count["cleared"],
        "replay_failed": count["replay_failed"],
        "replay_incomplete": sum(1 for c in outs if c.status == "replay_incomplete"),
        "evaluated_fresh": len(arm.decisions),
        "admitted": len(admitted),
        "admitted_ids": admitted,
        "admission_yield": f"{len(admitted)}/{slots}",
        "admission_yield_value": round(len(admitted) / slots, 4) if slots else 0.0,
        "replay_score": f"{cleared}/{slots}",
        "heldout_flips_fail_to_pass": sorted(recovered),
        "heldout_flips_pass_to_fail": sorted(broken),
        "c1_flips_all_evaluated": flips_report([recovery_evidence(d) for d in arm.decisions.values()]),
        "rejections_by_gate": _gate_failures(arm),
        "proposer_tokens": meter_payload(arm.meter),
        "per_cluster": [o.to_payload() for o in arm.outcomes],
    }


def _status_line(c: CandidateOutcome, d: Decision | None) -> str:
    if d is None:
        status = c.status
    elif d.admitted:
        status = "admitted"
    else:
        status = "rejected: " + ", ".join(k.name for k in d.checks if not k.passed)
    if c.replay:
        rep = c.replay
        status += (
            f" (replay: {rep.get('targets_hit')}/{rep.get('targets_total')} targets, "
            f"{len(rep.get('false_positives') or [])} false positives, {len(rep.get('errors') or [])} errors)"
        )
    if c.lint_hits:
        status += f" (copied literals: {[h['literal'] for h in c.lint_hits][:5]})"
    return status


def examples(arm: Arm) -> list[str]:
    """The loop's level-2 feedback lines, built the way the loop builds them for its challenger draft."""
    out: list[str] = []
    for o in arm.outcomes:
        for c in o.candidates:
            line = _status_line(c, arm.decisions.get(c.id))
            out.append(f"[champion] {c.kind} for cluster {o.cluster_id}: {line}")
        out += [f"[champion] {r}" for r in o.rejected]
    return out


def arm_score(arm: Arm, slots: int) -> ArmScore:
    r = arm_report(arm, slots)
    return ArmScore(
        arm.instructions.version,
        [o.cluster_id for o in arm.outcomes],
        slots=slots,
        proposed=r["proposed"],
        invalid=r["invalid"],
        duplicates=r["duplicates"],
        screened_out=r["screened"],
        replay_tested=r["replay_cleared"] + r["replay_failed"] + r["replay_incomplete"],
        replay_passed=r["replay_cleared"],
        evaluated=r["evaluated_fresh"],
        admitted=r["admitted"],
    )


def baseline(bench: Bench, experiment: str) -> tuple[Referee, list[Trace]]:
    """The parent's two fresh runs (P1 over every decision case, P2 over the held-out cases)."""
    decision_cases = [c for c in bench.cases if c.split in ("train", "heldout")]
    heldout = [c for c in bench.cases if c.split == "heldout"]
    p1 = evaluate_many([bench.seed], decision_cases, bench.run_case, bench.model, workers=bench.workers)[0]
    _persist(bench.store, p1, {"experiment": experiment, "role": "parent_run_1"})
    p2 = evaluate_many([bench.seed], heldout, bench.run_case, bench.model, workers=bench.workers)[0]
    _persist(bench.store, p2, {"experiment": experiment, "role": "parent_run_2"})
    benign_ids = {c.id for c in bench.cases if c.kind == "benign_twin"}
    train = p1.train_traces()
    ref = Referee(
        (p1, p2),
        evaluator_identity(bench.cases, bench.model_config),
        WalkCache(bench.simulator),
        training_literals(train),
        sorted(c.id for c in bench.cases if c.split == "train"),
        [t for t in p1.traces if t.outcome.passed],
        [t for t in p1.traces if t.case_id in benign_ids],
    )
    return ref, train


def _parent_score(ref: Referee) -> dict[str, int]:
    p1, p2 = ref.parent_runs
    return {
        "P1_heldout_passed": len(p1.passed("heldout")),
        "P2_heldout_passed": len(p2.passed("heldout")),
        "heldout_total": p1.count("heldout"),
        "P1_train_passed": len(p1.passed("train")),
        "train_total": p1.count("train"),
    }


def compare_level2(bench: Bench, experiment: str = "level2") -> dict[str, Any]:
    """Seed, loop challenger and searched instructions, each proposing freshly with the same budget on
    clusters none of them saw; every surviving candidate is judged by the real admit()."""
    started = time.perf_counter()
    ref, train = baseline(bench, experiment)
    benign_ids = {c.id for c in bench.cases if c.kind == "benign_twin"}
    diag = Meter()
    clusters = diagnose(train, bench.simulator, bench.improver, diag)
    family_of = {c.id: str(getattr(c, "family", "")) for c in bench.cases}
    a, b, mixed = split_clusters(clusters, family_of, bench.families_a)
    frozen = {"cluster_ids": [c.id for c in b], "digest": clusters_digest(b)}
    split = {
        "train_clusters": [c.id for c in a],
        "mixed_excluded": mixed,
        "families_a": sorted(bench.families_a),
    }
    bench.store.put_eval({"type": "level2_frozen_batch", "experiment": experiment, **frozen, **split})
    bench.log(f"clusters: {len(a)} train (A), {len(b)} frozen test (B), {len(mixed)} mixed and excluded")
    out: dict[str, Any] = {
        "type": "level2_comparison",
        "experiment": experiment,
        "frozen_batch": frozen,
        **split,
        "clusters": [c.to_payload() for c in clusters],
        "evaluator": ref.evaluator["version"],
        "diagnosis_cost": meter_payload(diag),
        "parent": bench.seed.version,
        "parent_score": _parent_score(ref),
    }
    if not a or not b:
        out |= {
            "claim": False,
            "reason": f"need clusters on both sides of the split (A={len(a)}, B={len(b)})",
        }
        bench.store.put_eval(out, eval_id=f"{experiment}-comparison")
        return out
    screen = Screen(
        bench.seed,
        [t for t in train if t.outcome.passed],
        [t for t in train if t.case_id in benign_ids],
        ref.walks,
        bench.runner,
        ref.literals,
    )
    firewall = Firewall.from_cases(bench.cases, b)

    # The challenger arm: the loop's own draft, from the seed's batch-A admission outcomes.
    seed_on_a = Arm("seed-on-A", SEED)
    propose_arm(seed_on_a, a, screen, bench)
    admit_all([seed_on_a], {c.id: c for c in a}, ref, bench, f"{experiment}:derive-challenger")
    meta = Meter()
    slots_a = bench.n * len(a)
    drafted = propose_instructions(
        SEED, {"champion": arm_score(seed_on_a, slots_a)}, examples(seed_on_a), bench.improver, meta
    )
    challenger = drafted or SEED
    bench.log(f"challenger {challenger.version} (parent {challenger.parent})")

    # The search arm: batch A only, replay evidence only.
    adapter = InstructionsAdapter(
        screen=screen, proposer=bench.improver, firewall=firewall, n=bench.n, propose_fn=bench.propose_fn
    )
    run = optimize_instructions(a, adapter, bench.improver, max_metric_calls=bench.search_calls)
    run_id = record_run(bench.store, run, context={"experiment": experiment})
    bench.log(f"search {run.instructions.version} after {run.record['metric_calls']} metric calls")

    arms = [Arm("seed", SEED), Arm("challenger", challenger), Arm("search", run.instructions)]
    for arm in arms:
        propose_arm(arm, b, screen, bench)
    admit_all(arms, {c.id: c for c in b}, ref, bench, experiment)
    slots = bench.n * len(b)
    out |= {
        "claim": True,
        "budget": {
            "candidates_per_cluster": bench.n,
            "clusters": len(b),
            "slots_per_arm": slots,
            "proposal_calls_per_arm": len(b),
            "repair_rounds": 0,
        },
        "arms": {arm.name: arm_report(arm, slots) for arm in arms},
        "derivation": {
            "challenger": {
                "from": "loop.propose_instructions over the seed's batch-A admission outcomes",
                "batch_a": arm_report(seed_on_a, slots_a),
                "meta_cost": meter_payload(meta),
                "text": challenger.text,
                "unchanged": drafted is None,
            },
            "search": {
                "search_run": run_id,
                "metric_calls": run.record["metric_calls"],
                "proposer_cost": run.record["proposer_cost"],
                "reflection_cost": run.record["reflection_cost"],
                "text": run.instructions.text,
                "improved": run.improved,
            },
        },
        "wall_s": round(time.perf_counter() - started, 1),
    }
    bench.store.put_eval(out, eval_id=f"{experiment}-comparison")
    return out


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    env = os.environ
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    p = argparse.ArgumentParser(prog="ratchet-level2")
    p.add_argument("--base-url", default=env.get("RATCHET_BASE_URL", "http://localhost:8801/v1"))
    p.add_argument("--model", default=env.get("RATCHET_MODEL", "gpt-oss-120b"))
    p.add_argument("--improver-model", default=env.get("RATCHET_IMPROVER_MODEL", "gpt-oss-120b"))
    p.add_argument("--uri", default=env.get("RATCHET_MONGODB_URI", ""))
    p.add_argument("--db", default=env.get("RATCHET_DB", "ratchet_l2"))
    p.add_argument("--families-a", default="approval_revoked,concurrent_worker,injection,order_modification")
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--search-calls", type=int, default=60)
    p.add_argument("--experiment", default=f"level2-{stamp}")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from ratchet_cases.cases import all_cases

    from ratchet.agent.guards import GuardRuntime
    from ratchet.agent.model import ModelClient, ModelConfig
    from ratchet.agent.seed import seed_harness
    from ratchet.agent.worker import run_case
    from ratchet.improve.sandbox import SandboxRunner
    from ratchet.improve.worldsim import WorldSimulator
    from ratchet.store import Store as AtlasStore

    a = _args(argv)
    key = os.environ.get("RATCHET_API_KEY", "")
    if not a.uri or not key:
        print("error: set RATCHET_MONGODB_URI (or --uri) and RATCHET_API_KEY", file=sys.stderr)
        return 2
    cases = all_cases()
    signer = Signer.for_decisions()
    model_config = {"base_url": a.base_url, "model": a.model, "improver_model": a.improver_model}
    with (
        AtlasStore(a.uri, a.db) as store,
        ModelClient(ModelConfig(base_url=a.base_url, model=a.model, api_key=key)) as model,
        ModelClient(ModelConfig(base_url=a.base_url, model=a.improver_model, api_key=key)) as improver,
        GuardRuntime() as runtime,
    ):
        store.setup(wait_search=False)
        bench = Bench(
            cases=cases,
            run_case=run_case,
            simulator=WorldSimulator(cases),
            runner=SandboxRunner(runtime),
            signer=signer,
            model=model,
            improver=improver,
            store=store,
            seed=seed_harness(),
            families_a=frozenset(f for f in a.families_a.split(",") if f),
            n=a.candidates,
            workers=a.workers,
            search_calls=a.search_calls,
            model_config=model_config,
            log=lambda m: print(m, flush=True),
        )
        result = compare_level2(bench, a.experiment)
        result["config"] = {
            **model_config,
            "db": a.db,
            "families_a": a.families_a,
            "candidates": a.candidates,
            "search_calls": a.search_calls,
            "signer_keyid": signer.keyid,
        }
        result["agent_model_cost"] = model.cost().__dict__
        result["improver_model_cost"] = improver.cost().__dict__
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
