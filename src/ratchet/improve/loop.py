"""The ratchet loop: run, diagnose, propose, replay, evaluate, admit, promote, repeat.

Two levels improve, and they are recorded and reported separately.

Level 1, the harness. Each generation turns the parent's training failures into candidate guards,
procedures and prompt edits. Guards are proven by counterfactual replay before any agent run; candidates
carrying literals copied from training traces are screened out; the rest are evaluated on training and
held-out cases; the admission rule decides, and every decision is signed. The parent is run twice on the
held-out cases so that a gain must clear the parent's own run-to-run noise.

Level 2, the improver. The instructions the improver proposes with are a versioned artifact. After a
generation the model drafts an edit (the challenger). Each version records the failure clusters whose
results it was shown while it was written. The challenger is judged against the champion only on a frozen
batch: clusters of a later generation that neither version has seen. Both propose for that batch against
the same parent; the challenger replaces the champion only if its admission yield (admitted / proposed) on
the batch is strictly higher. With no unseen cluster there is no comparison and no level-2 claim.

The firewall: the loop holds training cases only. Held-out cases, their runs and traces, the referee-side
replay, the admission rule, the decision signature and activation all belong to the referee
(`ratchet.improve.referee`), a separate process with its own key, which answers with totals.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ratchet_cases import released_digest
from ratchet_cases.types import Trace, sha256_hex

from ratchet.canonical import canonical_json
from ratchet.improve.activate import ActivationError
from ratchet.improve.diagnose import FailureCluster, diagnose
from ratchet.improve.evaluate import CaseLike, EvalReport, RunCase, evaluate_many
from ratchet.improve.evaluator import evaluator_identity
from ratchet.improve.llm import Chat, Meter, ask_json
from ratchet.improve.overfit import Literals, lint, training_literals
from ratchet.improve.propose import SEED, Candidate, Instructions, propose, revise_guard
from ratchet.improve.referee import Referee, Totals, Verdict
from ratchet.improve.replay import GuardRunner, ReplayReport, WalkCache, replay
from ratchet.improve.sim import Simulator
from ratchet.types import Guard, Harness, Procedure


def frozen_suite_digest() -> str:
    """The case suite digest the installed `probity-ratchet-cases` release was published with."""
    return released_digest()


class ParentChooser(Protocol):
    """The archive's parent selection: where ideas come from. It never changes the comparison parent."""

    def __call__(self, tip: str, generation: int) -> Any: ...


class ImproveStore(Protocol):
    def put_harness(self, harness: Harness) -> bool: ...
    def get_harness(self, version: str) -> Harness | None: ...
    def put_trace(self, trace: Trace) -> bool: ...
    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str: ...
    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str: ...


@dataclass
class Lab:
    """Everything the loop needs besides the store and the agent model."""

    cases: Sequence[CaseLike]  # training cases; held-out cases belong to the referee
    run_case: RunCase
    simulator: Simulator
    runner: GuardRunner
    referee: Referee
    improver: Chat | None = None
    workers: int = 8
    repeats: int = 1
    max_clusters: int = 4
    candidates_per_cluster: int = 4
    max_evals: int = 6
    repair_rounds: int = 1
    suite_digest: str | None = field(default_factory=frozen_suite_digest)
    model_config: dict[str, Any] = field(default_factory=dict)
    # Level 2 splits case families in two: the improver's instructions are tuned on clusters from the
    # first half and compared on clusters from the second. None means the sorted families, halved.
    evaluation_families: frozenset[str] | None = None

    def family_split(self) -> tuple[frozenset[str], frozenset[str]]:
        families = sorted({c.family for c in self.cases})
        evaluation = self.evaluation_families
        if evaluation is None:
            evaluation = frozenset(families[len(families) // 2 :])
        return frozenset(families) - evaluation, frozenset(evaluation)

    @property
    def train_cases(self) -> list[CaseLike]:
        """The cases the improver runs. The audit cohort is never used for any decision."""
        return [c for c in self.cases if c.split == "train"]

    log: Callable[[str], None] = lambda _msg: None


@dataclass
class ArmScore:
    """What one version of the improver's instructions achieved on a set of clusters."""

    instructions_version: str
    clusters: list[str] = field(default_factory=list)
    slots: int = 0
    proposed: int = 0
    invalid: int = 0
    duplicates: int = 0
    empty: int = 0
    rejected: int = 0
    screened_out: int = 0
    replay_tested: int = 0
    replay_passed: int = 0
    evaluated: int = 0
    admitted: int = 0
    stable_recoveries: int = 0

    @property
    def admission_yield(self) -> float:
        """Admitted over the fixed slots, so invalid, duplicate and empty proposals count against it."""
        denominator = self.slots or self.proposed
        return self.admitted / denominator if denominator else 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "instructions_version": self.instructions_version,
            "clusters": self.clusters,
            "slots": self.slots,
            "proposed": self.proposed,
            "invalid": self.invalid,
            "duplicates": self.duplicates,
            "empty": self.empty,
            "rejected": self.rejected,
            "screened_out": self.screened_out,
            "replay_tested": self.replay_tested,
            "replay_passed": self.replay_passed,
            "evaluated": self.evaluated,
            "admitted": self.admitted,
            "stable_recoveries": self.stable_recoveries,
            "admission_yield": round(self.admission_yield, 4),
        }


@dataclass
class GenerationRecord:
    generation: int
    parent: str
    level1: dict[str, Any] = field(default_factory=dict)
    level2: dict[str, Any] = field(default_factory=dict)
    clusters: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    replay: dict[str, Any] = field(default_factory=dict)
    improver_cost: dict[str, int] = field(default_factory=dict)
    agent_cost: dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0
    note: str = ""
    archive: dict[str, Any] = field(default_factory=dict)

    @property
    def promoted(self) -> str | None:
        value = self.level1.get("promoted")
        return str(value) if value else None

    def to_payload(self) -> dict[str, Any]:
        return {"type": "generation", **dict(self.__dict__)}


@dataclass
class EvolveResult:
    harness: Harness
    instructions: Instructions
    generations: list[GenerationRecord]


@dataclass
class _Tried:
    candidate: Candidate
    arm: str
    replay: ReplayReport | None = None
    status: str = "proposed"
    lint_hits: list[dict[str, str]] = field(default_factory=list)
    verdict: Verdict | None = None
    report: EvalReport | None = None  # the training run; held-out runs stay with the referee


@dataclass
class _Gen:
    """Per-generation state shared by the steps of one generation."""

    gen: int
    parent: Harness
    parent_eval: EvalReport  # the parent's training run
    parent_totals: tuple[Totals, Totals]
    evaluator: dict[str, Any]
    walks: WalkCache
    literals: Literals
    visible: list[str]
    benign_ids: set[str]
    rec: GenerationRecord
    meter: Meter
    inspiration: Harness | None = None
    archive_context: dict[str, Any] = field(default_factory=dict)
    task_stats: dict[tuple[str, str], dict[str, int]] = field(default_factory=dict)


def _score(r: EvalReport, heldout_passed: int, heldout_total: int) -> dict[str, int]:
    return {
        "train_passed": len(r.passed("train")),
        "train_total": r.count("train"),
        "heldout_passed": heldout_passed,
        "heldout_total": heldout_total,
    }


def _cost(r: EvalReport) -> dict[str, Any]:
    c = r.cost()
    return {
        "model_calls": c.model_calls,
        "input_tokens": c.input_tokens,
        "output_tokens": c.output_tokens,
        "tool_calls": c.tool_calls,
        "wall_s": c.wall_s,
    }


def _persist_eval(store: ImproveStore, report: EvalReport, context: dict[str, Any]) -> None:
    for t in report.traces:
        store.put_trace(t)
    store.put_eval({**report.to_payload(), **context})


def _proof_replay(tried: _Tried, g: _Gen, cluster: FailureCluster, train: list[Trace], lab: Lab) -> None:
    """The improver's own check, on training traces only, with repair rounds on failure."""
    passing = [t for t in train if t.outcome.passed]
    benign = [t for t in train if t.case_id in g.benign_ids]
    c = tried.candidate
    for attempt in range(lab.repair_rounds + 1):
        guard = c.harness.guards[-1]
        rep = replay(guard, cluster.members, cluster.traces, passing, g.walks, lab.runner, benign)
        tried.candidate, tried.replay = c, rep
        if rep.passed or attempt == lab.repair_rounds or lab.improver is None:
            break
        revised = revise_guard(c, rep, cluster, g.parent, lab.improver, walks=g.walks)
        if isinstance(revised, str):
            break
        c = revised
    tried.status = "replay_passed" if tried.replay and tried.replay.passed else "replay_failed"


Lane = tuple[str, set[str] | None, bool, int]  # arm, cluster ids, whether those ids are excluded, slots


def lanes_for(frozen_ids: set[str] | None, budget: int) -> list[Lane]:
    """Fixed evaluation slots. With a level-2 comparison running, challenger and champion get the same
    number of slots on the frozen batch, and the champion's work on the other clusters gets the rest."""
    if not frozen_ids:
        return [("champion", None, False, budget)]
    slot = max(1, budget // 3)
    return [
        ("challenger", frozen_ids, False, slot),
        ("champion", frozen_ids, False, slot),
        ("champion", frozen_ids, True, max(0, budget - 2 * slot)),
    ]


def _shortlist(tried: list[_Tried], lanes: list[Lane]) -> list[_Tried]:
    """Fill each lane round-robin over (kind, cluster) queues, so guards, procedures and prompt edits all get
    evaluated and one big cluster does not take every slot. Within a queue, guards that cleared training
    replay with the most targets come first."""
    kinds = ("guard", "procedure", "prompt_edit")
    chosen: list[_Tried] = []
    for arm, ids, exclude, slots in lanes:
        pool = [
            t for t in tried
            if t.arm == arm and t.status in ("proposed", "replay_passed") and t not in chosen
            and (ids is None or (t.candidate.cluster_id in ids) != exclude)
        ]  # fmt: skip
        pool.sort(key=lambda t: -len(t.candidate.targets))
        queues: dict[tuple[str, str], list[_Tried]] = {}
        for kind in kinds:
            for t in pool:
                if t.candidate.kind == kind:
                    queues.setdefault((kind, t.candidate.cluster_id), []).append(t)
        order = sorted(queues, key=lambda k: kinds.index(k[0]))
        by_kind = {kind: [k for k in order if k[0] == kind] for kind in kinds}
        interleaved = [k for group in _zip_longest(*by_kind.values()) for k in group if k is not None]
        taken = 0
        while taken < slots and any(queues.values()):
            for key in interleaved:
                if queues[key] and taken < slots:
                    chosen.append(queues[key].pop(0))
                    taken += 1
    return chosen


def _zip_longest(*groups: list[tuple[str, str]]) -> list[tuple[tuple[str, str] | None, ...]]:
    width = max((len(g) for g in groups), default=0)
    return [tuple(g[i] if i < len(g) else None for g in groups) for i in range(width)]


def compose(parent: Harness, picks: Sequence[Candidate]) -> Harness:
    """One child carrying every pick's change: their new guards, new procedures and prompt additions."""
    guards: list[Guard] = list(parent.guards)
    procs: list[Procedure] = list(parent.procedures)
    prompt = parent.system_prompt
    for c in picks:
        guards += [g for g in c.harness.guards if g.id not in {x.id for x in guards}]
        procs += [p for p in c.harness.procedures if p.id not in {x.id for x in procs}]
        if c.kind == "prompt_edit":
            prompt = f"{prompt}\n\n{c.change['append']}"
    return parent.child(guards=guards, procedures=procs, system_prompt=prompt, created_by="improver")


_META_SYSTEM = (
    "You maintain the instructions an improver model follows when it proposes harness changes. You edit them "
    "so that more of its candidates pass replay and admission. Reply with one JSON object only."
)


def propose_instructions(
    champion: Instructions, arms: dict[str, ArmScore], examples: list[str], model: Chat, meter: Meter
) -> Instructions | None:
    scores = "\n".join(f"{name}: {a.to_payload()}" for name, a in arms.items())
    user = "\n\n".join(
        [
            f"Current instructions (version {champion.version}):\n{champion.text}",
            f"Results last generation:\n{scores}",
            "What happened to individual candidates:\n" + ("\n".join(examples[:12]) or "(none)"),
            "Rewrite the instructions to raise the share of candidates that are admitted. Keep the three "
            "kinds (guard, procedure, prompt_edit); the guard API and the reply format are appended "
            'separately, so do not restate them. Reply JSON: {"text": "<new instructions>", '
            '"rationale": "<what you changed and why>"}',
        ]
    )
    try:
        reply = ask_json(model, _META_SYSTEM, user, meter)
    except Exception:
        return None
    text = str(reply.get("text", "")).strip()
    if not text or text == champion.text:
        return None
    return Instructions.build(text, parent=champion.version)


def _verdict(
    store: ImproveStore,
    lab: Lab,
    g: _Gen,
    harness: Harness,
    report: EvalReport,
    targets: Sequence[FailureCluster],
    guard_clusters: dict[str, str],
    context: dict[str, Any],
) -> Verdict:
    v = lab.referee.decide(g.gen, harness, report, targets, guard_clusters, g.literals, g.visible, context)
    g.rec.decisions.append(v.decision_id)
    # The improver keeps the referee's answer, totals only, beside its own runs.
    store.put_eval(
        {"type": "verdict", "generation": g.gen, "parent": g.parent.version, **asdict(v)},
        eval_id=f"verdict-{v.decision_id}",
    )
    return v


def _promote(
    store: ImproveStore,
    lab: Lab,
    model: Any,
    g: _Gen,
    admitted: list[_Tried],
    clusters: dict[str, FailureCluster],
) -> tuple[Harness, EvalReport, Verdict] | None:
    """The best admitted candidate, or the composition of all of them when that is admitted and better."""
    if not admitted:
        return None

    def key(t: _Tried) -> tuple[int, int, int]:
        assert t.report is not None and t.verdict is not None
        return (t.verdict.heldout_passed, len(t.report.passed("train")), -t.report.cost().model_calls)

    best = max(admitted, key=key)
    assert best.report is not None and best.verdict is not None
    picks = list({t.candidate.cluster_id: t for t in sorted(admitted, key=key)}.values())
    if len(picks) < 2:
        return best.candidate.harness, best.report, best.verdict
    combo = compose(g.parent, [t.candidate for t in picks])
    report = evaluate_many(
        [combo], lab.train_cases, lab.run_case, model, workers=lab.workers, repeats=lab.repeats
    )[0]
    _persist_eval(store, report, {"generation": g.gen, "role": "composite"})
    guarded = [t for t in picks if t.candidate.kind == "guard"]
    v = _verdict(
        store,
        lab,
        g,
        combo,
        report,
        [clusters[t.candidate.cluster_id] for t in guarded],
        {x.id: t.candidate.cluster_id for t in guarded for x in t.candidate.new_guards(g.parent)},
        {"generation": g.gen, "composite_of": [t.candidate.id for t in picks], **g.archive_context},
    )
    if v.admitted and v.heldout_passed > best.verdict.heldout_passed:
        return combo, report, v
    return best.candidate.harness, best.report, best.verdict


def _families(cluster: FailureCluster, family_of: dict[str, str]) -> set[str]:
    return {family_of[m.case_id] for m in cluster.members}


def _frozen_batch(clusters: list[FailureCluster], seen: set[str]) -> dict[str, Any]:
    batch = [c for c in clusters if c.id not in seen]
    return {
        "cluster_ids": [c.id for c in batch],
        "digest": sha256_hex(canonical_json([c.to_payload() for c in batch])),
    }


def evolve(
    store: ImproveStore,
    model: Any,
    generations: int = 3,
    *,
    lab: Lab,
    seed: Harness,
    instructions: Instructions = SEED,
    select_parent: ParentChooser | None = None,
) -> EvolveResult:
    """Run `generations` rounds of the ratchet from `seed`. Returns the final harness and every record.

    With `select_parent` (the version archive), each generation may draw ideas from an earlier admitted
    version: that version is shown to the proposer read-only, every candidate is still a child of the active
    harness and is compared with it, and the choice is recorded in the generation and in each decision."""
    improver: Chat = lab.improver or model
    lab.improver = improver
    parent, champion = seed, instructions
    challenger: Instructions | None = None
    seen: dict[str, set[str]] = {champion.version: set()}
    if any(c.split == "heldout" for c in lab.cases):
        raise ValueError("held-out cases belong to the referee; the loop takes training cases only")
    train_cases = lab.train_cases
    visible = sorted(c.id for c in train_cases)
    benign_ids = {c.id for c in lab.cases if c.kind == "benign_twin"}
    family_of = {c.id: c.family for c in lab.cases}
    tune_families, eval_families = lab.family_split()
    evaluator = lab.referee.start(evaluator_identity(train_cases, lab.model_config, lab.suite_digest))
    store.put_eval({"type": "evaluator", **evaluator}, eval_id=f"evaluator-{evaluator['version']}")
    store.put_harness(parent)
    active, revision = lab.referee.bootstrap(parent.version)
    if active != parent.version:
        raise ActivationError(f"the active harness is {active}, not the seed {parent.version}; rebase first")
    _record_instructions(store, champion, seen[champion.version], 0)
    records: list[GenerationRecord] = []
    for gen in range(1, generations + 1):
        started = time.perf_counter()
        meter = Meter()
        rec = GenerationRecord(gen, parent.version)
        # The parent runs on the training cases here and twice, fresh, on the held-out cases at the referee:
        # the noise floor is measured, never assumed.
        parent_eval = evaluate_many([parent], train_cases, lab.run_case, model, workers=lab.workers,
                                    repeats=lab.repeats)[0]  # fmt: skip
        _persist_eval(store, parent_eval, {"generation": gen, "role": "parent_train"})
        p1, p2 = lab.referee.parent_runs(gen, parent, parent_eval)
        rec.level1 = {
            "parent": parent.version,
            "evaluator": evaluator["version"],
            "parent_score": _score(parent_eval, p1.heldout_passed, p1.heldout_total),
            "parent_run_2_heldout_passed": p2.heldout_passed,
        }
        lab.log(f"gen {gen}: parent {parent.version} {rec.level1}")

        # The firewall: from here to the shortlist, only training traces are read.
        train = parent_eval.train_traces()
        g = _Gen(gen, parent, parent_eval, (p1, p2), evaluator, WalkCache(lab.simulator),
                 training_literals(train), visible, benign_ids, rec, meter)  # fmt: skip
        if select_parent is not None:
            choice = select_parent(parent.version, gen)
            rec.archive = choice.to_payload()
            g.archive_context = choice.decision_context()
            if choice.rebase_required:
                g.inspiration = store.get_harness(choice.version)
        clusters = diagnose(train, lab.simulator, improver, meter)[: lab.max_clusters]
        rec.clusters = [c.to_payload() for c in clusters]
        if not clusters:
            rec.note = "no training failures left to diagnose"
            rec.level2 = {"champion": champion.version, "claim": False, "reason": "no clusters"}
            _finish(store, rec, started, meter, [])
            records.append(rec)
            break

        # Level 2 setup: the frozen batch is fixed before any proposal is made.
        arms: dict[str, tuple[Instructions, ArmScore, list[FailureCluster]]] = {
            "champion": (champion, ArmScore(champion.version, [c.id for c in clusters]), clusters)
        }
        frozen: dict[str, Any] = {"cluster_ids": [], "digest": None}
        if challenger is not None:
            evaluation = [c for c in clusters if _families(c, family_of) <= eval_families]
            frozen = _frozen_batch(evaluation, seen[champion.version] | seen[challenger.version])
            batch = [c for c in clusters if c.id in frozen["cluster_ids"]]
            if batch:
                arms["challenger"] = (challenger, ArmScore(challenger.version, frozen["cluster_ids"]), batch)

        tried, rejected = _propose_all(arms, g, train, lab)
        failures = sum(1 for _, text in rejected if "model call failed" in text)
        if failures:
            rec.note = f"improver model calls failed for {failures} proposal task(s)"
        reports = _evaluate_and_admit(
            store,
            lab,
            model,
            g,
            tried,
            set(frozen["cluster_ids"]) if "challenger" in arms else None,
            clusters,
        )
        admitted = [t for t in tried if t.status == "admitted"]
        promoted = _promote(store, lab, model, g, admitted, {c.id: c for c in clusters})
        if promoted is not None:
            child, child_eval, verdict = promoted
            store.put_harness(child)
            decision_id, digest = verdict.decision_id, verdict.receipt_digest
            act = lab.referee.activate(
                candidate=child.version,
                expected=parent.version,
                expected_revision=revision,
                decision_id=decision_id,
                receipt_digest=digest,
                idempotency_key=f"activate:{decision_id}",
            )
            parent, revision = child, act.revision
            rec.level1 |= {
                "activation": {
                    "revision": act.revision,
                    "previous": act.previous,
                    "decision_id": decision_id,
                    "receipt_digest": digest,
                }
            }
            rec.level1 |= {
                "promoted": parent.version,
                "child_score": _score(child_eval, verdict.heldout_passed, verdict.heldout_total),
                "rates": verdict.rates,
            }
        rec.level1 |= {
            "candidates": len(tried),
            "evaluated": sum(1 for t in tried if t.verdict),
            "admitted": len(admitted),
        }

        tuning_ids = {c.id for c in clusters if _families(c, family_of) <= tune_families}
        champion, challenger, rec.level2 = _level2(
            store,
            arms,
            frozen,
            champion,
            challenger,
            seen,
            tried,
            rejected,
            improver,
            meter,
            gen,
            g.task_stats,
            lab.candidates_per_cluster,
            tuning_ids,
        )
        rec.level2["families"] = {"tuning": sorted(tune_families), "evaluation": sorted(eval_families)}
        rec.agent_cost = {"candidates": [_cost(r) for r in reports]}
        _finish(store, rec, started, meter, tried)
        lab.log(f"gen {gen}: level1 {rec.level1} level2 {rec.level2}")
        records.append(rec)
    return EvolveResult(parent, champion, records)


def _propose_all(
    arms: dict[str, tuple[Instructions, ArmScore, list[FailureCluster]]],
    g: _Gen,
    train: list[Trace],
    lab: Lab,
) -> tuple[list[_Tried], list[tuple[str, str]]]:
    assert lab.improver is not None
    passing = [t for t in train if t.outcome.passed]
    tried: list[_Tried] = []
    rejected: list[tuple[str, str]] = []
    for arm, (ins, score, batch) in arms.items():
        for cl in batch:
            prop = propose(
                cl,
                g.parent,
                lab.improver,
                instructions=ins,
                passing=passing,
                walks=g.walks,
                meter=g.meter,
                n=lab.candidates_per_cluster,
                inspiration=g.inspiration,
            )
            score.proposed += len(prop.candidates)
            g.task_stats[(arm, cl.id)] = {
                "valid": len(prop.candidates),
                "invalid": len(prop.rejected),
                "duplicates": prop.duplicates,
                "empty": int(not prop.candidates),
            }
            rejected += [(cl.id, f"[{arm}] {r['reason']}") for r in prop.rejected]
            tried += [_Tried(c, arm) for c in prop.candidates]
    clusters = {c.id: c for _, _, batch in arms.values() for c in batch}
    check_s = 0.0
    checks = 0
    for t in tried:
        score = arms[t.arm][1]
        t.lint_hits = lint(g.parent, t.candidate.harness, g.literals)
        if t.lint_hits:
            t.status = "screened: overfit_lint"
            score.screened_out += 1
            continue
        if t.candidate.kind != "guard":
            continue
        _proof_replay(t, g, clusters[t.candidate.cluster_id], train, lab)
        t.lint_hits = lint(g.parent, t.candidate.harness, g.literals)  # a repaired guard is screened again
        if t.lint_hits:
            t.status = "screened: overfit_lint"
            score.screened_out += 1
            continue
        score.replay_tested += 1
        score.replay_passed += t.status == "replay_passed"
        if t.replay is not None:
            check_s += t.replay.check_s
            checks += t.replay.checks
    g.rec.replay = {
        "guards_replayed": sum(1 for t in tried if t.replay),
        "checks": checks,
        "check_s": round(check_s, 4),
        "walk_s": round(g.walks.walk_s, 4),
        "checks_per_s": round(checks / check_s, 1) if check_s else 0.0,
    }
    return tried, rejected


def _evaluate_and_admit(
    store: ImproveStore,
    lab: Lab,
    model: Any,
    g: _Gen,
    tried: list[_Tried],
    frozen_ids: set[str] | None,
    clusters: list[FailureCluster],
) -> list[EvalReport]:
    """The shortlist runs on the training cases here; the referee runs each on the held-out cases, replays its
    guards over the protected traces, admits or rejects it and signs the decision."""
    shortlist = _shortlist(tried, lanes_for(frozen_ids, lab.max_evals))
    reports = evaluate_many(
        [t.candidate.harness for t in shortlist],
        lab.train_cases,
        lab.run_case,
        model,
        workers=lab.workers,
        repeats=lab.repeats,
    )
    by_id = {c.id: c for c in clusters}
    for t, report in zip(shortlist, reports, strict=True):
        t.report = report
        _persist_eval(store, report, {"generation": g.gen, "role": "candidate", "candidate": t.candidate.id})
        ctx = {"generation": g.gen, "arm": t.arm, **t.candidate.to_payload(), **g.archive_context}
        guards = {x.id: t.candidate.cluster_id for x in t.candidate.new_guards(g.parent)}
        t.verdict = _verdict(
            store, lab, g, t.candidate.harness, report, [by_id[t.candidate.cluster_id]], guards, ctx
        )
        t.status = "admitted" if t.verdict.admitted else "rejected: " + ", ".join(t.verdict.failed)
    return reports


def _level2(
    store: ImproveStore,
    arms: dict[str, tuple[Instructions, ArmScore, list[FailureCluster]]],
    frozen: dict[str, Any],
    champion: Instructions,
    challenger: Instructions | None,
    seen: dict[str, set[str]],
    tried: list[_Tried],
    rejected: list[tuple[str, str]],
    improver: Chat,
    meter: Meter,
    gen: int,
    task_stats: dict[tuple[str, str], dict[str, int]],
    slots_per_task: int,
    tuning_ids: set[str],
) -> tuple[Instructions, Instructions | None, dict[str, Any]]:
    """Score each arm on the frozen batch, keep or drop the challenger, and draft the next one."""
    frozen_ids = set(frozen["cluster_ids"])
    scored: dict[str, ArmScore] = {}
    for arm, (ins, _, _) in arms.items():
        s = ArmScore(ins.version, sorted(frozen_ids), slots=slots_per_task * len(frozen_ids))
        for cid in frozen_ids:
            st = task_stats.get((arm, cid), {})
            s.invalid += st.get("invalid", 0)
            s.duplicates += st.get("duplicates", 0)
            s.empty += st.get("empty", 0)
        for t in tried:
            if t.arm != arm or t.candidate.cluster_id not in frozen_ids:
                continue
            s.proposed += 1
            s.screened_out += t.status.startswith("screened")
            s.replay_tested += t.replay is not None
            s.replay_passed += bool(t.replay and t.replay.passed)
            s.evaluated += t.verdict is not None
            s.rejected += t.status.startswith(("rejected", "screened", "replay_failed"))
            if t.status == "admitted" and t.verdict is not None:
                s.admitted += 1
                s.stable_recoveries += t.verdict.stable_recoveries
        scored[arm] = s
    level2: dict[str, Any] = {
        "metric": "descendant admission yield",
        "champion": champion.version,
        "challenger": challenger.version if challenger else None,
        "frozen_batch": frozen,
        "seen": {
            v: sorted(seen.get(v, set()))
            for v in (champion.version, challenger.version if challenger else None)
            if v
        },
        "all_clusters_score": {name: s.to_payload() for name, (_, s, _) in arms.items()},
    }
    if challenger is not None and "challenger" in arms:
        ch, cs = scored["challenger"], scored["champion"]
        covered = {
            arm: sorted(
                {t.candidate.cluster_id for t in tried if t.arm == arm and t.status == "admitted"}
                & frozen_ids
            )
            for arm in ("champion", "challenger")
        }
        rule = {
            "more_distinct_admitted": ch.admitted > cs.admitted,
            "no_lost_cluster_coverage": set(covered["champion"]) <= set(covered["challenger"]),
            "no_fewer_stable_recoveries": ch.stable_recoveries >= cs.stable_recoveries,
        }
        keep = all(rule.values())
        level2 |= {
            "claim": True,
            "frozen_score": {"champion": cs.to_payload(), "challenger": ch.to_payload()},
            "covered_clusters": covered,
            "promotion_rule": rule,
            "kept": keep,
        }
        if keep:
            champion = challenger
        challenger = None
    elif challenger is not None:
        level2 |= {"claim": False, "reason": "no cluster unseen by both versions; challenger stays pending"}
    else:
        level2 |= {"claim": False, "reason": "no challenger this generation"}
    if challenger is None:
        tuning = [t for t in tried if t.candidate.cluster_id in tuning_ids]
        examples = [
            f"[{t.arm}] {t.candidate.kind} for cluster {t.candidate.cluster_id}: {t.status}"
            + (
                f" (replay: {t.replay.to_payload()['targets_hit']}/{len(t.replay.targets)} targets, "
                f"{len(t.replay.false_positives)} false positives, {len(t.replay.errors)} errors)"
                if t.replay
                else ""
            )
            + (f" (copied literals: {[h['literal'] for h in t.lint_hits][:5]})" if t.lint_hits else "")
            for t in tuning
        ] + [text for cid, text in rejected if cid in tuning_ids]
        shown = {t.candidate.cluster_id for t in tuning}
        challenger = propose_instructions(
            champion, {n: s for n, (_, s, _) in arms.items()}, examples, improver, meter
        )
        if challenger is not None:
            seen[challenger.version] = seen.get(champion.version, set()) | shown
            _record_instructions(store, challenger, seen[challenger.version], gen)
    level2 |= {
        "champion_after": champion.version,
        "next_challenger": challenger.version if challenger else None,
    }
    return champion, challenger, level2


def _record_instructions(store: ImproveStore, ins: Instructions, seen: set[str], gen: int) -> None:
    store.put_eval(
        {
            "type": "improver_instructions",
            "version": ins.version,
            "parent": ins.parent,
            "text": ins.text,
            "seen_clusters": sorted(seen),
            "generation": gen,
        }
    )


def _finish(
    store: ImproveStore, rec: GenerationRecord, started: float, meter: Meter, tried: list[_Tried]
) -> None:
    rec.candidates = [
        {
            **t.candidate.to_payload(),
            "arm": t.arm,
            "status": t.status,
            "lint_hits": t.lint_hits,
            "replay": t.replay.to_payload() if t.replay else None,
        }
        for t in tried
    ]
    c = meter.cost()
    rec.improver_cost = {
        "model_calls": c.model_calls,
        "input_tokens": c.input_tokens,
        "output_tokens": c.output_tokens,
    }
    rec.wall_s = round(time.perf_counter() - started, 3)
    store.put_eval(rec.to_payload(), eval_id=f"gen-{rec.parent}-{rec.generation}")
