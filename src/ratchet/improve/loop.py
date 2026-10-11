"""The ratchet loop: run, diagnose, propose, replay, evaluate, admit, promote, repeat.

Two levels improve, and they are scored separately.

Level 1, the harness: each generation turns the parent's training failures into candidate guards,
procedures and prompt edits; guards are proven by counterfactual replay before any agent run; candidates
are evaluated on training and held-out cases; the admission rule decides, and every decision is signed.

Level 2, the improver: the instructions the improver proposes with are a versioned artifact. After a
generation, the model proposes an edit to them (the challenger). In the next generation every cluster is
proposed for by both the current instructions (the champion) and the challenger, on the same clusters and
the same parent, and the challenger replaces the champion only if its admission yield (admitted /
proposed) is strictly higher.

The firewall: diagnosis and proposal read training traces only. Held-out cases and their traces reach the
admission check (the referee) and nothing else.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ratchet.improve.admit import Decision, admit
from ratchet.improve.diagnose import FailureCluster, diagnose
from ratchet.improve.evaluate import CaseLike, EvalReport, RunCase, evaluate_many
from ratchet.improve.llm import Chat, Meter, ask_json
from ratchet.improve.propose import SEED, Candidate, Instructions, propose, revise_guard
from ratchet.improve.receipt import Signer, receipt
from ratchet.improve.replay import GuardRunner, ReplayReport, WalkCache, replay
from ratchet.improve.sim import Simulator
from ratchet.types import Guard, Harness, Procedure, Trace


class ImproveStore(Protocol):
    def put_harness(self, harness: Harness) -> bool: ...
    def put_trace(self, trace: Trace) -> bool: ...
    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str: ...
    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str: ...


@dataclass
class Lab:
    """Everything the loop needs besides the store and the agent model."""

    cases: Sequence[CaseLike]
    run_case: RunCase
    simulator: Simulator
    runner: GuardRunner
    signer: Signer
    improver: Chat | None = None
    workers: int = 8
    repeats: int = 1
    max_clusters: int = 4
    candidates_per_cluster: int = 4
    max_evals: int = 6
    repair_rounds: int = 1
    log: Callable[[str], None] = lambda _msg: None


@dataclass
class ArmScore:
    """What one version of the improver's instructions achieved in one generation."""

    instructions_version: str
    proposed: int = 0
    replay_tested: int = 0
    replay_passed: int = 0
    evaluated: int = 0
    admitted: int = 0
    heldout_gain: int = 0

    @property
    def admission_yield(self) -> float:
        return self.admitted / self.proposed if self.proposed else 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "instructions_version": self.instructions_version,
            "proposed": self.proposed,
            "replay_tested": self.replay_tested,
            "replay_passed": self.replay_passed,
            "evaluated": self.evaluated,
            "admitted": self.admitted,
            "heldout_gain": self.heldout_gain,
            "admission_yield": round(self.admission_yield, 4),
        }


@dataclass
class GenerationRecord:
    generation: int
    parent: str
    promoted: str | None = None
    parent_score: dict[str, int] = field(default_factory=dict)
    child_score: dict[str, int] = field(default_factory=dict)
    clusters: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    replay: dict[str, Any] = field(default_factory=dict)
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    improver_cost: dict[str, int] = field(default_factory=dict)
    agent_cost: dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0
    note: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {"type": "generation", **{k: v for k, v in self.__dict__.items()}}


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
    decision: Decision | None = None
    report: EvalReport | None = None


def _score(r: EvalReport) -> dict[str, int]:
    return {
        "train_passed": len(r.passed("train")),
        "train_total": r.count("train"),
        "heldout_passed": len(r.passed("heldout")),
        "heldout_total": r.count("heldout"),
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


def _proof_replay(
    tried: _Tried,
    parent: Harness,
    cluster: FailureCluster,
    train_passing: list[Trace],
    walks: WalkCache,
    lab: Lab,
) -> None:
    """The improver's own check, on training traces only, with one repair round per failure."""
    c = tried.candidate
    for attempt in range(lab.repair_rounds + 1):
        guard = c.harness.guards[-1]
        rep = replay(guard, cluster.members, cluster.traces, train_passing, walks, lab.runner)
        tried.candidate, tried.replay = c, rep
        if rep.passed or attempt == lab.repair_rounds or lab.improver is None:
            break
        revised = revise_guard(c, rep, cluster, parent, lab.improver, walks=walks)
        if isinstance(revised, str):
            break
        c = revised
    tried.status = "replay_passed" if tried.replay and tried.replay.passed else "replay_failed"


def _shortlist(tried: list[_Tried], arms: list[str], budget: int) -> list[_Tried]:
    """Up to budget/len(arms) per arm: proven guards first (most targets), then procedures, then prompts."""
    rank = {"guard": 0, "procedure": 1, "prompt_edit": 2}
    per_arm = max(1, budget // max(1, len(arms)))
    chosen: list[_Tried] = []
    for arm in arms:
        pool = [t for t in tried if t.arm == arm and t.status in ("proposed", "replay_passed")]
        pool.sort(key=lambda t: (rank[t.candidate.kind], -len(t.candidate.targets)))
        # Round-robin over clusters so one big cluster does not take every slot.
        by_cluster: dict[str, list[_Tried]] = {}
        for t in pool:
            by_cluster.setdefault(t.candidate.cluster_id, []).append(t)
        while len([c for c in chosen if c.arm == arm]) < per_arm and any(by_cluster.values()):
            for queue in by_cluster.values():
                if queue and len([c for c in chosen if c.arm == arm]) < per_arm:
                    chosen.append(queue.pop(0))
    return chosen


def compose(parent: Harness, picks: Sequence[Candidate]) -> Harness:
    """One child carrying every pick's change: their new guards, new procedures and prompt additions."""
    guards: list[Guard] = list(parent.guards)
    procs: list[Procedure] = list(parent.procedures)
    prompt = parent.system_prompt
    for c in picks:
        guards += [g for g in c.harness.guards if g.id not in {x.id for x in guards}]
        procs += [p for p in c.harness.procedures if p.id not in {x.id for x in procs}]
        if c.kind == "prompt_edit":
            prompt = f"{prompt.rstrip()}\n\n{c.change['append']}"
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


def _admit_one(
    store: ImproveStore,
    lab: Lab,
    parent: Harness,
    parent_eval: EvalReport,
    harness: Harness,
    report: EvalReport,
    replays: dict[str, ReplayReport],
    context: dict[str, Any],
) -> Decision:
    d = admit(parent, harness, parent_eval=parent_eval, cand_eval=report, replays=replays, context=context)
    env = receipt(d, lab.signer)
    store.put_decision(
        {**d.to_payload(), "envelope": base64.b64encode(env).decode("ascii"), "keyid": lab.signer.keyid},
        decision_id=f"{d.parent}->{d.candidate}",
    )
    return d


def _referee_replays(
    tried: list[_Tried],
    parent: Harness,
    parent_eval: EvalReport,
    clusters: dict[str, FailureCluster],
    walks: WalkCache,
    lab: Lab,
) -> dict[str, ReplayReport]:
    """Rule 1 on the referee's side: every passing trace of the parent, held-out included."""
    passing = [t for t in parent_eval.traces if t.outcome.passed]
    out: dict[str, ReplayReport] = {}
    for t in tried:
        for g in t.candidate.new_guards(parent):
            cl = clusters[t.candidate.cluster_id]
            out[g.id] = replay(g, cl.members, cl.traces, passing, walks, lab.runner)
    return out


def evolve(
    store: ImproveStore,
    model: Any,
    generations: int = 3,
    *,
    lab: Lab,
    seed: Harness,
    instructions: Instructions = SEED,
    parent_eval: EvalReport | None = None,
) -> EvolveResult:
    """Run `generations` rounds of the ratchet from `seed`. Returns the final harness and every record."""
    improver: Chat = lab.improver or model
    lab.improver = improver
    parent, champion = seed, instructions
    challenger: Instructions | None = None
    store.put_harness(parent)
    store.put_eval(
        {
            "type": "improver_instructions",
            "version": champion.version,
            "parent": champion.parent,
            "text": champion.text,
        }
    )
    records: list[GenerationRecord] = []
    for gen in range(1, generations + 1):
        started = time.perf_counter()
        meter = Meter()
        rec = GenerationRecord(gen, parent.version)
        if parent_eval is None or parent_eval.harness_version != parent.version:
            parent_eval = evaluate_many(
                [parent], lab.cases, lab.run_case, model, workers=lab.workers, repeats=lab.repeats
            )[0]
            _persist_eval(store, parent_eval, {"generation": gen, "role": "parent"})
        rec.parent_score = _score(parent_eval)
        lab.log(f"gen {gen}: parent {parent.version} {rec.parent_score}")

        # The firewall: from here to the shortlist, only training traces are read.
        train = parent_eval.train_traces()
        train_passing = [t for t in train if t.outcome.passed]
        walks = WalkCache(lab.simulator)
        clusters = diagnose(train, lab.simulator, improver, meter)[: lab.max_clusters]
        rec.clusters = [c.to_payload() for c in clusters]
        if not clusters:
            rec.note = "no training failures left to diagnose"
            rec.wall_s = round(time.perf_counter() - started, 3)
            store.put_eval(rec.to_payload(), eval_id=f"gen-{parent.version}-{gen}")
            records.append(rec)
            break

        arms: dict[str, tuple[Instructions, ArmScore]] = {"champion": (champion, ArmScore(champion.version))}
        if challenger is not None:
            arms["challenger"] = (challenger, ArmScore(challenger.version))
        tried: list[_Tried] = []
        rejected: list[str] = []
        for cl in clusters:
            for arm, (ins, score) in arms.items():
                prop = propose(
                    cl,
                    parent,
                    improver,
                    instructions=ins,
                    passing=train_passing,
                    walks=walks,
                    meter=meter,
                    n=lab.candidates_per_cluster,
                )
                score.proposed += len(prop.candidates)
                rejected += [f"[{arm}] {r['reason']}" for r in prop.rejected]
                tried += [_Tried(c, arm) for c in prop.candidates]

        check_s = checks = 0.0
        for t in tried:
            if t.candidate.kind != "guard":
                continue
            _proof_replay(
                t, parent, clusters_by_id(clusters)[t.candidate.cluster_id], train_passing, walks, lab
            )
            arms[t.arm][1].replay_tested += 1
            arms[t.arm][1].replay_passed += t.status == "replay_passed"
            if t.replay is not None:
                check_s += t.replay.check_s
                checks += t.replay.checks
        rec.replay = {
            "guards_replayed": sum(1 for t in tried if t.replay),
            "checks": int(checks),
            "check_s": round(check_s, 4),
            "walk_s": round(walks.walk_s, 4),
            "checks_per_s": round(checks / check_s, 1) if check_s else 0.0,
        }

        shortlist = _shortlist(tried, list(arms), lab.max_evals)
        for t in shortlist:
            t.status = "evaluated"
        reports = evaluate_many(
            [t.candidate.harness for t in shortlist],
            lab.cases,
            lab.run_case,
            model,
            workers=lab.workers,
            repeats=lab.repeats,
        )
        referee = _referee_replays(shortlist, parent, parent_eval, clusters_by_id(clusters), walks, lab)
        admitted: list[_Tried] = []
        for t, report in zip(shortlist, reports, strict=True):
            t.report = report
            _persist_eval(
                store, report, {"generation": gen, "role": "candidate", "candidate": t.candidate.id}
            )
            ctx = {"generation": gen, "arm": t.arm, **t.candidate.to_payload()}
            new_ids = {g.id for g in t.candidate.new_guards(parent)}
            t.decision = _admit_one(
                store,
                lab,
                parent,
                parent_eval,
                t.candidate.harness,
                report,
                {k: v for k, v in referee.items() if k in new_ids},
                ctx,
            )
            rec.decisions.append(f"{t.decision.parent}->{t.decision.candidate}")
            score = arms[t.arm][1]
            score.evaluated += 1
            if t.decision.admitted:
                t.status = "admitted"
                score.admitted += 1
                score.heldout_gain += len(t.decision.check("heldout").evidence["gained"])
                admitted.append(t)
            else:
                t.status = "rejected: " + ", ".join(c.name for c in t.decision.checks if not c.passed)
        rec.candidates = [
            {
                **t.candidate.to_payload(),
                "arm": t.arm,
                "status": t.status,
                "replay": t.replay.to_payload() if t.replay else None,
            }
            for t in tried
        ]

        promoted = _promote(store, lab, model, gen, parent, parent_eval, admitted, referee, rec)
        if promoted is not None:
            parent, parent_eval = promoted
            store.put_harness(parent)
            rec.promoted = parent.version
            rec.child_score = _score(parent_eval)

        # Level 2: judge the challenger on this generation's paired results, then draft the next one.
        rec.arms = {name: s.to_payload() for name, (_, s) in arms.items()}
        if "challenger" in arms:
            ch, cs = arms["challenger"][1], arms["champion"][1]
            keep = ch.admission_yield > cs.admission_yield
            rec.meta = {
                "challenger": challenger.version if challenger else None,
                "champion": champion.version,
                "challenger_yield": round(ch.admission_yield, 4),
                "champion_yield": round(cs.admission_yield, 4),
                "kept": keep,
            }
            if keep and challenger is not None:
                champion = challenger
        examples = [
            f"[{t.arm}] {t.candidate.kind} for cluster {t.candidate.cluster_id}: {t.status}"
            + (
                f" (replay: {t.replay.to_payload()['targets_hit']}/{len(t.replay.targets)} targets, "
                f"{len(t.replay.false_positives)} false positives, {len(t.replay.errors)} errors)"
                if t.replay
                else ""
            )
            for t in tried
        ] + rejected
        challenger = propose_instructions(
            champion, {n: s for n, (_, s) in arms.items()}, examples, improver, meter
        )
        if challenger is not None:
            store.put_eval(
                {
                    "type": "improver_instructions",
                    "version": challenger.version,
                    "parent": challenger.parent,
                    "text": challenger.text,
                    "generation": gen,
                }
            )
        rec.meta = {
            **rec.meta,
            "next_challenger": challenger.version if challenger else None,
            "champion_after": champion.version,
        }
        c = meter.cost()
        rec.improver_cost = {
            "model_calls": c.model_calls,
            "input_tokens": c.input_tokens,
            "output_tokens": c.output_tokens,
        }
        rec.agent_cost = {"candidates": [_cost(r) for r in reports]}
        rec.wall_s = round(time.perf_counter() - started, 3)
        store.put_eval(rec.to_payload(), eval_id=f"gen-{rec.parent}-{gen}")
        lab.log(f"gen {gen}: promoted {rec.promoted} arms {rec.arms} meta {rec.meta}")
        records.append(rec)
    return EvolveResult(parent, champion, records)


def clusters_by_id(clusters: Sequence[FailureCluster]) -> dict[str, FailureCluster]:
    return {c.id: c for c in clusters}


def _promote(
    store: ImproveStore,
    lab: Lab,
    model: Any,
    gen: int,
    parent: Harness,
    parent_eval: EvalReport,
    admitted: list[_Tried],
    referee: dict[str, ReplayReport],
    rec: GenerationRecord,
) -> tuple[Harness, EvalReport] | None:
    """The best admitted candidate, or the composition of all of them when that is admitted and better."""
    if not admitted:
        return None

    def key(t: _Tried) -> tuple[int, int, int]:
        assert t.report is not None
        return (len(t.report.passed("heldout")), len(t.report.passed("train")), -t.report.cost().model_calls)

    best = max(admitted, key=key)
    assert best.report is not None
    picks = list({t.candidate.cluster_id: t for t in sorted(admitted, key=key)}.values())
    if len(picks) < 2:
        return best.candidate.harness, best.report
    combo = compose(parent, [t.candidate for t in picks])
    report = evaluate_many([combo], lab.cases, lab.run_case, model, workers=lab.workers, repeats=lab.repeats)[
        0
    ]
    _persist_eval(store, report, {"generation": gen, "role": "composite"})
    ctx = {"generation": gen, "composite_of": [t.candidate.id for t in picks]}
    d = _admit_one(store, lab, parent, parent_eval, combo, report, referee, ctx)
    rec.decisions.append(f"{d.parent}->{d.candidate}")
    if d.admitted and len(report.passed("heldout")) > len(best.report.passed("heldout")):
        return combo, report
    return best.candidate.harness, best.report
