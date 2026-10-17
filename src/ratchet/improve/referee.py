"""The held-out referee: the one process that runs, reads and grades held-out cases, under its own key.

The loop (the improver) holds training cases only. Everything that touches a held-out case or its trace runs
here: the parent's held-out runs, each candidate's held-out run and confirming run, the referee-side replay
over every passing and benign-twin trace (held-out included), the admission rule, the decision signature
(the referee's key, which the improver never holds) and activation. The improver sends training evidence
and gets back totals: held-out passes and counts, the admission flag, the names of failed gates and the
receipt digest. It never receives a held-out case id, outcome or trace.

`ProcessReferee` runs the referee as a separate Python process, built in that process by a factory named
`module:function` from a JSON config, and speaks newline-delimited JSON to it; the referee parses data only,
never pickles. With an Atlas store, the referee's records live in the held-out database, which the
improver's database role cannot read (tests/store/test_store_roles.py). `RefereeCore` is the logic itself,
used inside that process and directly in tests.

    python -m ratchet.improve.referee FACTORY      (started by ProcessReferee; the protocol is on stdio)
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

from ratchet_cases.types import Trace, sha256_hex

from ratchet.canonical import canonical_json
from ratchet.improve import wire
from ratchet.improve.activate import Activation, Activator
from ratchet.improve.admit import Check, Decision, ExtendedChange, admit, admit_v3, awaiting_confirmation
from ratchet.improve.diagnose import FailureCluster
from ratchet.improve.evaluate import CaseLike, EvalReport, RunCase, _one, evaluate_many
from ratchet.improve.evaluator import evaluator_identity, require_same
from ratchet.improve.overfit import Literals
from ratchet.improve.policies import RULING_2, c1_flips, normalized, rebuild
from ratchet.improve.receipt import Signer, receipt
from ratchet.improve.replay import GuardRunner, ReplayReport, WalkCache, replay
from ratchet.improve.results import Ledger
from ratchet.improve.sim import Simulator
from ratchet.types import Harness


class RefereeError(RuntimeError):
    """The referee refused a request or failed; the message is the referee's own."""


class RefereeStore(Protocol):
    def put_trace(self, trace: Trace) -> bool: ...
    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str: ...
    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str: ...


@dataclass
class RefereeParts:
    """What the referee holds. Built inside the referee's process by its factory."""

    cases: Sequence[CaseLike]
    run_case: RunCase
    model: Any
    simulator: Simulator
    runner: GuardRunner
    signer: Signer
    store: RefereeStore
    activator: Activator
    model_config: dict[str, Any] = field(default_factory=dict)
    suite_digest: str | None = None
    workers: int = 8
    repeats: int = 1
    # Candidates that add a tool or edit a procedure, and the admitted k-session bundle each is admitted on.
    session_bundles: Mapping[str, Path] = field(default_factory=dict)
    # Where the signed results file and each decision's verification bundle are written (None: not written).
    results_dir: Path | None = None
    # The stored decision a policy re-judges, by (database, candidate version); None: none can be named.
    decisions: Callable[[str, str], Mapping[str, Any] | None] | None = None
    # The stochastic policy's per-repetition worker clients (`ratchet.improve.endpoints.PerRepetition`).
    repetitions: Any = None
    close: Callable[[], None] = lambda: None


@dataclass(frozen=True)
class Totals:
    """What the improver may know about one run: counts, never which cases."""

    harness: str
    train_passed: int
    train_total: int
    heldout_passed: int
    heldout_total: int
    heldout_invalid: int


@dataclass(frozen=True)
class Verdict:
    """The referee's answer for one candidate: totals, the admission flag and the receipt that records it."""

    decision_id: str
    receipt_digest: str
    candidate: str
    admitted: bool
    failed: tuple[str, ...]
    heldout_passed: int
    heldout_total: int
    stable_recoveries: int
    protected_broken: int
    c1_fail_to_pass: int
    c1_pass_to_fail: int
    rates: dict[str, Any]


@dataclass(frozen=True)
class Request:
    """One candidate for `decide_many`: the improver's training run of it and what its guards were written
    for. `decision_id` names the stored decision (default `<parent>-><candidate>`)."""

    candidate: Harness
    train: EvalReport
    targets: Sequence[FailureCluster]
    guard_clusters: Mapping[str, str]
    literals: Literals
    visible_case_ids: Sequence[str]
    context: Mapping[str, Any]
    decision_id: str | None = None


class Referee(Protocol):
    """The improver's view of the referee."""

    def start(self, train_identity: Mapping[str, Any]) -> dict[str, Any]: ...
    def bootstrap(self, version: str) -> tuple[str | None, int]: ...
    def parent_runs(self, generation: int, parent: Harness, train: EvalReport) -> tuple[Totals, Totals]: ...
    def evaluate(self, role: str, harness: Harness, train: EvalReport) -> Totals: ...
    def decide(
        self,
        generation: int,
        candidate: Harness,
        train: EvalReport,
        targets: Sequence[FailureCluster],
        guard_clusters: Mapping[str, str],
        literals: Literals,
        visible_case_ids: Sequence[str],
        context: Mapping[str, Any],
    ) -> Verdict: ...
    def decide_many(self, generation: int, requests: Sequence[Request]) -> list[Verdict]: ...
    def results(self, name: str) -> dict[str, Any]: ...
    def activate(
        self,
        *,
        candidate: str,
        expected: str | None,
        expected_revision: int,
        decision_id: str,
        receipt_digest: str,
        idempotency_key: str,
    ) -> Activation: ...
    def active(self) -> tuple[str | None, int]: ...
    def public_key(self) -> str: ...
    def close(self) -> None: ...


def combine_identity(train: Mapping[str, Any], heldout: Mapping[str, Any]) -> dict[str, Any]:
    """The evaluator over both splits from the improver's identity of its training cases and the referee's of
    its held-out cases. Component digests are named by split; the suite digest must agree."""
    if train["components"].get("suite_file") != heldout["components"].get("suite_file"):
        raise RefereeError("the improver and the referee name different case suites")
    components = {f"train_{k}": v for k, v in train["components"].items() if k != "suite_file"}
    components |= {f"heldout_{k}": v for k, v in heldout["components"].items() if k != "suite_file"}
    components["suite_file"] = heldout["components"].get("suite_file")
    if dict(train["model_config"]) != dict(heldout["model_config"]):
        raise RefereeError("the improver and the referee run different model configurations")
    return {
        "version": sha256_hex(canonical_json(components))[:12],
        "components": components,
        "model_config": dict(heldout["model_config"]),
        "cases": int(train["cases"]) + int(heldout["cases"]),
    }


def _merge(train: EvalReport, heldout: EvalReport) -> EvalReport:
    if train.harness_version != heldout.harness_version:
        raise RefereeError("the training and held-out runs are of different harnesses")
    if any(r.split == "heldout" for r in train.results):
        raise RefereeError("the improver sent held-out results")
    return EvalReport(
        train.harness_version,
        [*train.results, *heldout.results],
        train.wall_s + heldout.wall_s,
        [*train.traces, *heldout.traces],
    )


def totals(r: EvalReport) -> Totals:
    held = [x for x in r.results if x.split == "heldout"]
    return Totals(
        r.harness_version,
        len(r.passed("train")),
        r.count("train"),
        len(r.passed("heldout")),
        r.count("heldout"),
        sum(1 for x in held if x.infra_failed or x.error),
    )


def _spec(spec: str) -> tuple[str, str]:
    db, sep, version = spec.partition(":")
    if not sep or not db or not version:
        raise RefereeError(f"a stored candidate is named db:version, got {spec!r}")
    return db, version


_V3_ROW = ("case_id", "parent", "parent_panels", "arm", "arm_panels", "parent_class", "arm_class", "decline")


def _v3_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in _V3_ROW}


def _v3_counts(cmp: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """What the improver may know of one arm against the parent reference: counts and pooled rates."""
    return {
        "arm": arm,
        "stable_recoveries": len(cmp["stable_recoveries"]),
        "stable_regressions": len(cmp["stable_regressions"]),
        "declines_over_limit": len(cmp["case_declines_over_limit"]),
        "incomplete_cases": len(cmp["incomplete_cases"]),
        "aggregate": cmp["aggregate"],
    }


class RefereeCore:
    """The referee's logic. One generation at a time: `parent_runs`, then `decide` for each candidate."""

    def __init__(self, parts: RefereeParts) -> None:
        if not parts.cases or any(c.split != "heldout" for c in parts.cases):
            raise ValueError("the referee holds held-out cases, and only those")
        self.parts = parts
        self.walks = WalkCache(parts.simulator)
        self.evaluator: dict[str, Any] | None = None
        self._train_identity: dict[str, Any] | None = None
        self._generation: tuple[int, Harness, tuple[EvalReport, EvalReport]] | None = None
        self.ledger = Ledger(parts.results_dir)

    def _heldout_identity(self) -> dict[str, Any]:
        return evaluator_identity(self.parts.cases, self.parts.model_config, self.parts.suite_digest)

    def start(self, train_identity: Mapping[str, Any]) -> dict[str, Any]:
        self._train_identity = dict(train_identity)
        self.evaluator = combine_identity(train_identity, self._heldout_identity())
        self.parts.store.put_eval(
            {"type": "evaluator", **self.evaluator}, eval_id=f"evaluator-{self.evaluator['version']}"
        )
        return self.evaluator

    def bootstrap(self, version: str) -> tuple[str | None, int]:
        return self.parts.activator.bootstrap(version)

    def _run(self, h: Harness) -> EvalReport:
        p = self.parts
        return evaluate_many([h], p.cases, p.run_case, p.model, workers=p.workers, repeats=p.repeats)[0]

    def _persist(self, report: EvalReport, context: dict[str, Any]) -> None:
        for t in report.traces:
            self.parts.store.put_trace(t)
        self.parts.store.put_eval({**report.to_payload(), **context})

    def parent_runs(self, generation: int, parent: Harness, train: EvalReport) -> tuple[Totals, Totals]:
        """The parent runs twice, fresh, on the held-out cases; the first run joins the improver's training
        run as P1, the second is P2. The noise floor is measured, never assumed."""
        if self.evaluator is None:
            raise RefereeError("start the referee first")
        p1 = _merge(train, self._run(parent))
        p2 = self._run(parent)
        self._persist(p1, {"generation": generation, "role": "parent_run_1"})
        self._persist(p2, {"generation": generation, "role": "parent_run_2"})
        self._generation = (generation, parent, (p1, p2))
        return totals(p1), totals(p2)

    def evaluate(self, role: str, harness: Harness, train: EvalReport) -> Totals:
        """One run of `harness` on the held-out cases, joined to the improver's training run and kept under
        `role` (a baseline, say). Outside any generation; the improver gets the counts."""
        report = _merge(train, self._run(harness))
        self._persist(report, {"role": role})
        return totals(report)

    def _sets(self) -> tuple[list[Trace], list[Trace]]:
        """The clearance traces: every passing parent trace and every benign-twin trace, held-out included."""
        assert self._generation is not None
        p1 = self._generation[2][0]
        benign = {r.case_id for r in p1.results if r.kind == "benign_twin"}
        return [t for t in p1.traces if t.outcome.passed], [t for t in p1.traces if t.case_id in benign]

    def _replays(
        self,
        generation: int,
        parent: Harness,
        candidate: Harness,
        targets: Sequence[FailureCluster],
        guard_clusters: Mapping[str, str],
    ) -> tuple[dict[str, ReplayReport], ReplayReport | None]:
        """Rule 1 on the referee's side: each added guard against the failures it was written for, then every
        guard of the candidate at once, over every passing and benign-twin parent trace."""
        passing, benign = self._sets()
        by_id = {c.id: c for c in targets}
        old = {g.id for g in parent.guards}
        added = [g for g in candidate.guards if g.id not in old]
        out: dict[str, ReplayReport] = {}
        for guard in added:
            cl = by_id.get(guard_clusters.get(guard.id, ""))
            if cl is None:
                raise RefereeError(f"guard {guard.id} names no target cluster")
            out[guard.id] = replay(
                guard, cl.members, cl.traces, passing, self.walks, self.parts.runner, benign
            )
            self._persist_replay(generation, parent, out[guard.id])
        composed = None
        if added and targets:
            members = [m for c in targets for m in c.members]
            failing = {k: v for c in targets for k, v in c.traces.items()}
            composed = replay(
                candidate.guards, members, failing, passing, self.walks, self.parts.runner, benign
            )
            self._persist_replay(generation, parent, composed)
        return out, composed

    def _persist_replay(self, generation: int, parent: Harness, report: ReplayReport) -> None:
        self.parts.store.put_eval(
            {
                "type": "replay",
                "role": "referee",
                "generation": generation,
                "parent": parent.version,
                **report.to_payload(),
            }
        )

    def decide(
        self,
        generation: int,
        candidate: Harness,
        train: EvalReport,
        targets: Sequence[FailureCluster],
        guard_clusters: Mapping[str, str],
        literals: Literals,
        visible_case_ids: Sequence[str],
        context: Mapping[str, Any],
        decision_id: str | None = None,
    ) -> Verdict:
        """Run the candidate on the held-out cases, replay its guards, admit or reject, and sign. A candidate
        that clears every gate on its first run gets one more fresh held-out run to confirm."""
        request = Request(
            candidate, train, targets, guard_clusters, literals, visible_case_ids, context, decision_id
        )
        return self.decide_many(generation, [request])[0]

    def _ready(self, generation: int) -> tuple[Harness, tuple[EvalReport, EvalReport]]:
        if self._generation is None or self._generation[0] != generation:
            raise RefereeError(f"generation {generation} has no parent runs")
        if self.evaluator is None or self._train_identity is None:
            raise RefereeError("start the referee first")
        require_same(self.evaluator, combine_identity(self._train_identity, self._heldout_identity()))
        return self._generation[1], self._generation[2]

    def _run_many(self, harnesses: Sequence[Harness]) -> list[EvalReport]:
        if not harnesses:
            return []
        p = self.parts
        return evaluate_many(harnesses, p.cases, p.run_case, p.model, workers=p.workers, repeats=p.repeats)

    def decide_many(
        self, generation: int, requests: Sequence[Request], policy: Mapping[str, Any] | None = None
    ) -> list[Verdict]:
        """`decide` for several candidates of one generation, with every first run in one bounded pool and
        every confirming run in another. `policy` (ruling 2 on a deterministic endpoint, say) is recorded on
        each decision."""
        parent, parent_runs = self._ready(generation)
        for r in requests:
            if r.candidate.parent != parent.version:
                raise RefereeError(f"candidate {r.candidate.version} is not a child of {parent.version}")
        firsts = [
            _merge(r.train, c)
            for r, c in zip(requests, self._run_many([r.candidate for r in requests]), strict=True)
        ]
        kws: list[dict[str, Any]] = []
        decisions: list[Decision] = []
        for r, c1 in zip(requests, firsts, strict=True):
            self._persist(c1, {"generation": generation, "role": "candidate", "harness": r.candidate.version})
            replays, composed = self._replays(generation, parent, r.candidate, r.targets, r.guard_clusters)
            kw: dict[str, Any] = {
                "parent_runs": parent_runs,
                "cand_eval": c1,
                "replays": replays,
                "composed": composed,
                "literals": r.literals,
                "visible_case_ids": r.visible_case_ids,
                "evaluator": self.evaluator,
                "context": dict(r.context),
            }
            bundle = self.parts.session_bundles.get(r.candidate.version)
            kw["extended"] = ExtendedChange(Path(bundle)) if bundle is not None else None
            kws.append(kw)
            decisions.append(admit(parent, r.candidate, confirm_eval=None, **kw))
        pending = [i for i, d in enumerate(decisions) if awaiting_confirmation(d)]
        confirms: dict[int, EvalReport] = {}
        for i, confirm in zip(pending, self._run_many([requests[i].candidate for i in pending]), strict=True):
            version = requests[i].candidate.version
            self._persist(confirm, {"generation": generation, "role": "confirm", "harness": version})
            confirms[i] = confirm
            decisions[i] = admit(parent, requests[i].candidate, confirm_eval=confirm, **kws[i])
        return [
            self._sign(d, r, parent, (firsts[i], confirms.get(i)), kws[i], policy)
            for i, (d, r) in enumerate(zip(decisions, requests, strict=True))
        ]

    def _sign(
        self,
        d: Decision,
        r: Request,
        parent: Harness,
        cand: tuple[EvalReport, EvalReport | None],
        kw: Mapping[str, Any],
        policy: Mapping[str, Any] | None,
    ) -> Verdict:
        if policy is not None:
            d = dataclasses.replace(d, policy=dict(policy))
        signer = self.parts.signer
        env = receipt(d, signer)
        decision_id = r.decision_id or f"{d.parent}->{d.candidate}"
        self.parts.store.put_decision(
            {**d.to_payload(), "envelope": base64.b64encode(env).decode("ascii"), "keyid": signer.keyid},
            decision_id=decision_id,
        )
        held = d.check("heldout").evidence
        up, down = c1_flips(held)
        self._record(d, decision_id, env, r.candidate, parent, cand, kw)
        c1 = totals(cand[0])
        return Verdict(
            decision_id,
            sha256_hex(env),
            d.candidate,
            d.admitted,
            tuple(c.name for c in d.checks if not c.passed),
            c1.heldout_passed,
            c1.heldout_total,
            len(held["stable_recoveries"]),
            len(held["protected_broken"]),
            len(up),
            len(down),
            dict(d.rates),
        )

    def _record(
        self,
        d: Decision,
        decision_id: str,
        env: bytes,
        candidate: Harness,
        parent: Harness,
        cand: tuple[EvalReport, EvalReport | None],
        kw: Mapping[str, Any],
    ) -> None:
        """The decision's verification bundle and its row in the results file: the per-case held-out table
        the improver never receives."""
        assert self._generation is not None
        p1, p2 = self._generation[2]
        named: dict[str, EvalReport] = {"parent_run_1": p1, "parent_run_2": p2, "candidate_run": cand[0]}
        if cand[1] is not None:
            named["candidate_confirm_run"] = cand[1]
        reports: dict[str, Any] = {k: e.to_payload() for k, e in named.items()}
        reports |= {f"replay:{gid}": rep.to_payload() for gid, rep in kw["replays"].items()}
        if kw["composed"] is not None:
            reports["replay:composed"] = kw["composed"].to_payload()
        traces = {k: [wire.trace_doc(t) for t in e.traces] for k, e in named.items()}
        harnesses = {"parent": wire.harness_doc(parent), "candidate": wire.harness_doc(candidate)}
        rel = self.ledger.bundle(decision_id, env, reports, traces, harnesses)
        held = d.check("heldout").evidence
        up, down = c1_flips(held)
        self.ledger.add(
            {
                "decision_id": decision_id,
                "policy": (d.policy or {}).get("name", "ruling-2"),
                "receipt_sha256": sha256_hex(env),
                "parent": d.parent,
                "candidate": d.candidate,
                "admitted": d.admitted,
                "failed": [c.name for c in d.checks if not c.passed],
                "bundle": rel,
                "heldout": {
                    "order": held["order"],
                    "outcomes": held["outcomes"],
                    "stable_recoveries": held["stable_recoveries"],
                    "protected_broken": held["protected_broken"],
                    "c1_fail_to_pass": up,
                    "c1_pass_to_fail": down,
                },
            }
        )

    def results(self, name: str) -> dict[str, Any]:
        """Sign the results file under the decision key and return what a driver may record."""
        try:
            return self.ledger.publish(name, self.parts.signer, self.evaluator)
        except ValueError as exc:
            raise RefereeError(str(exc)) from exc

    def candidate(self, spec: str, parent: Harness) -> dict[str, Any]:
        """The candidate a stored decision judged, rebuilt and checked by digest, and the part of the
        decision's context that is the improver's own (the change, its kind, the cluster and generation it
        came from). The decision's checks and their held-out evidence stay here."""
        db, version = _spec(spec)
        found = self._original(db, version)
        harness = rebuild(parent, found)
        ctx = found.get("context") or {}
        keep = ("kind", "change", "cluster_id", "generation", "experiment", "arm")
        return {
            "harness": wire.harness_doc(harness),
            "context": {k: ctx[k] for k in keep if k in ctx},
            "source": f"{db}:{found.get('parent')}->{found.get('candidate')}",
        }

    def _original(self, db: str, version: str) -> Mapping[str, Any]:
        if self.parts.decisions is None:
            raise RefereeError("this referee has no source of stored decisions")
        found = self.parts.decisions(db, version)
        if found is None:
            raise RefereeError(f"no decision for {db}:{version}")
        return found

    def _throughput(self, role: str, report: EvalReport) -> dict[str, Any]:
        n = len(report.results)
        return {
            "role": role,
            "runs": n,
            "wall_s": round(report.wall_s, 1),
            "runs_per_min": round(60 * n / report.wall_s, 2) if report.wall_s else None,
            "workers": self.parts.workers,
        }

    def deterministic(
        self,
        parent: Harness,
        request: Request,
        parent_train: EvalReport,
        repeat_heldout: int = 3,
    ) -> dict[str, Any]:
        """Ruling 2 on a deterministic endpoint (temperature 0, a fixed seed, endpoints in round robin).

        1. Repeatability: the parent runs twice on the first `repeat_heldout` held-out cases (0: all); each
           pair must give the same normalized trace.
        2. P1 and P2 (the parent's held-out runs); any held-out case on which they disagree stops the track.
        3. C1, replay, `admit()`, C2 when C1 clears every gate, and a receipt that names ruling 2, the
           deterministic evaluator and the endpoint configuration digest.

        The improver receives whether each stage held, counts and the verdict; the per-case digests and
        outcomes go to the results file."""
        if self.evaluator is None:
            raise RefereeError("start the referee first")
        digest = self.parts.model_config.get("endpoint_config_digest")
        out: dict[str, Any] = {"endpoint_config_digest": digest, "throughput": []}
        cases = self.parts.cases
        probe = list(cases[:repeat_heldout] if repeat_heldout > 0 else cases)
        first, second = (self._run_on(parent, probe, role, out) for role in ("repeat_a", "repeat_b"))
        digests = {t.case_id: normalized(t) for t in first.traces}
        again = {t.case_id: normalized(t) for t in second.traces}
        repeat = {c.id: digests.get(c.id) == again.get(c.id) for c in probe}
        self.ledger.note("repeatability", {"cases": repeat, "digests_a": digests, "digests_b": again})
        out["repeatability"] = {
            "probed": len(repeat),
            "repeatable": all(repeat.values()),
            "differing": sum(1 for v in repeat.values() if not v),
        }
        if not all(repeat.values()):
            return self._stop(
                out, "NOT REPEATABLE: two identical runs produced different traces; track stopped"
            )
        self.parent_runs(1, parent, parent_train)
        assert self._generation is not None
        p1, p2 = self._generation[2]
        out["throughput"] += [self._throughput("parent_run_1", p1), self._throughput("parent_run_2", p2)]
        p1h, p2h = ({r.case_id: r.passed for r in e.results if r.split == "heldout"} for e in (p1, p2))
        differ = sorted(k for k in p1h if p1h[k] != p2h.get(k))
        self.ledger.note("parent_discordance", differ)
        out["parent_discordance"] = len(differ)
        if differ:
            return self._stop(
                out, f"NOT REPEATABLE: P1 and P2 differ on {len(differ)} held-out case(s); track stopped"
            )
        policy = {**RULING_2, "endpoint_config_digest": digest}
        context = {**request.context, "policy": RULING_2, "endpoint_config_digest": digest}
        verdict = self.decide_many(1, [dataclasses.replace(request, context=context)], policy=policy)[0]
        out |= {"result": "decided", "verdict": asdict(verdict)}
        return out

    def _run_on(self, h: Harness, cases: Sequence[CaseLike], role: str, out: dict[str, Any]) -> EvalReport:
        p = self.parts
        report = evaluate_many([h], cases, p.run_case, p.model, workers=p.workers)[0]
        self._persist(
            report, {"role": role, "endpoint_config_digest": p.model_config.get("endpoint_config_digest")}
        )
        out["throughput"].append(self._throughput(role, report))
        return report

    def _stop(self, out: dict[str, Any], result: str) -> dict[str, Any]:
        out["result"] = result
        self.parts.store.put_eval({"type": "endpoint_verdict", "policy": RULING_2, **out})
        return out

    def stochastic(
        self,
        parent: Harness,
        candidates: Sequence[str],
        case_ids: Sequence[str],
        schedule_seed: int,
        *,
        deadline_s: float = float("inf"),
        scope_note: str = "",
        workers: int = 128,
    ) -> dict[str, Any]:
        """Stochastic policy v3, live: a parent reference arm, two identity-control arms and one arm per
        stored candidate (`db:version`), 40 fresh executions per case per arm, interleaved by a predeclared
        schedule; then `admit_v3` and a signed receipt per candidate.

        Every execution is stored as it completes (`v3_execution`), and its trace is appended to the
        cohort's trace file, so a stopped run keeps its evidence. The improver receives counts, pooled
        rates and the signed decisions; per-case tables go to the results file."""
        from ratchet.improve.stochastic import POLICY, CaseInfo, schedule, schedule_digest

        p = self.parts
        if p.repetitions is None:
            raise RefereeError("this referee has no per-repetition worker clients")
        by_id = {c.id: c for c in p.cases}
        unknown = sorted(set(case_ids) - set(by_id))
        if unknown or not case_ids:
            raise RefereeError(f"v3 evaluates this referee's held-out cases only; not held here: {unknown}")
        infos = [CaseInfo(c, by_id[c].kind, by_id[c].must_succeed) for c in case_ids]
        originals: dict[str, dict[str, Any]] = {}
        arms: dict[str, Harness] = {"parent": parent, "identity_1": parent, "identity_2": parent}
        for spec in candidates:
            db, version = _spec(spec)
            d = self._original(db, version)
            originals[version] = {**d, "source_db": db}
            arms[f"candidate:{version}"] = rebuild(parent, d)
        evaluator = evaluator_identity([by_id[c] for c in case_ids], p.model_config, p.suite_digest)
        order = schedule(list(arms), list(case_ids), schedule_seed)
        frozen = {
            "policy": POLICY,
            "schedule_seed": schedule_seed,
            "schedule_digest": schedule_digest(order),
            "cases": list(case_ids),
            "arms": {k: h.version for k, h in arms.items()},
            "evaluator": evaluator["version"],
            "model_config": p.model_config,
            "workers": workers,
            "infra_retries": 2,
            "scope_note": scope_note,
        }
        p.store.put_eval(
            {"type": "v3_frozen", **frozen}, eval_id=f"v3-frozen-{frozen['schedule_digest'][:16]}"
        )
        cohort_dir = self.ledger.cohort(frozen["schedule_digest"])
        results, wall = self._execute(arms, order, frozen, deadline_s, workers, cohort_dir)
        return self._v3_decide(parent, arms, results, infos, originals, frozen, evaluator, wall, cohort_dir)

    def _execute(
        self,
        arms: Mapping[str, Harness],
        order: Sequence[tuple[str, str, int]],
        frozen: Mapping[str, Any],
        deadline_s: float,
        workers: int,
        cohort_dir: str | None,
    ) -> tuple[dict[str, Any], float]:
        from ratchet.improve.stochastic import Arm, Execution, execution_record, panel_of

        p = self.parts
        by_id = {c.id: c for c in p.cases}
        seed_base = p.repetitions.seed_base
        results: dict[str, Arm] = {k: Arm(k, h.version) for k, h in arms.items()}
        lock = threading.Lock()
        started = time.perf_counter()

        def run_one(index: int, arm: str, case_id: str, rep: int) -> Execution:
            if time.perf_counter() - started > deadline_s:
                return Execution(
                    arm, case_id, rep, panel_of(rep), seed_base + rep, None, note="not run: deadline"
                )
            url, client = p.repetitions.client_for(index, rep)
            res, trace = _one(p.run_case, arms[arm], by_id[case_id], client, None)
            valid = res.mode == "FRESH" and not res.infra_failed and not res.error
            if trace is not None:
                p.store.put_trace(trace)
                self.ledger.append_trace(cohort_dir, arm, wire.trace_doc(trace))
            return Execution(
                arm, case_id, rep, panel_of(rep), seed_base + rep, res.passed if valid else None,
                res.run_id, res.interventions, res.attempts, "" if valid else (res.error or "invalid"), url,
            )  # fmt: skip

        def one(index: int, arm: str, case_id: str, rep: int) -> None:
            try:
                ex = run_one(index, arm, case_id, rep)
            except Exception as exc:  # recorded as an invalid execution
                note = f"driver error: {exc}"[:300]
                ex = Execution(arm, case_id, rep, panel_of(rep), seed_base + rep, None, note=note)
            p.store.put_eval(
                {"type": "v3_execution", "schedule_digest": frozen["schedule_digest"], **execution_record(ex)}
            )
            with lock:
                results[arm].executions.append(ex)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in [pool.submit(one, i, arm, c, r) for i, (arm, c, r) in enumerate(order)]:
                f.result()
        return results, time.perf_counter() - started

    def _v3_decide(
        self,
        parent: Harness,
        arms: Mapping[str, Harness],
        results: Mapping[str, Any],
        infos: Sequence[Any],
        originals: Mapping[str, Mapping[str, Any]],
        frozen: Mapping[str, Any],
        evaluator: Mapping[str, Any],
        wall: float,
        cohort_dir: str | None,
    ) -> dict[str, Any]:
        from ratchet.improve.stochastic import RUNS, compare, execution_record, qualify

        p = self.parts
        kinds = {c.id: c.kind for c in p.cases}
        cohort = qualify(results["parent"], [results["identity_1"], results["identity_2"]], infos)
        records = {
            k: [execution_record(e) for e in sorted(v.executions, key=lambda e: (e.case_id, e.rep))]
            for k, v in results.items()
        }
        arm_digests = {k: sha256_hex(canonical_json(v)) for k, v in records.items()}
        self.ledger.cohort_records(cohort_dir, records)
        invalid = [e for v in results.values() for e in v.executions if e.passed is None]
        self.ledger.note(
            "v3_identity_controls",
            {c["arm"]: [_v3_row(r) for r in c["cases"]] for c in cohort["controls"]},
        )
        summary: dict[str, Any] = {
            "frozen": {k: frozen[k] for k in ("schedule_digest", "arms", "evaluator", "policy")}
            | {"cases": len(frozen["cases"])},
            "wall_s": round(wall, 1),
            "executions": sum(len(v.executions) for v in results.values()),
            "invalid": len(invalid),
            "cohort": cohort["status"],
            "controls": [_v3_counts(c, c["arm"]) | {"qualifies": c["qualifies"]} for c in cohort["controls"]],
            "decisions": [],
        }
        for version, d in originals.items():
            name = f"candidate:{version}"
            cmp = compare(results["parent"], results[name], infos)
            benign = sum(
                e.interventions for e in results[name].executions if kinds[e.case_id] == "benign_twin"
            )
            checks = {c["name"]: c for c in d["checks"]}
            source = f"{d['source_db']}:{d['parent']}->{d['candidate']}"
            dec = admit_v3(
                parent,
                arms[name],
                cohort=cohort,
                comparison=cmp,
                benign_interventions=benign,
                replay_check=Check(
                    "replay",
                    checks["replay"]["passed"],
                    {**checks["replay"]["evidence"], "from_decision": source},
                ),
                overfit_check=Check(
                    "overfit_lint", checks["overfit_lint"]["passed"], checks["overfit_lint"]["evidence"]
                ),
                evaluator=evaluator,
                visible_case_ids=d.get("visible_case_ids", []),
                report_digests={f"arm:{k}": v for k, v in arm_digests.items()}
                | {"schedule": frozen["schedule_digest"]},
                corpus_digests=d.get("corpus_digests", {}),
                scope={
                    "cases": list(frozen["cases"]),
                    "note": frozen["scope_note"],
                    "runs_per_arm_per_case": RUNS,
                },
                exceptions=[f"{e.arm} {e.case_id} rep {e.rep}: {e.note}" for e in invalid],
                context={
                    "policy": "stochastic-v3",
                    "original_decision": f"{d['parent']}->{d['candidate']}",
                    "kind": d["context"]["kind"],
                },
            )
            env = receipt(dec, p.signer)
            decision_id = f"v3:{dec.parent}->{dec.candidate}:{frozen['schedule_digest'][:12]}"
            p.store.put_decision(
                {
                    **dec.to_payload(),
                    "envelope": base64.b64encode(env).decode("ascii"),
                    "keyid": p.signer.keyid,
                },
                decision_id=decision_id,
            )
            rel = self.ledger.bundle(
                decision_id,
                env,
                {},
                {},
                {"parent": wire.harness_doc(parent), "candidate": wire.harness_doc(arms[name])},
                cohort=cohort_dir,
                arms=["parent", name],
            )
            failed = [c.name for c in dec.checks if not c.passed]
            self.ledger.add(
                {
                    "decision_id": decision_id,
                    "policy": "stochastic-v3",
                    "receipt_sha256": sha256_hex(env),
                    "parent": dec.parent,
                    "candidate": dec.candidate,
                    "admitted": dec.admitted,
                    "failed": failed,
                    "bundle": rel,
                    "arm": name,
                    "schedule_digest": frozen["schedule_digest"],
                    "per_case": [_v3_row(r) for r in cmp["cases"]],
                    "stable_recoveries": cmp["stable_recoveries"],
                    "stable_regressions": cmp["stable_regressions"],
                    "declines_over_limit": cmp["case_declines_over_limit"],
                    "benign_interventions": benign,
                }
            )
            summary["decisions"].append(
                _v3_counts(cmp, name)
                | {
                    "decision_id": decision_id,
                    "candidate": version,
                    "admitted": dec.admitted,
                    "receipt_sha256": sha256_hex(env),
                    "failed": failed,
                    "benign_interventions": benign,
                }
            )
        p.store.put_eval({"type": "v3_summary", **summary})
        return summary

    def activate(
        self,
        *,
        candidate: str,
        expected: str | None,
        expected_revision: int,
        decision_id: str,
        receipt_digest: str,
        idempotency_key: str,
    ) -> Activation:
        return self.parts.activator.activate(
            candidate=candidate,
            expected=expected,
            expected_revision=expected_revision,
            decision_id=decision_id,
            receipt_digest=receipt_digest,
            idempotency_key=idempotency_key,
        )

    def active(self) -> tuple[str | None, int]:
        return self.parts.activator.active()

    def public_key(self) -> str:
        """The decision key's raw public key, hex: what a run publishes so its receipts can be checked."""
        from ratchet.verify.dsse import raw_public

        return raw_public(self.parts.signer.public_key).hex()

    def close(self) -> None:
        self.parts.close()

    def __enter__(self) -> RefereeCore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------------------------------------
# The process boundary
# ---------------------------------------------------------------------------------------------------------

Factory = Callable[[Mapping[str, Any]], RefereeParts]


def load_factory(name: str) -> Factory:
    module, _, attr = name.partition(":")
    if not module or not attr:
        raise RefereeError(f"factory {name!r} is not module:function")
    fn = getattr(importlib.import_module(module), attr)
    if not callable(fn):
        raise RefereeError(f"factory {name!r} is not callable")
    return fn  # type: ignore[no-any-return]


def request_doc(r: Request) -> dict[str, Any]:
    return {
        "candidate": wire.harness_doc(r.candidate),
        "train": wire.report_doc(r.train),
        "targets": [wire.cluster_doc(c) for c in r.targets],
        "guard_clusters": dict(r.guard_clusters),
        "literals": wire.literals_doc(r.literals),
        "visible_case_ids": list(r.visible_case_ids),
        "context": dict(r.context),
        "decision_id": r.decision_id,
    }


def request_of(a: Mapping[str, Any]) -> Request:
    return Request(
        wire.harness(a["candidate"]),
        wire.report(a["train"]),
        [wire.cluster(c) for c in a["targets"]],
        {str(k): str(v) for k, v in a["guard_clusters"].items()},
        wire.literals(a["literals"]),
        [str(x) for x in a["visible_case_ids"]],
        dict(a["context"]),
        None if a.get("decision_id") is None else str(a["decision_id"]),
    )


def _verdict(v: Mapping[str, Any]) -> Verdict:
    return Verdict(**{**v, "failed": tuple(v["failed"])})


def _rates(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dispatch(core: RefereeCore, request: Mapping[str, Any]) -> Any:
    method, a = request["method"], request.get("params", {})
    if method == "start":
        return core.start(a["train_identity"])
    if method == "bootstrap":
        return list(core.bootstrap(str(a["version"])))
    if method == "parent_runs":
        p1, p2 = core.parent_runs(int(a["generation"]), wire.harness(a["parent"]), wire.report(a["train"]))
        return [asdict(p1), asdict(p2)]
    if method == "evaluate":
        return asdict(core.evaluate(str(a["role"]), wire.harness(a["harness"]), wire.report(a["train"])))
    if method == "decide":
        return asdict(core.decide_many(int(a["generation"]), [request_of(a)])[0])
    if method == "decide_many":
        verdicts = core.decide_many(int(a["generation"]), [request_of(r) for r in a["requests"]])
        return [asdict(v) for v in verdicts]
    if method == "deterministic":
        return core.deterministic(
            wire.harness(a["parent"]),
            request_of(a["request"]),
            wire.report(a["parent_train"]),
            int(a["repeat_heldout"]),
        )
    if method == "stochastic":
        return core.stochastic(
            wire.harness(a["parent"]),
            [str(x) for x in a["candidates"]],
            [str(x) for x in a["case_ids"]],
            int(a["schedule_seed"]),
            deadline_s=float(a["deadline_s"]),
            scope_note=str(a["scope_note"]),
            workers=int(a["workers"]),
        )
    if method == "candidate":
        return core.candidate(str(a["spec"]), wire.harness(a["parent"]))
    if method == "results":
        return core.results(str(a["name"]))
    if method == "active":
        return list(core.active())
    if method == "public_key":
        return core.public_key()
    if method == "activate":
        return asdict(
            core.activate(
                **{
                    k: a[k]
                    for k in (
                        "candidate",
                        "expected",
                        "expected_revision",
                        "decision_id",
                        "receipt_digest",
                        "idempotency_key",
                    )
                }
            )
        )
    raise RefereeError(f"unknown method {method!r}")


def serve(factory: Factory, read: IO[str], write: IO[str]) -> int:
    """The referee's side: the first line is the config; then one JSON request per line, one answer each."""
    try:
        core = RefereeCore(factory(json.loads(read.readline())))
    except Exception as exc:
        return _refuse_start(write, exc)
    write.write(json.dumps({"ok": "ready"}) + "\n")
    write.flush()
    try:
        for line in read:
            request = json.loads(line)
            if request.get("method") == "close":
                write.write(json.dumps({"ok": None}) + "\n")
                write.flush()
                return 0
            try:
                answer = {"ok": _dispatch(core, request)}
            except Exception as exc:
                answer = {"error": f"{type(exc).__name__}: {exc}"}
            write.write(json.dumps(answer, default=str) + "\n")
            write.flush()
    finally:
        core.close()
    return 0


def _refuse_start(write: IO[str], exc: BaseException) -> int:
    write.write(json.dumps({"error": f"the referee could not start: {type(exc).__name__}: {exc}"}) + "\n")
    write.flush()
    return 1


class ProcessReferee:
    """The referee in its own process. Requests and answers are JSON lines on the child's stdin and stdout;
    anything the referee's code prints goes to its stderr."""

    def __init__(
        self,
        factory: str,
        config: Mapping[str, Any],
        *,
        env: Mapping[str, str] | None = None,
        python: str = sys.executable,
    ) -> None:
        self._proc = subprocess.Popen(
            [python, "-m", "ratchet.improve.referee", factory],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=dict(env) if env is not None else None,
        )
        try:
            self._send({**dict(config)})
            if self._read() != "ready":
                raise RefereeError("the referee did not start")
        except BaseException:
            self._end()
            raise

    def _send(self, doc: Mapping[str, Any]) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(doc) + "\n")
        self._proc.stdin.flush()

    def _read(self) -> Any:
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            try:
                status = self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                status = None
            raise RefereeError(f"the referee closed its output (exit status {status})")
        answer = json.loads(line)
        if "error" in answer:
            raise RefereeError(str(answer["error"]))
        return answer["ok"]

    def _call(self, method: str, **params: Any) -> Any:
        self._send({"method": method, "params": params})
        return self._read()

    def start(self, train_identity: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self._call("start", train_identity=dict(train_identity))
        return result

    def bootstrap(self, version: str) -> tuple[str | None, int]:
        active, revision = self._call("bootstrap", version=version)
        return active, int(revision)

    def parent_runs(self, generation: int, parent: Harness, train: EvalReport) -> tuple[Totals, Totals]:
        p1, p2 = self._call(
            "parent_runs",
            generation=generation,
            parent=wire.harness_doc(parent),
            train=wire.report_doc(train),
        )
        return Totals(**p1), Totals(**p2)

    def evaluate(self, role: str, harness: Harness, train: EvalReport) -> Totals:
        t = self._call("evaluate", role=role, harness=wire.harness_doc(harness), train=wire.report_doc(train))
        return Totals(**t)

    def decide(
        self,
        generation: int,
        candidate: Harness,
        train: EvalReport,
        targets: Sequence[FailureCluster],
        guard_clusters: Mapping[str, str],
        literals: Literals,
        visible_case_ids: Sequence[str],
        context: Mapping[str, Any],
        decision_id: str | None = None,
    ) -> Verdict:
        r = Request(
            candidate, train, targets, guard_clusters, literals, visible_case_ids, context, decision_id
        )
        return _verdict(self._call("decide", generation=generation, **request_doc(r)))

    def decide_many(self, generation: int, requests: Sequence[Request]) -> list[Verdict]:
        docs = self._call("decide_many", generation=generation, requests=[request_doc(r) for r in requests])
        return [_verdict(v) for v in docs]

    def deterministic(
        self, parent: Harness, request: Request, parent_train: EvalReport, repeat_heldout: int = 3
    ) -> dict[str, Any]:
        out: dict[str, Any] = self._call(
            "deterministic",
            parent=wire.harness_doc(parent),
            request=request_doc(request),
            parent_train=wire.report_doc(parent_train),
            repeat_heldout=repeat_heldout,
        )
        return out

    def stochastic(
        self,
        parent: Harness,
        candidates: Sequence[str],
        case_ids: Sequence[str],
        schedule_seed: int,
        *,
        deadline_s: float = float("inf"),
        scope_note: str = "",
        workers: int = 128,
    ) -> dict[str, Any]:
        out: dict[str, Any] = self._call(
            "stochastic",
            parent=wire.harness_doc(parent),
            candidates=list(candidates),
            case_ids=list(case_ids),
            schedule_seed=schedule_seed,
            deadline_s=deadline_s,
            scope_note=scope_note,
            workers=workers,
        )
        return out

    def candidate(self, spec: str, parent: Harness) -> dict[str, Any]:
        out: dict[str, Any] = self._call("candidate", spec=spec, parent=wire.harness_doc(parent))
        return out

    def results(self, name: str) -> dict[str, Any]:
        out: dict[str, Any] = self._call("results", name=name)
        return out

    def activate(
        self,
        *,
        candidate: str,
        expected: str | None,
        expected_revision: int,
        decision_id: str,
        receipt_digest: str,
        idempotency_key: str,
    ) -> Activation:
        a = self._call(
            "activate",
            candidate=candidate,
            expected=expected,
            expected_revision=expected_revision,
            decision_id=decision_id,
            receipt_digest=receipt_digest,
            idempotency_key=idempotency_key,
        )
        return Activation(**a)

    def active(self) -> tuple[str | None, int]:
        version, revision = self._call("active")
        return version, int(revision)

    def public_key(self) -> str:
        return str(self._call("public_key"))

    def close(self) -> None:
        if self._proc.poll() is None:
            with contextlib.suppress(RefereeError, BrokenPipeError):
                self._call("close")
        self._end()

    def _end(self) -> None:
        """Close the child's input, reap it (killed if it has not exited in a minute), close its output."""
        stdin = self._proc.stdin
        if stdin is not None and not stdin.closed:
            with contextlib.suppress(BrokenPipeError):
                stdin.close()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        if self._proc.stdout is not None:
            self._proc.stdout.close()

    def __enter__(self) -> ProcessReferee:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m ratchet.improve.referee module:function", file=sys.stderr)
        return 2
    # The protocol owns the real stdout; everything else the referee's code prints goes to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        factory = load_factory(args[0])
    except Exception as exc:
        return _refuse_start(protocol, exc)
    return serve(factory, sys.stdin, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
