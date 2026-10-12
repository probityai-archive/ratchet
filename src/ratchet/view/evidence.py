"""Gather, from the store, the evidence an admission receipt names, so the verifier can recompute from it.

Three kinds, each loaded only when the store holds exactly what the receipt describes:

- stored reports, found by the digest the receipt names for them (a stored document carries bookkeeping
  fields beside the report, so each document is reduced to the report's own fields and hashed);
- the parent and candidate harness documents (a candidate that was never stored is rebuilt from its decision
  and used only when the rebuilt digest equals the one the receipt signed);
- results recomputed here from recorded inputs: every added guard replayed over the recorded traces of its
  generation, and the overfitting screen over that generation's training traces.

Anything that cannot be loaded is simply absent. The verifier prints absent evidence as not supplied.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ratchet_cases.types import Trace, canonical_json, sha256_hex

from ratchet.improve.diagnose import FailurePoint
from ratchet.improve.evaluate import EvalReport
from ratchet.improve.overfit import lint, training_literals
from ratchet.improve.replay import GuardRunner, WalkCache, replay
from ratchet.improve.sim import Simulator
from ratchet.types import Guard, Harness, Procedure
from ratchet.view.data import Doc, ViewStore
from ratchet.view.matrix import CaseInfo

BOOKKEEPING = {"_id", "digest", "created_at"}
REPLAY_EXTRA = {"type", "role", "generation", "parent"}


@dataclass
class Evidence:
    reports: dict[str, Doc] = field(default_factory=dict)
    harnesses: dict[str, Doc] = field(default_factory=dict)
    recomputed: dict[str, Doc] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _digest(doc: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(dict(doc)))


def _eval_fields() -> set[str]:
    return set(EvalReport("x", [], 0.0).to_payload())


def stored_reports(store: ViewStore) -> dict[str, Doc]:
    """Every stored evaluation and replay report, by the digest of the report itself."""
    fields = _eval_fields()
    found: dict[str, Doc] = {}
    for doc in store.evals():
        kind = doc.get("type")
        if kind == "eval":
            report = {k: v for k, v in doc.items() if k in fields}
        elif kind == "replay":
            report = {k: v for k, v in doc.items() if k not in BOOKKEEPING | REPLAY_EXTRA}
        else:
            continue
        found[_digest(report)] = report
    return found


def harness_doc(h: Harness) -> Doc:
    return asdict(h)


def rebuild_candidate(parent: Harness, context: Mapping[str, Any], digest: str) -> Harness | None:
    """The candidate a decision describes, rebuilt from its parent and the recorded change; None unless the
    rebuilt content address equals the digest the receipt signed."""
    change = context.get("change") or {}
    try:
        if context.get("kind") == "guard":
            guard = Guard(**{k: change[k] for k in ("id", "description", "source", "origin_failure_ids")})
            built = parent.child(guards=[*parent.guards, guard], created_by="improver")
        elif context.get("kind") == "procedure":
            proc = Procedure(**{k: change[k] for k in ("id", "name", "when", "steps", "origin_failure_ids")})
            built = parent.child(procedures=[*parent.procedures, proc], created_by="improver")
        elif context.get("kind") == "prompt_edit":
            text = change["append"]
            built = parent.child(
                system_prompt=f"{parent.system_prompt.rstrip()}\n\n{text}", created_by="improver"
            )
        else:
            return None
    except (KeyError, TypeError):
        return None
    return built if built.digest == digest else None


def _parent_run(store: ViewStore, doc: Mapping[str, Any]) -> Doc | None:
    """The stored first fresh run of the parent in the decision's generation."""
    parent, gen = doc.get("parent"), (doc.get("context") or {}).get("generation")
    for ev in store.evals():
        is_run = ev.get("type") == "eval" and ev.get("role") == "parent_run_1"
        if is_run and ev.get("harness_version") == parent and (gen is None or ev.get("generation") == gen):
            return ev
    return None


def _generation_traces(
    store: ViewStore, doc: Mapping[str, Any]
) -> tuple[list[Trace], list[Trace], dict[str, str]]:
    """(training traces, all traces, kind of each case) of that run, as it recorded them."""
    ev = _parent_run(store, doc)
    if ev is None:
        return [], [], {}
    results = ev.get("results") or []
    traces = []
    for r in results:
        found = store.get_trace(r["run_id"], r["case_id"]) if r.get("run_id") else None
        if found is not None:
            traces.append(found)
    train = {r["case_id"] for r in results if r.get("split") == "train"}
    return (
        [t for t in traces if t.case_id in train],
        traces,
        {r["case_id"]: r.get("kind", "") for r in results},
    )


def _referee_doc(store: ViewStore, gid: str, doc: Mapping[str, Any]) -> Doc | None:
    gen = (doc.get("context") or {}).get("generation")
    for ev in store.evals():
        if ev.get("type") == "replay" and ev.get("guard_id") == gid and ev.get("generation", gen) == gen:
            return ev
    return None


def _targets(referee: Mapping[str, Any]) -> list[FailurePoint]:
    out = []
    for t in referee.get("targets") or []:
        run_id, _, _ = str(t["failure_id"]).rpartition("@")
        out.append(FailurePoint(run_id, str(t["case_id"]), int(t["failing_seq"]), "", "commission", ""))
    return out


def _recompute_replays(
    store: ViewStore,
    doc: Mapping[str, Any],
    new_guards: list[Guard],
    runner: GuardRunner,
    simulator: Simulator,
    out: Evidence,
) -> None:
    _, traces, kinds = _generation_traces(store, doc)
    if not traces:
        out.notes.append("the parent's recorded run is not in the store; replay not recomputed")
        return
    passing = [t for t in traces if t.outcome.passed]
    benign = [t for t in traces if kinds.get(t.case_id) == "benign_twin"]
    by_run = {t.run_id: t for t in traces}
    walks = WalkCache(simulator)
    for guard in new_guards:
        referee = _referee_doc(store, guard.id, doc)
        if referee is None:
            out.notes.append(f"no recorded replay of {guard.id}; its targets are unknown")
            continue
        targets = _targets(referee)
        failing = {t.run_id: by_run[t.run_id] for t in targets if t.run_id in by_run}
        report = replay(guard, targets, failing, passing, walks, runner, benign)
        out.recomputed[f"replay:{guard.id}"] = report.to_payload()


def collect(
    store: ViewStore,
    doc: Mapping[str, Any],
    runner: Callable[[], Any] | None = None,
    simulator: Simulator | None = None,
) -> Evidence:
    """Everything the store can supply for one decision document. `runner` is a context-manager factory for
    the guard runtime; without it (and a simulator) replay is not recomputed."""
    out = Evidence()
    by_digest = stored_reports(store)
    for name, digest in (doc.get("report_digests") or {}).items():
        if digest in by_digest:
            out.reports[name] = by_digest[digest]
    parent = store.get_harness(str(doc.get("parent")))
    if parent is None:
        out.notes.append("the parent harness is not in the store")
        return out
    out.harnesses["parent"] = harness_doc(parent)
    candidate = store.get_harness(str(doc.get("candidate"))) or rebuild_candidate(
        parent, doc.get("context") or {}, str(doc.get("candidate_digest"))
    )
    if candidate is None:
        out.notes.append("the candidate harness is neither stored nor rebuildable from its decision")
        return out
    out.harnesses["candidate"] = harness_doc(candidate)
    train, _, _ = _generation_traces(store, doc)
    if train:
        out.recomputed["overfit_lint"] = {"hits": lint(parent, candidate, training_literals(train))}
    old = {g.id for g in parent.guards}
    added = [g for g in candidate.guards if g.id not in old]
    if added and runner is not None and simulator is not None:
        with runner() as guard_runner:
            _recompute_replays(store, doc, added, guard_runner, simulator, out)
    return out


__all__ = ["CaseInfo", "Evidence", "collect", "rebuild_candidate", "stored_reports"]
