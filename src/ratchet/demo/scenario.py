"""`ratchet demo admit`: a labelled demo scenario that drives the real admission pipeline end to end.

The agent is the scripted demo worker (`ratchet.demo.worker`), which calls no model, so the parent's two
runs agree and the candidate's two runs agree by construction. Everything else is the production path: the
real case suite (the duplicate-refund traps of two families with their benign twins), the real worker and
graders, the sandboxed guard runtime, counterfactual replay, the admission rule, a signed receipt, and
activation by compare-and-swap in Atlas. Every stored record names the scripted worker, and the evaluator
identity carries it in its model config, so no output can be read as a live-model result.

Also stored: one replay-rejected candidate of the same parent (a guard that blocks a recorded success), so
the replay view shows a rejected column next to the admitted one.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ratchet.agent.guards import GuardRuntime
from ratchet.agent.seed import seed_harness
from ratchet.agent.worker import run_case
from ratchet.cases import all_cases
from ratchet.cases.model import Case
from ratchet.demo.worker import LABEL, NAME, ScriptedDemoWorker
from ratchet.improve.activate import Activator
from ratchet.improve.admit import Decision, admit, awaiting_confirmation
from ratchet.improve.diagnose import diagnose
from ratchet.improve.evaluate import CaseLike, EvalReport, evaluate_many
from ratchet.improve.evaluator import evaluator_identity
from ratchet.improve.loop import ImproveStore, frozen_suite_digest
from ratchet.improve.overfit import training_literals
from ratchet.improve.receipt import Signer, receipt
from ratchet.improve.replay import GuardRunner, ReplayReport, WalkCache, replay
from ratchet.improve.sim import Simulator
from ratchet.types import Guard, Harness, Trace, sha256_hex

CASE_IDS = (
    "refund_lost_reply.v1", "refund_lost_reply.v2", "refund_lost_reply.v3", "refund_lost_reply.v4",
    "concurrent_worker.v1", "concurrent_worker.v2", "concurrent_worker.v3", "concurrent_worker.v4",
)  # fmt: skip

SCENARIO = {
    "kind": "demo scenario",
    "agent": NAME,
    "label": LABEL,
    "note": "Scripted agent with fixed responses; not a live-model result.",
}
MODEL_CONFIG = {"model": NAME, "label": LABEL, "temperature": 0.0, "base_url": None}

ADMIT_GUARD = Guard(
    "g-demo-no-double-refund",
    "DEMO SCENARIO, scripted agent: block a refund the provider already records on that charge.",
    """def check(call, view, history):
    if call.name != "payments_refund":
        return None
    charge = call.args.get("charge_id")
    refunds = (view.get("payments") or {}).get("refunds") or {}
    for r in refunds.values():
        if r.get("charge_id") == charge and r.get("amount_cents") == call.args.get("amount_cents"):
            return ("the provider already records a refund of this amount on this charge; "
                    "read payments_list_refunds and do not pay twice")
    return None
""",
)

REJECT_GUARD = Guard(
    "g-demo-no-large-refund",
    "DEMO SCENARIO, scripted agent: block every refund of 50.00 or more.",
    """def check(call, view, history):
    if call.name == "payments_refund" and (call.args.get("amount_cents") or 0) >= 5000:
        return "refunds of 50.00 or more are not allowed"
    return None
""",
)


@dataclass
class DemoResult:
    parent: str
    admitted: Decision
    admitted_id: str
    rejected: Decision
    rejected_id: str
    failed_trace: str
    activation: dict[str, Any]


def demo_cases() -> list[Case]:
    cases = [c for c in all_cases() if c.id in CASE_IDS]
    if sorted(c.id for c in cases) != sorted(CASE_IDS):
        raise RuntimeError("the demo cases are missing from the case suite")
    return cases


def _persist(store: ImproveStore, report: EvalReport, role: str, **extra: Any) -> None:
    for t in report.traces:
        store.put_trace(t)
    store.put_eval({**report.to_payload(), "generation": 1, "role": role, "scenario": SCENARIO, **extra})


def _persist_replay(store: ImproveStore, parent: str, report: ReplayReport) -> None:
    """Stored in the loop's shape, so the verifier can match the report by its digest."""
    store.put_eval({"type": "replay", "role": "referee", "generation": 1, "parent": parent,
                    **report.to_payload()})  # fmt: skip


def _context(guard: Guard, candidate: Harness, cluster_id: str, targets: Sequence[str]) -> dict[str, Any]:
    return {
        "generation": 1,
        "arm": "demo",
        "id": f"c-{guard.id}",
        "kind": "guard",
        "cluster_id": cluster_id,
        "instructions_version": "demo-scenario",
        "harness_version": candidate.version,
        "change": {"id": guard.id, "description": guard.description, "source": guard.source,
                   "origin_failure_ids": list(targets)},
        "rationale": "Demo scenario: the scripted agent retries a refund the provider already holds.",
        "targets": list(targets),
        "scenario": SCENARIO,
    }  # fmt: skip


def _sign_and_store(store: ImproveStore, d: Decision, signer: Signer) -> tuple[str, str]:
    env = receipt(d, signer)
    decision_id = f"{d.parent}->{d.candidate}"
    store.put_decision(
        {**d.to_payload(), "envelope": base64.b64encode(env).decode("ascii"), "keyid": signer.keyid},
        decision_id=decision_id,
    )
    return decision_id, sha256_hex(env)


def run_demo(
    store: ImproveStore,
    activator: Activator,
    simulator: Simulator,
    runner: GuardRunner,
    signer: Signer,
    run: Callable[..., Trace] = run_case,
    cases: Sequence[CaseLike] | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> DemoResult:
    """Run the scenario and store every record. Raises when the active harness is not the seed parent."""
    cases = list(cases if cases is not None else demo_cases())
    model = ScriptedDemoWorker()
    parent = seed_harness()
    heldout = [c for c in cases if c.split == "heldout"]
    benign_ids = {c.id for c in cases if c.kind == "benign_twin"}
    evaluator = evaluator_identity(cases, MODEL_CONFIG, frozen_suite_digest())
    eval_id = f"evaluator-{evaluator['version']}"
    store.put_eval({"type": "evaluator", **evaluator, "scenario": SCENARIO}, eval_id=eval_id)
    store.put_harness(parent)
    active, revision = activator.bootstrap(parent.version)
    if active != parent.version:
        raise RuntimeError(f"the active harness is {active}, not the seed {parent.version}; use a fresh db")

    def ev(h: Harness, cs: Sequence[CaseLike]) -> EvalReport:
        return evaluate_many([h], cs, run, model, workers=4)[0]

    p1 = ev(parent, cases)
    _persist(store, p1, "parent_run_1")
    p2 = ev(parent, heldout)
    _persist(store, p2, "parent_run_2")
    log(f"parent {parent.version}: P1 and P2 recorded")
    train = p1.train_traces()
    walks = WalkCache(simulator)
    clusters = diagnose(train, simulator)
    if not clusters:
        raise RuntimeError("the scripted agent made no training failure; the scenario is broken")
    members = [m for c in clusters for m in c.members]
    failing = {k: v for c in clusters for k, v in c.traces.items()}
    passing = [t for t in p1.traces if t.outcome.passed]
    benign = [t for t in p1.traces if t.case_id in benign_ids]
    literals = training_literals(train)
    visible = sorted(c.id for c in cases if c.split == "train")
    targets = [m.id for m in members]

    def decide(guard: Guard) -> tuple[Harness, Decision, str]:
        cand = parent.child(guards=[*parent.guards, guard], created_by="demo-scenario")
        store.put_harness(cand)
        rep = replay(guard, members, failing, passing, walks, runner, benign)
        _persist_replay(store, parent.version, rep)
        composed = replay(cand.guards, members, failing, passing, walks, runner, benign)
        _persist_replay(store, parent.version, composed)
        c1 = ev(cand, cases)
        _persist(store, c1, "candidate", candidate=cand.version)
        kw: dict[str, Any] = {
            "parent_runs": (p1, p2), "cand_eval": c1, "replays": {guard.id: rep}, "composed": composed,
            "literals": literals, "visible_case_ids": visible, "evaluator": evaluator,
            "context": _context(guard, cand, clusters[0].id, targets),
        }  # fmt: skip
        d = admit(parent, cand, confirm_eval=None, **kw)
        if awaiting_confirmation(d):
            c2 = ev(cand, heldout)
            _persist(store, c2, "confirm", harness=cand.version)
            d = admit(parent, cand, confirm_eval=c2, **kw)
        _, digest = _sign_and_store(store, d, signer)
        log(f"candidate {cand.version}: admitted={d.admitted} "
            f"failed={[c.name for c in d.checks if not c.passed]}")  # fmt: skip
        return cand, d, digest

    rejected_h, rejected, _ = decide(REJECT_GUARD)
    admitted_h, decision, digest = decide(ADMIT_GUARD)
    admitted_id = f"{decision.parent}->{decision.candidate}"
    activation: dict[str, Any] = {}
    if decision.admitted:
        act = activator.activate(
            candidate=admitted_h.version, expected=parent.version, expected_revision=revision,
            decision_id=admitted_id, receipt_digest=digest, idempotency_key=f"activate:{admitted_id}",
        )  # fmt: skip
        activation = {"revision": act.revision, "previous": act.previous, "receipt_digest": digest}
    bad = next(t for t in p1.traces if not t.outcome.passed and t.case_id.endswith(".v1"))
    return DemoResult(
        parent=parent.version,
        admitted=decision,
        admitted_id=admitted_id,
        rejected=rejected,
        rejected_id=f"{rejected.parent}->{rejected_h.version}",
        failed_trace=f"{bad.run_id}/{bad.case_id}",
        activation=activation,
    )


def main(db: str, uri: str, reset: bool = False) -> int:
    """The CLI entry: Atlas store, sandboxed guard runtime, signing key from RATCHET_SIGNING_KEY."""
    from pymongo import MongoClient

    from ratchet.improve.activate import MongoActivator
    from ratchet.improve.sandbox import SandboxRunner
    from ratchet.improve.worldsim import WorldSimulator
    from ratchet.store import Store

    if not db.startswith("ratchet_demo"):
        print("error: the demo writes only to a database whose name starts with ratchet_demo")
        return 2
    signer = Signer.from_env()
    cases = demo_cases()
    client: MongoClient[dict[str, Any]] = MongoClient(uri)
    with client:
        if reset:  # the hackathon user may not drop databases, so empty every collection instead
            for name in client[db].list_collection_names():
                client[db][name].delete_many({})
        with Store(uri, db) as store, GuardRuntime() as runtime:
            store.setup(wait_search=False)
            r = run_demo(store, MongoActivator(client, db), WorldSimulator(cases), SandboxRunner(runtime),
                         signer, cases=cases, log=lambda m: print(m, flush=True))  # fmt: skip
    print(f"DEMO SCENARIO ({LABEL}) db={db}")
    print(f"parent              {r.parent}")
    print(f"admitted decision   {r.admitted_id} admitted={r.admitted.admitted}")
    print(f"rejected decision   {r.rejected_id} admitted={r.rejected.admitted}")
    print(f"failed parent trace {r.failed_trace}")
    print(f"activation          {r.activation or 'not activated'}")
    return 0 if r.admitted.admitted and r.activation else 1
