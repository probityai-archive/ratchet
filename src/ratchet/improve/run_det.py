"""Ruling-2 admission of one stored candidate on deterministic endpoints (temperature 0, fixed seed), with
gpt-oss-120b as the worker. The endpoint is either batch size one with no prefix cache, or vLLM's
batch-invariant mode (`--serve-config` records how it was served, and that record is part of the endpoint
configuration digest every receipt names).

1. Repeatability: held-out cases (three by default, `--repeat-heldout 0` for all) run twice each; every pair
   must produce the same normalized trace.
2. P1 (every decision case), P2 (held-out), C1 (every decision case), C2 (held-out, only when C1 clears).
   If P1 and P2 disagree on any held-out case, the endpoint is not repeatable and the track stops.
3. Replay of the candidate's guard over this cohort's recorded traces, then `admit()` and a signed receipt
   that names ruling 2, the deterministic evaluator and the endpoint configuration digest.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ratchet_cases.cases import all_cases
from ratchet_cases.types import Trace, sha256_hex

from ratchet.agent.guards import GuardRuntime
from ratchet.agent.model import ModelClient, ModelConfig, SeededTransport
from ratchet.agent.seed import seed_harness
from ratchet.agent.worker import run_case
from ratchet.canonical import canonical_json
from ratchet.improve.admit import admit
from ratchet.improve.diagnose import FailurePoint
from ratchet.improve.evaluate import EvalReport, evaluate_many
from ratchet.improve.evaluator import evaluator_identity
from ratchet.improve.loop import frozen_suite_digest
from ratchet.improve.overfit import training_literals
from ratchet.improve.receipt import Signer, receipt
from ratchet.improve.replay import WalkCache, replay
from ratchet.improve.run_v3 import rebuild
from ratchet.improve.sandbox import SandboxRunner
from ratchet.improve.worldsim import WorldSimulator
from ratchet.store.atlas import Store

RULING_2 = {"name": "ruling-2", "evaluator": "deterministic"}


def normalized(trace: Trace) -> str:
    """What must repeat: every event's kind and content, without timestamps, run ids or tool-call ids."""
    rows = []
    for e in trace.events:
        p = dict(e.payload)
        p.pop("latency_s", None)
        call = p.get("call")
        if isinstance(call, dict):
            p["call"] = {k: v for k, v in call.items() if k != "call_id"}
        res = p.get("result")
        if isinstance(res, dict):
            p["result"] = {k: v for k, v in res.items() if k != "call_id"}
        p.pop("call_id", None)
        if "tool_calls" in p:
            p["tool_calls"] = [{k: v for k, v in c.items() if k != "call_id"} for c in p["tool_calls"] or []]
        rows.append([e.kind, p])
    return sha256_hex(canonical_json([rows, trace.outcome.passed, trace.outcome.reasons]))


def _stop(store: Store, out: dict[str, Any], path: Path) -> int:
    """Record that the endpoint is not repeatable, in the database and the output file, and stop the track."""
    store.put_eval({"type": "endpoint_verdict", "policy": RULING_2, **out})
    path.write_text(json.dumps(out, indent=2, default=str))
    print("RESULT", out["result"], flush=True)
    return 0


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ratchet-improve-run-det")
    p.add_argument("--endpoints", required=True, help="comma-separated deterministic endpoints")
    p.add_argument("--model", default="gpt-oss-120b-det")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--candidate", required=True, help="db:candidate_version of a stored decision")
    p.add_argument("--db", default="ratchet_det")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--serve-config",
        type=Path,
        help="JSON record of how the endpoint was served; replaces the batch-size-one fields",
    )
    p.add_argument("--repeat-heldout", type=int, default=3, help="held-out cases to repeat; 0 = all")
    p.add_argument("--workers", type=int, help="concurrent cases (default: one per endpoint)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = _args(argv)
    uri = os.environ.get("RATCHET_MONGODB_URI", "")
    if not uri:
        print("error: set RATCHET_MONGODB_URI", file=sys.stderr)
        return 2
    every = all_cases()
    decision_cases = [c for c in every if c.split in ("train", "heldout")]
    heldout = [c for c in every if c.split == "heldout"]
    endpoints = [e.strip() for e in a.endpoints.split(",") if e.strip()]
    endpoint_config: dict[str, Any] = {
        "endpoints": endpoints,
        "model": a.model,
        "temperature": 0.0,
        "seed": a.seed,
    }
    if a.serve_config:
        endpoint_config["serving"] = json.loads(a.serve_config.read_text(encoding="utf-8"))
    else:
        endpoint_config |= {"batch_size": 1, "prefix_cache": False}
    config_digest = sha256_hex(canonical_json(endpoint_config))
    out: dict[str, Any] = {"endpoint_config": endpoint_config, "endpoint_config_digest": config_digest}
    db, version = a.candidate.split(":")
    with Store(uri, db) as s:
        found = s.db.decisions.find_one({"candidate": version}, {"envelope": 0})
        if found is None:
            print(f"error: no decision for {a.candidate}", file=sys.stderr)
            return 2
        original: dict[str, Any] = found
        gen = s.db.evals.find_one({"type": "generation", "generation": original["context"]["generation"]})
        if gen is None:
            print("error: the candidate's generation record is missing", file=sys.stderr)
            return 2
        cluster = next(c for c in gen["clusters"] if c["id"] == original["context"]["cluster_id"])
        members = [
            FailurePoint(m["run_id"], m["case_id"], m["seq"], m["reason"], m["mode"], m["tool"])
            for m in cluster["members"]
        ]
        targets = {m.run_id: s.get_trace(m.run_id, m.case_id) for m in members}
    parent = seed_harness()
    candidate = rebuild(parent, original)
    clients = [
        ModelClient(
            ModelConfig(base_url=e, model=a.model, api_key="unused", temperature=0.0),
            transport=SeededTransport(a.seed),
        )
        for e in endpoints
    ]
    workers = a.workers or len(clients)

    class Spread:
        """Round-robin over the deterministic endpoints."""

        def __init__(self) -> None:
            self.i = 0

        def __call__(self, harness: Any, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
            self.i += 1
            return run_case(harness, case, clients[self.i % len(clients)], store, run_id=run_id)

    runner = Spread()
    store = Store(uri, a.db)

    def ev(h: Any, cases: Sequence[Any], role: str) -> EvalReport:
        rep = evaluate_many([h], cases, runner, None, workers=workers)[0]
        for t in rep.traces:
            store.put_trace(t)
        store.put_eval({**rep.to_payload(), "role": role, "endpoint_config_digest": config_digest})
        n = len(rep.results)
        out.setdefault("throughput", {})[role] = {
            "runs": n,
            "wall_s": round(rep.wall_s, 1),
            "runs_per_min": round(60 * n / rep.wall_s, 2) if rep.wall_s else None,
            "workers": workers,
        }
        print(
            f"{role}: heldout {len(rep.passed('heldout'))}/{rep.count('heldout')} train "
            f"{len(rep.passed('train'))}/{rep.count('train')} wall {rep.wall_s:.0f}s",
            flush=True,
        )
        return rep

    # 1. Repeatability on held-out cases.
    probe = heldout[: a.repeat_heldout] if a.repeat_heldout > 0 else heldout
    first, second = ev(parent, probe, "repeat_a"), ev(parent, probe, "repeat_b")
    digests = {t.case_id: normalized(t) for t in first.traces}
    again = {t.case_id: normalized(t) for t in second.traces}
    repeat = {c.id: digests.get(c.id) == again.get(c.id) for c in probe}
    out["repeatability"] = {
        "cases": repeat,
        "repeatable": all(repeat.values()),
        "digests_a": digests,
        "digests_b": again,
    }
    print("REPEATABILITY", json.dumps(out["repeatability"]), flush=True)
    if not all(repeat.values()):
        out["result"] = "NOT REPEATABLE: two identical runs produced different traces; track stopped"
        return _stop(store, out, a.out)

    # 2. The parent twice; stop when the held-out outcomes disagree.
    p1 = ev(parent, decision_cases, "parent_run_1")
    p2 = ev(parent, heldout, "parent_run_2")
    p1h, p2h = ({r.case_id: r.passed for r in e.results if r.split == "heldout"} for e in (p1, p2))
    differ = sorted(k for k in p1h if p1h[k] != p2h.get(k))
    out["parent_discordance"] = differ
    if differ:
        out["result"] = f"NOT REPEATABLE: P1 and P2 differ on {len(differ)} held-out case(s); track stopped"
        return _stop(store, out, a.out)
    c1 = ev(candidate, decision_cases, "candidate_run")

    # 3. Replay over this cohort's records, then the decision.
    sim = WorldSimulator(every)
    benign_ids = {c.id for c in every if c.kind == "benign_twin"}
    passing = [t for t in p1.traces if t.outcome.passed]
    benign = [t for t in p1.traces if t.case_id in benign_ids]
    with GuardRuntime(workers=4) as rt:
        walks = WalkCache(sim)
        sandbox = SandboxRunner(rt)
        live_targets = {k: v for k, v in targets.items() if v is not None}
        new = candidate.guards[len(parent.guards) :]
        replays = {g.id: replay(g, members, live_targets, passing, walks, sandbox, benign) for g in new}
        composed = (
            replay(candidate.guards, members, live_targets, passing, walks, sandbox, benign) if new else None
        )
    literals = training_literals(p1.train_traces())
    model_config = {**endpoint_config, "endpoint_config_digest": config_digest}
    evaluator = evaluator_identity(every, model_config, frozen_suite_digest())
    visible = sorted(c.id for c in every if c.split == "train")
    kw: dict[str, Any] = {
        "parent_runs": (p1, p2),
        "cand_eval": c1,
        "replays": replays,
        "composed": composed,
        "literals": literals,
        "visible_case_ids": visible,
        "evaluator": evaluator,
        "context": {
            "policy": RULING_2,
            "endpoint_config_digest": config_digest,
            "original_decision": f"{db}:{original['parent']}->{original['candidate']}",
        },
    }
    d = admit(parent, candidate, confirm_eval=None, **kw)
    from ratchet.improve.admit import awaiting_confirmation

    c2 = None
    if awaiting_confirmation(d):
        c2 = ev(candidate, heldout, "candidate_confirm_run")
        d = admit(parent, candidate, confirm_eval=c2, **kw)
    from dataclasses import replace as dc_replace

    d = dc_replace(d, policy={**RULING_2, "endpoint_config_digest": config_digest})
    signer = Signer.for_decisions()
    env = receipt(d, signer)
    decision_id = f"det:{d.parent}->{d.candidate}:{config_digest[:12]}"
    store.put_decision(
        {**d.to_payload(), "envelope": base64.b64encode(env).decode("ascii"), "keyid": signer.keyid},
        decision_id=decision_id,
    )
    h = d.check("heldout").evidence
    out |= {
        "decision_id": decision_id,
        "admitted": d.admitted,
        "receipt_sha256": sha256_hex(env),
        "failed": [c.name for c in d.checks if not c.passed],
        "outcomes": h["outcomes"],
        "stable_recoveries": h["stable_recoveries"],
        "protected_broken": h["protected_broken"],
        "signer_keyid": signer.keyid,
    }
    a.out.write_text(json.dumps(out, indent=2, default=str))
    print("RESULT", json.dumps({k: out[k] for k in ("decision_id", "admitted", "failed")}), flush=True)
    for c in clients:
        c.close()
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
