"""Ruling-2 admission of one stored candidate on deterministic endpoints (temperature 0, fixed seed), through
the held-out referee.

    uv run python -m ratchet.improve.run_det --endpoints-file det.json --candidate db:version --out x.json

The endpoint configuration is a JSON file (`ratchet.improve.endpoints`): the endpoints, the model, the
temperature, the fixed seed, and either how the endpoint was served (`serving`) or the batch-size-one
fields. Its digest, `endpoint_config_digest`, is part of the evaluator and of every receipt.

This process runs the training cases only. The referee (`ProcessReferee`, factory
`ratchet.improve.referee_live:policy_parts`) opens the same endpoint file, refuses it when its digest is not
the one named here, and runs the held-out side in its own process:

1. Repeatability: held-out cases (three by default, `--repeat-heldout 0` for all) run twice each; every pair
   must produce the same normalized trace (`ratchet.improve.policies.normalized`).
2. P1 and P2 on the held-out cases; if they disagree on any, the endpoint is not repeatable and the track
   stops.
3. C1, replay of the candidate's guards over the parent's recorded traces, `admit()`, C2 when C1 clears
   every gate, and a receipt signed with the decision key that names ruling 2, the deterministic evaluator
   and the endpoint configuration digest.

The candidate is rebuilt by the referee from its stored decision and checked by digest; this process
receives the harness and the training cluster its change targets, never the decision's checks. It
receives counts and the signed verdict. Every per-case table goes to the referee's signed results file
under `--results-dir`, which `python -m ratchet.verify.results` checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ratchet.improve import endpoints as ep
from ratchet.improve.diagnose import FailureCluster, FailurePoint
from ratchet.improve.evaluate import CaseLike, EvalReport, RunCase, evaluate_many
from ratchet.improve.evaluator import evaluator_identity
from ratchet.improve.loop import frozen_suite_digest
from ratchet.improve.overfit import training_literals
from ratchet.improve.referee import ProcessReferee, Request
from ratchet.types import Harness

FACTORY = "ratchet.improve.referee_live:policy_parts"


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ratchet-improve-run-det")
    p.add_argument("--endpoints-file", type=Path, required=True, help="the endpoint configuration (JSON)")
    p.add_argument("--candidate", required=True, help="db:candidate_version of a stored decision")
    p.add_argument("--db", default="ratchet_det", help="training records; the referee writes <db>_heldout")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, help="the referee's results (default: next to --out)")
    p.add_argument("--repeat-heldout", type=int, default=3, help="held-out cases to repeat; 0 = all")
    p.add_argument("--workers", type=int, help="concurrent cases (default: one per endpoint)")
    return p.parse_args(argv)


def target_cluster(store: Any, context: dict[str, Any], train_ids: set[str]) -> FailureCluster:
    """The training failure cluster the candidate's change was written for, with its recorded traces. A
    member outside the training split is refused: this process reads training material only."""
    gen = store.db.evals.find_one({"type": "generation", "generation": context["generation"]})
    if gen is None:
        raise SystemExit("error: the candidate's generation record is missing")
    doc = next(c for c in gen["clusters"] if c["id"] == context["cluster_id"])
    members = [
        FailurePoint(m["run_id"], m["case_id"], m["seq"], m["reason"], m["mode"], m["tool"])
        for m in doc["members"]
    ]
    outside = sorted({m.case_id for m in members} - train_ids)
    if outside:
        raise SystemExit(f"error: the target cluster names non-training cases {outside}")
    traces = {m.run_id: store.get_trace(m.run_id, m.case_id) for m in members}
    return FailureCluster(
        doc["id"],
        {str(k): str(v) for k, v in (doc.get("key") or {}).items()},
        members,
        str(doc.get("summary", "")),
        {k: v for k, v in traces.items() if v is not None},
    )


def _train(
    h: Harness,
    cases: Sequence[CaseLike],
    run: RunCase,
    store: Any,
    role: str,
    workers: int,
    out: dict[str, Any],
) -> EvalReport:
    rep = evaluate_many([h], cases, run, None, workers=workers)[0]
    for t in rep.traces:
        store.put_trace(t)
    store.put_eval({**rep.to_payload(), "role": role})
    out["throughput"].append({"role": role, "runs": len(rep.results), "wall_s": round(rep.wall_s, 1)})
    print(f"{role}: train {len(rep.passed('train'))}/{rep.count('train')} wall {rep.wall_s:.0f}s", flush=True)
    return rep


def main(argv: Sequence[str] | None = None) -> int:
    from ratchet_cases.cases import all_cases

    from ratchet.agent.seed import seed_harness
    from ratchet.agent.worker import run_case
    from ratchet.improve import wire
    from ratchet.store.atlas import Store

    a = _args(argv)
    uri = os.environ.get("RATCHET_MONGODB_URI", "")
    if not uri:
        print("error: set RATCHET_MONGODB_URI", file=sys.stderr)
        return 2
    endpoints = ep.load(a.endpoints_file)
    if endpoints.doc.get("seed") is None:
        print("error: a deterministic endpoint configuration names a seed", file=sys.stderr)
        return 2
    train_cases = [c for c in all_cases() if c.split == "train"]
    workers = a.workers or len(endpoints.urls)
    results_dir = a.results_dir or a.out.parent / f"{a.out.stem}-results"
    out: dict[str, Any] = {
        "endpoint_config": endpoints.doc,
        "endpoint_config_digest": endpoints.digest,
        "throughput": [],
    }
    referee_config = {
        "cases": "heldout",
        "store": {"kind": "atlas", "db": f"{a.db}_heldout"},
        "endpoints": endpoints.reference(a.endpoints_file),
        "worker": "round_robin",
        "workers": workers,
        "suite_digest": frozen_suite_digest(),
        "results_dir": str(results_dir),
    }
    parent = seed_harness()
    spread = ep.RoundRobin(endpoints, run_case)
    try:
        with ProcessReferee(FACTORY, referee_config) as referee, Store(uri, a.db) as store:
            found = referee.candidate(a.candidate, parent)
            candidate = wire.harness(found["harness"])
            with Store(uri, a.candidate.split(":")[0]) as source:
                cluster = target_cluster(source, found["context"], {c.id for c in train_cases})
            referee.start(evaluator_identity(train_cases, endpoints.model_config(), frozen_suite_digest()))
            p_train = _train(parent, train_cases, spread, store, "parent_train", workers, out)
            c_train = _train(candidate, train_cases, spread, store, "candidate_train", workers, out)
            old = {g.id for g in parent.guards}
            request = Request(
                candidate,
                c_train,
                [cluster],
                {g.id: cluster.id for g in candidate.guards if g.id not in old},
                training_literals(p_train.train_traces()),
                sorted(c.id for c in train_cases),
                {"original_decision": found["source"]},
                f"det:{parent.version}->{candidate.version}:{endpoints.digest[:12]}",
            )
            held = referee.deterministic(parent, request, p_train, a.repeat_heldout)
            out |= {k: v for k, v in held.items() if k != "throughput"}
            out["throughput"] += held["throughput"]
            out["results"] = referee.results(a.out.stem)
    finally:
        spread.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    verdict = out.get("verdict") or {}
    print(
        "RESULT",
        json.dumps({"result": out["result"], **{k: verdict.get(k) for k in ("decision_id", "admitted")}}),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
