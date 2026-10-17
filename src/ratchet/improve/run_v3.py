"""Evaluate stored candidates under stochastic policy v3, live, through the held-out referee: a parent
reference arm, two identity-control arms and each candidate arm, 40 fresh executions per case per arm,
interleaved by a predeclared schedule.

    uv run python -m ratchet.improve.run_v3 --endpoints-file v3.json --cases a,b \\
        --candidate db:ad50dac8194c --out x.json

The endpoint configuration (`ratchet.improve.endpoints`) names the endpoints, the model and `seed_base`:
repetition `r` of every arm is sent with seed `seed_base + r`. With `--replicas-file`, the referee re-reads
the endpoint list before every execution, so replicas of the same served model join as they answer.

Every v3 case is held out, so this process runs nothing. The referee (`ProcessReferee`, factory
`ratchet.improve.referee_live:policy_parts`) opens the endpoint file, refuses it when its digest is not the
one named here, rebuilds each candidate from its stored decision, runs the schedule, stores every execution
as it completes (`v3_executions`, so a stopped run keeps its evidence) and signs one receipt per candidate.
This process receives counts, pooled rates and the signed decisions; the per-case tables go to the
referee's signed results file under `--results-dir`, which `python -m ratchet.verify.results` checks.
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
from ratchet.improve.loop import frozen_suite_digest
from ratchet.improve.referee import ProcessReferee

FACTORY = "ratchet.improve.referee_live:policy_parts"


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ratchet-improve-run-v3")
    p.add_argument("--endpoints-file", type=Path, required=True, help="the endpoint configuration (JSON)")
    p.add_argument("--replicas-file", type=Path, help="re-read before every execution: one endpoint per line")
    p.add_argument("--workers", type=int, default=128)
    p.add_argument("--schedule-seed", type=int, default=20261011)
    p.add_argument("--db", default="ratchet_v3", help="the referee's held-out database")
    p.add_argument("--cases", required=True, help="comma-separated held-out case ids")
    p.add_argument(
        "--candidate", action="append", default=[], help="db:candidate_version of a stored decision"
    )
    p.add_argument("--deadline-s", type=float, default=float("inf"))
    p.add_argument("--scope-note", default="")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, help="the referee's results (default: next to --out)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from ratchet.agent.seed import seed_harness

    a = _args(argv)
    if not os.environ.get("RATCHET_MONGODB_URI") and not os.environ.get("RATCHET_REFEREE_MONGODB_URI"):
        print("error: set RATCHET_MONGODB_URI", file=sys.stderr)
        return 2
    endpoints = ep.load(a.endpoints_file)
    if endpoints.doc.get("seed_base") is None:
        print("error: a v3 endpoint configuration names seed_base", file=sys.stderr)
        return 2
    case_ids = [c.strip() for c in a.cases.split(",") if c.strip()]
    results_dir = a.results_dir or a.out.parent / f"{a.out.stem}-results"
    referee_config = {
        "cases": case_ids,
        "store": {"kind": "atlas", "db": a.db},
        "endpoints": endpoints.reference(a.endpoints_file),
        "worker": "per_repetition",
        "replicas_file": str(a.replicas_file) if a.replicas_file else None,
        "suite_digest": frozen_suite_digest(),
        "results_dir": str(results_dir),
    }
    with ProcessReferee(FACTORY, referee_config) as referee:
        summary: dict[str, Any] = referee.stochastic(
            seed_harness(),
            a.candidate,
            case_ids,
            a.schedule_seed,
            deadline_s=a.deadline_s,
            scope_note=a.scope_note,
            workers=a.workers,
        )
        summary["endpoint_config_digest"] = endpoints.digest
        summary["results"] = referee.results(a.out.stem)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    decisions = [(x["candidate"], x["admitted"], x["failed"]) for x in summary["decisions"]]
    print("DONE", json.dumps({"cohort": summary["cohort"], "decisions": decisions}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
