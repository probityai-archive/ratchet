"""`python -m ratchet.bench.agentdojo {baseline,evolve} --suite banking --out private/bench/agentdojo/<run>`.

The endpoint and key come from `RATCHET_BASE_URL` and `RATCHET_API_KEY`; models from the flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ratchet.bench.agentdojo.cases import DEFAULT_ATTACK, DEFAULT_VERSION, build_suite
from ratchet.bench.agentdojo.experiment import (
    Settings,
    admitted_changes,
    baseline,
    evolve_from,
    lineage,
    provenance,
    version_metrics,
)
from ratchet.bench.agentdojo.report import render
from ratchet.bench.agentdojo.run import DojoRunner, JsonStore

REFERENCE_CLONE = Path.home() / "Documents" / "git-clones" / "agentdojo"


def parse(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m ratchet.bench.agentdojo")
    ap.add_argument("command", choices=["baseline", "evolve"])
    ap.add_argument("--suite", default="banking")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--improver-model", default="gpt-5.6-terra")
    ap.add_argument("--benchmark-version", default=DEFAULT_VERSION)
    ap.add_argument("--attack", default=DEFAULT_ATTACK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--max-users", type=int, default=None)
    return ap.parse_args(list(argv))


def settings_from(args: argparse.Namespace, env: dict[str, str]) -> Settings:
    missing = [k for k in ("RATCHET_BASE_URL", "RATCHET_API_KEY") if not env.get(k)]
    if missing:
        raise SystemExit(f"missing environment variables: {', '.join(missing)}")
    return Settings(
        suite=args.suite,
        out=args.out,
        base_url=env["RATCHET_BASE_URL"],
        api_key=env["RATCHET_API_KEY"],
        model=args.model,
        improver_model=args.improver_model,
        benchmark_version=args.benchmark_version,
        attack=args.attack,
        seed=args.seed,
        workers=args.workers,
        generations=args.generations if args.command == "evolve" else 0,
        max_users=args.max_users,
    )


def run(command: str, s: Settings, log: Any = print) -> dict[str, Any]:
    s.out.mkdir(parents=True, exist_ok=True)
    dojo = build_suite(
        s.suite, benchmark_version=s.benchmark_version, attack=s.attack, seed=s.seed, max_users=s.max_users
    )
    prov = provenance(s, dojo, REFERENCE_CLONE if REFERENCE_CLONE.is_dir() else None)
    (s.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    runner, store = DojoRunner(dojo), JsonStore(s.out)
    model, improver = s.client(s.model), s.client(s.improver_model)
    started = time.perf_counter()
    v0, v0_eval = baseline(dojo, runner, model, store, s.workers)
    elapsed = time.perf_counter() - started
    log(f"baseline {v0.version}: held-out {v0_eval.to_payload()['heldout']} in {elapsed:.0f}s")
    result = None
    if command == "evolve":
        result = evolve_from(dojo, runner, model, improver, store, v0, s, log)
    versions = lineage(v0, result)
    results = {
        "run_id": s.out.name,
        "command": command,
        "provenance": prov,
        "versions": versions,
        "metrics": version_metrics(dojo, s.out, versions),
        "admitted": admitted_changes(s.out, versions[1:]),
        "generations": [r.to_payload() for r in result.generations] if result else [],
        "wall_s": round(time.perf_counter() - started, 1),
        "notes": [
            "Metrics are on held-out cases only: held-out user tasks paired with held-out injection tasks "
            "(attacked) or with none (utility).",
            "Every evaluation the run made is listed, in order: the baseline, the loop's two fresh parent "
            "runs per generation, and each promoted version's candidate, composite and confirmation runs.",
        ],
    }
    (s.out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (s.out / "summary.md").write_text(render(results))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse(sys.argv[1:] if argv is None else argv)
    results = run(args.command, settings_from(args, dict(os.environ)))
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
