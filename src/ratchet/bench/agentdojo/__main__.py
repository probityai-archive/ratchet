"""`python -m ratchet.bench.agentdojo {baseline,evolve,summarize} --suite banking --out <run dir>`.

The endpoint and key come from `RATCHET_BASE_URL` and `RATCHET_API_KEY`; models from the flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ratchet.bench.agentdojo.cases import DEFAULT_ATTACK, DEFAULT_VERSION, DojoSuite, build_suite
from ratchet.bench.agentdojo.experiment import (
    Settings,
    admitted_changes,
    baseline,
    evolve_from,
    harness_from,
    lineage,
    provenance,
    read_jsonl,
    recorded_lineage,
    suite_digest,
    version_metrics,
)
from ratchet.bench.agentdojo.report import document, render
from ratchet.bench.agentdojo.run import DojoRunner, JsonStore
from ratchet.improve.evaluate import evaluate_many

STOPPED = "STOPPED.txt"  # written by the operator of a run that was stopped on purpose, with the reason
REFERENCE_CLONE = Path.home() / "Documents" / "git-clones" / "agentdojo"


def parse(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m ratchet.bench.agentdojo")
    ap.add_argument("command", choices=["baseline", "evolve", "summarize", "report", "transfer"])
    ap.add_argument("--suite", default="banking")
    ap.add_argument("--out", type=Path, help="run directory (report: the Markdown document to write)")
    ap.add_argument("--runs", type=Path, nargs="*", default=[], help="report: run directories, in order")
    ap.add_argument("--source", type=Path, default=None, help="transfer: the run whose versions to test")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--improver-model", default="gpt-5.6-terra")
    ap.add_argument("--benchmark-version", default=DEFAULT_VERSION)
    ap.add_argument("--attack", default=DEFAULT_ATTACK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--max-users", type=int, default=None)
    args = ap.parse_args(list(argv))
    if args.out is None:
        ap.error("--out is required")
    return args


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
    return write_results(s.out, dojo, prov, versions, command, "complete")


NOTES = [
    "Metrics are on held-out cases only: held-out user tasks paired with held-out injection tasks "
    "(attacked) or with none (utility).",
    "Every evaluation the run made is listed, in order: the baseline, the loop's two fresh parent "
    "runs per generation, and each promoted version's candidate, composite and confirmation runs.",
]


def write_results(
    out: Path, dojo: DojoSuite, prov: dict[str, Any], versions: list[str], command: str, status: str
) -> dict[str, Any]:
    records = [e for e in read_jsonl(out / "evals.jsonl") if e.get("type") == "generation"]
    results = {
        "run_id": out.name,
        "command": command,
        "status": status,
        "provenance": prov,
        "versions": versions,
        "metrics": version_metrics(dojo, out, versions),
        "admitted": admitted_changes(out, versions[1:]),
        "generations": records,
        "notes": NOTES,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "summary.md").write_text(render(results))
    return results


def summarize(out: Path) -> dict[str, Any]:
    """Results of a run from its files alone, finished or still running. Refuses a mismatched suite."""
    prov = json.loads((out / "provenance.json").read_text())
    dojo = build_suite(
        prov["suite"],
        benchmark_version=prov["benchmark_version"],
        attack=prov["attack"],
        seed=prov["seed"],
        max_users=prov.get("max_users"),
    )
    if suite_digest(dojo) != prov["suite_digest"]:
        raise SystemExit(f"{out}: the rebuilt suite does not match the run's suite digest")
    versions = recorded_lineage(out)
    previous = json.loads((out / "results.json").read_text()) if (out / "results.json").is_file() else {}
    command = str(previous.get("command") or "summarize")
    if previous.get("status") == "complete":
        status = "complete"
    elif (out / STOPPED).is_file():
        status = "stopped: " + (out / STOPPED).read_text().strip()
    else:
        status = "in progress"
    return write_results(out, dojo, prov, versions, command, status)


def transfer(source: Path, s: Settings) -> dict[str, Any]:
    """Every version a run promoted, run fresh on its held-out cases under a different attack.

    Same suite, split seed and agent model as the source run; only the attack text differs, and the
    improver never saw it. Held-out cases only."""
    src = json.loads((source / "provenance.json").read_text())
    s = replace(s, suite=src["suite"], model=src["model"], seed=src["seed"], max_users=src.get("max_users"))
    if s.attack == src["attack"]:
        raise SystemExit(f"--attack must differ from the source run's attack ({src['attack']})")
    s.out.mkdir(parents=True, exist_ok=True)
    store = JsonStore(s.out)
    dojo = build_suite(
        s.suite,
        benchmark_version=src["benchmark_version"],
        attack=s.attack,
        seed=s.seed,
        max_users=s.max_users,
    )
    prov = {
        **provenance(s, dojo, REFERENCE_CLONE if REFERENCE_CLONE.is_dir() else None),
        "transfer_from": source.name,
        "trained_attack": src["attack"],
        "improver_model": src["improver_model"],
    }
    (s.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    stored = {h["version"]: harness_from(h) for h in read_jsonl(source / "harnesses.jsonl")}
    versions = recorded_lineage(source)
    harnesses = [stored[v] for v in versions]
    for h in harnesses:
        store.put_harness(h)
    held = dojo.by_split("heldout")
    reports = evaluate_many(harnesses, held, DojoRunner(dojo).run_case, s.client(s.model), workers=s.workers)
    for r in reports:
        for t in r.traces:
            store.put_trace(t)
        store.put_eval({**r.to_payload(), "role": f"transfer:{s.attack}"})
    return write_results(s.out, dojo, prov, versions, "transfer", "complete")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse(sys.argv[1:] if argv is None else argv)
    if args.command == "report":
        args.out.write_text(document([summarize(r) for r in args.runs]))
        print(args.out.read_text())
        return 0
    if args.command == "summarize":
        results = summarize(args.out)
    elif args.command == "transfer":
        if args.source is None:
            raise SystemExit("transfer needs --source <run dir>")
        results = transfer(args.source, settings_from(args, dict(os.environ)))
    else:
        results = run(args.command, settings_from(args, dict(os.environ)))
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
