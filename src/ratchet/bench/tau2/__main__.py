"""Command line: `python -m ratchet.bench.tau2 fetch-data|run|finalize|summary|report|package ...`.

fetch-data  fetch the task data of the pinned tau2-bench commit into the cache (sparse git checkout)
run         run one harness over every base task of a domain (or named tasks) at N trials; each finished
            conversation is written at once, in tau2-bench's directory format and as a Ratchet trace
finalize    rebuild a run's `results.json` from its simulation files (after an interrupted run)
summary     counts and pass^k for every run under `--out`, with denominators
report      write the summary as a Markdown results page (numbers generated, never typed)
package     build a leaderboard submission (custom type) from finished runs: `submission.json` and one
            tau2-bench trajectory file per domain

Runs land under `--out` (default `private/bench/tau2`, never committed):
  runs/<domain>/<harness version>/<name>/   config.json, harness.json, results.json, simulations/,
                                            traces.jsonl, runs.jsonl, attempts.jsonl
Endpoints and models come from flags or the environment (`RATCHET_BASE_URL`, `RATCHET_API_KEY`); the key
is never written to a record. With `RATCHET_MONGODB_URI` set, every trace is also inserted into the
evaluator-only database (`--trace-db`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import ratchet.bench.tau2  # noqa: F401  # sets TAU2_DATA_DIR before tau2-bench is imported
from ratchet.agent.model import ModelClient, ModelConfig
from ratchet.bench.tau2.data import (
    TAU2_COMMIT,
    data_digest,
    data_dir,
    fetch_data,
    installed_commit,
    is_complete,
)
from ratchet.bench.tau2.results import (
    RunWriter,
    finalize,
    read_jsonl,
    summarize_run,
    tau2_info,
)
from ratchet.bench.tau2.runner import (
    MAX_ERRORS,
    MAX_STEPS,
    Tau2Runner,
    configure_user_endpoint,
    litellm_model,
    run_config,
    seed_harness,
)
from ratchet.bench.tau2.tasks import (
    DEFAULT_SEED,
    DOMAINS,
    Domain,
    Tau2Case,
    domain_cases,
    load_tasks,
    trial_seeds,
)
from ratchet.improve.evaluate import INFRA_RETRIES, is_infra_failure
from ratchet.types import Guard, Harness, Procedure, Trace

Log = Callable[[str], None]


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# -- harnesses ------------------------------------------------------------------------------------------


def harness_from(raw: dict[str, Any]) -> Harness:
    raw = raw.get("harness", raw)
    h = Harness(
        version=raw["version"],
        parent=raw["parent"],
        system_prompt=raw["system_prompt"],
        procedures=[Procedure(**p) for p in raw["procedures"]],
        guards=[Guard(**g) for g in raw["guards"]],
        tool_allowlist=list(raw["tool_allowlist"]),
        created_by=raw["created_by"],
        digest=raw["digest"],
    )
    if not h.verify_digest():
        raise ValueError(f"harness {h.version} does not match its digest")
    return h


def rebase(source: Harness, target: Harness) -> Harness:
    """`source`'s guards and procedures appended to `target` (a domain's seed harness): the learned,
    executable and instructional changes travel; the domain's prompt and tool allowlist stay."""
    have_g = {g.id for g in target.guards}
    have_p = {p.id for p in target.procedures}
    return target.child(
        guards=[*target.guards, *(g for g in source.guards if g.id not in have_g)],
        procedures=[*target.procedures, *(p for p in source.procedures if p.id not in have_p)],
        created_by=source.created_by,
    )


def load_harness(spec: str, domain: Domain, *, onto_seed: bool = False) -> Harness:
    """`seed`, or a harness JSON file (a bare harness or `{"harness": ...}`), optionally rebased onto the
    domain's seed harness."""
    if spec == "seed":
        return seed_harness(domain)
    h = harness_from(json.loads(Path(spec).read_text(encoding="utf-8")))
    return rebase(h, seed_harness(domain)) if onto_seed else h


def harness_record(h: Harness) -> dict[str, Any]:
    return {
        "version": h.version,
        "digest": h.digest,
        "parent": h.parent,
        "created_by": h.created_by,
        "guards": [g.id for g in h.guards],
        "procedures": [p.id for p in h.procedures],
        "tools": len(h.tool_allowlist),
    }


# -- run ------------------------------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return done.stdout.strip()


def quiet_tau2_logs() -> None:
    """tau2-bench logs through loguru; keep its default level (ERROR) on stderr."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="ERROR")


def set_judge(model: str) -> None:
    """The model tau2-bench's natural-language assertion judge calls (through litellm)."""
    import tau2.evaluator.evaluator_nl_assertions as nl  # type: ignore[import-untyped]  # no type info

    nl.DEFAULT_LLM_NL_ASSERTIONS = litellm_model(model)


class TraceSink:
    """Inserts traces into the evaluator-only database (plain inserts, no schema side effects)."""

    def __init__(self, uri: str, db: str, run_name: str) -> None:
        from pymongo import MongoClient

        self._client: Any = MongoClient(uri, serverSelectionTimeoutMS=10_000)
        self._coll = self._client[db]["tau2_traces"]
        self.run_name = run_name
        self.written = 0

    def put(self, trace: Trace, extra: dict[str, Any]) -> None:
        doc = {**asdict(trace), **extra, "_id": f"{self.run_name}:{trace.run_id}", "run_name": self.run_name}
        self._coll.insert_one(doc)
        self.written += 1

    def close(self) -> None:
        self._client.close()


def run_unit(
    runner: Tau2Runner, harness: Harness, unit: Tau2Case, model: Any, writer: RunWriter
) -> tuple[dict[str, Any], Trace | None]:
    """One trial unit under the predeclared retry rule: a conversation lost to the model endpoint (an
    exception, or a run cut by a model error) is re-run at most `INFRA_RETRIES` times; a completed one,
    pass or fail, never is. Every attempt is recorded."""
    trace: Trace | None = None
    error = ""
    attempts = 0
    while attempts <= INFRA_RETRIES:
        attempts += 1
        run_id = f"run-{harness.version}-{uuid.uuid4().hex[:12]}"
        try:
            trace = runner(harness, unit, model, None, run_id=run_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            trace = None
            writer.put_attempt({"case_id": unit.id, "trial": unit.trial, "run_id": run_id, "error": error})
            continue
        writer.put_trace(trace)
        lost = is_infra_failure(trace)
        writer.put_attempt(
            {
                "case_id": unit.id,
                "trial": unit.trial,
                "run_id": run_id,
                "passed": trace.outcome.passed,
                "lost": lost,
            },
        )
        if not lost:
            break
    sim = runner.simulations.pop(trace.run_id, None) if trace is not None else None
    lost = trace is None or is_infra_failure(trace)
    if sim is not None and not lost:
        writer.put_simulation(sim)
    info = sim.reward_info if sim is not None else None
    row = {
        "case_id": unit.id,
        "task_id": unit.task_id,
        "domain": unit.domain,
        "split": unit.split,
        "trial": unit.trial,
        "seed": unit.seed,
        "run_id": trace.run_id if trace is not None else None,
        "passed": bool(trace is not None and not lost and trace.outcome.passed),
        "reward": float(info.reward) if info is not None and not lost else None,
        "termination": str(sim.termination_reason.value) if sim is not None else None,
        "reasons": list(trace.outcome.reasons[:4]) if trace is not None else [f"run error: {error}"],
        "attempts": attempts,
        "infra_failed": lost,
        "error": error or None,
        "cost": asdict(trace.outcome.cost) if trace is not None else None,
    }
    return row, trace


def _run_dir(args: argparse.Namespace, harness: Harness) -> Path:
    return Path(args.out, "runs", args.domain, harness.version, args.name)


def _select(args: argparse.Namespace) -> list[Tau2Case]:
    ids = [t for t in args.tasks.split(",") if t] if args.tasks else None
    cases = domain_cases(args.domain, ids)
    return cases[: args.limit] if args.limit else cases


def cmd_run(args: argparse.Namespace) -> int:
    quiet_tau2_logs()
    base_url = args.base_url or os.environ.get("RATCHET_BASE_URL", "")
    api_key = os.environ.get("RATCHET_API_KEY", "")
    if not base_url or not api_key:
        raise SystemExit("set RATCHET_BASE_URL and RATCHET_API_KEY (or pass --base-url)")
    user_url = args.user_base_url or base_url
    path = data_dir()
    if not is_complete(path):
        fetch_data(path)
    harness = load_harness(args.harness, args.domain, onto_seed=args.onto_seed)
    cases = _select(args)
    seeds = trial_seeds(args.seed, args.trials)
    units = [replace(c, trial=t, seed=seeds[t]) for t in range(args.trials) for c in cases]
    root = _run_dir(args, harness)
    done_units = {(r["case_id"], r["trial"]) for r in read_jsonl(root / "runs.jsonl")}
    if done_units and not args.resume:
        raise SystemExit(f"{root} already has results; pass --resume to finish it, or pick another --name")
    writer = RunWriter(root)
    judge = args.judge_model or args.user_model
    config = {
        "domain": args.domain,
        "harness": harness_record(harness),
        "harness_spec": args.harness,
        "onto_seed": args.onto_seed,
        "agent_model": args.agent_model,
        "user_model": args.user_model,
        "judge_model": judge,
        "temperature": args.temperature,
        "user_temperature": 0.0,
        "agent_base_url": base_url,
        "user_base_url": user_url,
        "trials": args.trials,
        "seed": args.seed,
        "trial_seeds": seeds,
        "tasks": len(cases),
        "task_ids": [c.task_id for c in cases],
        "task_set": "base",
        "max_steps": MAX_STEPS,
        "max_errors": MAX_ERRORS,
        "infra_retries": INFRA_RETRIES,
        "workers": args.workers,
        "tau2_commit_pinned": TAU2_COMMIT,
        "tau2_commit_installed": installed_commit(),
        "tau2_data_digest": data_digest(path, [args.domain]),
        "ratchet_commit": _git_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if not (root / "config.json").exists():
        (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (root / "harness.json").write_text(json.dumps(asdict(harness), indent=2), encoding="utf-8")
    configure_user_endpoint(user_url, api_key)
    set_judge(judge)
    tconfig = run_config(
        args.domain,
        agent_model=args.agent_model,
        user_model=args.user_model,
        temperature=args.temperature,
        trials=args.trials,
        seed=args.seed,
    )
    runner = Tau2Runner(tconfig)
    todo = [u for u in units if (u.id, u.trial) not in done_units]
    model = ModelClient(
        ModelConfig(base_url=base_url, model=args.agent_model, api_key=api_key, temperature=args.temperature)
    )
    sink = None
    uri = os.environ.get("RATCHET_MONGODB_URI")
    if uri and args.trace_db:
        sink = TraceSink(uri, args.trace_db, f"{args.domain}/{harness.version}/{args.name}")
    counter = {"n": 0}
    lock = threading.Lock()

    def one(unit: Tau2Case) -> None:
        row, trace = run_unit(runner, harness, unit, model, writer)
        writer.put_row(row)
        if sink is not None and trace is not None:
            sink.put(trace, {"row": row})
        with lock:
            counter["n"] += 1
            verdict = "LOST" if row["infra_failed"] else ("PASS" if row["passed"] else "fail")
            _log(f"[{counter['n']}/{len(todo)}] t{unit.trial} {unit.id} {verdict} {row['reasons'][:1]}")

    _log(
        f"{args.domain} {harness.version}: {len(todo)} units "
        f"({len(cases)} tasks x {args.trials} trials) -> {root}"
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(one, todo))
    model.close()
    if sink is not None:
        _log(f"stored {sink.written} traces in {args.trace_db}.tau2_traces")
        sink.close()
    info = tau2_info(
        tconfig, agent_implementation=f"probity-ratchet:{harness.version}", commit=installed_commit()
    )
    finalize(root, info, [t for t in load_tasks(args.domain) if str(t.id) in set(config["task_ids"])])
    print(json.dumps(summarize_run(root), indent=2), flush=True)
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    root = Path(args.run)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    tconfig = run_config(
        config["domain"],
        agent_model=config["agent_model"],
        user_model=config["user_model"],
        temperature=config["temperature"],
        trials=config["trials"],
        seed=config["seed"],
    )
    info = tau2_info(
        tconfig,
        agent_implementation=f"probity-ratchet:{config['harness']['version']}",
        commit=config["tau2_commit_installed"],
    )
    ids = set(config["task_ids"])
    finalize(root, info, [t for t in load_tasks(config["domain"]) if str(t.id) in ids])
    print(json.dumps(summarize_run(root), indent=2))
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    from ratchet.bench.tau2.evolve import run_evolve, signer_from_env

    quiet_tau2_logs()
    base_url = args.base_url or os.environ.get("RATCHET_BASE_URL", "")
    api_key = os.environ.get("RATCHET_API_KEY", "")
    if not base_url or not api_key:
        raise SystemExit("set RATCHET_BASE_URL and RATCHET_API_KEY (or pass --base-url)")
    user_url = args.user_base_url or base_url
    path = data_dir()
    if not is_complete(path):
        fetch_data(path)
    root = Path(args.out, "evolve", args.domain, args.name)
    if (root / "traces.jsonl").exists():
        raise SystemExit(f"{root} already has a run; pick another --name")
    root.mkdir(parents=True, exist_ok=True)
    cases = _select(args)
    judge = args.judge_model or args.user_model
    config = {
        "domain": args.domain,
        "agent_model": args.agent_model,
        "user_model": args.user_model,
        "judge_model": judge,
        "improver_model": args.improver_model,
        "temperature": args.temperature,
        "agent_base_url": base_url,
        "user_base_url": user_url,
        "improver_base_url": args.improver_base_url or base_url,
        "tasks": len(cases),
        "train": sum(1 for c in cases if c.split == "train"),
        "heldout": sum(1 for c in cases if c.split == "heldout"),
        "generations": args.generations,
        "max_evals": args.max_evals,
        "max_clusters": args.max_clusters,
        "candidates_per_cluster": args.candidates_per_cluster,
        "workers": args.workers,
        "tau2_commit_installed": installed_commit(),
        "tau2_data_digest": data_digest(path, [args.domain]),
        "ratchet_commit": _git_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    configure_user_endpoint(user_url, api_key)
    set_judge(judge)
    tconfig = run_config(args.domain, agent_model=args.agent_model, user_model=args.user_model,
                         temperature=args.temperature)  # fmt: skip
    model = ModelClient(
        ModelConfig(base_url=base_url, model=args.agent_model, api_key=api_key, temperature=args.temperature)
    )
    improver_key = os.environ.get("RATCHET_IMPROVER_API_KEY", api_key)
    improver = ModelClient(
        ModelConfig(
            base_url=args.improver_base_url or base_url, model=args.improver_model, api_key=improver_key
        )
    )
    log_path = root / "log.txt"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    out = run_evolve(
        root,
        cases=cases,
        runner=Tau2Runner(tconfig),
        model=model,
        improver=improver,
        seed=seed_harness(args.domain),
        config=config,
        generations=args.generations,
        workers=args.workers,
        max_clusters=args.max_clusters,
        candidates_per_cluster=args.candidates_per_cluster,
        max_evals=args.max_evals,
        signer=signer_from_env(),
        log=log,
    )
    log(f"active harness {out['active']}, frozen at {root / 'frozen.json'}")
    return 0


def all_runs(out: Path) -> list[Path]:
    return sorted(p.parent for p in Path(out).glob("runs/*/*/*/config.json"))


def cmd_summary(args: argparse.Namespace) -> int:
    print(json.dumps([summarize_run(p) for p in all_runs(Path(args.out))], indent=2))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    path = fetch_data(data_dir())
    print(f"{path} {data_digest(path)}")
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--domain", choices=list(DOMAINS), default="retail")
    p.add_argument("--agent-model", default="gpt-5.6-luna")
    p.add_argument("--user-model", default="gpt-5.6-luna")
    p.add_argument("--judge-model", default=None, help="assertion judge model (default: the user model)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--base-url", default=None)
    p.add_argument("--user-base-url", default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)


def parser() -> argparse.ArgumentParser:
    from ratchet.bench.tau2.report import add_package_parser, add_report_parser

    p = argparse.ArgumentParser(prog="python -m ratchet.bench.tau2")
    p.add_argument("--out", default="private/bench/tau2")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-data")
    f.set_defaults(func=cmd_fetch)
    r = sub.add_parser("run")
    _common(r)
    r.add_argument("--harness", default="seed", help="'seed' or a harness JSON file")
    r.add_argument("--onto-seed", action="store_true", help="rebase the harness file onto the domain seed")
    r.add_argument("--trials", type=int, default=4)
    r.add_argument("--tasks", default="", help="comma-separated tau2-bench task ids (default: every task)")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--name", default="run-1")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--trace-db", default="ratchet_heldout", help="evaluator-only database; empty to skip")
    r.set_defaults(func=cmd_run)
    e = sub.add_parser("evolve")
    _common(e)
    e.add_argument("--name", default="evolve-1")
    e.add_argument("--tasks", default="")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--workers", type=int, default=16)
    e.add_argument("--improver-model", default="gpt-5.6-terra")
    e.add_argument("--improver-base-url", default=None)
    e.add_argument("--generations", type=int, default=1)
    e.add_argument("--max-evals", type=int, default=3)
    e.add_argument("--max-clusters", type=int, default=4)
    e.add_argument("--candidates-per-cluster", type=int, default=3)
    e.set_defaults(func=cmd_evolve)
    fz = sub.add_parser("finalize")
    fz.add_argument("run")
    fz.set_defaults(func=cmd_finalize)
    s = sub.add_parser("summary")
    s.set_defaults(func=cmd_summary)
    add_report_parser(sub)
    add_package_parser(sub)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
