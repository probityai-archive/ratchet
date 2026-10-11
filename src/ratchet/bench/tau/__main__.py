"""Command line: `python -m ratchet.bench.tau trial|evolve|summary ...`.

trial    run one harness once over a split of one domain; each trace is written the moment its run ends
evolve   run `ratchet.improve.loop.evolve` from the domain's seed harness over the train / held-out split
summary  pass^1 .. pass^k per domain, harness and split, from every recorded run (trials and evolve runs)
report   the same as a Markdown page with every evaluation and admission decision

Every run lands under `--out` (default `private/bench/tau`, never committed):
  runs/<domain>/<harness>/<split>-trial-<n>/   config.json, harnesses.jsonl, traces.jsonl, evals.jsonl
  evolve/<domain>/<name>/                      config.json, the evolve store (jsonl), log.txt, result.json
Endpoint and models come from flags or the environment (`RATCHET_BASE_URL`, `RATCHET_API_KEY`); the key is
never written to a record.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ratchet.agent.guards import GuardRuntime
from ratchet.agent.model import ModelClient, ModelConfig
from ratchet.bench.tau.report import collect, render, summarize
from ratchet.bench.tau.runner import TauRunner, configure_user_endpoint, seed_harness, tau_bench_commit
from ratchet.bench.tau.store import JsonlStore
from ratchet.bench.tau.tasks import DEFAULT_SEED, DOMAINS, Domain, TauCase, default_cases, split_record
from ratchet.improve.activate import MemoryActivator
from ratchet.improve.evaluate import evaluate_many
from ratchet.improve.loop import Lab, evolve
from ratchet.improve.receipt import Signer
from ratchet.improve.sandbox import SandboxRunner
from ratchet.improve.sim import RecordedViews
from ratchet.types import Guard, Harness, Procedure, Trace


def harness_from(raw: dict[str, Any]) -> Harness:
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


def load_harness(spec: str, domain: Domain, root: Path) -> Harness:
    """`seed` (the domain's seed harness), or a harness version recorded anywhere under `root`."""
    if spec == "seed":
        return seed_harness(domain)
    for path in sorted(root.rglob("harnesses.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            if raw["version"] == spec:
                return harness_from(raw)
    raise SystemExit(f"no harness {spec!r} recorded under {root}")


class RecordingRunner:
    """Runs a case, appends its trace to the store the moment it ends, and prints progress."""

    def __init__(self, runner: TauRunner, store: JsonlStore, total: int) -> None:
        self.runner, self.store, self.total = runner, store, total
        self.done = 0
        self._lock = threading.Lock()

    def __call__(self, harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
        try:
            trace = self.runner(harness, case, model, store, run_id=run_id)
        except Exception as exc:
            self._tick(f"{harness.version} {case.id} ERROR {type(exc).__name__}: {str(exc)[:200]}")
            raise
        self.store.put_trace(trace)
        verdict = "PASS" if trace.outcome.passed else "fail"
        self._tick(f"{harness.version} {case.id} {verdict} {trace.outcome.reasons[:1]}")
        return trace

    def _tick(self, line: str) -> None:
        with self._lock:
            self.done += 1
            print(f"[{self.done}/{self.total}] {line}", flush=True)


def _endpoint(args: argparse.Namespace) -> tuple[str, str]:
    base_url = args.base_url or os.environ.get("RATCHET_BASE_URL", "")
    api_key = os.environ.get("RATCHET_API_KEY", "")
    if not base_url or not api_key:
        raise SystemExit("set RATCHET_BASE_URL and RATCHET_API_KEY (or pass --base-url)")
    return base_url, api_key


def _all_cases(args: argparse.Namespace) -> list[TauCase]:
    return default_cases(args.domain, seed=args.seed, n_train=args.n_train, n_heldout=args.n_heldout)


def _config(args: argparse.Namespace, cases: list[TauCase], **extra: Any) -> dict[str, Any]:
    return {
        "domain": args.domain,
        "agent_model": args.agent_model,
        "user_model": args.user_model,
        "temperature": args.temperature,
        "tau_bench_commit": tau_bench_commit(),
        "workers": args.workers,
        "cases": len(cases),
        "split_record": split_record(_all_cases(args), args.seed),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    }


# The shared endpoint rate-limits under load ("All credentials ... are cooling down"); a benchmark run waits
# that out instead of scoring an infrastructure error as an agent failure.
def _client(base_url: str, api_key: str, model: str, temperature: float | None = None) -> ModelClient:
    cfg = ModelConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_attempts=12,
        max_backoff_total_s=900.0,
        backoff_cap_s=60.0,
    )
    return ModelClient(cfg)


def _agent(args: argparse.Namespace, base_url: str, api_key: str) -> ModelClient:
    configure_user_endpoint(base_url, api_key)
    return _client(base_url, api_key, args.agent_model, args.temperature)


def cmd_trial(args: argparse.Namespace) -> int:
    base_url, api_key = _endpoint(args)
    root = Path(args.out)
    harness = load_harness(args.harness, args.domain, root)
    cases = [c for c in _all_cases(args) if args.split == "all" or c.split == args.split]
    if args.limit:
        cases = cases[: args.limit]
    run_dir = root / "runs" / args.domain / harness.version / f"{args.split}-trial-{args.trial}"
    if (run_dir / "traces.jsonl").exists():
        raise SystemExit(f"{run_dir} already has traces; pick another --trial")
    store = JsonlStore(run_dir)
    store.put_harness(harness)
    config = _config(args, cases, harness=harness.version, split=args.split, trial=args.trial)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = _agent(args, base_url, api_key)
    runner = RecordingRunner(TauRunner(user_model=args.user_model), store, len(cases))
    report = evaluate_many([harness], cases, runner, model, workers=args.workers)[0]
    payload = {**report.to_payload(), "config": config, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    store.put_eval(payload, eval_id=f"{harness.version}-{args.split}-trial-{args.trial}")
    errors = sum(1 for r in report.results if r.error)
    print(
        f"{args.domain} {harness.version}: "
        f"heldout {len(report.passed('heldout'))}/{report.count('heldout')}, "
        f"train {len(report.passed('train'))}/{report.count('train')}, run errors {errors}, "
        f"wall {report.wall_s:.0f}s",
        flush=True,
    )
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    base_url, api_key = _endpoint(args)
    root = Path(args.out) / "evolve" / args.domain / args.name
    if (root / "traces.jsonl").exists():
        raise SystemExit(f"{root} already has a run; pick another --name")
    cases = _all_cases(args)
    store = JsonlStore(root)
    record = split_record(cases, args.seed)
    config = _config(
        args,
        cases,
        improver_model=args.improver_model,
        generations=args.generations,
        max_evals=args.max_evals,
        max_clusters=args.max_clusters,
        candidates_per_cluster=args.candidates_per_cluster,
    )
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = _agent(args, base_url, api_key)
    improver = _client(base_url, api_key, args.improver_model)
    # One `trial` of evolve's evaluations is about len(cases) runs; the counter just shows progress.
    runner = RecordingRunner(TauRunner(user_model=args.user_model), store, len(cases))
    signer = Signer.from_env() if os.environ.get("RATCHET_SIGNING_KEY") else Signer.generate()
    log_path = root / "log.txt"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # Activation follows the same rules as the Atlas activator (stored, admitting, signed decision; compare
    # and swap on the active pointer) over this run's own decision log.
    activator = MemoryActivator(store.decision)
    started = time.perf_counter()
    with GuardRuntime() as runtime:
        lab = Lab(
            cases=cases,
            run_case=runner,
            simulator=RecordedViews(),
            runner=SandboxRunner(runtime),
            signer=signer,
            activator=activator,
            improver=improver,
            workers=args.workers,
            max_clusters=args.max_clusters,
            candidates_per_cluster=args.candidates_per_cluster,
            max_evals=args.max_evals,
            suite_digest=record["digest"],
            model_config={
                k: config[k] for k in ("agent_model", "user_model", "temperature", "tau_bench_commit")
            },
            log=log,
        )
        result = evolve(store, model, args.generations, lab=lab, seed=seed_harness(args.domain))
    out = {
        "config": config,
        "final_harness": result.harness.version,
        "instructions": result.instructions.version,
        "generations": [r.to_payload() for r in result.generations],
        "keyid": signer.keyid,
        "active": list(activator.active()),
        "wall_s": round(time.perf_counter() - started, 1),
    }
    (root / "result.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log(f"final harness {result.harness.version}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    print(json.dumps(summarize(collect(Path(args.out)), args.max_k), indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    text = render(Path(args.out), args.domain)
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--domain", choices=list(DOMAINS), default="retail")
    p.add_argument("--agent-model", default="gpt-5.6-luna")
    p.add_argument("--user-model", default="gpt-5.6-luna")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--base-url", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-heldout", type=int, default=None)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ratchet.bench.tau")
    p.add_argument("--out", default="private/bench/tau")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trial")
    _common(t)
    t.add_argument("--harness", default="seed")
    t.add_argument("--split", choices=["heldout", "train", "all"], default="all")
    t.add_argument("--trial", type=int, default=1)
    t.add_argument("--limit", type=int, default=0)
    t.set_defaults(func=cmd_trial)
    e = sub.add_parser("evolve")
    _common(e)
    e.add_argument("--name", default="run-1")
    e.add_argument("--improver-model", default="gpt-5.6-terra")
    e.add_argument("--generations", type=int, default=1)
    e.add_argument("--max-evals", type=int, default=3)
    e.add_argument("--max-clusters", type=int, default=4)
    e.add_argument("--candidates-per-cluster", type=int, default=3)
    e.set_defaults(func=cmd_evolve)
    s = sub.add_parser("summary")
    s.add_argument("--max-k", type=int, default=4)
    s.set_defaults(func=cmd_summary)
    r = sub.add_parser("report")
    r.add_argument("--domain", choices=list(DOMAINS), default="retail")
    r.add_argument("--write", default="")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
