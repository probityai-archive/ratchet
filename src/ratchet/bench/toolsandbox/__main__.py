"""Command line: `python -m ratchet.bench.toolsandbox trial|evolve|summary|report ...`.

trial    run one harness once over a split of the scenario set; each trace is written the moment its run ends
evolve   run `ratchet.improve.loop.evolve` from the seed harness over a train / held-out sample of the set
summary  the benchmark's category scores per recorded run, as JSON
report   the same as a Markdown page (and a CSV of every scenario of every run)

Every run lands under `--out` (default `private/bench/toolsandbox`, never committed):
  runs/<harness>/<split>-trial-<n>/   config.json, cases.json, harnesses.jsonl, traces.jsonl, evals.jsonl,
                                      scenarios.jsonl
  evolve/<name>/                      config.json, the evolve store (jsonl), log.txt, result.json
Endpoint and models come from flags or the environment (`RATCHET_BASE_URL`, `RATCHET_API_KEY`); the key is
never written to a record. A RapidAPI key is taken from `RAPID_API_KEY`, else the keyring; with none, the
scenarios whose necessary tools need it are recorded NOT RUN and stay in every denominator.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ratchet_cases.types import Trace

from ratchet.agent.guards import GuardRuntime
from ratchet.agent.model import ModelClient, ModelConfig
from ratchet.bench.toolsandbox.cases import (
    DEFAULT_SEED,
    ToolSandboxCase,
    build_cases,
    keyring_key,
    split_record,
)
from ratchet.bench.toolsandbox.grading import Graded
from ratchet.bench.toolsandbox.metrics import summarize
from ratchet.bench.toolsandbox.replay import ToolSandboxSimulator
from ratchet.bench.toolsandbox.report import collect, render, render_csv
from ratchet.bench.toolsandbox.runner import (
    ToolSandboxRunner,
    endpoint_user,
    rebase,
    seed_harness,
    tool_sandbox_commit,
)
from ratchet.bench.toolsandbox.store import RunStore
from ratchet.bench.transfer.run import harness_from
from ratchet.improve.activate import MemoryActivator
from ratchet.improve.evaluate import evaluate_many
from ratchet.improve.loop import Lab, evolve
from ratchet.improve.receipt import Signer
from ratchet.improve.referee import ProcessReferee, RefereeParts
from ratchet.improve.sandbox import SandboxRunner
from ratchet.types import Harness


def load_harness(spec: str, root: Path, *, onto_seed: bool = False) -> Harness:
    """`seed` (the benchmark's own agent prompt), `@file` (an exported harness or admitted candidate, as
    `{"harness": {...}}` or the harness alone), or a harness version recorded anywhere under `root`."""
    if spec == "seed":
        return seed_harness()
    if spec.startswith("@"):
        raw = json.loads(Path(spec[1:]).read_text(encoding="utf-8"))
        loaded = harness_from(raw.get("harness", raw))
        return rebase(loaded, seed_harness()) if onto_seed else loaded
    for path in sorted(root.rglob("harnesses.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            if raw["version"] == spec:
                return harness_from(raw)
    raise SystemExit(f"no harness {spec!r} recorded under {root}")


def ratchet_commit() -> str | None:
    """The commit of the checkout this run came from (a box's is a snapshot commit), if git answers."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def parse_shard(text: str) -> tuple[int, int]:
    """`K/N`: shard K (counting from 0) of N."""
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError:
        raise SystemExit(f"--shard needs K/N, got {text!r}") from None
    if count < 1 or not 0 <= index < count:
        raise SystemExit(f"--shard {text}: K must be from 0 to N-1")
    return index, count


def shard_of(cases: list[ToolSandboxCase], index: int, count: int) -> list[ToolSandboxCase]:
    """Every `count`th case starting at `index`: the shards are disjoint and together are every case."""
    return [c for i, c in enumerate(cases) if i % count == index]


def select(cases: list[ToolSandboxCase], split: str, limit: int, seed: int) -> list[ToolSandboxCase]:
    """The cases of one split (or `all`); with `limit`, a seeded sample of that many runnable cases, so a
    smoke run spans the scenario families instead of taking the first few names."""
    chosen = [c for c in cases if split == "all" or c.split == split]
    if limit:
        runnable = [c for c in chosen if not c.not_run]
        chosen = sorted(random.Random(seed).sample(runnable, min(limit, len(runnable))), key=lambda c: c.id)
    return chosen


class RecordingRunner:
    """Runs a case, appends its trace and its benchmark grade the moment it ends, and prints progress."""

    def __init__(self, runner: ToolSandboxRunner, store: RunStore, total: int) -> None:
        self.runner, self.store, self.total = runner, store, total
        self.done = 0
        self._lock = threading.Lock()
        runner.on_graded = self._graded

    def _graded(self, case: ToolSandboxCase, trace: Trace, graded: Graded) -> None:
        self.store.put_graded(case, trace, graded)

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


# The shared endpoint rate-limits under load; a benchmark run waits that out instead of scoring an
# infrastructure error as an agent failure.
def _client(base_url: str, api_key: str, model: str, temperature: float | None) -> ModelClient:
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


def cmd_trial(args: argparse.Namespace) -> int:
    base_url, api_key = _endpoint(args)
    root = Path(args.out)
    harness = load_harness(args.harness, root, onto_seed=args.onto_seed)
    key = keyring_key()
    if key:
        os.environ["RAPID_API_KEY"] = key
    everything = build_cases(seed=args.seed, n_train=args.n_train)
    selected = select(everything, args.split, args.limit, args.seed)
    index, count = parse_shard(args.shard)
    cases = shard_of(selected, index, count)
    suffix = f"-shard-{index}-of-{count}" if count > 1 else ""
    run_dir = root / "runs" / harness.version / f"{args.split}-trial-{args.trial}{suffix}"
    if (run_dir / "traces.jsonl").exists():
        raise SystemExit(f"{run_dir} already has traces; pick another --trial")
    store = RunStore(run_dir)
    store.put_harness(harness)
    store.write_cases(selected)
    config = {
        "agent_model": args.agent_model,
        "user_model": args.user_model,
        "agent_endpoint": base_url,
        "user_endpoint": args.user_base_url or base_url,
        "temperature": args.temperature,
        "tool_sandbox_commit": tool_sandbox_commit(),
        "ratchet_commit": ratchet_commit(),
        "harness": harness.version,
        "harness_source": args.harness,
        "onto_seed": args.onto_seed,
        "split": args.split,
        "trial": args.trial,
        "limit": args.limit,
        "workers": args.workers,
        "cases": len(selected),
        "shard": {"index": index, "count": count, "cases": len(cases)},
        "rapid_api_key": "present" if key else "absent",
        "split_record": split_record(everything, args.seed),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = _client(base_url, api_key, args.agent_model, args.temperature)
    user = endpoint_user(args.user_base_url or base_url, api_key, args.user_model)
    runner = RecordingRunner(ToolSandboxRunner(user=user), store, len(cases))
    report = evaluate_many([harness], cases, runner, model, workers=args.workers)[0]
    payload = {**report.to_payload(), "config": config, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    store.put_eval(payload, eval_id=f"{harness.version}-{args.split}-trial-{args.trial}")
    errors = sum(1 for r in report.results if r.error)
    print(
        f"{harness.version}: heldout {len(report.passed('heldout'))}/{report.count('heldout')}, "
        f"train {len(report.passed('train'))}/{report.count('train')}, run errors {errors}, "
        f"wall {report.wall_s:.0f}s",
        flush=True,
    )
    return 0


def evolve_cases(
    everything: list[ToolSandboxCase], train_bases: int, heldout_bases: int, seed: int
) -> list[ToolSandboxCase]:
    """Every form of a seeded sample of runnable bases: `train_bases` train, `heldout_bases` held out.

    A scenario that cannot run is left out, since a case that can never pass would only teach the improver to
    chase it. The sample keeps the split `build_cases` made, so no base is on both sides.
    """
    runnable = [c for c in everything if not c.not_run]
    out: list[ToolSandboxCase] = []
    for split, count in (("train", train_bases), ("heldout", heldout_bases)):
        bases = sorted({c.base for c in runnable if c.split == split})
        chosen = set(random.Random(seed).sample(bases, min(count, len(bases))))
        out += [c for c in runnable if c.split == split and c.base in chosen]
    return out


def _evolve_cases(args: argparse.Namespace) -> list[Any]:
    key = keyring_key()
    if key:
        os.environ["RAPID_API_KEY"] = key
    everything = build_cases(seed=args.seed, n_train=args.n_train)
    return evolve_cases(everything, args.train_bases, args.heldout_bases, args.seed)


def _model_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "agent_model": args.agent_model,
        "user_model": args.user_model,
        "temperature": args.temperature,
        "tool_sandbox_commit": tool_sandbox_commit(),
    }


def referee_parts(config: dict[str, Any]) -> RefereeParts:
    """The held-out side of `evolve`, rebuilt in the referee's process from the same command line."""
    args = parser().parse_args(config["argv"])
    base_url, api_key = _endpoint(args)
    cases = _evolve_cases(args)
    held = [c for c in cases if c.split == "heldout"]
    store = RunStore(Path(config["root"]))
    stack = ExitStack()
    runtime = stack.enter_context(GuardRuntime())
    user = endpoint_user(args.user_base_url or base_url, api_key, args.user_model)
    return RefereeParts(
        cases=held,
        run_case=RecordingRunner(ToolSandboxRunner(user=user), store, len(held)),
        model=_client(base_url, api_key, args.agent_model, args.temperature),
        simulator=ToolSandboxSimulator(),
        runner=SandboxRunner(runtime),
        signer=Signer.for_decisions(),
        store=store,
        # The same rules as the Atlas activator over this run's own decision log.
        activator=MemoryActivator(store.decision),
        model_config=_model_config(args),
        suite_digest=split_record(cases, args.seed)["digest"],
        workers=args.workers,
        close=stack.close,
    )


def cmd_evolve(args: argparse.Namespace) -> int:
    base_url, api_key = _endpoint(args)
    root = Path(args.out) / "evolve" / args.name
    if (root / "traces.jsonl").exists():
        raise SystemExit(f"{root} already has a run; pick another --name")
    cases = _evolve_cases(args)
    store = RunStore(root)
    store.write_cases(cases)
    record = split_record(cases, args.seed)
    config = {
        "agent_model": args.agent_model,
        "user_model": args.user_model,
        "agent_endpoint": base_url,
        "user_endpoint": args.user_base_url or base_url,
        "improver_model": args.improver_model,
        "temperature": args.temperature,
        "tool_sandbox_commit": tool_sandbox_commit(),
        "ratchet_commit": ratchet_commit(),
        "workers": args.workers,
        "cases": len(cases),
        "split_record": record,
        "generations": args.generations,
        "max_evals": args.max_evals,
        "max_clusters": args.max_clusters,
        "candidates_per_cluster": args.candidates_per_cluster,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = _client(base_url, api_key, args.agent_model, args.temperature)
    improver = _client(base_url, api_key, args.improver_model, None)
    training = [c for c in cases if c.split != "heldout"]
    runner = RecordingRunner(
        ToolSandboxRunner(user=endpoint_user(args.user_base_url or base_url, api_key, args.user_model)),
        store,
        len(training),
    )
    log_path = root / "log.txt"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    started = time.perf_counter()
    # Held-out runs, admission, decisions and activation: the referee, in its own process and directory.
    referee_config = {"argv": args.argv, "root": str(root / "referee")}
    with (
        ProcessReferee("ratchet.bench.toolsandbox.__main__:referee_parts", referee_config) as referee,
        GuardRuntime() as runtime,
    ):
        lab = Lab(
            cases=training,
            run_case=runner,
            simulator=ToolSandboxSimulator(),
            runner=SandboxRunner(runtime),
            referee=referee,
            improver=improver,
            workers=args.workers,
            max_clusters=args.max_clusters,
            candidates_per_cluster=args.candidates_per_cluster,
            max_evals=args.max_evals,
            suite_digest=record["digest"],
            model_config=_model_config(args),
            log=log,
        )
        result = evolve(store, model, args.generations, lab=lab, seed=seed_harness())
        public_key, active = referee.public_key(), referee.active()
    out = {
        "config": config,
        "final_harness": result.harness.version,
        "instructions": result.instructions.version,
        "generations": [r.to_payload() for r in result.generations],
        "decision_public_key": public_key,
        "active": list(active),
        "wall_s": round(time.perf_counter() - started, 1),
    }
    (root / "result.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log(f"final harness {result.harness.version}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    out = [
        {
            "run": run.path,
            "harness": run.report["harness_version"],
            "split": run.config.get("split"),
            "categories": summarize(run.rows),
        }
        for run in collect(Path(args.out))
    ]
    print(json.dumps(out, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    root = Path(args.out)
    text = render(root, Path(args.csv).name if args.csv else "toolsandbox-scenarios.csv")
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.csv:
        Path(args.csv).write_text(render_csv(root), encoding="utf-8")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ratchet.bench.toolsandbox")
    p.add_argument("--out", default="private/bench/toolsandbox")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trial")
    t.add_argument("--agent-model", default="gpt-5.6-luna")
    t.add_argument("--user-model", default="gpt-5.6-luna")
    t.add_argument("--temperature", type=float, default=0.0)
    t.add_argument("--base-url", default=None)
    t.add_argument("--user-base-url", default=None, help="the user simulator's endpoint, if not --base-url")
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--seed", type=int, default=DEFAULT_SEED)
    t.add_argument("--n-train", type=int, default=26, help="base scenarios assigned to train")
    t.add_argument(
        "--harness", default="seed", help="seed, a recorded version, or @file with an exported harness"
    )
    t.add_argument("--split", choices=["heldout", "train", "all"], default="all")
    t.add_argument("--trial", type=int, default=1)
    t.add_argument("--limit", type=int, default=0, help="a seeded sample of this many runnable scenarios")
    t.add_argument(
        "--onto-seed",
        action="store_true",
        help="with @file: append the file's guards and procedures to this benchmark's seed harness",
    )
    t.add_argument(
        "--shard", default="0/1", help="run shard K of N (K/N); shards are disjoint and cover the set"
    )
    t.set_defaults(func=cmd_trial)
    e = sub.add_parser("evolve")
    e.add_argument("--agent-model", default="gpt-5.6-luna")
    e.add_argument("--user-model", default="gpt-5.6-luna")
    e.add_argument("--improver-model", default="gpt-5.6-terra")
    e.add_argument("--temperature", type=float, default=0.0)
    e.add_argument("--base-url", default=None)
    e.add_argument("--user-base-url", default=None, help="the user simulator's endpoint, if not --base-url")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--seed", type=int, default=DEFAULT_SEED)
    e.add_argument("--n-train", type=int, default=26, help="base scenarios assigned to train")
    e.add_argument("--train-bases", type=int, default=8, help="train base scenarios the loop works on")
    e.add_argument("--heldout-bases", type=int, default=16, help="held-out base scenarios it is admitted on")
    e.add_argument("--name", default="run-1")
    e.add_argument("--generations", type=int, default=1)
    e.add_argument("--max-evals", type=int, default=3)
    e.add_argument("--max-clusters", type=int, default=4)
    e.add_argument("--candidates-per-cluster", type=int, default=3)
    e.set_defaults(func=cmd_evolve)
    s = sub.add_parser("summary")
    s.set_defaults(func=cmd_summary)
    r = sub.add_parser("report")
    r.add_argument("--write", default="")
    r.add_argument("--csv", default="")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
