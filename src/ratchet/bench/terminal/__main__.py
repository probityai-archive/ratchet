"""Command line: `python -m ratchet.bench.terminal split|trial|evolve|rates ...`.

split    print (or `--write`) the seeded phase sides; with `--suite`, also check the suite against them
oracle   run each task's reference solution through the world's own tools and grade it (no model involved)
trial    run one harness over named tasks, a phase side, or both phases; each trace is stored as it ends
evolve   run `ratchet.improve.loop.evolve` for phase 1 (from the seed) or phase 2 (from a phase-1 harness)
rates    pass rates with denominators for a harness over the phase task sets, from every recorded run

Runs land under `--out` (default `private/bench/terminal`, never committed):
  runs/<harness>/<label>-trial-<n>/   config.json, harnesses.jsonl, traces.jsonl, evals.jsonl
  evolve/phase-<p>/<name>/            config.json, the evolve store (jsonl), log.txt, result.json
The endpoint comes from `RATCHET_BASE_URL` and `RATCHET_API_KEY`; the key is never written to a record.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ratchet.agent.guards import GuardRuntime
from ratchet.agent.model import ModelClient, ModelConfig
from ratchet.bench.tau.store import JsonlStore
from ratchet.bench.terminal import report
from ratchet.bench.terminal.container import Container
from ratchet.bench.terminal.grade import reasons_of, summary, verify
from ratchet.bench.terminal.runner import TerminalRunner, package_version, seed_harness, start_container
from ratchet.bench.terminal.tasks import (
    DEFAULT_SEED,
    HELDOUT_PER_PHASE,
    PHASE_1,
    PHASE_2,
    TaskSpec,
    TerminalCase,
    case_id,
    committed_record,
    load_suite,
    phase_cases,
    phases,
    split_record,
    task_cases,
)
from ratchet.bench.terminal.world import RUN_COMMAND, TerminalWorld
from ratchet.improve.activate import MemoryActivator
from ratchet.improve.evaluate import evaluate_many
from ratchet.improve.loop import Lab, evolve
from ratchet.improve.receipt import Signer
from ratchet.improve.sandbox import SandboxRunner
from ratchet.improve.sim import RecordedViews
from ratchet.types import Guard, Harness, Procedure, ToolCall, Trace

DEFAULT_SUITE = os.path.expanduser("~/Documents/git-clones/terminal-bench-2")


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


def load_harness(spec: str, root: Path) -> Harness:
    """`seed`, or a harness version recorded anywhere under `root`."""
    if spec == "seed":
        return seed_harness()
    for path in sorted(root.rglob("harnesses.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            if raw["version"] == spec:
                return harness_from(raw)
    raise SystemExit(f"no harness {spec!r} recorded under {root}")


class TerminalStore(JsonlStore):
    """The benchmark run store, plus the harness lookup the improvement loop reads back."""

    def get_harness(self, version: str) -> Harness | None:
        found = [h for h in self.read("harnesses") if h["version"] == version]
        return harness_from(found[-1]) if found else None


class RecordingRunner:
    """Runs a case, appends its trace to the store the moment it ends, and prints progress."""

    def __init__(self, runner: TerminalRunner, store: JsonlStore) -> None:
        self.runner, self.store = runner, store
        self.done = 0
        self._lock = threading.Lock()

    def __call__(self, harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
        started = time.time()
        try:
            trace = self.runner(harness, case, model, store, run_id=run_id)
        except Exception as exc:
            self._tick(f"{harness.version} {case.id} ERROR {type(exc).__name__}: {str(exc)[:300]}")
            raise
        self.store.put_trace(trace)
        verdict = "PASS" if trace.outcome.passed else "fail"
        wall = time.time() - started
        self._tick(f"{harness.version} {case.id} {verdict} {wall:.0f}s {trace.outcome.reasons[:2]}")
        return trace

    def _tick(self, line: str) -> None:
        with self._lock:
            self.done += 1
            print(f"[{self.done}] {time.strftime('%H:%M:%S')} {line}", flush=True)


def _endpoint() -> tuple[str, str]:
    base_url = os.environ.get("RATCHET_BASE_URL", "")
    api_key = os.environ.get("RATCHET_API_KEY", "")
    if not base_url or not api_key:
        raise SystemExit("set RATCHET_BASE_URL and RATCHET_API_KEY")
    return base_url, api_key


def _client(model: str, temperature: float | None = None) -> ModelClient:
    base_url, api_key = _endpoint()
    cfg = ModelConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        timeout_s=600.0,
        max_attempts=12,
        max_backoff_total_s=900.0,
        backoff_cap_s=60.0,
    )
    return ModelClient(cfg)


def _config(args: argparse.Namespace, cases: Sequence[TerminalCase], **extra: Any) -> dict[str, Any]:
    return {
        "agent_model": args.agent_model,
        "temperature": args.temperature,
        "suite": args.suite,
        "suite_commit": _suite_commit(Path(args.suite)),
        "package": f"terminal-bench=={package_version()}",
        "workers": args.workers,
        "cases": [c.id for c in cases],
        "split_record": split_record(args.seed, _heldout(args)),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    }


def _heldout(args: argparse.Namespace) -> dict[int, int]:
    """Held-out tasks per phase: `--heldout 6,5`, or the first split's 4,4."""
    raw = getattr(args, "heldout", "")
    if not raw:
        return dict(HELDOUT_PER_PHASE)
    one, two = (int(x) for x in raw.split(","))
    return {1: one, 2: two}


def _suite_commit(root: Path) -> str:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return "unknown"
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref: "):
        target = root / ".git" / ref[5:]
        return target.read_text(encoding="utf-8").strip() if target.is_file() else "unknown"
    return ref


def cmd_split(args: argparse.Namespace) -> int:
    record = split_record(args.seed, _heldout(args))
    if args.suite:
        found = phases(load_suite(Path(args.suite)))
        record["suite_check"] = {str(p): len(n) for p, n in found.items()}
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


def _trial_cases(args: argparse.Namespace) -> tuple[list[TerminalCase], str]:
    tasks = load_suite(Path(args.suite))
    if args.tasks:
        names = [n.strip() for n in args.tasks.split(",") if n.strip()]
        return task_cases(tasks, names), args.label or "tasks"
    if args.set == "union":
        return task_cases(tasks, [*PHASE_1, *PHASE_2]), "union"
    if args.set in ("phase1", "phase2"):
        return task_cases(tasks, PHASE_1 if args.set == "phase1" else PHASE_2), args.set
    phase, side = args.set.split("-")
    cases = [c for c in phase_cases(tasks, int(phase[-1]), args.seed, _heldout(args)) if c.split == side]
    return cases, args.set


def cmd_trial(args: argparse.Namespace) -> int:
    root = Path(args.out)
    harness = load_harness(args.harness, root)
    cases, label = _trial_cases(args)
    run_dir = root / "runs" / harness.version / f"{label}-trial-{args.trial}"
    if (run_dir / "traces.jsonl").exists():
        raise SystemExit(f"{run_dir} already has traces; pick another --trial")
    store = TerminalStore(run_dir)
    store.put_harness(harness)
    config = _config(args, cases, harness=harness.version, label=label, trial=args.trial)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = _client(args.agent_model, args.temperature)
    with GuardRuntime() as runtime:
        runner = RecordingRunner(TerminalRunner(runtime=runtime), store)
        rep = evaluate_many([harness], cases, runner, model, workers=args.workers, repeats=args.repeats)[0]
    payload = {**rep.to_payload(), "config": config, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    store.put_eval(payload, eval_id=f"{harness.version}-{label}-trial-{args.trial}")
    errors = sum(1 for r in rep.results if r.error)
    print(
        f"{harness.version} {label}: {len(rep.passed())}/{rep.count()} passed, run errors {errors}",
        flush=True,
    )
    return 0


def oracle_one(task: TaskSpec, factory: Callable[[TaskSpec], Container] = start_container) -> dict[str, Any]:
    """Run the reference solution with `run_command` in a fresh container, then the task's tests."""
    started = time.time()
    container = factory(task)
    try:
        world = TerminalWorld(container, task.name, task.agent_timeout_s)
        container.put_dir(task.path / "solution", "/solution")
        cmd = {"cmd": "bash /solution/solve.sh", "timeout_s": task.agent_timeout_s}
        result = world.call(ToolCall(RUN_COMMAND, cmd, "oracle"))
        verdict = verify(container, task)
        return {
            "task": task.name,
            "solution_exit_code": (result.data or {}).get("exit_code"),
            "solution_output": str((result.data or {}).get("output", ""))[-3000:],
            "tests_output": verdict.stdout_tail[-3000:],
            "view": world.view().to_json(),
            **summary(verdict),
            "reasons": reasons_of(verdict, None),
            "wall_s": round(time.time() - started, 1),
        }
    finally:
        container.close()


def cmd_oracle(args: argparse.Namespace) -> int:
    tasks = load_suite(Path(args.suite))
    names = [n for n in args.tasks.split(",") if n] if args.tasks else [*PHASE_1, *PHASE_2]
    picked = [task_cases(tasks, [n])[0].task for n in names]
    out = Path(args.out) / "oracle" / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def one(task: TaskSpec) -> dict[str, Any]:
        try:
            row = oracle_one(task)
        except Exception as exc:
            row = {"task": task.name, "passed": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        with lock, (out / "oracle.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"{time.strftime('%H:%M:%S')} {task.name} reward={row.get('reward')} {row.get('error') or ''}")
        return row

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(one, picked))
    passed = sum(1 for r in rows if r.get("passed"))
    print(f"oracle: {passed}/{len(rows)} reference solutions pass their tests; records in {out}", flush=True)
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    root = Path(args.out) / "evolve" / f"phase-{args.phase}" / args.name
    if (root / "traces.jsonl").exists():
        raise SystemExit(f"{root} already has a run; pick another --name")
    parent = load_harness(args.parent, Path(args.out))
    if args.phase == 2 and parent.created_by == "seed":
        raise SystemExit("phase 2 starts from the harness phase 1 admitted, not from the seed")
    cases = phase_cases(load_suite(Path(args.suite)), args.phase, args.seed, _heldout(args))
    store = TerminalStore(root)
    config = _config(
        args,
        cases,
        phase=args.phase,
        parent=parent.version,
        sides={c.id: c.split for c in cases},
        improver_model=args.improver_model,
        generations=args.generations,
        max_evals=args.max_evals,
        max_clusters=args.max_clusters,
        candidates_per_cluster=args.candidates_per_cluster,
    )
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = _client(args.agent_model, args.temperature)
    improver = _client(args.improver_model)
    signer = Signer.from_env() if os.environ.get("RATCHET_SIGNING_KEY") else Signer.generate()
    log_path = root / "log.txt"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    activator = MemoryActivator(store.decision)
    started = time.perf_counter()
    with GuardRuntime() as runtime:
        lab = Lab(
            cases=cases,
            run_case=RecordingRunner(TerminalRunner(runtime=runtime), store),
            simulator=RecordedViews(),
            runner=SandboxRunner(runtime),
            signer=signer,
            activator=activator,
            improver=improver,
            workers=args.workers,
            max_clusters=args.max_clusters,
            candidates_per_cluster=args.candidates_per_cluster,
            max_evals=args.max_evals,
            suite_digest=config["split_record"]["digest"],
            model_config={k: config[k] for k in ("agent_model", "temperature", "suite_commit", "package")},
            log=log,
        )
        result = evolve(store, model, args.generations, lab=lab, seed=parent)
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


def cmd_rates(args: argparse.Namespace) -> int:
    rows = report.collect(Path(args.out))
    sources = tuple(s for s in args.source.split(",") if s)
    sides = committed_record()["phases"]
    sets = {
        "phase 1 (12 tasks)": [case_id(n) for n in PHASE_1],
        "phase 2 (10 tasks)": [case_id(n) for n in PHASE_2],
        "union (22 tasks)": [case_id(n) for n in (*PHASE_1, *PHASE_2)],
        "phase-1 held-out (audit in phase 2)": [case_id(n) for n in sides["1"]["heldout"]],
    }
    out: dict[str, Any] = {}
    columns: list[tuple[str, report.Rate]] = []
    for harness in args.harness.split(","):
        results = report.results_for(rows, harness, sources)
        out[harness] = {}
        for label, ids in sets.items():
            r = report.rate(results, ids)
            out[harness][label] = {"text": r.text(), "per_task": r.per_task}
            columns.append((f"`{harness}` {label}", r))
    if not args.markdown:
        print(json.dumps(out, indent=2))
        return 0
    print(report.table(columns))
    union = [case_id(n) for n in (*PHASE_1, *PHASE_2)]
    per = [
        (f"`{h}`", report.rate(report.results_for(rows, h, sources), union)) for h in args.harness.split(",")
    ]
    print()
    print(report.per_task_table(per, union))
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--suite", default=DEFAULT_SUITE)
    p.add_argument("--agent-model", default="gpt-5.5")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--heldout", default="", help="held-out tasks per phase, e.g. 6,5 (default 4,4)")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ratchet.bench.terminal")
    p.add_argument("--out", default="private/bench/terminal")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--suite", default="")
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--heldout", default="")
    s.add_argument("--write", default="")
    s.set_defaults(func=cmd_split)
    o = sub.add_parser("oracle")
    o.add_argument("--suite", default=DEFAULT_SUITE)
    o.add_argument("--tasks", default="", help="comma-separated task names (default: both phases)")
    o.add_argument("--workers", type=int, default=4)
    o.set_defaults(func=cmd_oracle)
    t = sub.add_parser("trial")
    _common(t)
    t.add_argument("--harness", default="seed")
    t.add_argument("--tasks", default="", help="comma-separated task names")
    t.add_argument("--label", default="")
    choices = [
        "phase1",
        "phase2",
        "union",
        "phase1-train",
        "phase1-heldout",
        "phase2-train",
        "phase2-heldout",
    ]
    t.add_argument("--set", choices=[*choices, "phase2-audit"], default="union")
    t.add_argument("--trial", type=int, default=1)
    t.add_argument("--repeats", type=int, default=1)
    t.set_defaults(func=cmd_trial)
    e = sub.add_parser("evolve")
    _common(e)
    e.add_argument("--phase", type=int, choices=[1, 2], required=True)
    e.add_argument("--parent", default="seed")
    e.add_argument("--name", default="run-1")
    e.add_argument("--improver-model", default="gpt-5.5")
    e.add_argument("--generations", type=int, default=1)
    e.add_argument("--max-evals", type=int, default=3)
    e.add_argument("--max-clusters", type=int, default=4)
    e.add_argument("--candidates-per-cluster", type=int, default=3)
    e.set_defaults(func=cmd_evolve)
    r = sub.add_parser("rates")
    r.add_argument("--harness", required=True, help="comma-separated harness versions")
    r.add_argument("--source", default="", help="comma-separated source prefixes under --out")
    r.add_argument("--markdown", action="store_true", help="print tables instead of JSON")
    r.set_defaults(func=cmd_rates)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
