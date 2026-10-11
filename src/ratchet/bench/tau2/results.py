"""Write a run in tau2-bench's own results format, and compute pass^k the way tau2-bench does.

A run directory holds tau2-bench's directory format (the format `tau2 run` writes under
`data/simulations/<run>/`): `results.json` (run info, tasks, simulation index) and `simulations/<id>.json`,
one scored `SimulationRun` per conversation, written the moment it ends. Next to them Ratchet keeps its
own records: `traces.jsonl` (every Ratchet trace), `runs.jsonl` (one row per trial unit, including units
lost to the model endpoint, which have no simulation) and `config.json`.

pass^k comes from tau2-bench's `compute_metrics` over the simulations. Units the endpoint lost after the
predeclared retries have no simulation, so tau2-bench's numbers cover completed conversations only; the
summary therefore also reports pass^k with every lost unit counted as a failure, and both denominators.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from math import comb
from pathlib import Path
from typing import Any

import ratchet.bench.tau2  # noqa: F401  # sets TAU2_DATA_DIR before tau2-bench is imported
from ratchet.types import Trace

SIMULATIONS = "simulations"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def tau2_info(config: Any, *, agent_implementation: str, commit: str) -> Any:
    """tau2-bench's run `Info` for `config`, naming the Ratchet harness as the agent implementation."""
    from tau2.runner.helpers import get_info  # type: ignore[import-untyped]  # no type info

    info = get_info(config)
    info.git_commit = commit
    info.agent_info.implementation = agent_implementation
    return info


class RunWriter:
    """Appends one finished unit at a time to a run directory. Safe to call from several threads."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        (self.root / SIMULATIONS).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _append(self, name: str, record: Any) -> None:
        line = json.dumps(record, sort_keys=True, default=str)
        with self._lock, (self.root / name).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def put_simulation(self, simulation: Any) -> None:
        _write_atomic(self.root / SIMULATIONS / f"{simulation.id}.json", simulation.model_dump_json(indent=2))

    def put_trace(self, trace: Trace) -> None:
        self._append("traces.jsonl", asdict(trace))

    def put_attempt(self, record: Mapping[str, Any]) -> None:
        self._append("attempts.jsonl", dict(record))

    def put_row(self, row: Mapping[str, Any]) -> None:
        self._append("runs.jsonl", dict(row))

    def rows(self) -> list[dict[str, Any]]:
        return read_jsonl(self.root / "runs.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_simulations(root: Path) -> list[Any]:
    from tau2.data_model.simulation import SimulationRun  # type: ignore[import-untyped]  # no type info

    files = sorted((Path(root) / SIMULATIONS).glob("*.json"))
    return [SimulationRun.model_validate_json(f.read_text(encoding="utf-8")) for f in files]


def finalize(root: Path, info: Any, tasks: Sequence[Any]) -> Any:
    """Write `results.json` (tau2-bench directory format) over every simulation file in `root`."""
    from tau2.data_model.simulation import Results

    sims = sorted(load_simulations(root), key=lambda s: (s.trial or 0, str(s.task_id)))
    results = Results(info=info, tasks=list(tasks), simulations=sims)
    results.save(Path(root), format="dir")
    return results


def load_results(root: Path) -> Any:
    from tau2.data_model.simulation import Results

    return Results.load(Path(root))


def tau2_metrics(results: Any) -> dict[str, Any]:
    """tau2-bench's own metrics: average reward and pass^k for every k up to the trial count."""
    from tau2.metrics.agent_metrics import compute_metrics  # type: ignore[import-untyped]  # no type info

    m = compute_metrics(results)
    return {
        "avg_reward": round(float(m.avg_reward), 4),
        "pass_hat_k": {int(k): round(float(v), 4) for k, v in m.pass_hat_ks.items()},
    }


def pass_hat_k(successes: Mapping[str, list[bool]], k: int) -> float:
    """Mean over tasks of C(c, k) / C(n, k): the chance k trials drawn without replacement all pass."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not successes:
        raise ValueError("no tasks")
    total = 0.0
    for task, runs in successes.items():
        if len(runs) < k:
            raise ValueError(f"task {task} has {len(runs)} trials, fewer than k={k}")
        total += comb(sum(runs), k) / comb(len(runs), k)
    return total / len(successes)


def by_task(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for r in rows:
        out.setdefault(str(r["task_id"]), []).append(bool(r["passed"]))
    return out


def summarize_run(root: Path) -> dict[str, Any]:
    """Counts, tau2-bench's pass^k over completed conversations, and pass^k with lost units as failures."""
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    rows = read_jsonl(root / "runs.jsonl")
    lost = [r for r in rows if r.get("infra_failed")]
    trials = int(config["trials"])
    out: dict[str, Any] = {
        "run": str(root),
        "domain": config["domain"],
        "harness": config["harness"]["version"],
        "agent_model": config["agent_model"],
        "user_model": config["user_model"],
        "tasks": config["tasks"],
        "trials": trials,
        "units_planned": config["tasks"] * trials,
        "units_recorded": len(rows),
        "simulations": len(rows) - len(lost),
        "successes": sum(1 for r in rows if r["passed"]),
        "lost_to_endpoint": len(lost),
    }
    if (root / "results.json").exists():
        out["tau2"] = tau2_metrics(load_results(root))
    grouped = by_task(rows)
    complete = all(len(v) == trials for v in grouped.values()) and len(grouped) == config["tasks"]
    if rows and complete:
        out["pass_hat_k_lost_as_fail"] = {k: round(pass_hat_k(grouped, k), 4) for k in range(1, trials + 1)}
    return out
