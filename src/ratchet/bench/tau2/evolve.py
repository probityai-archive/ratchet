"""Run Ratchet's improvement loop on a tau2-bench domain and export the admitted harness.

`evolve` runs `ratchet.improve.loop.evolve` from the domain's seed harness over the domain's base tasks:
tau2-bench's train split is what the improver reads, its test split is the held-out side of every
admission. Every conversation the loop runs is kept twice, as a Ratchet trace in the run's store and as
tau2-bench's scored `SimulationRun` under `simulations/`. Decisions are signed with the pinned
decision key (from `RATCHET_SIGNING_KEY` or the keyring; any other key is refused) and activation follows
the decision log, as in every Ratchet run.

The run ends by writing `frozen.json`: the active harness and the signed decision that admitted it (or the
seed harness and no decision when nothing was admitted). That file is what a full run takes with
`--harness`, so the frozen harness is exactly the bytes the loop activated.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from ratchet_cases.types import Trace

import ratchet.bench.tau2  # noqa: F401  # sets TAU2_DATA_DIR before tau2-bench is imported
from ratchet.agent.guards import GuardRuntime
from ratchet.bench.tau.store import JsonlStore
from ratchet.bench.tau2.results import RunWriter
from ratchet.bench.tau2.runner import EndpointPool, Tau2Runner
from ratchet.bench.tau2.tasks import Tau2Case
from ratchet.improve.activate import MemoryActivator
from ratchet.improve.loop import Lab, evolve
from ratchet.improve.receipt import Signer
from ratchet.improve.referee import Referee, RefereeParts
from ratchet.improve.sandbox import SandboxRunner
from ratchet.improve.sim import RecordedViews
from ratchet.types import Harness

Log = Callable[[str], None]


class RecordingRunner:
    """A `RunCase` that keeps every conversation: the trace in the store, the simulation on disk."""

    def __init__(
        self, runner: Tau2Runner | EndpointPool, store: JsonlStore, writer: RunWriter, log: Log
    ) -> None:
        self.runner, self.store, self.writer, self.log = runner, store, writer, log
        self.done = 0
        self._lock = threading.Lock()

    def __call__(self, harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
        try:
            trace = self.runner(harness, case, model, store, run_id=run_id)
        except Exception as exc:
            self._tick(f"{harness.version} {case.id} ERROR {type(exc).__name__}: {str(exc)[:200]}")
            raise
        self.store.put_trace(trace)
        sim = self.runner.pop_simulation(run_id)
        if sim is not None:
            self.writer.put_simulation(sim)
        verdict = "PASS" if trace.outcome.passed else "fail"
        self._tick(f"{harness.version} {case.id} {verdict} {trace.outcome.reasons[:1]}")
        return trace

    def _tick(self, line: str) -> None:
        with self._lock:
            self.done += 1
            self.log(f"[{self.done}] {line}")


def signer_from_env() -> Signer:
    return Signer.for_decisions()


def referee_parts_for(
    root: Path,
    cases: list[Tau2Case],
    runner: Tau2Runner | EndpointPool,
    model: Any,
    config: dict[str, Any],
    workers: int,
) -> RefereeParts:
    """The referee of a tau2 run: the held-out cases, a store of its own under `root`, the decision key."""
    store = JsonlStore(root)
    stack = ExitStack()
    runtime = stack.enter_context(GuardRuntime())
    runner.runtime = runtime
    held = [c for c in cases if c.split == "heldout"]
    return RefereeParts(
        cases=held,
        run_case=RecordingRunner(runner, store, RunWriter(root), print),
        model=model,
        simulator=RecordedViews(),
        runner=SandboxRunner(runtime),
        signer=signer_from_env(),
        store=store,
        activator=MemoryActivator(store.decision),
        model_config=_model_config(config),
        suite_digest=config["tau2_data_digest"],
        workers=workers,
        close=stack.close,
    )


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: config[k] for k in ("agent_model", "user_model", "judge_model", "temperature")}


def frozen_record(
    store: JsonlStore, decisions: JsonlStore, active: str | None, seed: Harness
) -> dict[str, Any]:
    """The active harness (from the improver's store) and the decision that admitted it (from the referee's;
    none for the seed)."""
    harness = store.get_harness(active) if active else None
    if harness is None or harness.version == seed.version:
        return {"harness": asdict(seed), "decision_id": None, "decision": None}
    found = [
        r
        for r in decisions.read("decisions")
        if (r["decision"] or {}).get("candidate") == harness.version and (r["decision"] or {}).get("admitted")
    ]
    last = found[-1] if found else {"decision_id": None, "decision": None}
    return {"harness": asdict(harness), "decision_id": last["decision_id"], "decision": last["decision"]}


def run_evolve(
    root: Path,
    *,
    cases: list[Tau2Case],
    runner: Tau2Runner | EndpointPool,
    model: Any,
    improver: Any,
    seed: Harness,
    config: dict[str, Any],
    generations: int,
    workers: int,
    max_clusters: int,
    candidates_per_cluster: int,
    max_evals: int,
    referee: Referee,
    log: Log,
) -> dict[str, Any]:
    """Run the loop over the training `cases`; held-out runs, decisions and activation are the referee's,
    which keeps them under `root/referee/`."""
    store = JsonlStore(root)
    # The public half of the key that signs this run's decisions, so its receipts verify on their own.
    key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(referee.public_key()))
    pem = key.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    (root / "signer.pub").write_bytes(pem)
    recording = RecordingRunner(runner, store, RunWriter(root), log)
    started = time.perf_counter()
    with GuardRuntime() as runtime:
        runner.runtime = runtime
        lab = Lab(
            cases=cases,
            run_case=recording,
            simulator=RecordedViews(),
            runner=SandboxRunner(runtime),
            referee=referee,
            improver=improver,
            workers=workers,
            max_clusters=max_clusters,
            candidates_per_cluster=candidates_per_cluster,
            max_evals=max_evals,
            suite_digest=config["tau2_data_digest"],
            model_config=_model_config(config),
            log=log,
        )
        result = evolve(store, model, generations, lab=lab, seed=seed)
    active = referee.active()[0]
    frozen = frozen_record(store, JsonlStore(root / "referee"), active, seed)
    (root / "frozen.json").write_text(json.dumps(frozen, indent=2, default=str), encoding="utf-8")
    envelope = (frozen["decision"] or {}).get("envelope")
    if envelope:
        (root / "receipt.dsse.json").write_bytes(base64.b64decode(envelope))
    out = {
        "config": config,
        "final_harness": result.harness.version,
        "active": active,
        "frozen_decision": frozen["decision_id"],
        "instructions": result.instructions.version,
        "generations": [r.to_payload() for r in result.generations],
        "decision_public_key": referee.public_key(),
        "wall_s": round(time.perf_counter() - started, 1),
    }
    (root / "result.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
