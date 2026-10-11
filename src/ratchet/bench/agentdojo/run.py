"""Run AgentDojo cases through Ratchet's worker loop, replay them for diagnosis, and keep every record.

`DojoRunner.run_case` has the signature the improver's evaluator calls, so the same object serves a baseline
evaluation and `ratchet.improve.loop.evolve`. `DojoSimulator` rebuilds an episode and re-applies its executed
calls, grading with AgentDojo's checks after each one, so diagnosis can name the call after which the
attacker's goal first held. `JsonStore` writes harnesses, traces, evaluations and decisions as JSON lines.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from agentdojo.agent_pipeline.agent_pipeline import load_system_message

from ratchet.agent.model import ChatModel
from ratchet.agent.worker import RunLimits
from ratchet.agent.worker import run_case as worker_run_case
from ratchet.bench.agentdojo.cases import DojoSuite
from ratchet.bench.agentdojo.world import DojoWorld, outcome
from ratchet.cases.model import Case
from ratchet.improve.sim import Step, Walk, result_ids, tool_call_of
from ratchet.types import Harness, Trace, to_jsonable
from ratchet.world import World

# AgentDojo's pipeline makes one model call, then up to 15 tool-execution rounds of one call each.
DOJO_LIMITS = RunLimits(max_steps=16)


def seed_harness(dojo: DojoSuite) -> Harness:
    """Harness v0 for a suite: AgentDojo's default system message, every suite tool, no guards."""
    return Harness.build(
        parent=None,
        system_prompt=str(load_system_message(None)),
        procedures=[],
        guards=[],
        tool_allowlist=[str(f.name) for f in dojo.suite.tools],
        created_by="seed",
    )


class DojoRunner:
    """Runs one case in a fresh episode. Every run builds its own world; nothing is shared across runs."""

    def __init__(self, dojo: DojoSuite, limits: RunLimits = DOJO_LIMITS) -> None:
        self.dojo = dojo
        self.limits = limits

    def world(self, case_id: str) -> DojoWorld:
        return DojoWorld(self.dojo.suite, self.dojo.tasks[case_id])

    def run_case(self, harness: Harness, case: Case, model: ChatModel, store: Any, *, run_id: str) -> Trace:
        world = self.world(case.id)
        # The worker loop is typed against the simulated company's World; DojoWorld provides the five
        # members the loop calls (tools, tool_names, call, view, digest), and the grader below reads the
        # DojoWorld it closes over rather than the argument.
        return worker_run_case(
            harness,
            case,
            model,
            None,
            run_id=run_id,
            limits=self.limits,
            world_factory=lambda _case: cast(World, world),
            grader=lambda _case, _world, trace: outcome(world, trace),
        )


class DojoSimulator:
    """Rebuild an episode from its case id and re-apply each executed call, grading after every one."""

    def __init__(self, runner: DojoRunner) -> None:
        self.runner = runner

    def walk(self, trace: Trace) -> Walk:
        world = self.runner.world(trace.case_id)
        done = result_ids(trace.events)
        initial = tuple(world.grade("").reasons())
        steps: list[Step] = []
        for i, e in enumerate(trace.events):
            if e.kind != "tool_call":
                continue
            call = tool_call_of(e)
            recorded = e.payload.get("view")
            view = recorded if isinstance(recorded, dict) else world.snapshot()
            executed = call.call_id in done
            if executed:
                world.call(call)
            after = tuple(world.grade("").reasons())
            steps.append(Step(e.seq, call, view, trace.events[:i], executed, after))
        return Walk(trace.run_id, steps, initial, True)


class JsonStore:
    """The improver's store protocol over JSON-lines files in one directory. Thread-safe appends."""

    FILES = ("harnesses.jsonl", "traces.jsonl", "evals.jsonl", "decisions.jsonl")

    def __init__(self, root: Path) -> None:
        """A store over an empty run directory. Appending a second run to an old one would mix their
        records, so a directory that already holds any record file is refused."""
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        used = [name for name in self.FILES if (root / name).exists()]
        if used:
            raise FileExistsError(f"{root} already holds a run ({', '.join(used)}); use a new directory")
        self._lock = threading.Lock()
        self._evals = 0
        self._harnesses: set[str] = set()
        self._traces: set[str] = set()
        self._decisions: dict[str, dict[str, Any]] = {}

    def _append(self, name: str, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, default=str)
        with self._lock, (self.root / name).open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def put_harness(self, harness: Harness) -> bool:
        if harness.version in self._harnesses:
            return False
        self._harnesses.add(harness.version)
        self._append("harnesses.jsonl", asdict(harness))
        return True

    def put_trace(self, trace: Trace) -> bool:
        if trace.run_id in self._traces:
            return False
        self._traces.add(trace.run_id)
        self._append("traces.jsonl", asdict(trace))
        return True

    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str:
        with self._lock:
            self._evals += 1
            eid = eval_id or f"eval-{self._evals:04d}"
        self._append("evals.jsonl", {"eval_id": eid, **_payload(report)})
        return eid

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        did = decision_id or f"decision-{self._evals:04d}"
        record = {"decision_id": did, **_payload(decision)}
        self._append("decisions.jsonl", record)
        self._decisions[did] = record
        return did

    def decision(self, decision_id: str) -> dict[str, Any] | None:
        """A decision stored by this run, for the activator's binding check."""
        return self._decisions.get(decision_id)


def _payload(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_payload"):
        out: dict[str, Any] = obj.to_payload()
        return out
    value = to_jsonable(obj)
    return dict(value) if isinstance(value, Mapping) else {"value": value}
