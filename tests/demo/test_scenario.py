"""The demo scenario admits through the real gates, rejects the bad guard, and labels every record."""

from __future__ import annotations

import base64
import json
from typing import Any

from ratchet_cases.types import Trace

from ratchet.agent.guards import GuardRuntime
from ratchet.demo.scenario import demo_cases, run_demo
from ratchet.demo.worker import NAME, ScriptedDemoWorker
from ratchet.improve.activate import MemoryActivator
from ratchet.improve.receipt import Signer
from ratchet.improve.sandbox import SandboxRunner
from ratchet.improve.worldsim import WorldSimulator
from ratchet.types import Harness
from ratchet.verify.dsse import parse


class MemStore:
    def __init__(self) -> None:
        self.harnesses: dict[str, Harness] = {}
        self.traces: dict[str, Trace] = {}
        self.evals: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}

    def get_harness(self, version: str) -> Harness | None:
        return self.harnesses.get(version)

    def put_harness(self, harness: Harness) -> bool:
        self.harnesses[harness.version] = harness
        return True

    def put_trace(self, trace: Trace) -> bool:
        self.traces[trace.run_id] = trace
        return True

    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str:
        self.evals.append(dict(report))
        return eval_id or str(len(self.evals))

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        self.by_id[str(decision_id)] = dict(decision)
        return str(decision_id)


def test_the_scripted_worker_is_deterministic_and_names_itself() -> None:
    msgs = [{"role": "user", "content": "Ticket T-1042: please help."}]
    a, b = ScriptedDemoWorker().complete(msgs, []), ScriptedDemoWorker().complete(msgs, [])
    assert a.message == b.message and a.tool_calls[0].name == "tickets_get"
    assert ScriptedDemoWorker().name == NAME == "scripted-demo-worker"


def test_the_demo_admits_one_guard_rejects_the_other_and_activates() -> None:
    store, signer = MemStore(), Signer.generate()
    activator = MemoryActivator(store.by_id.get)
    cases = demo_cases()
    with GuardRuntime() as runtime:
        r = run_demo(store, activator, WorldSimulator(cases), SandboxRunner(runtime), signer, cases=cases)
    assert r.admitted.admitted and not r.rejected.admitted
    assert not r.rejected.check("replay").passed
    fresh = r.admitted.check("heldout").evidence
    assert len(fresh["stable_recoveries"]) == 2 and not fresh["protected_broken"]
    assert fresh["parent_discordance"] == 0
    assert activator.active() == (r.admitted.candidate, 2)
    doc = store.by_id[r.admitted_id]
    payload = json.loads(parse(base64.b64decode(doc["envelope"])).payload)
    assert payload["evaluator"]["model_config"]["model"] == NAME
    assert payload["context"]["scenario"]["agent"] == NAME
    assert payload["evaluator"]["components"]["suite_file"]
    assert r.failed_trace.endswith("/refund_lost_reply.v1")
