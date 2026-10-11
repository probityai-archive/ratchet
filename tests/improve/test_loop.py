from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from typing import Any

from ratchet.improve.activate import MemoryActivator
from ratchet.improve.loop import Lab, compose, evolve
from ratchet.improve.receipt import Signer, verify
from ratchet.types import Harness, Trace

from .fakes import (
    BLOCK_ALL_REFUNDS,
    NO_DUP_GUARD,
    FakeCase,
    FakeRunCase,
    FakeSimulator,
    FakeTurn,
    LocalRunner,
)

CASES = [
    FakeCase("timeout-1", "train"), FakeCase("timeout-2", "train"),
    FakeCase("plain-3", "train", kind="benign_twin"),
    FakeCase("omit-7", "train"),
    FakeCase("timeout-4", "heldout"), FakeCase("timeout-5", "heldout"),
    FakeCase("plain-6", "heldout", kind="benign_twin"), FakeCase("omit-9", "heldout"),
    FakeCase("omit-10", "heldout"),
]  # fmt: skip
HELDOUT_IDS = ["timeout-4", "timeout-5", "plain-6", "omit-9", "omit-10"]

SEED = Harness.build(
    parent=None, system_prompt="You handle refunds.", procedures=[], guards=[],
    tool_allowlist=["payments_get_charge", "payments_list_refunds", "payments_refund"], created_by="seed",
)  # fmt: skip


class RoutingChat:
    """Answers by request type. Only instructions containing IMPROVED yield a useful procedure."""

    def __init__(self) -> None:
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> FakeTurn:
        self.requests.append([dict(m) for m in messages])
        system, user = str(messages[0]["content"]), str(messages[-1]["content"])
        if "failure mechanism in one sentence" in system:
            return FakeTurn(json.dumps({"mechanism": "summarised"}))
        if "maintain the instructions" in system:
            return FakeTurn(json.dumps({"text": "IMPROVED: always propose a procedure for omissions."}))
        if "Return the corrected guard" in user:
            return FakeTurn(json.dumps({"description": "narrowed", "source": NO_DUP_GUARD}))
        if "duplicate refund" in user:
            return FakeTurn(json.dumps({"candidates": [
                {"kind": "guard", "description": "too broad", "source": BLOCK_ALL_REFUNDS},
                {"kind": "prompt_edit", "text": "Be careful with refunds."},
            ]}))  # fmt: skip
        if "IMPROVED" in user:
            return FakeTurn(json.dumps({"candidates": [
                {"kind": "procedure", "name": "finish the refund", "when": "a refund is owed",
                 "steps": ["payments_get_charge", "payments_refund"]},
            ]}))  # fmt: skip
        return FakeTurn(json.dumps({"candidates": [{"kind": "prompt_edit", "text": "Think harder."}]}))

    def prompts(self) -> str:
        return "\n".join(str(m.get("content")) for r in self.requests for m in r)


class MemStore:
    def __init__(self) -> None:
        self.harnesses: dict[str, Harness] = {}
        self.traces: dict[str, Trace] = {}
        self.evals: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}

    def put_harness(self, harness: Harness) -> bool:
        new = harness.version not in self.harnesses
        self.harnesses[harness.version] = harness
        return new

    def put_trace(self, trace: Trace) -> bool:
        self.traces[trace.run_id] = trace
        return True

    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str:
        self.evals.append(dict(report))
        return eval_id or str(len(self.evals))

    def decision(self, decision_id: str) -> dict[str, Any] | None:
        return self.by_id.get(decision_id)

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        self.decisions.append(dict(decision))
        self.by_id[decision_id or str(len(self.decisions))] = dict(decision)
        return decision_id or str(len(self.decisions))


def _lab(store: MemStore, chat: RoutingChat, signer: Signer, **kw: Any) -> Lab:
    kw = {"max_clusters": 1, **kw}
    return Lab(
        cases=CASES,
        run_case=FakeRunCase(),
        simulator=FakeSimulator(),
        runner=LocalRunner(),
        signer=signer,
        activator=MemoryActivator(store.decision),
        improver=chat,
        workers=4,
        max_evals=4,
        suite_digest="suite",
        **kw,
    )


def test_two_generations_ratchet_both_levels() -> None:
    chat, store, signer = RoutingChat(), MemStore(), Signer.generate()
    logs: list[str] = []
    result = evolve(
        store, model=None, generations=3, lab=_lab(store, chat, signer, log=logs.append), seed=SEED
    )
    g1, g2, g3 = result.generations

    # Generation 1, level 1: the broad guard fails replay, is repaired into the precise one, and is admitted.
    guard_row = next(c for c in g1.candidates if c["kind"] == "guard")
    assert guard_row["status"] == "admitted"
    assert guard_row["replay"]["passed"] and guard_row["replay"]["targets_hit"] == 2
    assert g1.promoted is not None
    assert g1.level1["child_score"]["heldout_passed"] >= g1.level1["parent_run_2_heldout_passed"] + 2
    assert g1.level1["rates"]["candidate"]["false_intervention_rate"] == 0.0
    assert g1.replay["checks"] > 0 and g1.replay["checks_per_s"] > 0
    # Level 2 makes no claim without a challenger, and drafts one from what it saw.
    assert g1.level2["claim"] is False and g1.level2["next_challenger"] is not None
    dup = g1.clusters[0]["id"]

    # Generation 2: the omission cluster is unseen by both versions, so it is the frozen batch. Only the
    # challenger's procedure is admitted, so the challenger becomes the champion.
    assert g2.level2["claim"] is True and dup not in g2.level2["frozen_batch"]["cluster_ids"]
    assert g2.level2["seen"][g2.level2["challenger"]] == [dup]
    fs = g2.level2["frozen_score"]
    assert fs["challenger"]["admitted"] == 1 and fs["champion"]["admitted"] == 0
    assert g2.level2["kept"] is True and g2.level2["champion_after"] == g2.level2["challenger"]
    assert result.instructions.text.startswith("IMPROVED")
    assert g2.promoted == result.harness.version and result.harness.procedures and result.harness.guards

    assert g3.note.startswith("no training failures")
    # Each promotion was activated by compare-and-swap after its receipt was stored.
    assert g1.level1["activation"]["revision"] == 2 and g2.level1["activation"]["revision"] == 3
    assert g2.level1["activation"]["previous"] == g1.promoted

    # Every decision carries a receipt that verifies, admitted and rejected alike, and names what the
    # improver could see and the frozen suite.
    assert store.decisions and all(
        verify(base64.b64decode(d["envelope"]), {signer.keyid: signer.public_key})["candidate"]
        == d["candidate"]
        for d in store.decisions
    )
    assert any(not d["admitted"] for d in store.decisions)
    assert all(
        d["visible_case_ids"] == ["omit-7", "plain-3", "timeout-1", "timeout-2"] for d in store.decisions
    )
    versions = {d["evaluator"]["version"] for d in store.decisions}
    assert len(versions) == 1 and all(
        d["evaluator"]["components"]["suite_file"] == "suite" for d in store.decisions
    )
    roles = {e.get("role") for e in store.evals if e["type"] == "eval"}
    assert {"parent_run_1", "parent_run_2", "candidate", "confirm"} <= roles
    assert {e["type"] for e in store.evals} >= {"eval", "generation", "improver_instructions", "evaluator"}
    l2 = fs["challenger"]
    assert l2["slots"] == 4 and l2["invalid"] == 0 and l2["empty"] == 0 and fs["champion"]["rejected"] == 1
    assert logs and result.harness.version in store.harnesses


def test_the_improver_never_sees_heldout_cases() -> None:
    chat = RoutingChat()
    store = MemStore()
    evolve(store, model=None, generations=2, lab=_lab(store, chat, Signer.generate()), seed=SEED)
    text = chat.prompts()
    assert "timeout-1" in text or "omit-7" in text  # training material is shown
    for case_id in HELDOUT_IDS:
        assert case_id not in text


def test_meta_failure_is_tolerated() -> None:
    class NoMeta(RoutingChat):
        def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> FakeTurn:
            if "maintain the instructions" in str(messages[0]["content"]):
                self.requests.append([dict(m) for m in messages])
                return FakeTurn("no json here")
            return super().complete(messages, tools)

    store = MemStore()
    result = evolve(store, None, 1, lab=_lab(store, NoMeta(), Signer.generate()), seed=SEED)
    assert result.generations[0].level2["next_challenger"] is None
    assert result.generations[0].level1["parent_score"]["train_total"] == 4


def test_compose_merges_changes() -> None:
    from ratchet.improve.propose import Candidate

    g = SEED.child(system_prompt=SEED.system_prompt + "\n\nX", created_by="improver")
    c = Candidate("c", "prompt_edit", "k", "v", g, {"append": "X"}, "")
    both = compose(SEED, [c, c])
    assert both.system_prompt.count("X") == 2 and both.parent == SEED.version


def test_fixes_for_two_clusters_are_composed_when_the_composite_does_better() -> None:
    class Both(RoutingChat):
        def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> FakeTurn:
            user = str(messages[-1]["content"])
            if "Propose" in user and "not refunded" in user:
                messages = [*messages[:-1], {"role": "user", "content": user + "\nIMPROVED"}]
            return super().complete(messages, tools)

    store = MemStore()
    result = evolve(store, None, 1, lab=_lab(store, Both(), Signer.generate(), max_clusters=4), seed=SEED)
    g1 = result.generations[0]
    assert (
        g1.level1["child_score"]["heldout_passed"] == 5
        and result.harness.guards
        and result.harness.procedures
    )
    assert any("composite_of" in d["context"] for d in store.decisions)


def test_audit_cases_are_never_run() -> None:
    class Recording(FakeRunCase):
        def __init__(self) -> None:
            super().__init__()
            self.seen: set[str] = set()

        def __call__(self, harness: Any, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
            self.seen.add(case.id)
            return super().__call__(harness, case, model, store, run_id=run_id)

    run = Recording()
    store = MemStore()
    lab = _lab(store, RoutingChat(), Signer.generate())
    lab.cases = [*CASES, FakeCase("timeout-99", "audit")]
    lab.run_case = run
    evolve(store, None, 1, lab=lab, seed=SEED)
    assert run.seen and "timeout-99" not in run.seen


LEAKY_GUARD = """
import json


def check(call, view, history):
    if call.name != "payments_refund":
        return None
    charge = call.args.get("charge_id", "")
    if any(r["charge_id"] == charge and r["status"] == "succeeded" for r in view["refunds"]):
        return "duplicate"
    if int(charge.split("_")[1]) >= 4:
        return "LEAK " + json.dumps(view)
    return None
"""


def test_a_guard_that_leaks_protected_inputs_through_its_reason_never_reaches_the_improver() -> None:
    import re

    class Leaky(RoutingChat):
        def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> FakeTurn:
            user = str(messages[-1]["content"])
            if "Propose" in user and "duplicate refund" in user:
                self.requests.append([dict(m) for m in messages])
                return FakeTurn(
                    json.dumps({"candidates": [{"kind": "guard", "description": "d", "source": LEAKY_GUARD}]})
                )
            return super().complete(messages, tools)

    chat, store = Leaky(), MemStore()
    result = evolve(store, None, 2, lab=_lab(store, chat, Signer.generate()), seed=SEED)
    leaked_row = next(c for c in result.generations[0].candidates if c["kind"] == "guard")
    assert leaked_row["replay"]["verdict"] == "CLEARED"  # it hides on training traces
    decision = next(d for d in store.decisions if d["context"].get("kind") == "guard")
    assert not decision["admitted"]
    fires = decision["checks"][0]["evidence"]["reports"][0]["false_fires"]
    assert fires and all(f.startswith("run-") for f in fires)  # the referee caught it on held-out traces
    protected = re.compile(r"\bch_(4|5|6|9|10)\b")
    referee = [e for e in store.evals if e["type"] == "replay" and e["role"] == "referee"]
    leaks = [fp["reason"] for e in referee for fp in e["false_positives"] if fp["reason"].startswith("LEAK")]
    assert leaks and protected.search(" ".join(leaks))  # the leak attempt really carried protected data
    assert not protected.search(chat.prompts())  # and none of it reached the improver
