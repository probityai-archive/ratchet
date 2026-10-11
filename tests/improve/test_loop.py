from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from typing import Any

from ratchet.improve.evaluate import evaluate
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
]  # fmt: skip
HELDOUT_IDS = ["timeout-4", "timeout-5", "plain-6", "omit-9"]

SEED = Harness.build(
    parent=None, system_prompt="You handle refunds.", procedures=[], guards=[],
    tool_allowlist=["get_charge", "list_refunds", "refund"], created_by="seed",
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
                 "steps": ["get_charge", "refund"]},
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

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        self.decisions.append(dict(decision))
        return decision_id or str(len(self.decisions))


def _lab(chat: RoutingChat, signer: Signer, **kw: Any) -> Lab:
    kw = {"max_clusters": 1, **kw}
    return Lab(cases=CASES, run_case=FakeRunCase(), simulator=FakeSimulator(), runner=LocalRunner(),
               signer=signer, improver=chat, workers=4, max_evals=4, suite_digest="suite", **kw)  # fmt: skip


def test_two_generations_ratchet_both_levels() -> None:
    chat, store, signer = RoutingChat(), MemStore(), Signer.generate()
    logs: list[str] = []
    result = evolve(store, model=None, generations=3, lab=_lab(chat, signer, log=logs.append), seed=SEED)
    g1, g2, g3 = result.generations

    # Generation 1, level 1: the broad guard fails replay, is repaired into the precise one, and is admitted.
    guard_row = next(c for c in g1.candidates if c["kind"] == "guard")
    assert guard_row["status"] == "admitted"
    assert guard_row["replay"]["passed"] and guard_row["replay"]["targets_hit"] == 2
    assert g1.promoted is not None
    assert g1.level1["child_score"]["heldout_passed"] > g1.level1["parent_rerun_heldout_passed"]
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
    assert all(d["suite_digest"] == "suite" for d in store.decisions)
    roles = {e.get("role") for e in store.evals if e["type"] == "eval"}
    assert {"parent", "parent_rerun", "candidate"} <= roles
    assert {e["type"] for e in store.evals} >= {"eval", "generation", "improver_instructions"}
    assert logs and result.harness.version in store.harnesses


def test_the_improver_never_sees_heldout_cases() -> None:
    chat = RoutingChat()
    evolve(MemStore(), model=None, generations=2, lab=_lab(chat, Signer.generate()), seed=SEED)
    text = chat.prompts()
    assert "timeout-1" in text or "omit-7" in text  # training material is shown
    for case_id in HELDOUT_IDS:
        assert case_id not in text


def test_parent_eval_is_reused_and_meta_failure_is_tolerated() -> None:
    class NoMeta(RoutingChat):
        def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> FakeTurn:
            if "maintain the instructions" in str(messages[0]["content"]):
                self.requests.append([dict(m) for m in messages])
                return FakeTurn("no json here")
            return super().complete(messages, tools)

    run = FakeRunCase()
    pe = evaluate(SEED, CASES, run, None)
    lab = _lab(NoMeta(), Signer.generate())
    result = evolve(MemStore(), None, 1, lab=lab, seed=SEED, parent_eval=pe)
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
    result = evolve(store, None, 1, lab=_lab(Both(), Signer.generate(), max_clusters=4), seed=SEED)
    g1 = result.generations[0]
    assert (
        g1.level1["child_score"]["heldout_passed"] == 4
        and result.harness.guards
        and result.harness.procedures
    )
    assert any("composite_of" in d["context"] for d in store.decisions)
