from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ratchet.agent.guards import GuardRuntime
from ratchet.improve import instruction_search as search
from ratchet.improve.diagnose import FailureCluster, FailurePoint, diagnose
from ratchet.improve.instruction_search import (
    Bench,
    CandidateOutcome,
    ClusterOutcome,
    Firewall,
    HeldOutRefusedError,
    InstructionsAdapter,
    ReflectionLM,
    Screen,
    compare_level2,
    optimize_instructions,
    record_run,
    reflection_record,
    search_challenger,
    split_clusters,
)
from ratchet.improve.llm import Chat, Meter
from ratchet.improve.overfit import training_literals
from ratchet.improve.propose import SEED, Instructions, Proposal, make_candidate, propose
from ratchet.improve.receipt import Signer
from ratchet.improve.replay import GuardRunner, ReplayReport, WalkCache
from ratchet.improve.sandbox import SandboxRunner
from ratchet.types import Cost, Guard, Harness, Trace

from .fakes import (
    BLOCK_ALL_REFUNDS,
    NO_DUP_GUARD,
    FakeRunCase,
    FakeSimulator,
    FakeTurn,
    LocalRunner,
    ScriptedChat,
    bad_timeout,
    good_plain,
    good_timeout,
)

PARENT = Harness.build(
    parent=None,
    system_prompt="You handle refunds.",
    procedures=[],
    guards=[],
    tool_allowlist=["payments_get_charge", "payments_list_refunds", "payments_refund"],
    created_by="seed",
)


@dataclass(frozen=True)
class Case:
    id: str
    split: Any
    family: str
    kind: str = "trap"
    must_succeed: bool = True


CASES = [
    Case("timeout-1", "train", "dup"),
    Case("timeout-2", "train", "dup"),
    Case("plain-3", "train", "dup", kind="benign_twin"),
    Case("omit-7", "train", "omit"),
    Case("omit-8", "train", "omit"),
    Case("timeout-4", "heldout", "dup"),
    Case("timeout-5", "heldout", "dup"),
    Case("plain-6", "heldout", "dup", kind="benign_twin"),
    Case("omit-9", "heldout", "omit"),
    Case("omit-10", "heldout", "omit"),
    Case("omit-11", "audit", "omit"),
]


class Router:
    """The improver for these tests. Instructions containing IMPROVED yield good candidates."""

    def __init__(self) -> None:
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> FakeTurn:
        self.requests.append([dict(m) for m in messages])
        system, user = str(messages[0]["content"]), str(messages[-1]["content"])
        if "Provide the new instructions within" in user:
            return FakeTurn(
                "Here you go:\n```\nIMPROVED: key guards on refunds already at the provider.\n```"
            )
        if "failure mechanism in one sentence" in system:
            return FakeTurn(json.dumps({"mechanism": "summarised"}))
        if "maintain the instructions" in system:
            return FakeTurn(json.dumps({"text": "CHALLENGER: be careful.", "rationale": "r"}))
        good = "IMPROVED" in user
        if "duplicate refund" in user:
            guard = NO_DUP_GUARD if good else BLOCK_ALL_REFUNDS
            items: list[dict[str, Any]] = [{"kind": "guard", "description": "d", "source": guard}]
            if not good:
                items.append({"kind": "prompt_edit", "text": "Be careful with refunds."})
            return FakeTurn(json.dumps({"candidates": items}))
        if good:
            proc = {
                "kind": "procedure",
                "name": "finish the refund",
                "when": "a refund is owed",
                "steps": ["payments_get_charge", "payments_refund"],
            }
            return FakeTurn(json.dumps({"candidates": [proc]}))
        return FakeTurn(json.dumps({"candidates": [{"kind": "prompt_edit", "text": "Think harder."}]}))

    def proposals(self) -> int:
        return sum(1 for r in self.requests if "You are the improver" in str(r[0]["content"]))


class MemStore:
    def __init__(self) -> None:
        self.traces: dict[str, Trace] = {}
        self.evals: list[dict[str, Any]] = []
        self.decisions: dict[str, dict[str, Any]] = {}

    def put_trace(self, trace: Trace) -> bool:
        self.traces[trace.run_id] = trace
        return True

    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str:
        self.evals.append(dict(report))
        return eval_id or f"e{len(self.evals)}"

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        key = decision_id or str(len(self.decisions))
        self.decisions[key] = dict(decision)
        return key


def train_traces() -> list[Trace]:
    return [
        bad_timeout(1, "bad-1"),
        bad_timeout(2, "bad-2"),
        good_timeout(3, "good-3"),
        good_plain(3, "plain-3"),
    ]


def _cluster(traces: list[Trace]) -> FailureCluster:
    return diagnose(traces, FakeSimulator())[0]


def _screen(traces: list[Trace], runner: GuardRunner | None = None) -> Screen:
    passing = [t for t in traces if t.outcome.passed]
    return Screen(
        PARENT,
        passing,
        [t for t in passing if t.case_id.startswith("plain")],
        WalkCache(FakeSimulator()),
        runner or LocalRunner(),
        training_literals(traces),
    )


TRAIN_IDS = frozenset({"timeout-1", "timeout-2", "timeout-3", "plain-3"})


def _adapter(traces: list[Trace], chat: Chat, **kw: Any) -> InstructionsAdapter:
    return InstructionsAdapter(screen=_screen(traces), proposer=chat, firewall=Firewall(TRAIN_IDS), **kw)


def test_firewall_refuses_heldout_clusters_frozen_clusters_and_foreign_traces() -> None:
    traces = train_traces()
    cluster = _cluster(traces)
    fw = Firewall.from_cases([Case("timeout-1", "train", "dup"), Case("timeout-2", "train", "dup")])
    fw.check_clusters([cluster])
    with pytest.raises(HeldOutRefusedError, match="non-training cases"):
        Firewall(frozenset({"timeout-1"})).check_clusters([cluster])
    with pytest.raises(HeldOutRefusedError, match="frozen level-2 test batch"):
        Firewall(TRAIN_IDS, frozenset({cluster.id})).check_clusters([cluster])
    with pytest.raises(HeldOutRefusedError, match="clearance traces"):
        InstructionsAdapter(
            screen=_screen([*traces, good_plain(9, "p9")]), proposer=Router(), firewall=Firewall(TRAIN_IDS)
        )


def test_search_refuses_a_heldout_cluster_before_any_proposal() -> None:
    traces = train_traces()
    chat = Router()
    held = _cluster([bad_timeout(4, "bad-4")])  # case timeout-4 is not in the training set
    adapter = _adapter(traces, chat)
    with pytest.raises(HeldOutRefusedError):
        optimize_instructions([held], adapter, chat, max_metric_calls=4)
    with pytest.raises(HeldOutRefusedError):
        adapter.evaluate([held], {"instructions": SEED.text})
    assert chat.requests == []
    with pytest.raises(ValueError, match="at least one"):
        optimize_instructions([], adapter, chat, max_metric_calls=4)


def _fake_propose(guards: Sequence[str], extra: Sequence[dict[str, Any]] = (), dup: int = 0) -> Any:
    def fn(
        cluster: FailureCluster,
        harness: Harness,
        model: Chat,
        *,
        instructions: Instructions,
        passing: Sequence[Trace],
        walks: WalkCache | None,
        meter: Meter | None,
        n: int,
    ) -> Proposal:
        assert meter is not None
        meter.add(FakeTurn("x", 11, 5))
        out = Proposal(duplicates=dup)
        for i, src in enumerate(guards):
            made = make_candidate(
                {"kind": "guard", "description": f"g{i}", "source": src}, cluster, harness, instructions
            )
            assert not isinstance(made, str)
            out.candidates.append(made)
        for raw in extra:
            made = make_candidate(raw, cluster, harness, instructions)
            if isinstance(made, str):
                out.rejected.append({"reason": made})
            else:
                out.candidates.append(made)
        return out

    return fn


def _fake_replay(verdicts: Mapping[str, ReplayReport]) -> Any:
    def fn(
        guards: Guard | Sequence[Guard],
        targets: Sequence[FailurePoint],
        failing: Mapping[str, Trace],
        passing: Iterable[Trace],
        walks: WalkCache,
        runner: GuardRunner,
        benign: Iterable[Trace] = (),
    ) -> ReplayReport:
        assert isinstance(guards, Guard)
        return verdicts[str(guards.description)]

    return fn


def _report(gid: str, *, hit: bool, fps: int = 0, errors: int = 0) -> ReplayReport:
    targets = [
        {
            "failure_id": "bad-1@3",
            "case_id": "timeout-1",
            "failing_seq": 3,
            "first_block_seq": 3 if hit else None,
            "hit": hit,
        }
    ]
    false_positives = [
        {"run_id": "plain-3", "case_id": "plain-3", "seq": 3, "reason": "no refunds", "benign_twin": True}
        for _ in range(fps)
    ]
    errs = [{"run_id": "good-3", "seq": 1, "error": "KeyError"} for _ in range(errors)]
    passed = hit and not fps and not errors
    return ReplayReport(gid, passed, targets, false_positives, errs, [], 3, 9, 0.01, 0.0, traces_checked=3)


def test_adapter_scores_cleared_guards_over_requested_slots() -> None:
    traces = train_traces()
    cluster = _cluster(traces)
    screen = _screen(traces)
    screen.replay_fn = _fake_replay(
        {
            "g0": _report("a", hit=True),
            "g1": _report("b", hit=False, fps=1),
            "g2": _report("c", hit=True, errors=1),
        }
    )
    extra: list[dict[str, Any]] = [
        {"kind": "procedure", "name": "p", "when": "w", "steps": ["s"]},
        {"kind": "prompt_edit", "text": "Never refund ch_1 twice."},
        {"kind": "procedure"},
    ]
    adapter = InstructionsAdapter(
        screen=screen,
        proposer=Router(),
        firewall=Firewall(TRAIN_IDS),
        n=8,
        propose_fn=_fake_propose(
            [NO_DUP_GUARD, BLOCK_ALL_REFUNDS, "def check(call, view, history):\n    return None\n"],
            extra,
            dup=1,
        ),
    )
    batch = adapter.evaluate([cluster], {"instructions": SEED.text}, capture_traces=True)
    assert batch.scores == [1 / 8]
    out = batch.outputs[0]
    assert [c.status for c in out.candidates] == [
        "cleared",
        "replay_failed",
        "replay_incomplete",
        "not_replayable",
        "screened",
    ]
    assert out.cleared == 1 and out.rejected and out.duplicates == 1
    assert out.proposer == {"model_calls": 1, "input_tokens": 11, "output_tokens": 5}
    assert adapter.meter.input_tokens == 11
    assert batch.trajectories == batch.outputs
    assert adapter.evaluations[0]["instructions_version"] == SEED.version
    assert adapter.evaluate([cluster], {"instructions": SEED.text}).trajectories is None

    rows = adapter.make_reflective_dataset({"instructions": SEED.text}, batch, ["instructions"])
    feedback = rows["instructions"][0]["Feedback"]
    assert "Score 0.12: 1 of 8" in feedback
    assert "MISSED target bad-1@3" in feedback
    assert "FALSE FIRE in benign-twin run plain-3" in feedback
    assert "ERROR in run good-3" in feedback
    assert "SCREENED OUT, copies training literals ['ch_1']" in feedback
    assert "natural-language trigger" in feedback
    assert "invalid item: procedure rejected" in feedback
    assert "1 duplicate candidate(s)" in feedback
    generated = rows["instructions"][0]["Generated Outputs"]
    assert any(g.startswith("procedure p (when w)") for g in generated)
    assert any(g.startswith("prompt_edit:") for g in generated)


def test_false_fire_feedback_names_the_blocked_call() -> None:
    traces = train_traces()
    cluster = _cluster(traces)
    adapter = _adapter(traces, Router(), propose_fn=_fake_propose([BLOCK_ALL_REFUNDS]))
    out = adapter.evaluate([cluster], {"instructions": SEED.text}, capture_traces=True).outputs[0]
    cand = out.candidates[0]
    assert cand.status == "replay_failed" and cand.blocked_calls
    assert cand.blocked_calls[0]["call"]["name"] == "payments_refund"
    assert '"name": "payments_refund"' in reflection_record(out)["Feedback"]


def test_empty_proposal_and_score_with_no_slots() -> None:
    out = ClusterOutcome("c", "v", 0, [], [], 0, {})
    assert out.score == 0.0
    rec = reflection_record(out)
    assert rec["Generated Outputs"] == "(none)" and "no usable candidate" in rec["Feedback"]
    assert CandidateOutcome("x", "guard", "cleared", {}).to_payload()["status"] == "cleared"


def test_a_changed_text_is_a_new_content_addressed_version_descending_from_the_seed() -> None:
    adapter = _adapter(train_traces(), Router())
    assert adapter.instructions_for(SEED.text) is SEED
    a = adapter.instructions_for(SEED.text + "\nKey on provider state.")
    b = adapter.instructions_for(SEED.text + "\nKey on provider state!")
    assert a.version != b.version != SEED.version
    assert a.parent == SEED.version and a == Instructions.build(a.text, parent=SEED.version)
    assert set(adapter.texts) == {SEED.version, a.version, b.version}


def test_reflection_lm_counts_real_tokens_and_accepts_messages() -> None:
    chat = ScriptedChat(["one", "two"])
    lm = ReflectionLM(chat)
    assert lm("hello") == "one"
    assert lm([{"role": "user", "content": "hi"}]) == "two"
    assert chat.requests[0] == [{"role": "user", "content": "hello"}]
    assert (lm.total_tokens_in, lm.total_tokens_out, lm.total_cost) == (14, 6, 0.0)


def test_search_improves_instructions_and_records_the_pareto_front() -> None:
    traces = train_traces()
    cluster = _cluster(traces)
    chat = Router()
    adapter = _adapter(traces, chat, n=2)
    run = optimize_instructions([cluster], adapter, chat, max_metric_calls=6, use_merge=False)
    assert run.improved and run.instructions.text.startswith("IMPROVED")
    assert run.instructions.parent == SEED.version
    rec = run.record
    assert rec["seed_version"] == SEED.version and rec["best_version"] == run.instructions.version
    versions = [c["version"] for c in rec["candidates"]]
    assert versions[0] == SEED.version and run.instructions.version in versions
    assert rec["candidates"][0]["per_cluster"] == {cluster.id: 0.0}
    best = next(c for c in rec["candidates"] if c["version"] == run.instructions.version)
    assert best["per_cluster"] == {cluster.id: 0.5} and best["parents"] == [SEED.version]
    assert rec["pareto_front"] == {cluster.id: [run.instructions.version]}
    assert rec["metric_calls"] >= 2 and rec["reflection_cost"]["model_calls"] >= 1
    assert rec["proposer_cost"]["model_calls"] == chat.proposals()

    store = MemStore()
    rid = record_run(store, run, generation=2, context={"experiment": "x"})
    assert store.evals[0]["type"] == "instruction_search_run" and store.evals[0]["experiment"] == "x"
    ins = store.evals[1]
    assert ins["type"] == "improver_instructions" and ins["search_run"] == rid
    assert ins["version"] == run.instructions.version and ins["seen_clusters"] == [cluster.id]
    assert ins["experiment"] == "x"


def test_search_challenger_hook_returns_none_when_nothing_beats_the_champion() -> None:
    traces = train_traces()
    cluster = _cluster(traces)
    store = MemStore()

    class Stubborn(Router):
        def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> FakeTurn:
            if "Provide the new instructions within" in str(messages[-1]["content"]):
                self.requests.append([dict(m) for m in messages])
                return FakeTurn("```\nStill the same idea, reworded.\n```")
            return super().complete(messages, tools)

    kw: dict[str, Any] = {
        "parent": PARENT,
        "train": traces,
        "benign_ids": {"plain-3"},
        "walks": WalkCache(FakeSimulator()),
        "runner": LocalRunner(),
        "firewall": Firewall(TRAIN_IDS),
        "n": 2,
        "max_metric_calls": 4,
    }
    assert search_challenger(store, SEED, [cluster], improver=Stubborn(), **kw) is None
    got = search_challenger(store, SEED, [cluster], improver=Router(), generation=3, **kw)
    assert got is not None and got.text.startswith("IMPROVED")
    assert store.evals[-1]["generation"] == 3


def test_integration_real_sandbox_on_two_recorded_traces() -> None:
    traces = [bad_timeout(1, "bad-1"), good_plain(3, "plain-3")]
    cluster = _cluster(traces)
    chat = ScriptedChat(
        [
            {
                "candidates": [
                    {"kind": "guard", "description": "no duplicate refund", "source": NO_DUP_GUARD},
                    {"kind": "guard", "description": "too broad", "source": BLOCK_ALL_REFUNDS},
                ]
            }
        ]
    )
    with GuardRuntime(workers=1) as rt:
        screen = _screen(traces, SandboxRunner(rt))
        adapter = InstructionsAdapter(
            screen=screen, proposer=chat, firewall=Firewall(TRAIN_IDS), n=2, propose_fn=propose
        )
        out = adapter.evaluate([cluster], {"instructions": SEED.text}, capture_traces=True).outputs[0]
    assert [c.status for c in out.candidates] == ["cleared", "replay_failed"]
    assert out.score == 0.5
    assert out.candidates[1].blocked_calls[0]["benign_twin"] is True


def test_c1_flips_count_both_directions_against_either_parent_run() -> None:
    ev = {
        "outcomes": {
            "a": ["fail", "fail", "pass", "invalid"],
            "b": ["pass", "fail", "fail", "pass"],
            "c": ["fail", "pass", "pass", "pass"],
            "d": ["fail", "fail", "fail", "fail"],
        }
    }
    assert search.c1_flips(ev) == (["a"], ["b"])
    rep = search.flips_report([ev, {}])
    assert rep["candidates"] == 2 and rep["c1_fail_to_pass_total"] == 1 and rep["c1_pass_to_fail_total"] == 1
    assert rep["candidates_with_2plus_recoveries_and_no_c1_break"] == 0


def test_split_clusters_keeps_mixed_clusters_out_of_both_sides() -> None:
    def cl(cid: str, *cases: str) -> FailureCluster:
        return FailureCluster(cid, {}, [FailurePoint(f"r-{c}", c, 1, "x", "commission", "t") for c in cases])

    family = {"a1": "A", "a2": "A", "b1": "B"}
    a, b, mixed = split_clusters(
        [cl("x", "a1", "a2"), cl("y", "b1"), cl("z", "a1", "b1")], family, frozenset({"A"})
    )
    assert [c.id for c in a] == ["x"] and [c.id for c in b] == ["y"] and mixed == ["z"]


def _bench(store: MemStore, chat: Chat, **kw: Any) -> Bench:
    args: dict[str, Any] = {"families_a": frozenset({"dup"}), "n": 2, "workers": 4, "search_calls": 6} | kw
    return Bench(
        cases=CASES,
        run_case=FakeRunCase(),
        simulator=FakeSimulator(),
        runner=LocalRunner(),
        signer=Signer.generate(),
        model=None,
        improver=chat,
        store=store,
        seed=PARENT,
        **args,
    )


def test_three_arm_comparison_under_the_real_admission_rule() -> None:
    store = MemStore()
    chat = Router()
    logs: list[str] = []
    out = compare_level2(_bench(store, chat, log=logs.append), "exp")
    assert out["claim"] is True
    assert out["budget"]["slots_per_arm"] == 2 and out["budget"]["clusters"] == 1
    frozen = out["frozen_batch"]["cluster_ids"]
    assert len(frozen) == 1 and out["train_clusters"] and not set(frozen) & set(out["train_clusters"])
    arms = out["arms"]
    assert arms["search"]["admitted"] == 1 and arms["search"]["admission_yield"] == "1/2"
    assert arms["search"]["heldout_flips_fail_to_pass"] == ["omit-10", "omit-9"]
    assert arms["search"]["heldout_flips_pass_to_fail"] == []
    flips = arms["search"]["c1_flips_all_evaluated"]
    assert flips["candidates"] == 1 and flips["c1_fail_to_pass_cases"] == ["omit-10", "omit-9"]
    assert flips["candidates_with_2plus_recoveries_and_no_c1_break"] == 1
    assert arms["seed"]["admitted"] == 0 and sum(arms["seed"]["rejections_by_gate"].values()) >= 1
    assert arms["challenger"]["instructions_version"] != SEED.version
    assert arms["challenger"]["admitted"] == 0
    for name in ("seed", "challenger", "search"):
        assert arms[name]["proposer_tokens"]["model_calls"] == 1
    der = out["derivation"]
    assert der["challenger"]["unchanged"] is False and der["challenger"]["meta_cost"]["model_calls"] == 1
    assert der["challenger"]["batch_a"]["replay_failed"] == 1
    assert der["search"]["improved"] is True
    kinds = {e.get("type") for e in store.evals}
    assert {
        "level2_frozen_batch",
        "instruction_search_run",
        "improver_instructions",
        "level2_comparison",
    } <= kinds
    assert any(k.startswith("exp:search:") for k in store.decisions)
    assert all("envelope" in d for d in store.decisions.values())
    assert any(k.startswith("exp:derive-challenger:") for k in store.decisions)
    admitted = [d for d in store.decisions.values() if d["admitted"]]
    assert len(admitted) == 1 and admitted[0]["context"]["arm"] == "search"
    # no searched or drafted text ever saw the frozen batch: every proposal prompt for it came after both
    search_prompts = [
        r for r in chat.requests if "Provide the new instructions within" in str(r[-1]["content"])
    ]
    assert search_prompts and all(frozen[0] not in str(r) for r in search_prompts)
    assert logs and "frozen test (B)" in logs[0]


def test_comparison_without_clusters_on_both_sides_makes_no_claim() -> None:
    store = MemStore()
    out = compare_level2(_bench(store, Router(), families_a=frozenset({"dup", "omit"})), "exp2")
    assert out["claim"] is False and "B=0" in out["reason"]
    assert store.evals[-1]["type"] == "level2_comparison"


class _Ctx:
    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_main_runs_the_comparison_and_writes_the_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import ratchet.agent.guards as guards_mod
    import ratchet.agent.model as model_mod
    import ratchet.agent.seed as seed_mod
    import ratchet.agent.worker as worker_mod
    import ratchet.cases as cases_mod
    import ratchet.improve.sandbox as sandbox_mod
    import ratchet.improve.worldsim as worldsim_mod
    import ratchet.store as store_mod

    class FakeStore(MemStore, _Ctx):
        def __init__(self, uri: str, db: str) -> None:
            super().__init__()

        def setup(self, *, wait_search: bool = True) -> None:
            return None

    class FakeClient(Router, _Ctx):
        def __init__(self, cfg: Any) -> None:
            super().__init__()

        def cost(self) -> Cost:
            return Cost(model_calls=len(self.requests))

    class FakeRuntime(_Ctx):
        pass

    monkeypatch.setattr(store_mod, "Store", FakeStore)
    monkeypatch.setattr(model_mod, "ModelClient", FakeClient)
    monkeypatch.setattr(guards_mod, "GuardRuntime", FakeRuntime)
    monkeypatch.setattr(sandbox_mod, "SandboxRunner", lambda rt: LocalRunner())
    monkeypatch.setattr(worker_mod, "run_case", FakeRunCase())
    monkeypatch.setattr(cases_mod, "all_cases", lambda: CASES)
    monkeypatch.setattr(worldsim_mod, "WorldSimulator", lambda cases: FakeSimulator())
    monkeypatch.setattr(seed_mod, "seed_harness", lambda: PARENT)
    monkeypatch.setenv("RATCHET_API_KEY", "k")
    monkeypatch.delenv("RATCHET_SIGNING_KEY", raising=False)
    target = tmp_path / "l2.json"
    argv = [
        "--uri",
        "mongodb://x",
        "--families-a",
        "dup",
        "--candidates",
        "2",
        "--search-calls",
        "6",
        "--experiment",
        "e",
        "--out",
        str(target),
    ]
    assert search.main(argv) == 0
    written = json.loads(target.read_text())
    assert written["claim"] is True and written["config"]["db"] == "ratchet_l2"
    assert written["arms"]["search"]["admitted"] == 1


def test_main_refuses_without_store_or_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RATCHET_MONGODB_URI", raising=False)
    monkeypatch.delenv("RATCHET_API_KEY", raising=False)
    assert search.main(["--out", str(tmp_path / "x.json")]) == 2
