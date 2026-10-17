"""Recall against a real Atlas (the local container): real vector and keyword search, real stored decisions.

The scenario is built with the production code paths: `diagnose` for clusters, `make_candidate` for the
candidates, `replay` for the proofs, `admit` for the decisions and `GenerationRecord` for the history. Only
the agent and the guard sandbox are the test fakes the rest of the improver tests use.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import re
import secrets
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import quote, urlsplit

import pytest
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from ratchet_cases.types import Trace

from ratchet.improve.admit import admit
from ratchet.improve.diagnose import FailureCluster, diagnose
from ratchet.improve.evaluate import EvalReport, evaluate
from ratchet.improve.loop import GenerationRecord
from ratchet.improve.overfit import training_literals
from ratchet.improve.propose import SEED, Candidate, build_prompt, make_candidate
from ratchet.improve.recall import (
    PriorFix,
    Recall,
    RecallChat,
    cluster_of,
    cluster_of_prompt,
    failure_id,
    failure_text,
    render_prior_fixes,
)
from ratchet.improve.replay import ReplayReport, WalkCache, replay
from ratchet.store import atlas_admin
from ratchet.store.atlas import Store
from ratchet.store.embed import EMBEDDING_DIMS, InputType, VoyageEmbedder
from ratchet.types import Harness

from .fakes import (
    BLOCK_ALL_REFUNDS,
    NO_DUP_GUARD,
    FakeCase,
    FakeRunCase,
    FakeSimulator,
    LocalRunner,
    ScriptedChat,
    fake_cases,
)

LOCAL_URI = "mongodb://ratchet:ratchet-local-dev@127.0.0.1:27018/?directConnection=true"
UNAUTHORIZED = 13
_WORD = re.compile(r"[a-z0-9_]+")
TOOLS = ["payments_get_charge", "payments_refund", "payments_list_refunds"]
PARENT = Harness.build(
    parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=TOOLS, created_by="seed"
)
CASES = [
    *fake_cases(),
    FakeCase("omit-7", "train"),
    FakeCase("omit-8", "heldout"),
    FakeCase("omit-9", "heldout"),
]
TRAIN = sorted(c.id for c in CASES if c.split == "train")
HELDOUT = [c for c in CASES if c.split == "heldout"]
HELDOUT_IDS = {c.id for c in HELDOUT}
HELDOUT_MARKER = "heldout-sentinel-4f1c9e"


class HashEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words are close in cosine distance."""

    model = "test-hash-1024"

    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        return [self._one(t) for t in texts]

    @staticmethod
    def _one(text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIMS
        for word in _WORD.findall(text.lower()):
            vec[int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % EMBEDDING_DIMS] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def _atlas_up() -> str:
    client: MongoClient[dict[str, Any]] = MongoClient(LOCAL_URI, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except Exception:
        pytest.fail("local Atlas is not reachable; run: docker compose -f docker/compose.yaml up -d --wait")
    finally:
        client.close()
    return LOCAL_URI


@pytest.fixture
def store() -> Iterator[Store]:
    name = f"ratchet_test_{uuid.uuid4().hex[:10]}"
    s = Store(_atlas_up(), name, embedder=HashEmbedder())
    s.setup()
    yield s
    s._client.drop_database(name)
    s._client.drop_database(f"{name}_heldout")
    s.close()


@dataclass
class World:
    """The seeded history: clusters, candidates and the ids that must stay hidden."""

    parent_eval: EvalReport
    clusters: dict[str, FailureCluster]
    candidates: dict[str, Candidate]
    decisions: dict[str, dict[str, Any]]
    heldout_run_ids: set[str]


def _passing(pe: EvalReport) -> list[Trace]:
    return [t for t in pe.traces if t.outcome.passed]


def _decide(
    c: Candidate, pe: EvalReport, rerun: EvalReport, cluster: FailureCluster, generation: int
) -> tuple[dict[str, Any], ReplayReport | None]:
    walks = WalkCache(FakeSimulator())
    benign = [t for t in pe.traces if t.case_id.startswith("plain")]
    replays = {
        g.id: replay(g, cluster.members, cluster.traces, _passing(pe), walks, LocalRunner(), benign)
        for g in c.new_guards(PARENT)
    }
    composed = (
        replay(c.harness.guards, cluster.members, cluster.traces, _passing(pe), walks, LocalRunner(), benign)
        if c.harness.guards
        else None
    )
    ce = evaluate(c.harness, CASES, FakeRunCase(), None)
    confirm = evaluate(c.harness, HELDOUT, FakeRunCase(), None)
    d = admit(
        PARENT,
        c.harness,
        parent_runs=(pe, rerun),
        cand_eval=ce,
        confirm_eval=confirm,
        replays=replays,
        composed=composed,
        literals=training_literals(pe.train_traces()),
        visible_case_ids=TRAIN,
        evaluator={"version": "ev1"},
        context={"generation": generation, "arm": "champion", **c.to_payload()},
    )
    proof = None
    if c.kind == "guard":
        train_passing = [t for t in _passing(pe) if t.case_id in TRAIN]
        train_benign = [t for t in pe.traces if t.case_id in TRAIN and t.case_id.startswith("plain")]
        proof = replay(
            c.harness.guards[-1], cluster.members, cluster.traces, train_passing, walks, LocalRunner(),
            train_benign,
        )  # fmt: skip
    return d.to_payload(), proof


def _seed_world(store: Store) -> World:
    pe = evaluate(PARENT, CASES, FakeRunCase(), None)
    rerun = evaluate(PARENT, HELDOUT, FakeRunCase(), None)
    for t in pe.traces:
        store.put_trace(t)
    clusters = {c.key["mode"] + c.key["tool"]: c for c in diagnose(pe.train_traces(), FakeSimulator())}
    timeout, omit = clusters["commissionpayments_refund"], clusters["omission<end>"]
    raws: list[tuple[FailureCluster, dict[str, Any]]] = [
        (timeout, {"kind": "guard", "description": "no duplicate refund", "source": NO_DUP_GUARD}),
        (timeout, {"kind": "guard", "description": "block every refund", "source": BLOCK_ALL_REFUNDS}),
        (omit, {"kind": "procedure", "name": "refund", "when": "customer asks", "steps": ["get", "refund"]}),
        (omit, {"kind": "prompt_edit", "text": "Remember to refund the customer."}),
    ]
    rec = GenerationRecord(1, PARENT.version, clusters=[c.to_payload() for c in clusters.values()])
    candidates: dict[str, Candidate] = {}
    decisions: dict[str, dict[str, Any]] = {}
    for cluster, raw in raws:
        made = make_candidate(raw, cluster, PARENT, SEED)
        assert isinstance(made, Candidate), made
        payload, proof = _decide(made, pe, rerun, cluster, 1)
        store.put_decision(payload, decision_id=f"{PARENT.version}->{made.harness.version}")
        status = "admitted" if payload["admitted"] else "rejected: " + ", ".join(
            c["name"] for c in payload["checks"] if not c["passed"]
        )  # fmt: skip
        rec.candidates.append({**made.to_payload(), "arm": "champion", "status": status, "lint_hits": [],
                               "replay": proof.to_payload() if proof else None})  # fmt: skip
        candidates[made.change.get("id", "") or made.id] = made
        decisions[made.id] = payload
    store.put_eval(rec.to_payload(), eval_id=f"gen-{PARENT.version}-1")
    held = {t.run_id for t in pe.traces if t.case_id in HELDOUT_IDS}
    return World(pe, clusters, {c.id: c for c in candidates.values()}, decisions, held)


def _eventually(fn: Callable[[], Any], ok: Callable[[Any], bool], timeout_s: float = 60.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while True:
        got = fn()
        if ok(got) or time.monotonic() > deadline:
            return got
        time.sleep(0.25)


@pytest.fixture
def world(store: Store) -> World:
    return _seed_world(store)


def _recall(store: Store, w: World, **kw: Any) -> Recall:
    r = Recall(store, train_case_ids=TRAIN, **kw)
    r.remember(list(w.clusters.values()))
    return r


def _heldout_gate(w: World) -> str:
    """The name admission gives its held-out comparison: the check whose evidence lists stable recoveries."""
    decision = next(iter(w.decisions.values()))
    return str(next(c["name"] for c in decision["checks"] if "stable_recoveries" in c["evidence"]))


def _find(w: World, cluster: str) -> FailureCluster:
    return w.clusters[cluster]


def _recalled(r: Recall, cluster: FailureCluster, harness: Harness | None = None) -> Any:
    return _eventually(lambda: r.recall(cluster, harness), lambda x: x.considered >= 2)


def _by_kind(fixes: Sequence[PriorFix]) -> dict[str, PriorFix]:
    return {f.candidate_id: f for f in fixes}


def test_the_scenario_has_the_outcomes_the_tests_rely_on(world: World) -> None:
    by = {c.kind: world.decisions[c.id] for c in world.candidates.values()}
    assert by["procedure"]["admitted"] is True
    assert by["prompt_edit"]["admitted"] is False
    verdicts = {
        c.change.get("description"): (world.decisions[c.id]["admitted"]) for c in world.candidates.values()
    }
    assert verdicts["no duplicate refund"] is True and verdicts["block every refund"] is False


def test_failure_ids_round_trip_and_foreign_ids_are_ignored(world: World) -> None:
    for c in world.clusters.values():
        assert cluster_of(failure_id(c)) == c.id
    assert cluster_of("f-dup") is None and cluster_of("cluster-") is None and cluster_of("cluster-x-") is None


def test_remember_stores_only_training_clusters_and_is_idempotent(store: Store, world: World) -> None:
    held_cluster = replace(
        _find(world, "commissionpayments_refund"),
        id="heldoutcl0",
        members=[replace(m, case_id="timeout-4") for m in _find(world, "commissionpayments_refund").members],
    )
    r = Recall(store, train_case_ids=TRAIN)
    ids = r.remember([*world.clusters.values(), held_cluster])
    assert sorted(ids) == sorted(failure_id(c) for c in world.clusters.values())
    assert r.remember(list(world.clusters.values())) == ids
    assert store.db["failures"].count_documents({}) == 2
    assert r.remember([held_cluster]) == []
    doc = store.db["failures"].find_one({"_id": failure_id(_find(world, "omission<end>"))})
    assert doc is not None and doc["trace_ref"].endswith("/omit-7")


def test_recall_ranks_clean_admissions_first_then_rejections_with_their_reasons(
    store: Store, world: World
) -> None:
    r = _recall(store, world)
    got = _recalled(r, _find(world, "commissionpayments_refund"))
    fixes = got.fixes
    assert got.considered == 4 and got.omitted == 0 and len(fixes) == 4
    assert [f.outcome for f in fixes] == ["admitted_clean", "admitted_clean", "rejected", "rejected"]
    assert [f.rank for f in fixes] == [1, 2, 3, 4] and {f.of_total for f in fixes} == {4}
    # the two admissions tie on tier, so the retrieved cluster nearest the query comes first
    assert fixes[0].kind == "guard" and fixes[0].cluster_id == _find(world, "commissionpayments_refund").id
    guard = fixes[0]
    assert guard.helpful == 2 and guard.stable_recoveries == 2 and guard.harmful == 0
    assert guard.targets_hit == guard.targets_total == 2 and guard.false_fires == 0
    broad = next(f for f in fixes if f.description == "block every refund")
    assert broad.outcome == "rejected" and {"replay", "no_regression", "benign_twins"} <= set(
        broad.gates_failed
    )
    assert broad.false_fires > 0 and broad.harmful >= 3 and broad.benign_twin_interventions >= 1
    edit = next(f for f in fixes if f.kind == "prompt_edit")
    assert (
        edit.outcome == "rejected"
        and _heldout_gate(world) in edit.gates_failed
        and edit.text.startswith("Remember")
    )
    proc = next(f for f in fixes if f.kind == "procedure")
    assert (
        proc.outcome == "admitted_clean" and proc.steps == ("get", "refund") and proc.when == "customer asks"
    )


def test_an_admission_with_false_fires_ranks_below_a_clean_one(store: Store, world: World) -> None:
    guard = next(c for c in world.candidates.values() if c.change.get("description") == "no duplicate refund")
    noisy = copy.deepcopy(world.decisions[guard.id])
    next(c for c in noisy["checks"] if c["name"] == "replay")["evidence"]["composed"]["false_fires"] = 1
    store.put_decision(noisy, decision_id=f"{PARENT.version}->{guard.harness.version}-noisy")
    decisions = store.db["decisions"]
    decisions.delete_one({"_id": f"{PARENT.version}->{guard.harness.version}"})
    r = _recall(store, world)
    fixes = _recalled(r, _find(world, "commissionpayments_refund")).fixes
    assert [f.outcome for f in fixes][:2] == ["admitted_clean", "admitted"]
    assert fixes[1].composed_false_fires == 1 and fixes[1].candidate_id == guard.id


def test_held_out_material_never_reaches_a_prior_fix(store: Store, world: World) -> None:
    r = _recall(store, world)
    got = _recalled(r, _find(world, "commissionpayments_refund"))
    everything = repr(got.fixes) + render_prior_fixes(got.fixes)
    for run_id in world.heldout_run_ids:
        assert run_id not in everything
    for case_id in HELDOUT_IDS:
        assert case_id not in everything
    broad = next(f for f in got.fixes if f.description == "block every refund")
    shown = {i.split("@")[0] for i in broad.false_fire_ids}
    train_runs = {t.run_id for t in world.parent_eval.traces if t.case_id in TRAIN}
    assert shown and shown <= train_runs and not shown & world.heldout_run_ids
    assert broad.false_fires_withheld > 0 and broad.false_fires >= len(shown) + broad.false_fires_withheld
    assert any("no refunds" in reason for reason in broad.false_fire_reasons)


def test_an_unverifiable_trace_id_is_withheld_not_shown(store: Store, world: World) -> None:
    store.db["traces"].delete_many({})
    r = _recall(store, world)
    got = _recalled(r, _find(world, "commissionpayments_refund"))
    broad = next(f for f in got.fixes if f.description == "block every refund")
    referee_only = [f for f in got.fixes if f.decisions and f.false_fires]
    assert referee_only and all(f.false_fire_ids == () or f.false_fire_ids for f in got.fixes)
    # the proof replay on training traces names its case, so those ids survive; the referee's do not
    assert all(i.split("@")[0] not in world.heldout_run_ids for i in broad.false_fire_ids)
    assert broad.false_fires_withheld > 0


def test_nothing_is_shown_when_no_training_set_is_known(store: Store, world: World) -> None:
    store.db["decisions"].update_many({}, {"$set": {"visible_case_ids": []}})
    store.db["evals"].update_many({"type": "generation"}, {"$unset": {"candidates.$[].replay": ""}})
    r = Recall(store)
    r.remember([])
    r._store_train_loaded = True
    got = r.recall(_find(world, "commissionpayments_refund"))
    assert got.fixes == [] or all(f.false_fire_ids == () for f in got.fixes)


def test_training_cases_in_the_store_count_as_visible(store: Store, world: World) -> None:
    store.put_case({"id": "timeout-1", "split": "train", "family": "refund"})
    r = Recall(store)
    assert r.train_ids() == {"timeout-1"}
    assert r.train_ids() == {"timeout-1"}


def test_a_small_budget_omits_fixes_and_cuts_sources_and_says_so(store: Store, world: World) -> None:
    cluster = _find(world, "commissionpayments_refund")
    big = NO_DUP_GUARD + "# pad\n" * 400
    store.db["evals"].update_many(
        {"type": "generation"},
        {"$set": {"candidates.$[c].change.source": big}},
        array_filters=[{"c.kind": "guard"}],
    )
    roomy = _recalled(_recall(store, world, max_source_chars=800), cluster)
    full = len(render_prior_fixes(roomy.fixes))
    assert roomy.omitted == 0 and roomy.source_chars_omitted > 0
    r = _recall(store, world, budget_chars=full * 2 // 3, max_source_chars=800)
    got = _recalled(r, cluster)
    assert got.considered == 4 and got.omitted == 4 - len(got.fixes) and got.source_chars_omitted > 0
    assert "omitted" in got.statement
    text = render_prior_fixes(got.fixes)
    assert f"{len(got.fixes)} of 4 shown" in text and "source characters cut" in text
    assert len(text) <= full * 2 // 3
    none = _recalled(_recall(store, world, budget_chars=10), cluster)
    assert none.fixes == [] and none.omitted == 4 and render_prior_fixes(none.fixes) == ""


def test_a_source_too_long_for_the_budget_is_cut_further(store: Store, world: World) -> None:
    cluster = _find(world, "commissionpayments_refund")
    base = _recalled(_recall(store, world), cluster)
    guard = next(f for f in base.fixes if f.kind == "guard" and f.description == "no duplicate refund")
    long_source = NO_DUP_GUARD + "# pad pad pad pad pad pad\n" * 80
    store.db["evals"].update_many(
        {"type": "generation"},
        {"$set": {"candidates.$[c].change.source": long_source}},
        array_filters=[{"c.id": guard.candidate_id}],
    )
    first = _recalled(_recall(store, world), cluster)
    one = next(f for f in first.fixes if f.candidate_id == guard.candidate_id)
    assert one.source_chars_omitted == len(long_source) - 1500
    budget = len(render_prior_fixes([one])) - 300
    got = _recalled(_recall(store, world, budget_chars=budget), cluster)
    kept = next(f for f in got.fixes if f.candidate_id == guard.candidate_id)
    assert kept.source_chars_omitted > one.source_chars_omitted
    assert len(render_prior_fixes(got.fixes)) <= budget


def test_active_items_carry_helpful_and_harmful_counters(store: Store, world: World) -> None:
    good = next(c for c in world.candidates.values() if c.change.get("description") == "no duplicate refund")
    bad = next(c for c in world.candidates.values() if c.change.get("description") == "block every refund")
    proc = next(c for c in world.candidates.values() if c.kind == "procedure")
    both = PARENT.child(
        guards=[*good.harness.guards, *bad.harness.guards],
        procedures=proc.harness.procedures,
        created_by="improver",
    )
    composite = copy.deepcopy(world.decisions[good.id])
    composite["context"]["composite_of"] = [good.id, bad.id]
    store.put_decision(composite, decision_id="composite")
    r = _recall(store, world)
    got = _recalled(r, _find(world, "commissionpayments_refund"), both)
    active = {f.candidate_id: f for f in got.fixes if f.outcome == "active"}
    assert set(active) == {g.id for g in both.guards} | {p.id for p in both.procedures}
    g = active[good.harness.guards[-1].id]
    assert (g.helpful, g.harmful, g.decisions) == (2, 0, 1)
    b = active[bad.harness.guards[-1].id]
    assert b.helpful == 0 and b.harmful >= 3 and b.decisions == 1
    p = active[proc.harness.procedures[-1].id]
    assert p.helpful == 2 and p.kind == "procedure"
    text = render_prior_fixes(got.fixes)
    assert "Track record" in text and "helpful 2, harmful 0 over 1 decisions" in text
    assert render_prior_fixes([g]).count("Track record") == 1
    assert not [f for f in got.fixes[: -len(active)] if f.outcome == "active"]


def test_a_harness_without_items_or_none_has_no_counters(store: Store, world: World) -> None:
    r = _recall(store, world)
    cluster = _find(world, "commissionpayments_refund")
    assert not [f for f in _recalled(r, cluster, PARENT).fixes if f.outcome == "active"]
    assert not [f for f in _recalled(r, cluster).fixes if f.outcome == "active"]


def test_a_candidate_without_a_decision_is_a_lesson_with_its_status(store: Store, world: World) -> None:
    cluster = _find(world, "commissionpayments_refund")
    store.db["evals"].update_many(
        {"type": "generation"},
        {
            "$push": {
                "candidates": {
                    "id": "c-screened",
                    "kind": "guard",
                    "cluster_id": cluster.id,
                    "change": {
                        "id": "g-screened",
                        "description": "copies ids",
                        "source": "def check(a, b, c): ...",
                    },
                    "status": "screened: overfit_lint",
                    "replay": None,
                }
            }
        },
    )
    r = _recall(store, world)
    got = _recalled(r, cluster)
    lesson = _by_kind(got.fixes)["c-screened"]
    assert lesson.outcome == "rejected" and lesson.gates_failed == ("screened: overfit_lint",)
    assert lesson.decisions == 0 and lesson.false_fires == 0


def test_a_decision_without_a_generation_record_still_yields_a_fix(store: Store, world: World) -> None:
    store.db["evals"].delete_many({"type": "generation"})
    r = _recall(store, world)
    got = _recalled(r, _find(world, "commissionpayments_refund"))
    assert got.considered >= 3 and all(f.generation == 1 for f in got.fixes)
    assert {f.kind for f in got.fixes} >= {"guard", "procedure"}


def test_recall_with_an_empty_memory_returns_nothing(store: Store, world: World) -> None:
    store.db["failures"].delete_many({})
    got = Recall(store, train_case_ids=TRAIN).recall(_find(world, "commissionpayments_refund"))
    assert got.fixes == [] and got.considered == 0 and got.retrieved_clusters == []
    assert "0 of 0" in got.statement


def test_foreign_failure_documents_are_ignored(store: Store, world: World) -> None:
    store.put_failure("duplicate refund after timeout", case_family="refund", failure_id="not-ours")
    r = _recall(store, world)
    got = _recalled(r, _find(world, "commissionpayments_refund"))
    assert got.considered == 4 and "not-ours" not in repr(got.retrieved_clusters)


def test_rendered_section_states_its_bounds_and_names_every_kind(store: Store, world: World) -> None:
    r = _recall(store, world)
    got = _recalled(r, _find(world, "commissionpayments_refund"))
    text = render_prior_fixes(got.fixes)
    assert text.startswith("Prior fixes for similar failures (4 of 4 shown")
    needles = (
        "ADMITTED, no false fires",
        "REJECTED",
        "procedure refund (when customer asks)",
        "prompt text: Remember",
        "source:",
        "wrongly blocked training runs",
        "not training runs",
    )
    for needle in needles:
        assert needle in text, needle
    assert render_prior_fixes([]) == ""


def test_the_failure_text_names_the_mechanism(world: World) -> None:
    c = _find(world, "omission<end>")
    assert c.summary in failure_text(c) and "reason:" in failure_text(c)


def _with_user(uri: str, user: str, password: str) -> str:
    sp = urlsplit(uri)
    creds = f"{user}:{quote(password, safe='')}"
    return f"mongodb://{creds}@{sp.hostname}:{sp.port}/?directConnection=true&authSource=admin"


@pytest.fixture
def improver(store: Store) -> Iterator[Store]:
    """A store connected as a local user whose role mirrors ratchetImprover: readWrite on the main db only."""
    admin = store._client.admin
    role, user, password = f"imp_{uuid.uuid4().hex[:8]}", f"u_{uuid.uuid4().hex[:8]}", secrets.token_hex(12)
    admin.command("createRole", role, privileges=[], roles=[{"role": "readWrite", "db": store.db.name}])
    admin.command("createUser", user, pwd=password, roles=[{"role": role, "db": "admin"}])
    s = Store(_with_user(LOCAL_URI, user, password), store.db.name, embedder=HashEmbedder())
    yield s
    s.close()
    admin.command("dropUser", user)
    admin.command("dropRole", role)


def test_recall_under_the_improver_identity_cannot_see_held_out_data_and_matches_the_admin_view(
    store: Store, world: World, improver: Store
) -> None:
    store.put_case({"id": "timeout-1", "split": "train", "family": "refund", "customer_message": "refund"})
    store.put_case(
        {"id": "timeout-4", "split": "heldout", "family": "refund", "customer_message": HELDOUT_MARKER}
    )
    with pytest.raises(OperationFailure) as denied:
        improver.cases("heldout")
    assert denied.value.code == UNAUTHORIZED
    cluster = _find(world, "commissionpayments_refund")
    as_admin = _recalled(_recall(store, world), cluster)
    r = Recall(improver, train_case_ids=TRAIN)
    r.remember(list(world.clusters.values()))
    as_improver = _eventually(
        lambda: r.recall(cluster),
        lambda got: [f.similarity for f in got.fixes] == [f.similarity for f in as_admin.fixes],
    )
    from dataclasses import asdict

    assert [f.candidate_id for f in as_improver.fixes] == [f.candidate_id for f in as_admin.fixes]
    for mine, theirs in zip(as_improver.fixes, as_admin.fixes, strict=True):
        differing = {k: (v, asdict(theirs)[k]) for k, v in asdict(mine).items() if v != asdict(theirs)[k]}
        assert differing == {}
    everything = repr(as_improver.fixes) + render_prior_fixes(as_improver.fixes)
    assert HELDOUT_MARKER not in everything
    assert not any(run_id in everything for run_id in world.heldout_run_ids)
    assert not any(case in everything for case in HELDOUT_IDS)
    shown = {i.split("@")[0] for f in as_improver.fixes for i in f.false_fire_ids}
    assert not shown & world.heldout_run_ids


def test_the_improver_can_write_failure_memory_but_not_the_held_out_database(
    store: Store, world: World, improver: Store
) -> None:
    r = Recall(improver, train_case_ids=TRAIN)
    assert len(r.remember(list(world.clusters.values()))) == 2
    with pytest.raises(OperationFailure) as denied:
        improver.heldout["cases"].insert_one({"_id": "x"})
    assert denied.value.code == UNAUTHORIZED


def test_the_search_is_hybrid_and_returns_only_clusters_we_wrote(store: Store, world: World) -> None:
    r = _recall(store, world)
    omit = _find(world, "omission<end>")
    got = _recalled(r, omit)
    assert omit.id in got.retrieved_clusters and all(isinstance(c, str) for c in got.retrieved_clusters)
    near = max(got.fixes, key=lambda f: f.similarity)
    assert near.similarity > 0.0


def test_priorfix_tier_orders_unknown_outcomes_last() -> None:
    base = PriorFix("guard", "c", "k", "s", "active")
    assert base.tier == 3 and replace(base, outcome="admitted_clean").tier == 0


def test_a_prompt_gives_back_the_cluster_it_was_written_for(world: World) -> None:
    c = _find(world, "commissionpayments_refund")
    prompt = build_prompt(c, PARENT, SEED, [], None, 3)
    back = cluster_of_prompt(prompt)
    assert back is not None and back.id == c.id and back.summary == c.summary and back.key == c.key
    assert failure_text(back) == failure_text(c) and failure_id(back) == failure_id(c)
    assert cluster_of_prompt("nothing to see") is None
    broken = prompt.replace("Mechanism key: {", "Mechanism key: {{")
    assert cluster_of_prompt(broken) is None


def test_recall_chat_adds_the_section_before_the_ask_and_only_to_proposals(
    store: Store, world: World
) -> None:
    c = _find(world, "commissionpayments_refund")
    r = _recall(store, world)
    _recalled(r, c)
    inner = ScriptedChat([{"candidates": []}, {"mechanism": "x"}])
    chat = RecallChat(inner, r, PARENT)
    prompt = build_prompt(c, PARENT, SEED, [], None, 3)
    chat.complete([{"role": "system", "content": "s"}, {"role": "user", "content": prompt}], [])
    sent = str(inner.requests[0][-1]["content"])
    assert "Prior fixes for similar failures" in sent and sent.endswith(
        prompt[prompt.rfind("\n\nPropose ") :]
    )
    assert sent.index("Prior fixes") < sent.index("Propose 3 candidates")
    assert chat.sections[c.id].considered == 4
    chat.complete([{"role": "user", "content": "Summarise this failure"}], [])
    assert inner.requests[1][-1]["content"] == "Summarise this failure"
    other = Recall(store, train_case_ids=TRAIN)
    quiet = ScriptedChat([{"candidates": []}])
    store.db["failures"].delete_many({})
    RecallChat(quiet, other).complete([{"role": "user", "content": prompt}], [])
    assert quiet.requests[0][-1]["content"] == prompt
    unparsed = ScriptedChat([{}])
    RecallChat(unparsed, other).complete([{"role": "user", "content": "x\n\nPropose 2 candidates."}], [])
    assert unparsed.requests[0][-1]["content"] == "x\n\nPropose 2 candidates."


def _keyring(account: str) -> str:
    out = subprocess.run(
        ["secret-tool", "lookup", "service", "mongodb", "account", account], capture_output=True, text=True
    ).stdout.strip()
    assert out, f"keyring entry {account} is missing"
    return out


def test_sandbox_recall_as_the_real_improver_never_returns_held_out_material() -> None:
    """The restricted `ratchet-improver` user on the hackathon cluster: seeded by the admin, read by the
    improver, with a held-out case and a held-out trace id the improver must never see.

    It writes to the shared cluster, so it runs only when RATCHET_TEST_SANDBOX=1.
    """
    if os.environ.get("RATCHET_TEST_SANDBOX") != "1":
        pytest.skip("shared Atlas cluster: set RATCHET_TEST_SANDBOX=1 to run")
    admin_uri = _keyring("hackathon-uri")
    host = urlsplit(admin_uri).hostname
    improver_uri = (
        f"mongodb+srv://{atlas_admin.IMPROVER_USER}:{quote(_keyring('hackathon-db-improver'), safe='')}"
        f"@{host}/?retryWrites=true&w=majority"
    )
    voyage = subprocess.run(
        ["secret-tool", "lookup", "service", "voyageai", "account", "api-key"], capture_output=True, text=True
    ).stdout.strip()
    if voyage:
        os.environ["VOYAGE_API_KEY"] = voyage
    sentinel = f"sentinel-{uuid.uuid4().hex[:8]}"
    embedder = VoyageEmbedder() if voyage else HashEmbedder()
    with Store(admin_uri, "ratchet", embedder=embedder) as admin:
        w = _seed_world(admin)
        admin.put_case({"id": sentinel, "split": "heldout", "family": "refund", "customer_message": sentinel})
        cleanup: list[tuple[str, dict[str, Any]]] = [
            ("traces", {"run_id": {"$in": [t.run_id for t in w.parent_eval.traces]}}),
            ("decisions", {"parent": PARENT.version}),
            ("evals", {"_id": f"gen-{PARENT.version}-1"}),
            ("failures", {"_id": {"$in": [failure_id(c) for c in w.clusters.values()]}}),
        ]
        try:
            with Store(improver_uri, "ratchet", embedder=embedder) as improver:
                with pytest.raises(OperationFailure) as denied:
                    improver.cases("heldout")
                assert denied.value.code == UNAUTHORIZED
                r = Recall(improver, train_case_ids=TRAIN)
                assert len(r.remember(list(w.clusters.values()))) == 2
                cluster = _find(w, "commissionpayments_refund")
                got = _eventually(lambda: r.recall(cluster), lambda g: g.considered >= 4, 120.0)
                assert got.considered == 4 and got.fixes[0].outcome == "admitted_clean"
                everything = repr(got.fixes) + render_prior_fixes(got.fixes)
                assert sentinel not in everything
                assert not any(run_id in everything for run_id in w.heldout_run_ids)
                assert not any(case in everything for case in HELDOUT_IDS)
        finally:
            for collection, query in cleanup:
                admin.db[collection].delete_many(query)
            admin.heldout["cases"].delete_one({"_id": sentinel})
            getattr(embedder, "close", lambda: None)()
