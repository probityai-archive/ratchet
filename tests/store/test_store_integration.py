"""Integration tests against the local Atlas container (mongod + mongot)."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest
from conftest import HashEmbedder

from ratchet.store import IntegrityError, Store, VoyageEmbedder, atlas_setup
from ratchet.types import Cost, Event, Guard, Harness, Outcome, Procedure, Trace

pytestmark = pytest.mark.integration


def _seed() -> Harness:
    return Harness.build(
        parent=None,
        system_prompt="seed",
        procedures=[],
        guards=[],
        tool_allowlist=["refund"],
        created_by="seed",
    )


def _trace(
    run_id: str = "r1", case_id: str = "c1", version: str = "v", passed: bool = True, n: int = 3
) -> Trace:
    events = [
        Event(seq=i, kind="tool_call", payload={"i": i, "pad": "x" * 50}, ts=float(i)) for i in range(n)
    ]
    out = Outcome(
        passed=passed, reasons=[] if passed else ["bad"], final_state_digest="d", cost=Cost(tool_calls=n)
    )
    return Trace(
        run_id=run_id, case_id=case_id, harness_version=version, model="m", events=events, outcome=out
    )


def _eventually(fn: Any, timeout_s: float = 30.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while True:
        result = fn()
        if result or time.monotonic() > deadline:
            return result
        time.sleep(0.25)


def test_setup_is_idempotent_and_search_indexes_become_queryable(store: Store) -> None:
    store.setup()
    store.setup()
    assert atlas_setup.search_indexes_ready(store.db)
    assert set(atlas_setup.COLLECTIONS) <= set(store.db.list_collection_names())
    names = {i["name"] for i in store.db["failures"].list_search_indexes()}
    assert names == {atlas_setup.FAILURES_VECTOR_INDEX, atlas_setup.FAILURES_TEXT_INDEX}


def _vector_index(store: Store) -> dict[str, Any]:
    found = {i["name"]: dict(i) for i in store.db["failures"].list_search_indexes()}
    return found[atlas_setup.FAILURES_VECTOR_INDEX]


def test_setup_replaces_a_changed_search_definition_and_leaves_an_unchanged_one(store: Store) -> None:
    store.setup()
    version = _vector_index(store)["latestVersion"]
    atlas_setup.ensure_schema(store.db)
    assert _vector_index(store)["latestVersion"] == version
    vector, text = atlas_setup.SEARCH_INDEXES
    narrowed = (*vector[:3], {"fields": vector[3]["fields"][:2]})
    atlas_setup.ensure_schema(store.db, (narrowed, text))
    assert len(_vector_index(store)["latestDefinition"]["fields"]) == 2
    atlas_setup.ensure_schema(store.db)
    assert len(_vector_index(store)["latestDefinition"]["fields"]) == 3


def test_put_harness_is_idempotent_and_verified_on_read(store: Store) -> None:
    seed = _seed()
    assert store.put_harness(seed) is True
    assert store.put_harness(seed) is False
    assert store.db["harness_versions"].count_documents({}) == 1
    assert store.get_harness(seed.version) == seed
    assert store.get_harness("nope") is None
    store.db["harness_versions"].update_one({"_id": seed.version}, {"$set": {"system_prompt": "tampered"}})
    with pytest.raises(IntegrityError):
        store.get_harness(seed.version)


def test_put_harness_rejects_same_version_with_other_content(store: Store) -> None:
    seed = _seed()
    store.put_harness(seed)
    store.db["harness_versions"].update_one({"_id": seed.version}, {"$set": {"digest": "other"}})
    with pytest.raises(IntegrityError, match="different content"):
        store.put_harness(seed)


def test_harness_roundtrips_procedures_and_guards(store: Store) -> None:
    g = Guard(
        id="g1",
        description="d",
        source="def check(call, view, history):\n    return None\n",
        origin_failure_ids=["f1"],
    )
    p = Procedure(id="p1", name="n", when="w", steps=["a", "b"], origin_failure_ids=["f1"])
    h = _seed().child(guards=[g], procedures=[p], created_by="improver")
    store.put_harness(h)
    assert store.get_harness(h.version) == h


def test_lineage_is_a_tree_via_graphlookup_with_ancestry(store: Store) -> None:
    seed = _seed()
    a = seed.child(system_prompt="a", created_by="improver")
    b = seed.child(system_prompt="b", created_by="improver")
    a1 = a.child(system_prompt="a1", created_by="improver")
    other_root = Harness.build(
        parent=None, system_prompt="other", procedures=[], guards=[], tool_allowlist=[], created_by="human"
    )
    for h in (seed, a, b, a1, other_root):
        store.put_harness(h)
    roots = store.lineage()
    assert {r.version for r in roots} == {seed.version, other_root.version}
    tree = next(r for r in roots if r.version == seed.version)
    assert {c.version for c in tree.children} == {a.version, b.version}
    leaf = next(c for c in tree.children if c.version == a.version).children
    assert [n.version for n in leaf] == [a1.version]
    assert {n.version for n in tree.walk()} == {seed.version, a.version, b.version, a1.version}
    assert [n.version for n in store.lineage(a.version)[0].walk()] == [a.version, a1.version]
    assert store.ancestry(a1.version) == [a1.version, a.version, seed.version]
    assert store.ancestry("missing") == []


def test_put_trace_idempotent_and_roundtrips(store: Store) -> None:
    t = _trace()
    assert store.put_trace(t) is True
    assert store.put_trace(t) is False
    assert store.db["traces"].count_documents({}) == 1
    assert store.get_trace("r1", "c1") == t
    assert store.get_trace("r1", "zzz") is None
    with pytest.raises(IntegrityError, match="different content"):
        store.put_trace(_trace(passed=False))


def test_large_traces_are_chunked_and_read_back_whole(atlas_uri: str, hash_embedder: HashEmbedder) -> None:
    s = Store(atlas_uri, "ratchet_test_chunks", embedder=hash_embedder, max_inline_bytes=500)
    try:
        t = _trace(n=60)
        assert s.put_trace(t) is True
        assert s.put_trace(t) is False
        header = s.db["traces"].find_one({"_id": "r1/c1"})
        assert header is not None and "events" not in header and header["chunks"] >= 1
        assert s.db["trace_chunks"].count_documents({}) == header["chunks"]
        assert s.get_trace("r1", "c1") == t
    finally:
        s._client.drop_database("ratchet_test_chunks")
        s.close()


def test_traces_filter_limit_and_order(store: Store) -> None:
    for i in range(4):
        store.put_trace(
            _trace(run_id=f"r{i}", case_id=f"c{i % 2}", version="va" if i < 2 else "vb", passed=i % 2 == 0)
        )
    assert len(store.traces()) == 4
    assert {t.run_id for t in store.traces({"harness_version": "vb"})} == {"r2", "r3"}
    assert [t.run_id for t in store.traces({"passed": False})] == ["r1", "r3"]
    assert [t.run_id for t in store.traces({"case_id": "c0"}, newest_first=True)] == ["r2", "r0"]
    assert len(store.traces(limit=1)) == 1


def test_runs_evals_and_decisions_are_idempotent(store: Store) -> None:
    assert store.put_run("r1", harness_version="v", model="m", meta={"cases": 3}) is True
    assert store.put_run("r1", harness_version="v", model="m", meta={"cases": 3}) is False
    run = store.get_run("r1")
    assert run is not None and run["meta"] == {"cases": 3}
    assert store.get_run("none") is None

    @dataclass
    class Report:
        candidate: str
        passed: int

    rid = store.put_eval(Report(candidate="c1", passed=5))
    assert store.put_eval(Report(candidate="c1", passed=5)) == rid
    assert store.put_eval({"candidate": "c2", "passed": 1}, eval_id="e2") == "e2"
    assert [e["candidate"] for e in store.evals({"candidate": "c1"})] == ["c1"]
    with pytest.raises(IntegrityError):
        store.put_eval({"candidate": "other"}, eval_id="e2")

    did = store.put_decision({"candidate": "c1", "admitted": True, "checks": {"replay": True}})
    assert store.put_decision({"candidate": "c1", "admitted": True, "checks": {"replay": True}}) == did
    assert store.db["decisions"].count_documents({}) == 1
    assert store.decisions({"admitted": True})[0]["checks"] == {"replay": True}
    with pytest.raises(TypeError):
        store.put_decision("not a record")


def test_checkpoint_resume_are_atomic_upserts(store: Store) -> None:
    assert store.resume("r1") is None
    assert store.checkpoint("r1", {"step": 1}) == 1
    assert store.checkpoint("r1", {"step": 2, "done": ["c1"]}) == 2
    assert store.resume("r1") == {"step": 2, "done": ["c1"]}
    threads = [threading.Thread(target=store.checkpoint, args=("r2", {"n": i})) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert store.db["checkpoints"].find_one({"_id": "r2"})["seq"] == 8  # type: ignore[index]
    assert store.db["checkpoints"].count_documents({"_id": "r2"}) == 1


FAILURES = [
    ("refund timed out after the provider committed and the agent refunded again", "refund", "f-dup"),
    ("agent emailed the customer before the approval was granted", "approval", "f-mail"),
    ("ticket closed while the payment dispute was still open", "dispute", "f-ticket"),
]


def _load_failures(store: Store) -> None:
    for text, family, fid in FAILURES:
        store.put_failure(
            text, case_family=family, trace_ref=f"r/{fid}", harness_version="v1", failure_id=fid
        )
    store.setup()


def test_similar_failures_vector_search_ranks_the_nearest_first(store: Store) -> None:
    _load_failures(store)
    hits = _eventually(lambda: store.similar_failures("refund timed out so the agent refunded twice", 3))
    assert hits[0].id == "f-dup" and hits[0].trace_ref == "r/f-dup" and hits[0].harness_version == "v1"
    assert hits[0].score >= hits[-1].score
    scoped = store.similar_failures("refund timed out", 3, case_family="approval")
    assert [h.id for h in scoped] == ["f-mail"]
    assert "f-dup" not in [h.id for h in store.similar_failures("refund timed out", 3, exclude_id="f-dup")]


def test_hybrid_search_fuses_keyword_and_vector_hits(store: Store) -> None:
    _load_failures(store)
    query = "dispute ticket closed early"
    hits = _eventually(lambda: store.similar_failures_hybrid(query, 3))
    assert hits[0].id == "f-ticket"
    assert hits[0].score == pytest.approx(2 / 61)  # first in both rankings
    assert len({h.id for h in hits}) == len(hits)
    scoped = store.similar_failures_hybrid(query, 3, case_family="dispute", exclude_id="f-mail")
    assert [h.id for h in scoped] == ["f-ticket"]


def test_put_failure_is_idempotent_and_embeds_once(store: Store, hash_embedder: HashEmbedder) -> None:
    fid = store.put_failure("same text", case_family="x", trace_ref="r/1")
    assert store.put_failure("same text", case_family="x", trace_ref="r/1") == fid
    assert store.db["failures"].count_documents({}) == 1
    doc = store.db["failures"].find_one({"_id": fid})
    assert doc is not None and len(doc["embedding"]) == 1024 and doc["embedding_model"] == hash_embedder.model
    assert sum(n for kind, n in hash_embedder.calls if kind == "document") == 1


def test_embedding_cache_prevents_re_embedding_across_stores(
    atlas_uri: str, hash_embedder: HashEmbedder
) -> None:
    first = Store(atlas_uri, "ratchet_test_cache", embedder=hash_embedder)
    try:
        first.put_failure("cached text", case_family="x", failure_id="a")
        assert hash_embedder.calls == [("document", 1)]
        second = Store(atlas_uri, "ratchet_test_cache", embedder=hash_embedder)
        second.put_failure("cached text", case_family="y", failure_id="b")
        assert hash_embedder.calls == [("document", 1)]
        assert second.embedder.stats.hits == 1 and second.embedder.stats.misses == 0
        second.close()
        assert first.db["embedding_cache"].count_documents({}) == 1
    finally:
        first._client.drop_database("ratchet_test_cache")
        first.close()


def test_watch_yields_change_stream_events(store: Store) -> None:
    store.setup(wait_search=False)
    seen: list[Any] = []

    def reader() -> None:
        seen.extend(store.watch("harness_versions", timeout_s=8))

    th = threading.Thread(target=reader)
    th.start()
    time.sleep(1.5)
    seed = _seed()
    store.put_harness(seed)
    th.join()
    assert [e.operation for e in seen] == ["insert"]
    assert seen[0].doc_id == seed.version and seen[0].document["created_by"] == "seed"
    assert "embedding" not in (seen[0].document or {})


@pytest.mark.live
def test_voyage_embeddings_drive_semantic_retrieval(atlas_uri: str) -> None:
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.fail("VOYAGE_API_KEY is not set; export it for the live tests")
    s = Store(atlas_uri, "ratchet_test_voyage", embedder=VoyageEmbedder())
    try:
        _load_failures(s)
        hits = _eventually(
            lambda: s.similar_failures("a payment was refunded twice after a network timeout", 3)
        )
        assert hits[0].id == "f-dup"
        hybrid = s.similar_failures_hybrid("customer was emailed prematurely", 3)
        assert hybrid[0].id == "f-mail"
    finally:
        s._client.drop_database("ratchet_test_voyage")
        s.close()
