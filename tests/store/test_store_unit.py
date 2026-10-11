"""Unit tests: no network, no database."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from ratchet.store import (
    EmbeddingError,
    IntegrityError,
    MissingKeyError,
    Store,
    VoyageEmbedder,
    atlas,
    atlas_setup,
    rrf_fuse,
)
from ratchet.store.embed import EMBEDDING_DIMS, CachedEmbedder, cache_key
from ratchet.types import Cost, Event, Harness, Outcome, Trace


def _trace(n_events: int = 3) -> Trace:
    events = [Event(seq=i, kind="model", payload={"text": f"e{i}"}, ts=float(i)) for i in range(n_events)]
    out = Outcome(passed=False, reasons=["r"], final_state_digest="d", cost=Cost(model_calls=1))
    return Trace(run_id="r1", case_id="c1", harness_version="v1", model="m", events=events, outcome=out)


def _voyage(handler: Any, **kw: Any) -> VoyageEmbedder:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return VoyageEmbedder("secret-key", client=client, backoff_s=0.0, **kw)


def _ok(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    data = [{"index": i, "embedding": [0.5] * EMBEDDING_DIMS} for i in range(len(body["input"]))]
    return httpx.Response(200, json={"data": data[::-1]})


def test_rrf_fuse_rewards_agreement_and_is_deterministic() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["c", "a", "d"]])
    ids = [i for i, _ in fused]
    assert ids[0] == "a" and set(ids) == {"a", "b", "c", "d"}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert rrf_fuse([["x", "y"], ["y", "x"]]) == rrf_fuse([["x", "y"], ["y", "x"]])
    assert rrf_fuse([]) == []


def test_cache_key_separates_model_input_type_and_text() -> None:
    keys = {cache_key("m", "document", "t"), cache_key("m", "query", "t"), cache_key("n", "document", "t")}
    assert len(keys) == 3 and cache_key("m", "document", "t") == cache_key("m", "document", "t")


def test_voyage_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        VoyageEmbedder()


def test_voyage_reads_key_from_env_and_orders_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "from-env")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(request)

    emb = VoyageEmbedder(client=httpx.Client(transport=httpx.MockTransport(handler)))
    vectors = emb.embed(["a", "b"], "query")
    assert len(vectors) == 2 and len(vectors[0]) == EMBEDDING_DIMS
    assert seen[0].headers["authorization"] == "Bearer from-env"
    assert json.loads(seen[0].content)["input_type"] == "query"


def test_voyage_batches_over_the_service_limit() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(json.loads(request.content)["input"]))
        return _ok(request)

    assert len(_voyage(handler).embed([str(i) for i in range(300)], "document")) == 300
    assert sizes == [128, 128, 44]


def test_voyage_retries_transient_errors_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429)
        if attempts["n"] == 2:
            raise httpx.ConnectError("boom")
        return _ok(request)

    assert len(_voyage(handler).embed(["a"], "document")) == 1 and attempts["n"] == 3


def test_voyage_error_never_contains_the_key() -> None:
    emb = _voyage(lambda r: httpx.Response(401, text="bad key secret-key"))
    with pytest.raises(EmbeddingError) as err:
        emb.embed(["a"], "document")
    assert "secret-key" not in str(err.value) and "401" in str(err.value)


def test_voyage_gives_up_after_retries() -> None:
    emb = _voyage(lambda r: httpx.Response(503), retries=2)
    with pytest.raises(EmbeddingError, match="after retries"):
        emb.embed(["a"], "document")


def test_voyage_rejects_wrong_dimensions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

    with pytest.raises(EmbeddingError, match="shape"):
        _voyage(handler).embed(["a"], "document")


class _FakeCollection:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def find(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        return [self.docs[k] for k in query["_id"]["$in"] if k in self.docs]

    def bulk_write(self, ops: Sequence[Any], ordered: bool = True) -> None:
        for op in ops:
            self.docs[op._filter["_id"]] = {"_id": op._filter["_id"], **op._doc["$setOnInsert"]}


class _CountingEmbedder:
    model = "count"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        self.texts.extend(texts)
        return [[float(len(t))] for t in texts]


def test_cached_embedder_only_embeds_misses() -> None:
    inner, coll = _CountingEmbedder(), _FakeCollection()
    cached = CachedEmbedder(inner, coll)  # type: ignore[arg-type]
    assert cached.embed(["aa", "b"], "document") == [[2.0], [1.0]]
    assert cached.embed(["b", "ccc", "aa"], "document") == [[1.0], [3.0], [2.0]]
    assert inner.texts == ["aa", "b", "ccc"]
    assert (cached.stats.hits, cached.stats.misses) == (2, 3)
    cached.embed(["b"], "query")  # a different input type is a different vector
    assert inner.texts[-1] == "b"


def test_trace_roundtrip_and_digest_check() -> None:
    from dataclasses import asdict

    t = _trace()
    header = {
        "_id": "r1/c1",
        "run_id": "r1",
        "case_id": "c1",
        "harness_version": "v1",
        "model": "m",
        "outcome": asdict(t.outcome),
        "digest": atlas.trace_digest(t),
    }
    events = [asdict(e) for e in t.events]
    assert atlas._trace_from_parts(header, events) == t
    events[0]["payload"] = {"text": "tampered"}
    with pytest.raises(IntegrityError):
        atlas._trace_from_parts(header, events)


def test_split_events_keeps_order_and_bounds_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(atlas, "_CHUNK_BYTES", 100)
    events = [{"seq": i, "pad": "x" * 40} for i in range(10)]
    chunks = atlas._split_events(events)
    assert len(chunks) > 1 and [e for c in chunks for e in c] == events
    assert atlas._split_events([]) == [[]]


def test_harness_doc_roundtrip() -> None:
    h = Harness.build(
        parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=["b", "a"], created_by="seed"
    )
    assert atlas._harness_from_doc(atlas._harness_doc(h)) == h


def test_put_harness_rejects_a_bad_digest_before_any_io() -> None:
    h = Harness.build(
        parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=[], created_by="seed"
    )
    bad = Harness(**{**h.__dict__, "system_prompt": "changed"})
    with Store("mongodb://127.0.0.1:1", "unused") as store, pytest.raises(IntegrityError):
        store.put_harness(bad)


def test_watch_rejects_unknown_collections() -> None:
    with Store("mongodb://127.0.0.1:1", "unused") as store, pytest.raises(ValueError, match="cannot watch"):
        next(store.watch("embedding_cache"))


def test_every_index_targets_a_known_collection_and_search_indexes_match_the_store() -> None:
    names = set(atlas_setup.COLLECTIONS)
    assert {c for c, _, _ in atlas_setup.BTREE_INDEXES} <= names
    assert {c for c, *_ in atlas_setup.SEARCH_INDEXES} <= names
    vector = atlas_setup.SEARCH_INDEXES[0][3]["fields"][0]
    assert vector["path"] == "embedding" and vector["numDimensions"] == EMBEDDING_DIMS
    assert set(atlas.WATCHABLE) <= names


def test_setup_main_requires_a_uri(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("RATCHET_MONGODB_URI", raising=False)
    assert atlas_setup.main([]) == 2
    assert "--uri" in capsys.readouterr().err
