"""Fusion probe logic against fake collections: scoring, error capture, seeding and the full run."""

from __future__ import annotations

from typing import Any

import pytest
from atlas_fakes import FakeClient, FakeCollection, FakeDb, const_embedder, op_failure

from ratchet.atlas_features import fusion
from ratchet.atlas_features.dataset import DOCS, QUERIES

VECTORS = {q.id: [0.0, 1.0, 0.0] for q in QUERIES}


def gold_first(pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the gold document for whichever query text appears in the pipeline."""
    blob = repr(pipeline)
    for q in QUERIES:
        if q.text in blob:
            return [{"_id": q.gold}]
    return [{"_id": "od1"}]


def test_run_mode_scores_a_perfect_retriever() -> None:
    coll = FakeCollection("c", FakeDb(), gold_first)
    score = fusion.run_mode(coll, "rank_fusion", QUERIES, VECTORS)  # type: ignore[arg-type]
    assert (score.ok, score.recall_at_3, score.recall_at_1, score.mrr_at_3) == (True, 1.0, 1.0, 1.0)
    assert score.family_hit_at_3 == 1.0
    assert score.misses == []


def test_run_mode_counts_misses_by_query_and_kind() -> None:
    coll = FakeCollection("c", FakeDb(), lambda _p: [{"_id": "od1"}])
    score = fusion.run_mode(coll, "vector_only", QUERIES, VECTORS)  # type: ignore[arg-type]
    assert score.recall_at_3 == 0.0
    assert len(score.misses or []) == len(QUERIES)
    assert score.by_kind is not None and set(score.by_kind) == {q.kind for q in QUERIES}


def test_run_mode_records_the_server_error_verbatim() -> None:
    err = op_failure(40324, "Unrecognized pipeline stage name: '$scoreFusion'")
    coll = FakeCollection("c", FakeDb(), lambda _p: err)
    score = fusion.run_mode(coll, "score_fusion", QUERIES, VECTORS)  # type: ignore[arg-type]
    assert not score.ok
    assert score.error == "code 40324: Unrecognized pipeline stage name: '$scoreFusion'"


def test_score_fusion_response_reports_code_name_and_message() -> None:
    err = op_failure(40324, "Unrecognized pipeline stage name: '$scoreFusion'")
    coll = FakeCollection("c", FakeDb(), lambda _p: err)
    got = fusion.score_fusion_response(coll, VECTORS)  # type: ignore[arg-type]
    assert got == {
        "ok": False,
        "code": 40324,
        "codeName": "Code40324",
        "errmsg": str(err.details and err.details["errmsg"]),
    }
    ok = fusion.score_fusion_response(FakeCollection("c", FakeDb(), gold_first), VECTORS)  # type: ignore[arg-type]
    assert ok["ok"] is True


def test_rank_fusion_sample_asks_for_score_details() -> None:
    coll = FakeCollection("c", FakeDb(), lambda _p: [{"_id": "rl1", "details": {}}])
    rows = fusion.rank_fusion_sample(coll, VECTORS)  # type: ignore[arg-type]
    assert rows == [{"_id": "rl1", "details": {}}]
    assert coll.pipelines[0][0]["$rankFusion"]["scoreDetails"] is True


def test_manual_rank_fusion_unions_both_legs() -> None:
    pipeline = fusion.pipeline_manual_rank_fusion("refund", [0.0])
    assert any("$unionWith" in stage for stage in pipeline)
    assert pipeline[-1] == {"$limit": fusion.TOP_K}


def test_text_and_vector_only_pipelines_use_one_leg_each() -> None:
    assert "$vectorSearch" in fusion.pipeline_vector("t", [0.0])[0]
    assert "$search" in fusion.pipeline_text("t", [0.0])[0]


def test_seed_writes_every_document_once() -> None:
    coll = FakeCollection("c", FakeDb())
    fusion.seed(coll, const_embedder(None).__enter__())  # type: ignore[arg-type]
    assert len(coll.docs) == len(DOCS)
    assert len(coll.docs["rl1"]["embedding"]) == 3


def test_ensure_indexes_creates_only_missing_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fusion, "wait_queryable", lambda *_a, **_k: 1.5)
    coll = FakeCollection("c", FakeDb())
    coll.search_indexes = [{"name": fusion.VEC_INDEX}]
    assert fusion.ensure_indexes(coll, "db") == 1.5  # type: ignore[arg-type]
    assert [m.document["name"] for m in coll.created_indexes] == [fusion.TXT_INDEX]


def test_run_reports_the_operators_that_work(monkeypatch: pytest.MonkeyPatch) -> None:
    db = FakeDb()
    err = op_failure(40324, "Unrecognized pipeline stage name: '$scoreFusion'")

    def agg(pipeline: list[dict[str, Any]]) -> Any:
        return err if "$scoreFusion" in repr(pipeline[0]) else gold_first(pipeline)

    db[fusion.COLLECTION].aggregate_fn = agg
    monkeypatch.setattr(fusion, "connect", lambda: FakeClient())
    monkeypatch.setattr(fusion, "probe_db", lambda _c: db)
    monkeypatch.setattr(fusion, "embedder", const_embedder)
    monkeypatch.setattr(fusion, "wait_queryable", lambda *_a, **_k: 0.1)
    result = fusion.run()
    assert result.works is True
    assert result.data["server_version"] == "9.0.4"
    assert result.data["score_fusion_response"]["ok"] is False
    assert "rank_fusion" in result.summary
    assert result.error is not None and "score_fusion" in result.error


def test_run_fails_when_no_operator_works(monkeypatch: pytest.MonkeyPatch) -> None:
    db = FakeDb()
    db[fusion.COLLECTION].aggregate_fn = lambda _p: op_failure(1, "nope")
    monkeypatch.setattr(fusion, "connect", lambda: FakeClient())
    monkeypatch.setattr(fusion, "probe_db", lambda _c: db)
    monkeypatch.setattr(fusion, "embedder", const_embedder)
    monkeypatch.setattr(fusion, "wait_queryable", lambda *_a, **_k: 0.1)
    # rank_fusion_sample runs an unguarded aggregate, so a server refusing it surfaces to the caller
    with pytest.raises(Exception, match="nope"):
        fusion.run()
