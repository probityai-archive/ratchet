"""Fusion probe: pipeline shapes and dataset integrity (unit), and the live operators (sandbox)."""

from __future__ import annotations

from typing import Any

import pytest
from pymongo.database import Database
from pymongo.errors import OperationFailure

from ratchet.atlas_features import fusion
from ratchet.atlas_features.common import embedder
from ratchet.atlas_features.dataset import DOCS, QUERIES
from ratchet.store.embed import EMBEDDING_DIMS


def test_dataset_gold_documents_exist_and_ids_are_unique() -> None:
    ids = [d.id for d in DOCS]
    assert len(ids) == len(set(ids))
    assert all(q.gold in set(ids) for q in QUERIES)
    assert len({q.id for q in QUERIES}) == len(QUERIES)


def test_rank_fusion_pipeline_has_both_legs_and_weights() -> None:
    stage = fusion.pipeline_rank_fusion("refund timeout", [0.0, 1.0])[0]["$rankFusion"]
    legs = stage["input"]["pipelines"]
    assert set(legs) == {"vec", "txt"}
    assert "$vectorSearch" in legs["vec"][0]
    assert "$search" in legs["txt"][0]
    assert stage["combination"]["weights"] == {"vec": 1.0, "txt": 1.0}


def test_score_fusion_pipeline_normalizes_scores() -> None:
    stage = fusion.pipeline_score_fusion("q", [0.0])[0]["$scoreFusion"]
    assert stage["input"]["normalization"] == "sigmoid"
    assert set(stage["input"]["pipelines"]) == {"vec", "txt"}


@pytest.mark.sandbox
def test_rank_fusion_runs_on_the_sandbox_and_finds_gold(sandbox_db: Database[dict[str, Any]]) -> None:
    coll = sandbox_db[fusion.COLLECTION]
    with embedder(sandbox_db) as emb:
        fusion.seed(coll, emb)
        fusion.ensure_indexes(coll, sandbox_db.name)
        qvecs = emb.embed([q.text for q in QUERIES], "query")
    vectors = dict(zip([q.id for q in QUERIES], qvecs, strict=True))
    score = fusion.run_mode(coll, "rank_fusion", QUERIES, vectors)
    assert score.ok, score.error
    assert score.recall_at_3 >= 0.85
    assert score.family_hit_at_3 >= 0.95


@pytest.mark.sandbox
def test_score_fusion_matches_the_server_version(sandbox_db: Database[dict[str, Any]]) -> None:
    """Documented to need 8.3 or later: code 40324 before that, a working stage with good recall after."""
    coll = sandbox_db[fusion.COLLECTION]
    version = tuple(int(p) for p in sandbox_db.client.admin.command("buildInfo")["version"].split(".")[:2])
    unit = [1.0] + [0.0] * (EMBEDDING_DIMS - 1)
    try:
        list(coll.aggregate(fusion.pipeline_score_fusion("refund timeout", unit)))
    except OperationFailure as exc:
        assert version < (8, 3), f"$scoreFusion failed on a server that should support it: {exc.code}"
        assert exc.code == 40324
        return
    assert version >= (8, 3), "$scoreFusion ran on a server older than 8.3"
    with embedder(sandbox_db) as emb:
        qvecs = emb.embed([q.text for q in QUERIES], "query")
    vectors = dict(zip([q.id for q in QUERIES], qvecs, strict=True))
    score = fusion.run_mode(coll, "score_fusion", QUERIES, vectors)
    assert score.ok, score.error
    assert score.recall_at_3 >= 0.85
