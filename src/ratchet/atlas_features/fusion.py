"""Probe 1: `$rankFusion` and `$scoreFusion` over failure summaries (vector + Atlas Search text).

    uv run python -m ratchet.atlas_features.fusion

Builds `fusion_probe` in `ratchet_featureprobe` from the labelled set in `dataset.py`, runs the vector-only,
the text-only, both fusion operators and a hand-built rank-fusion baseline over every query, and
reports recall@3 (the gold document in the top 3) with the median latency of each.
"""

from __future__ import annotations

import contextlib
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from ratchet.atlas_features.common import (
    ProbeResult,
    connect,
    embedder,
    probe_db,
    timed,
    wait_queryable,
)
from ratchet.atlas_features.dataset import DOCS, FAMILY_OF, QUERIES, Query
from ratchet.store.embed import EMBEDDING_DIMS, Embedder

COLLECTION = "fusion_probe"
VEC_INDEX = "fusion_vec"
TXT_INDEX = "fusion_txt"
TOP_K = 3
NUM_CANDIDATES = 50
FUSION_POOL = 10  # each leg keeps this many candidates before fusion
FUSION_OPERATORS = ("rank_fusion", "score_fusion")
WEIGHTS = {"vec": 1.0, "txt": 1.0}

VECTOR_DEFINITION: dict[str, Any] = {
    "fields": [
        {"type": "vector", "path": "embedding", "numDimensions": EMBEDDING_DIMS, "similarity": "cosine"},
        {"type": "filter", "path": "case_family"},
    ]
}
TEXT_DEFINITION: dict[str, Any] = {
    "mappings": {
        "dynamic": False,
        "fields": {"summary": {"type": "string"}, "case_family": {"type": "token"}},
    }
}

Pipeline = list[dict[str, Any]]


def vector_leg(vector: list[float]) -> Pipeline:
    return [
        {
            "$vectorSearch": {
                "index": VEC_INDEX,
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": NUM_CANDIDATES,
                "limit": FUSION_POOL,
            }
        }
    ]


def text_leg(text: str) -> Pipeline:
    return [
        {"$search": {"index": TXT_INDEX, "text": {"query": text, "path": "summary"}}},
        {"$limit": FUSION_POOL},
    ]


def project_id() -> dict[str, Any]:
    return {"$project": {"_id": 1}}


def pipeline_vector(text: str, vector: list[float]) -> Pipeline:
    return [*vector_leg(vector), {"$limit": TOP_K}, project_id()]


def pipeline_text(text: str, vector: list[float]) -> Pipeline:
    return [*text_leg(text), {"$limit": TOP_K}, project_id()]


def rank_fusion_with(weights: dict[str, float]) -> Callable[[str, list[float]], Pipeline]:
    def build(text: str, vector: list[float]) -> Pipeline:
        return [
            {
                "$rankFusion": {
                    "input": {"pipelines": {"vec": vector_leg(vector), "txt": text_leg(text)}},
                    "combination": {"weights": dict(weights)},
                    "scoreDetails": False,
                }
            },
            {"$limit": TOP_K},
            project_id(),
        ]

    return build


pipeline_rank_fusion = rank_fusion_with(WEIGHTS)


def score_fusion_with(
    weights: dict[str, float], normalization: str = "sigmoid"
) -> Callable[[str, list[float]], Pipeline]:
    def build(text: str, vector: list[float]) -> Pipeline:
        return [
            {
                "$scoreFusion": {
                    "input": {
                        "pipelines": {"vec": vector_leg(vector), "txt": text_leg(text)},
                        "normalization": normalization,
                    },
                    "combination": {"weights": dict(weights)},
                    "scoreDetails": False,
                }
            },
            {"$limit": TOP_K},
            project_id(),
        ]

    return build


pipeline_score_fusion = score_fusion_with(WEIGHTS)


def pipeline_manual_rank_fusion(text: str, vector: list[float]) -> Pipeline:
    """Rank fusion by hand with `$unionWith`, the formula `$rankFusion` documents (weight / (60 + rank))."""

    def leg(pipeline: Pipeline) -> Pipeline:
        return [
            *pipeline,
            {"$group": {"_id": None, "docs": {"$push": "$_id"}}},
            {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
            {"$project": {"_id": "$docs", "fused": {"$divide": [1.0, {"$add": [60, "$rank", 1]}]}}},
        ]

    return [
        *leg(vector_leg(vector)),
        {"$unionWith": {"coll": COLLECTION, "pipeline": leg(text_leg(text))}},
        {"$group": {"_id": "$_id", "score": {"$sum": "$fused"}}},
        {"$sort": {"score": -1}},
        {"$limit": TOP_K},
    ]


MODES: dict[str, Callable[[str, list[float]], Pipeline]] = {
    "vector_only": pipeline_vector,
    "text_only": pipeline_text,
    "rank_fusion": pipeline_rank_fusion,
    "rank_fusion_vec2": rank_fusion_with({"vec": 2.0, "txt": 1.0}),
    "rank_fusion_txt2": rank_fusion_with({"vec": 1.0, "txt": 2.0}),
    "score_fusion": pipeline_score_fusion,
    "score_fusion_vec2": score_fusion_with({"vec": 2.0, "txt": 1.0}),
    "score_fusion_minmax_vec2": score_fusion_with({"vec": 2.0, "txt": 1.0}, "minMaxScaler"),
    "manual_rank_fusion": pipeline_manual_rank_fusion,
}


def seed(coll: Collection[dict[str, Any]], emb: Embedder) -> None:
    """Idempotent: write the labelled documents with document-side embeddings."""
    vectors = emb.embed([d.summary for d in DOCS], "document")
    for doc, vec in zip(DOCS, vectors, strict=True):
        coll.replace_one(
            {"_id": doc.id},
            {"_id": doc.id, "summary": doc.summary, "case_family": doc.family, "embedding": vec},
            upsert=True,
        )


def ensure_indexes(coll: Collection[dict[str, Any]], db_name: str) -> float:
    have = {i["name"] for i in coll.list_search_indexes()}
    if VEC_INDEX not in have:
        coll.create_search_index(
            SearchIndexModel(definition=VECTOR_DEFINITION, name=VEC_INDEX, type="vectorSearch")
        )
    if TXT_INDEX not in have:
        coll.create_search_index(SearchIndexModel(definition=TEXT_DEFINITION, name=TXT_INDEX, type="search"))
    return wait_queryable(coll.database, COLLECTION, [VEC_INDEX, TXT_INDEX])


@dataclass
class ModeScore:
    mode: str
    ok: bool
    recall_at_3: float = 0.0
    family_hit_at_3: float = 0.0
    recall_at_1: float = 0.0
    mrr_at_3: float = 0.0
    median_ms: float = 0.0
    by_kind: dict[str, float] | None = None
    error: str | None = None
    misses: list[str] | None = None


def run_mode(
    coll: Collection[dict[str, Any]],
    mode: str,
    queries: tuple[Query, ...],
    vectors: dict[str, list[float]],
) -> ModeScore:
    build = MODES[mode]
    hits: list[bool] = []
    first: list[bool] = []
    rr: list[float] = []
    fam_hits: list[bool] = []
    kinds: dict[str, list[bool]] = {}
    lat: list[float] = []
    misses: list[str] = []
    for q in queries:
        pipeline = build(q.text, vectors[q.id])
        with timed() as elapsed:
            try:
                got = [str(r["_id"]) for r in coll.aggregate(pipeline)]
            except OperationFailure as exc:
                errmsg = (exc.details or {}).get("errmsg", str(exc))
                return ModeScore(mode, False, error=f"code {exc.code}: {errmsg}")
        lat.append(elapsed() * 1000)
        hit = q.gold in got
        hits.append(hit)
        first.append(got[:1] == [q.gold])
        rr.append(1.0 / (got.index(q.gold) + 1) if hit else 0.0)
        fam_hits.append(any(FAMILY_OF[g] == FAMILY_OF[q.gold] for g in got))
        kinds.setdefault(q.kind, []).append(hit)
        if not hit:
            misses.append(q.id)
    return ModeScore(
        mode,
        True,
        recall_at_3=sum(hits) / len(hits),
        recall_at_1=sum(first) / len(first),
        mrr_at_3=sum(rr) / len(rr),
        family_hit_at_3=sum(fam_hits) / len(fam_hits),
        median_ms=statistics.median(lat),
        by_kind={k: round(sum(v) / len(v), 3) for k, v in kinds.items()},
        misses=misses,
    )


def rank_fusion_sample(
    coll: Collection[dict[str, Any]], vectors: dict[str, list[float]]
) -> list[dict[str, Any]]:
    """One `$rankFusion` result with `scoreDetails`, as the server returned it."""
    q = QUERIES[0]
    pipeline = pipeline_rank_fusion(q.text, vectors[q.id])
    pipeline[0]["$rankFusion"]["scoreDetails"] = True
    pipeline[-1] = {"$project": {"_id": 1, "details": {"$meta": "scoreDetails"}}}
    return [dict(r) for r in coll.aggregate(pipeline)]


def score_fusion_response(
    coll: Collection[dict[str, Any]], vectors: dict[str, list[float]]
) -> dict[str, Any]:
    """The exact server answer to a minimal `$scoreFusion`: result ids, or code, codeName and errmsg."""
    q = QUERIES[0]
    try:
        return {
            "ok": True,
            "ids": [str(r["_id"]) for r in coll.aggregate(pipeline_score_fusion(q.text, vectors[q.id]))],
        }
    except OperationFailure as exc:
        d = exc.details or {}
        return {"ok": False, "code": exc.code, "codeName": d.get("codeName"), "errmsg": d.get("errmsg")}


def run() -> ProbeResult:
    client = connect()
    try:
        db = probe_db(client)
        coll = db[COLLECTION]
        with embedder(db) as emb, timed() as elapsed:
            seed(coll, emb)
            index_wait = ensure_indexes(coll, db.name)
            vectors = dict(
                zip([q.id for q in QUERIES], emb.embed([q.text for q in QUERIES], "query"), strict=True)
            )
            # one warm-up call per mode so index load does not pollute latency
            for mode in MODES:
                with contextlib.suppress(OperationFailure):
                    next(iter(coll.aggregate(MODES[mode](QUERIES[0].text, vectors[QUERIES[0].id]))), None)
            scores = [run_mode(coll, m, QUERIES, vectors) for m in MODES]
        data = {
            "server_version": client.admin.command("buildInfo")["version"],
            "rank_fusion_sample": rank_fusion_sample(coll, vectors),
            "score_fusion_response": score_fusion_response(coll, vectors),
            "docs": len(DOCS),
            "queries": len(QUERIES),
            "index_wait_s": round(index_wait, 1),
            "scores": [s.__dict__ for s in scores],
        }
        working = [s.mode for s in scores if s.ok and s.mode in FUSION_OPERATORS]
        return ProbeResult(
            probe="fusion",
            works=bool(working),
            summary=f"fusion operators that run: {working or 'none'}",
            latency_s=elapsed(),
            pipeline={m: MODES[m]("<query text>", [0.0] * 3) for m in ("rank_fusion", "score_fusion")},
            data=data,
            error="; ".join(f"{s.mode}: {s.error}" for s in scores if not s.ok and s.mode in FUSION_OPERATORS)
            or None,
        )
    finally:
        client.close()


def main() -> int:
    start = time.perf_counter()
    result = run()
    path = result.save(suffix=f"-{result.data['server_version']}")
    result.save()
    took = time.perf_counter() - start
    print(f"{result.probe}: works={result.works} {result.summary} ({took:.1f}s) -> {path}")
    for s in result.data["scores"]:
        print(s)
    return 0 if result.works else 1


if __name__ == "__main__":
    sys.exit(main())
