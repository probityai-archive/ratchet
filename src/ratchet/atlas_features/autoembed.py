"""Probe 2: Atlas Vector Search automated embedding (`autoEmbed` index, Voyage models run by Atlas).

    uv run python -m ratchet.atlas_features.autoembed

Stores only text, creates an `autoEmbed` vector index, and queries with `query` text so neither the write path
nor the read path embeds anything client-side. Measures index build time, query latency, whether an insert
and an update become searchable without any client code, recall@3 against the client-embedded baseline of
the fusion probe, and whether `$rankFusion` accepts an autoEmbed leg.
"""

from __future__ import annotations

import statistics
import sys
import time
from typing import Any

from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from ratchet.atlas_features.common import ProbeResult, connect, probe_db, timed, wait_queryable
from ratchet.atlas_features.dataset import DOCS, QUERIES
from ratchet.atlas_features.fusion import TEXT_DEFINITION, TXT_INDEX, text_leg

COLLECTION = "autoembed_probe"
MODELS = ("voyage-4", "voyage-4-large")
TOP_K = 3

Pipeline = list[dict[str, Any]]


def auto_definition(model: str) -> dict[str, Any]:
    return {"fields": [{"type": "autoEmbed", "modality": "text", "path": "summary", "model": model}]}


def index_name(model: str) -> str:
    return "auto_" + model.replace("-", "_")


def auto_leg(model: str, text: str, limit: int = 10) -> Pipeline:
    """The vector leg with no client-side embedding: Atlas embeds `query` with the index's model."""
    return [
        {
            "$vectorSearch": {
                "index": index_name(model),
                "path": "summary",
                "query": text,
                "numCandidates": 50,
                "limit": limit,
            }
        }
    ]


def auto_rank_fusion(model: str, text: str) -> Pipeline:
    return [
        {
            "$rankFusion": {
                "input": {"pipelines": {"vec": auto_leg(model, text), "txt": text_leg(text)}},
                "combination": {"weights": {"vec": 2.0, "txt": 1.0}},
            }
        },
        {"$limit": TOP_K},
        {"$project": {"_id": 1}},
    ]


def ensure(coll: Collection[dict[str, Any]]) -> dict[str, float]:
    """Seed text-only documents and create the indexes; return seconds until each became queryable."""
    for d in DOCS:
        coll.replace_one(
            {"_id": d.id}, {"_id": d.id, "summary": d.summary, "case_family": d.family}, upsert=True
        )
    have = {i["name"] for i in coll.list_search_indexes()}
    started = time.monotonic()
    for model in MODELS:
        if index_name(model) not in have:
            coll.create_search_index(
                SearchIndexModel(
                    definition=auto_definition(model), name=index_name(model), type="vectorSearch"
                )
            )
    if TXT_INDEX not in have:
        coll.create_search_index(SearchIndexModel(definition=TEXT_DEFINITION, name=TXT_INDEX, type="search"))
    waits: dict[str, float] = {}
    for name in [index_name(m) for m in MODELS] + [TXT_INDEX]:
        wait_queryable(coll.database, COLLECTION, [name], timeout_s=600)
        waits[name] = round(time.monotonic() - started, 1)
    return waits


def _ids(coll: Collection[dict[str, Any]], pipeline: Pipeline) -> list[str]:
    return [str(r["_id"]) for r in coll.aggregate([*pipeline, {"$project": {"_id": 1}}])]


def score_mode(coll: Collection[dict[str, Any]], build: Any) -> dict[str, Any]:
    hits, firsts, lat = [], [], []
    for q in QUERIES:
        with timed() as elapsed:
            try:
                got = _ids(coll, [*build(q.text), {"$limit": TOP_K}])
            except OperationFailure as exc:
                return {"ok": False, "error": f"code {exc.code}: {(exc.details or {}).get('errmsg', exc)}"}
        lat.append(elapsed() * 1000)
        hits.append(q.gold in got)
        firsts.append(got[:1] == [q.gold])
    return {
        "ok": True,
        "recall_at_3": round(sum(hits) / len(hits), 3),
        "recall_at_1": round(sum(firsts) / len(firsts), 3),
        "median_ms": round(statistics.median(lat), 1),
    }


def sync_check(coll: Collection[dict[str, Any]], model: str, timeout_s: float = 180.0) -> dict[str, Any]:
    """Insert a new document and update it; report seconds until each text is found by meaning alone.

    The two topics are far from every labelled failure, so the marker is the top hit once embedded.
    """
    marker = "zzprobe"
    coll.delete_one({"_id": marker})
    coll.insert_one(
        {"_id": marker, "summary": "A Kubernetes pod was evicted during the nightly backup window."}
    )
    inserted = _wait_found(
        coll, model, "container scheduler removed a workload while saving data", marker, timeout_s
    )
    coll.update_one(
        {"_id": marker},
        {"$set": {"summary": "The sourdough starter needs feeding with flour and warm water."}},
    )
    updated = _wait_found(coll, model, "bread baking yeast culture feeding schedule", marker, timeout_s)
    coll.delete_one({"_id": marker})
    return {"insert_searchable_s": inserted, "update_searchable_s": updated, "timeout_s": timeout_s}


def _wait_found(
    coll: Collection[dict[str, Any]], model: str, query: str, doc_id: str, timeout_s: float
) -> float | None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if doc_id in _ids(coll, [*auto_leg(model, query), {"$limit": 1}]):
            return round(time.monotonic() - start, 1)
        time.sleep(1.0)
    return None


def run() -> ProbeResult:
    client = connect()
    try:
        db = probe_db(client)
        coll = db[COLLECTION]
        with timed() as elapsed:
            waits = ensure(coll)
            scores = {m: score_mode(coll, lambda t, m=m: auto_leg(m, t)) for m in MODELS}
            fusion = score_mode(coll, lambda t: auto_rank_fusion(MODELS[0], t))
            sync = sync_check(coll, MODELS[0])
        works = all(s["ok"] for s in scores.values())
        return ProbeResult(
            probe="autoembed",
            works=works,
            summary="autoEmbed index built and queried with text only" if works else "autoEmbed failed",
            latency_s=elapsed(),
            pipeline={
                "index": auto_definition(MODELS[0]),
                "query": auto_leg(MODELS[0], "<query text>"),
                "hybrid": auto_rank_fusion(MODELS[0], "<query text>"),
            },
            data={
                "index_ready_s": waits,
                "docs": len(DOCS),
                "queries": len(QUERIES),
                "recall": scores,
                "auto_rank_fusion": fusion,
                "sync": sync,
            },
            error=None if works else str(scores),
        )
    finally:
        client.close()


def main() -> int:
    result = run()
    path = result.save()
    print(f"{result.probe}: works={result.works} {result.summary} -> {path}")
    for key, value in result.data.items():
        print(key, value)
    return 0 if result.works else 1


if __name__ == "__main__":
    sys.exit(main())
