"""Create the Ratchet collections, indexes and Atlas search indexes. Safe to re-run.

    uv run python -m ratchet.store --uri "$RATCHET_MONGODB_URI" --db ratchet

Every query the store runs has an index here. Search indexes are created only when missing, and replaced
when their definition changed, so running this against a live cluster never drops data or an index.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from functools import partial
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from ratchet.store.embed import EMBEDDING_DIMS

COLLECTIONS = (
    "harness_versions",
    "runs",
    "traces",
    "trace_chunks",
    "cases",
    "evals",
    "decisions",
    "failures",
    "checkpoints",
    "active",
    "activations",
    "embedding_cache",
)

_SEARCH_UNREACHABLE = 508  # SearchIndexManagementHostUnreachable

HELDOUT_COLLECTION = "cases"
FAILURES_VECTOR_INDEX = "failures_vector"
FAILURES_TEXT_INDEX = "failures_text"

# (collection, keys, options). Each line names the query it serves.
BTREE_INDEXES: tuple[tuple[str, list[tuple[str, int]], dict[str, Any]], ...] = (
    # lineage() roots and children: $graphLookup walks parent -> _id
    ("harness_versions", [("parent", ASCENDING)], {"name": "by_parent"}),
    # content addressing: the digest is unique, so two versions can never claim the same content
    ("harness_versions", [("digest", ASCENDING)], {"name": "by_digest", "unique": True}),
    ("harness_versions", [("created_by", ASCENDING), ("created_at", DESCENDING)], {"name": "by_author"}),
    # runs of one harness version, newest first
    ("runs", [("harness_version", ASCENDING), ("created_at", DESCENDING)], {"name": "by_version"}),
    # traces(filter): by run, by case across versions, by outcome of one version
    ("traces", [("run_id", ASCENDING), ("case_id", ASCENDING)], {"name": "by_run_case", "unique": True}),
    ("traces", [("case_id", ASCENDING), ("harness_version", ASCENDING)], {"name": "by_case_version"}),
    ("traces", [("harness_version", ASCENDING), ("passed", ASCENDING)], {"name": "by_version_passed"}),
    ("traces", [("created_at", DESCENDING)], {"name": "by_time"}),
    # chunked traces: read back in order
    ("trace_chunks", [("trace_id", ASCENDING), ("n", ASCENDING)], {"name": "by_trace", "unique": True}),
    # evals and decisions: per candidate, per parent, newest first
    ("evals", [("candidate", ASCENDING), ("created_at", DESCENDING)], {"name": "by_candidate"}),
    ("evals", [("parent", ASCENDING), ("created_at", DESCENDING)], {"name": "by_parent"}),
    ("decisions", [("candidate", ASCENDING), ("created_at", DESCENDING)], {"name": "by_candidate"}),
    ("decisions", [("admitted", ASCENDING), ("created_at", DESCENDING)], {"name": "by_admitted"}),
    # failures: by family and by the trace they came from
    ("failures", [("case_family", ASCENDING), ("created_at", DESCENDING)], {"name": "by_family"}),
    ("failures", [("trace_ref", ASCENDING)], {"name": "by_trace_ref"}),
)

_VECTOR_DEFINITION: dict[str, Any] = {
    "fields": [
        {"type": "vector", "path": "embedding", "numDimensions": EMBEDDING_DIMS, "similarity": "cosine"},
        {"type": "filter", "path": "case_family"},
        {"type": "filter", "path": "harness_version"},
    ]
}
_TEXT_DEFINITION: dict[str, Any] = {
    "mappings": {
        "dynamic": False,
        "fields": {
            "summary": {"type": "string"},
            "case_family": {"type": "token"},
            "harness_version": {"type": "token"},
        },
    }
}
SEARCH_INDEXES: tuple[tuple[str, str, str, dict[str, Any]], ...] = (
    ("failures", FAILURES_VECTOR_INDEX, "vectorSearch", _VECTOR_DEFINITION),
    ("failures", FAILURES_TEXT_INDEX, "search", _TEXT_DEFINITION),
)


def _retry_unreachable[T](call: Callable[[], T], timeout_s: float, poll_s: float = 1.0) -> T:
    """Run one search-index management call, retrying while the search service is unreachable (code 508).

    A local cluster answers mongod before mongot is ready, and Atlas's index management service can drop a
    single request; any other error, or the deadline, is raised.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return call()
        except OperationFailure as exc:
            if exc.code != _SEARCH_UNREACHABLE or time.monotonic() > deadline:
                raise
            time.sleep(poll_s)


def _wait_dropped(db: Database[dict[str, Any]], collection: str, name: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while name in _existing_search_indexes(db, collection):
        if time.monotonic() > deadline:
            raise TimeoutError(f"search index {name} still present after {timeout_s:.0f}s")
        time.sleep(0.5)


def _existing_search_indexes(
    db: Database[dict[str, Any]], collection: str, timeout_s: float = 60.0
) -> dict[str, dict[str, Any]]:
    return _retry_unreachable(
        lambda: {i["name"]: dict(i) for i in db[collection].list_search_indexes()}, timeout_s
    )


def ensure_schema(
    db: Database[dict[str, Any]],
    search_indexes: tuple[tuple[str, str, str, dict[str, Any]], ...] = SEARCH_INDEXES,
    timeout_s: float = 60.0,
) -> None:
    """Create missing collections, B-tree indexes and search indexes. Idempotent."""
    have = set(db.list_collection_names())
    for name in COLLECTIONS:
        if name not in have:
            db.create_collection(name)
    for collection, keys, options in BTREE_INDEXES:
        db[collection].create_index(keys, **options)
    for collection, name, kind, definition in search_indexes:
        existing = _existing_search_indexes(db, collection, timeout_s)
        model = SearchIndexModel(definition=definition, name=name, type=kind)
        if name not in existing:
            _retry_unreachable(partial(db[collection].create_search_index, model), timeout_s)
        elif existing[name].get("latestDefinition") != definition:
            # updateSearchIndex does not know the index type on every deployment, so replace the index
            _retry_unreachable(partial(db[collection].drop_search_index, name), timeout_s)
            _wait_dropped(db, collection, name, timeout_s)
            _retry_unreachable(partial(db[collection].create_search_index, model), timeout_s)


def ensure_heldout(heldout_db: Database[dict[str, Any]]) -> None:
    """Create the held-out case collection in its own database, which a role can deny."""
    if HELDOUT_COLLECTION not in heldout_db.list_collection_names():
        heldout_db.create_collection(HELDOUT_COLLECTION)


def search_indexes_ready(db: Database[dict[str, Any]]) -> bool:
    """True when every search index exists and is queryable."""
    for collection, name, _kind, _definition in SEARCH_INDEXES:
        found = _existing_search_indexes(db, collection).get(name)
        if found is None or not found.get("queryable"):
            return False
    return True


def wait_search_indexes(db: Database[dict[str, Any]], timeout_s: float = 120.0, poll_s: float = 0.5) -> None:
    """Block until every search index is queryable; raise TimeoutError otherwise."""
    deadline = time.monotonic() + timeout_s
    while not search_indexes_ready(db):
        if time.monotonic() > deadline:
            raise TimeoutError(f"search indexes not queryable after {timeout_s:.0f}s")
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create Ratchet collections and indexes on a cluster.")
    parser.add_argument("--uri", default=os.environ.get("RATCHET_MONGODB_URI", ""))
    parser.add_argument("--db", default=os.environ.get("RATCHET_MONGODB_DB", "ratchet"))
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for search indexes")
    args = parser.parse_args(argv)
    if not args.uri:
        print("error: pass --uri or set RATCHET_MONGODB_URI", file=sys.stderr)
        return 2
    client: MongoClient[dict[str, Any]] = MongoClient(args.uri, tz_aware=True)
    try:
        db = client[args.db]
        ensure_schema(db)
        ensure_heldout(client[f"{args.db}_heldout"])
        wait_search_indexes(db, args.timeout)
        print(
            f"ok: {len(COLLECTIONS)} collections, {len(BTREE_INDEXES)} indexes, "
            f"{len(SEARCH_INDEXES)} search indexes"
        )
    finally:
        client.close()
    return 0
