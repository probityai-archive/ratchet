"""MongoDB Atlas persistence for Ratchet: lineage, traces, evaluations, decisions and failure memory.

Design notes
- Every write is idempotent. Documents are addressed by content or by a natural id, written with
  `$setOnInsert`, and a second write of the same thing is a no-op; a second write of DIFFERENT content under
  the same id raises `IntegrityError` instead of overwriting history.
- A trace is ONE document with its events embedded, because replay reads a whole trace at once and a
  one-document read is one round trip with no join. Only a trace whose events exceed `max_inline_bytes`
  (default 4 MB, a quarter of BSON's 16 MB cap) is split into ordered chunks in `trace_chunks`; the header
  document is written last, so a header is the commit marker and a crash mid-write leaves no visible trace.
- Harness versions are verified against their digest on every read.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from ratchet.store import atlas_setup
from ratchet.store.embed import CachedEmbedder, Embedder, VoyageEmbedder
from ratchet.types import (
    Cost,
    Event,
    Guard,
    Harness,
    Outcome,
    Procedure,
    Trace,
    canonical_json,
    sha256_hex,
    to_jsonable,
)

Doc = dict[str, Any]
WATCHABLE = ("harness_versions", "runs", "traces", "evals", "decisions", "failures", "checkpoints")
DEFAULT_MAX_INLINE_BYTES = 4_000_000
_CHUNK_BYTES = 1_000_000
RRF_K = 60


class IntegrityError(RuntimeError):
    """Stored or submitted content does not match its digest, or collides with different content."""


@dataclass(frozen=True)
class LineageNode:
    version: str
    parent: str | None
    created_by: str
    digest: str
    created_at: datetime | None = None
    children: list[LineageNode] = field(default_factory=list)

    def walk(self) -> Iterator[LineageNode]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True)
class SimilarFailure:
    id: str
    summary: str
    case_family: str
    trace_ref: str | None
    harness_version: str | None
    score: float


@dataclass(frozen=True)
class ChangeEvent:
    operation: str
    collection: str
    doc_id: Any
    document: Doc | None
    token: Doc


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _now() -> datetime:
    return datetime.now(UTC)


def _harness_doc(h: Harness) -> Doc:
    return {
        "_id": h.version,
        "parent": h.parent,
        "digest": h.digest,
        "created_by": h.created_by,
        "system_prompt": h.system_prompt,
        "procedures": [asdict(p) for p in h.procedures],
        "guards": [asdict(g) for g in h.guards],
        "tool_allowlist": list(h.tool_allowlist),
    }


def _harness_from_doc(doc: Doc) -> Harness:
    return Harness(
        version=doc["_id"],
        parent=doc["parent"],
        system_prompt=doc["system_prompt"],
        procedures=[Procedure(**p) for p in doc["procedures"]],
        guards=[Guard(**g) for g in doc["guards"]],
        tool_allowlist=list(doc["tool_allowlist"]),
        created_by=doc["created_by"],
        digest=doc["digest"],
    )


def trace_id(run_id: str, case_id: str) -> str:
    return f"{run_id}/{case_id}"


def trace_digest(trace: Trace) -> str:
    return sha256_hex(canonical_json(asdict(trace)))


def _trace_from_parts(header: Doc, events: list[Doc]) -> Trace:
    out = header["outcome"]
    trace = Trace(
        run_id=header["run_id"],
        case_id=header["case_id"],
        harness_version=header["harness_version"],
        model=header["model"],
        events=[Event(**e) for e in events],
        outcome=Outcome(
            passed=out["passed"],
            reasons=list(out["reasons"]),
            final_state_digest=out["final_state_digest"],
            cost=Cost(**out["cost"]),
        ),
    )
    if trace_digest(trace) != header["digest"]:
        raise IntegrityError(f"trace {header['_id']} does not match its stored digest")
    return trace


def _split_events(events: list[Doc]) -> list[list[Doc]]:
    chunks: list[list[Doc]] = [[]]
    size = 0
    for event in events:
        n = len(canonical_json(event))
        if chunks[-1] and size + n > _CHUNK_BYTES:
            chunks.append([])
            size = 0
        chunks[-1].append(event)
        size += n
    return chunks


class Store:
    """The Atlas-backed record of every harness version, run, trace, evaluation, decision and failure."""

    def __init__(
        self,
        uri: str,
        db: str,
        *,
        embedder: Embedder | None = None,
        max_inline_bytes: int = DEFAULT_MAX_INLINE_BYTES,
        index_timeout_s: float = 120.0,
    ) -> None:
        self._client: MongoClient[Doc] = MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=10_000)
        self.db = self._client[db]
        self._raw_embedder = embedder
        self._embedder: CachedEmbedder | None = None
        self._owned_embedder: VoyageEmbedder | None = None
        self._max_inline = max_inline_bytes
        self._index_timeout_s = index_timeout_s
        self._schema_ready = False
        self._search_ready = False

    # -- lifecycle ---------------------------------------------------------------------------------

    def close(self) -> None:
        """Close the database client and any embedder this store created itself."""
        if self._owned_embedder is not None:
            self._owned_embedder.close()
        self._client.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def setup(self, *, wait_search: bool = True) -> None:
        """Create collections and indexes (idempotent). Runs lazily before the first write."""
        atlas_setup.ensure_schema(self.db)
        self._schema_ready = True
        if wait_search:
            atlas_setup.wait_search_indexes(self.db, self._index_timeout_s)
            self._search_ready = True

    def _c(self, name: str) -> Collection[Doc]:
        if not self._schema_ready:
            self.setup(wait_search=False)
        return self.db[name]

    @property
    def embedder(self) -> CachedEmbedder:
        if self._embedder is None:
            if self._raw_embedder is None:
                self._owned_embedder = VoyageEmbedder()
            inner = self._raw_embedder or self._owned_embedder
            assert inner is not None
            self._embedder = CachedEmbedder(inner, self._c("embedding_cache"))
        return self._embedder

    # -- harness versions --------------------------------------------------------------------------

    def put_harness(self, harness: Harness) -> bool:
        """Store a harness version. Returns True when newly written, False when it already existed."""
        if not harness.verify_digest():
            raise IntegrityError(f"harness {harness.version} does not match its digest")
        coll = self._c("harness_versions")
        doc = _harness_doc(harness)
        res = coll.update_one(
            {"_id": harness.version}, {"$setOnInsert": {**doc, "created_at": _now()}}, upsert=True
        )
        if res.upserted_id is not None:
            return True
        stored = coll.find_one({"_id": harness.version}, {"digest": 1})
        if stored is None or stored["digest"] != harness.digest:
            raise IntegrityError(f"version {harness.version} already holds different content")
        return False

    def get_harness(self, version: str) -> Harness | None:
        doc = self._c("harness_versions").find_one({"_id": version})
        if doc is None:
            return None
        harness = _harness_from_doc(doc)
        if not harness.verify_digest():
            raise IntegrityError(f"stored harness {version} fails its digest check")
        return harness

    def lineage(self, root: str | None = None) -> list[LineageNode]:
        """The version tree, via one `$graphLookup` per root. `root` limits it to one subtree."""
        match: Doc = {"_id": root} if root is not None else {"parent": None}
        pipeline: list[Doc] = [
            {"$match": match},
            {
                "$graphLookup": {
                    "from": "harness_versions",
                    "startWith": "$_id",
                    "connectFromField": "_id",
                    "connectToField": "parent",
                    "as": "descendants",
                    "depthField": "depth",
                }
            },
            {
                "$project": {
                    "parent": 1,
                    "digest": 1,
                    "created_by": 1,
                    "created_at": 1,
                    "descendants": self._light("descendants"),
                }
            },
        ]
        roots: list[LineageNode] = []
        for doc in self._c("harness_versions").aggregate(pipeline):
            nodes = {doc["_id"]: self._node(doc)}
            for d in sorted(doc["descendants"], key=lambda x: (x["depth"], x["created_at"], x["_id"])):
                nodes[d["_id"]] = self._node(d)
            for node in list(nodes.values())[1:]:
                nodes[node.parent or ""].children.append(node)
            roots.append(nodes[doc["_id"]])
        return sorted(roots, key=lambda n: (n.created_at or _now(), n.version))

    def ancestry(self, version: str) -> list[str]:
        """Versions from `version` up to its root, nearest first, via `$graphLookup` on `parent`."""
        pipeline: list[Doc] = [
            {"$match": {"_id": version}},
            {
                "$graphLookup": {
                    "from": "harness_versions",
                    "startWith": "$parent",
                    "connectFromField": "parent",
                    "connectToField": "_id",
                    "as": "ancestors",
                    "depthField": "depth",
                }
            },
            {
                "$project": {
                    "ancestors": {
                        "$map": {
                            "input": "$ancestors",
                            "as": "a",
                            "in": {"_id": "$$a._id", "depth": "$$a.depth"},
                        }
                    }
                }
            },
        ]
        for doc in self._c("harness_versions").aggregate(pipeline):
            return [version, *[a["_id"] for a in sorted(doc["ancestors"], key=lambda a: a["depth"])]]
        return []

    @staticmethod
    def _light(field_name: str) -> Doc:
        keep = ("_id", "parent", "digest", "created_by", "created_at", "depth")
        return {
            "$map": {
                "input": f"${field_name}",
                "as": "d",
                "in": {k: f"$$d.{k}" for k in keep},
            }
        }

    @staticmethod
    def _node(doc: Doc) -> LineageNode:
        return LineageNode(
            version=doc["_id"],
            parent=doc.get("parent"),
            created_by=doc["created_by"],
            digest=doc["digest"],
            created_at=doc.get("created_at"),
        )

    # -- runs and traces ---------------------------------------------------------------------------

    def put_run(
        self, run_id: str, *, harness_version: str, model: str, meta: Mapping[str, Any] | None = None
    ) -> bool:
        res = self._c("runs").update_one(
            {"_id": run_id},
            {
                "$setOnInsert": {
                    "harness_version": harness_version,
                    "model": model,
                    "meta": dict(meta or {}),
                    "created_at": _now(),
                }
            },
            upsert=True,
        )
        return res.upserted_id is not None

    def get_run(self, run_id: str) -> Doc | None:
        return self._c("runs").find_one({"_id": run_id})

    def put_trace(self, trace: Trace) -> bool:
        """Store a trace. Returns True when newly written. Same trace twice is a no-op."""
        tid = trace_id(trace.run_id, trace.case_id)
        digest = trace_digest(trace)
        coll = self._c("traces")
        existing = coll.find_one({"_id": tid}, {"digest": 1})
        if existing is not None:
            if existing["digest"] != digest:
                raise IntegrityError(f"trace {tid} already holds different content")
            return False
        events = [asdict(e) for e in trace.events]
        header: Doc = {
            "run_id": trace.run_id,
            "case_id": trace.case_id,
            "harness_version": trace.harness_version,
            "model": trace.model,
            "passed": trace.outcome.passed,
            "outcome": asdict(trace.outcome),
            "n_events": len(events),
            "digest": digest,
            "created_at": _now(),
        }
        if sum(len(canonical_json(e)) for e in events) > self._max_inline:
            chunks = _split_events(events)
            self._c("trace_chunks").bulk_write(
                [
                    _upsert_one({"_id": f"{tid}#{n}"}, {"trace_id": tid, "n": n, "events": chunk})
                    for n, chunk in enumerate(chunks)
                ],
                ordered=False,
            )
            header["chunks"] = len(chunks)
        else:
            header["events"] = events
        try:
            coll.insert_one({"_id": tid, **header})
        except DuplicateKeyError:
            # a concurrent writer committed first; fine when it wrote the same content
            stored = coll.find_one({"_id": tid}, {"digest": 1})
            if stored is None or stored["digest"] != digest:
                raise IntegrityError(f"trace {tid} already holds different content") from None
            return False
        return True

    def get_trace(self, run_id: str, case_id: str) -> Trace | None:
        header = self._c("traces").find_one({"_id": trace_id(run_id, case_id)})
        return None if header is None else self._load_trace(header)

    def traces(
        self, filter: Mapping[str, Any] | None = None, *, limit: int = 0, newest_first: bool = False
    ) -> list[Trace]:
        """Traces matching a MongoDB filter over header fields (run_id, case_id, harness_version, passed)."""
        cursor = self._c("traces").find(dict(filter or {}))
        cursor = cursor.sort("created_at", -1 if newest_first else 1)
        if limit:
            cursor = cursor.limit(limit)
        return [self._load_trace(h) for h in cursor]

    def _load_trace(self, header: Doc) -> Trace:
        if "events" in header:
            events = header["events"]
        else:
            chunks = self._c("trace_chunks").find({"trace_id": header["_id"]}).sort("n", 1)
            events = [e for chunk in chunks for e in chunk["events"]]
        return _trace_from_parts(header, events)

    # -- evaluations and decisions -----------------------------------------------------------------

    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str:
        """Store an evaluation report (a dataclass or mapping). The id defaults to the content hash."""
        return self._put_record("evals", report, eval_id)

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        """Store an admission decision (a dataclass or mapping). The id defaults to the content hash."""
        return self._put_record("decisions", decision, decision_id)

    def evals(self, filter: Mapping[str, Any] | None = None) -> list[Doc]:
        return list(self._c("evals").find(dict(filter or {})).sort("created_at", 1))

    def decisions(self, filter: Mapping[str, Any] | None = None) -> list[Doc]:
        return list(self._c("decisions").find(dict(filter or {})).sort("created_at", 1))

    def _put_record(self, collection: str, record: Any, record_id: str | None) -> str:
        body = to_jsonable(record)
        if not isinstance(body, Mapping):
            raise TypeError(
                f"{collection} record must be a dataclass or mapping, got {type(record).__name__}"
            )
        body = dict(body)
        digest = sha256_hex(canonical_json(body))
        rid = record_id or digest
        coll = self._c(collection)
        res = coll.update_one(
            {"_id": rid}, {"$setOnInsert": {**body, "digest": digest, "created_at": _now()}}, upsert=True
        )
        if res.upserted_id is None:
            stored = coll.find_one({"_id": rid}, {"digest": 1})
            if stored is None or stored["digest"] != digest:
                raise IntegrityError(f"{collection} record {rid} already holds different content")
        return rid

    # -- failure memory ----------------------------------------------------------------------------

    def put_failure(
        self,
        summary: str,
        *,
        case_family: str,
        trace_ref: str | None = None,
        harness_version: str | None = None,
        failure_id: str | None = None,
    ) -> str:
        """Store a failure summary with its embedding. Idempotent on `failure_id` (default: content hash)."""
        fid = failure_id or sha256_hex(canonical_json([summary, case_family, trace_ref]))
        coll = self._c("failures")
        if coll.find_one({"_id": fid}, {"_id": 1}) is not None:
            return fid
        vector = self.embedder.embed([summary], "document")[0]
        coll.update_one(
            {"_id": fid},
            {
                "$setOnInsert": {
                    "summary": summary,
                    "case_family": case_family,
                    "trace_ref": trace_ref,
                    "harness_version": harness_version,
                    "embedding": vector,
                    "embedding_model": self.embedder.model,
                    "created_at": _now(),
                }
            },
            upsert=True,
        )
        return fid

    def _ensure_search(self) -> None:
        if not self._search_ready:
            self.setup(wait_search=True)

    def similar_failures(
        self, text: str, k: int = 5, *, case_family: str | None = None, exclude_id: str | None = None
    ) -> list[SimilarFailure]:
        """The k stored failures nearest to `text` by `$vectorSearch` over Voyage embeddings."""
        self._ensure_search()
        vector = self.embedder.embed([text], "query")[0]
        return self._vector_hits(vector, k, case_family, exclude_id)

    def similar_failures_hybrid(
        self, text: str, k: int = 5, *, case_family: str | None = None, exclude_id: str | None = None
    ) -> list[SimilarFailure]:
        """`$search` keyword hits and `$vectorSearch` hits fused with reciprocal rank fusion."""
        self._ensure_search()
        vector = self.embedder.embed([text], "query")[0]
        pool = max(k * 4, 20)
        by_vector = self._vector_hits(vector, pool, case_family, exclude_id)
        by_text = self._text_hits(text, pool, case_family, exclude_id)
        known = {h.id: h for h in [*by_text, *by_vector]}
        fused = rrf_fuse([[h.id for h in by_vector], [h.id for h in by_text]])
        return [SimilarFailure(**{**asdict(known[i]), "score": score}) for i, score in fused[:k]]

    def _vector_hits(
        self, vector: list[float], k: int, case_family: str | None, exclude_id: str | None
    ) -> list[SimilarFailure]:
        stage: Doc = {
            "index": atlas_setup.FAILURES_VECTOR_INDEX,
            "path": "embedding",
            "queryVector": vector,
            "numCandidates": max(k * 20, 100),
            "limit": k + (1 if exclude_id else 0),
        }
        if case_family is not None:
            stage["filter"] = {"case_family": case_family}
        return self._hits(
            [{"$vectorSearch": stage}, {"$addFields": {"score": {"$meta": "vectorSearchScore"}}}],
            k,
            exclude_id,
        )

    def _text_hits(
        self, text: str, k: int, case_family: str | None, exclude_id: str | None
    ) -> list[SimilarFailure]:
        compound: Doc = {"must": [{"text": {"query": text, "path": "summary"}}]}
        if case_family is not None:
            compound["filter"] = [{"equals": {"path": "case_family", "value": case_family}}]
        pipeline: list[Doc] = [
            {"$search": {"index": atlas_setup.FAILURES_TEXT_INDEX, "compound": compound}},
            {"$limit": k + (1 if exclude_id else 0)},
            {"$addFields": {"score": {"$meta": "searchScore"}}},
        ]
        return self._hits(pipeline, k, exclude_id)

    def _hits(self, pipeline: list[Doc], k: int, exclude_id: str | None) -> list[SimilarFailure]:
        pipeline = [*pipeline, {"$project": {"embedding": 0}}]
        out = [
            SimilarFailure(
                id=d["_id"],
                summary=d["summary"],
                case_family=d["case_family"],
                trace_ref=d.get("trace_ref"),
                harness_version=d.get("harness_version"),
                score=float(d["score"]),
            )
            for d in self._c("failures").aggregate(pipeline)
            if d["_id"] != exclude_id
        ]
        return out[:k]

    # -- checkpoints -------------------------------------------------------------------------------

    def checkpoint(self, run_id: str, state: Mapping[str, Any]) -> int:
        """Atomically replace the checkpoint for a run; returns its new sequence number."""
        doc = self._c("checkpoints").find_one_and_update(
            {"_id": run_id},
            {"$set": {"state": dict(state), "updated_at": _now()}, "$inc": {"seq": 1}},
            upsert=True,
            return_document=True,
            projection={"seq": 1},
        )
        return int(doc["seq"]) if doc else 1

    def resume(self, run_id: str) -> Doc | None:
        """The last checkpointed state for a run, or None when the run never checkpointed."""
        doc = self._c("checkpoints").find_one({"_id": run_id})
        return None if doc is None else dict(doc["state"])

    # -- change streams ----------------------------------------------------------------------------

    def watch(
        self,
        collection: str,
        *,
        pipeline: list[Doc] | None = None,
        resume_after: Doc | None = None,
        timeout_s: float | None = None,
        poll_ms: int = 250,
    ) -> Iterator[ChangeEvent]:
        """Yield change-stream events for a collection until `timeout_s` passes (or forever when None)."""
        if collection not in WATCHABLE:
            raise ValueError(f"cannot watch {collection!r}; choose one of {', '.join(WATCHABLE)}")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._c(collection).watch(
            pipeline or [], full_document="updateLookup", resume_after=resume_after, max_await_time_ms=poll_ms
        ) as stream:
            while deadline is None or time.monotonic() < deadline:
                change = stream.try_next()
                if change is None:
                    continue
                document = change.get("fullDocument")
                if document is not None:
                    document = {k: v for k, v in document.items() if k not in ("embedding", "events")}
                yield ChangeEvent(
                    operation=change["operationType"],
                    collection=collection,
                    doc_id=change["documentKey"]["_id"],
                    document=document,
                    token=change["_id"],
                )


def _upsert_one(filter: Doc, fields: Doc) -> UpdateOne:
    return UpdateOne(filter, {"$setOnInsert": fields}, upsert=True)


__all__ = ["ChangeEvent", "IntegrityError", "LineageNode", "SimilarFailure", "Store", "rrf_fuse"]
