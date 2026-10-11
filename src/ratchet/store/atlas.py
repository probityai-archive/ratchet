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
from ratchet.store.embed import CachedEmbedder, Embedder, Reranker, VoyageEmbedder, VoyageReranker
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
class NewFailure:
    summary: str
    case_family: str
    trace_ref: str | None = None
    harness_version: str | None = None
    failure_id: str | None = None

    @property
    def id(self) -> str:
        return self.failure_id or sha256_hex(canonical_json([self.summary, self.case_family, self.trace_ref]))


@dataclass(frozen=True)
class ChangeEvent:
    operation: str
    collection: str
    doc_id: Any
    document: Doc | None
    token: Doc


def _filters(case_family: str | None, harness_version: str | None) -> dict[str, str]:
    return {
        name: value
        for name, value in (("case_family", case_family), ("harness_version", harness_version))
        if value is not None
    }


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
        heldout_db: str | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        max_inline_bytes: int = DEFAULT_MAX_INLINE_BYTES,
        index_timeout_s: float = 120.0,
    ) -> None:
        self._client: MongoClient[Doc] = MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=10_000)
        self.db = self._client[db]
        # held-out cases live in their own database so a database role can deny the improver every read
        self.heldout = self._client[heldout_db or f"{db}_heldout"]
        self._raw_embedder = embedder
        self._embedder: CachedEmbedder | None = None
        self._reranker = reranker
        self._owned_reranker: VoyageReranker | None = None
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
        if self._owned_reranker is not None:
            self._owned_reranker.close()
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
        """A collection for writing: the schema is created first, once per store."""
        if not self._schema_ready:
            self.setup(wait_search=False)
        return self.db[name]

    def _r(self, name: str) -> Collection[Doc]:
        """A collection for reading: never creates or changes schema, so a read cannot fail on index setup."""
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
        doc = self._r("harness_versions").find_one({"_id": version})
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
        for doc in self._r("harness_versions").aggregate(pipeline):
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
        for doc in self._r("harness_versions").aggregate(pipeline):
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
        return self._r("runs").find_one({"_id": run_id})

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
        header = self._r("traces").find_one({"_id": trace_id(run_id, case_id)})
        return None if header is None else self._load_trace(header)

    def traces(
        self, filter: Mapping[str, Any] | None = None, *, limit: int = 0, newest_first: bool = False
    ) -> list[Trace]:
        """Traces matching a MongoDB filter over header fields (run_id, case_id, harness_version, passed)."""
        cursor = self._r("traces").find(dict(filter or {}))
        cursor = cursor.sort("created_at", -1 if newest_first else 1)
        if limit:
            cursor = cursor.limit(limit)
        return [self._load_trace(h) for h in cursor]

    def _load_trace(self, header: Doc) -> Trace:
        if "events" in header:
            events = header["events"]
        else:
            chunks = self._r("trace_chunks").find({"trace_id": header["_id"]}).sort("n", 1)
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
        return list(self._r("evals").find(dict(filter or {})).sort("created_at", 1))

    def decisions(self, filter: Mapping[str, Any] | None = None) -> list[Doc]:
        return list(self._r("decisions").find(dict(filter or {})).sort("created_at", 1))

    def put_case(self, case: Any) -> str:
        """Store a case. Held-out cases go to the separate held-out database; the rest to `cases`.

        The case is a dataclass or mapping with `id` and `split` ("train" or "heldout").
        """
        body = to_jsonable(case)
        if not isinstance(body, Mapping) or "id" not in body or body.get("split") not in ("train", "heldout"):
            raise TypeError("a case needs an id and a split of 'train' or 'heldout'")
        if body["split"] == "heldout":
            atlas_setup.ensure_heldout(self.heldout)
            return self._put_record(self.heldout[atlas_setup.HELDOUT_COLLECTION], body, body["id"])
        return self._put_record(self._c("cases"), body, body["id"])

    def cases(self, split: str = "train") -> list[Doc]:
        """Cases of one split. Reading "heldout" needs a role that can read the held-out database."""
        coll = self.heldout[atlas_setup.HELDOUT_COLLECTION] if split == "heldout" else self._r("cases")
        return list(coll.find({"split": split}).sort("_id", 1))

    def _put_record(self, collection: str | Collection[Doc], record: Any, record_id: str | None) -> str:
        body = to_jsonable(record)
        if not isinstance(body, Mapping):
            raise TypeError(
                f"{collection} record must be a dataclass or mapping, got {type(record).__name__}"
            )
        body = dict(body)
        digest = sha256_hex(canonical_json(body))
        rid = record_id or digest
        coll = self._c(collection) if isinstance(collection, str) else collection
        res = coll.update_one(
            {"_id": rid}, {"$setOnInsert": {**body, "digest": digest, "created_at": _now()}}, upsert=True
        )
        if res.upserted_id is None:
            stored = coll.find_one({"_id": rid}, {"digest": 1})
            if stored is None or stored["digest"] != digest:
                raise IntegrityError(f"{coll.name} record {rid} already holds different content")
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
        failure = NewFailure(summary, case_family, trace_ref, harness_version, failure_id)
        self.put_failures([failure])
        return failure.id

    def put_failures(self, failures: Sequence[NewFailure]) -> list[str]:
        """Store many failures, embedding all the new ones in one batch. Idempotent per failure id."""
        coll = self._c("failures")
        ids = [f.id for f in failures]
        have = {d["_id"] for d in coll.find({"_id": {"$in": ids}}, {"_id": 1})}
        fresh = {f.id: f for f in failures if f.id not in have}
        if fresh:
            vectors = self.embedder.embed([f.summary for f in fresh.values()], "document")
            now = _now()
            coll.bulk_write(
                [
                    _upsert_one(
                        {"_id": fid},
                        {
                            "summary": f.summary,
                            "case_family": f.case_family,
                            "trace_ref": f.trace_ref,
                            "harness_version": f.harness_version,
                            "embedding": vector,
                            "embedding_model": self.embedder.model,
                            "created_at": now,
                        },
                    )
                    for (fid, f), vector in zip(fresh.items(), vectors, strict=True)
                ],
                ordered=False,
            )
        return ids

    def _ensure_search(self) -> None:
        if not self._search_ready:
            self.setup(wait_search=True)

    def similar_failures(
        self,
        text: str,
        k: int = 5,
        *,
        case_family: str | None = None,
        harness_version: str | None = None,
        exclude_id: str | None = None,
    ) -> list[SimilarFailure]:
        """The k stored failures nearest to `text` by `$vectorSearch` over Voyage embeddings."""
        self._ensure_search()
        vector = self.embedder.embed([text], "query")[0]
        filters = _filters(case_family, harness_version)
        return self._vector_hits(vector, k, filters, exclude_id)

    def similar_failures_hybrid(
        self,
        text: str,
        k: int = 5,
        *,
        case_family: str | None = None,
        harness_version: str | None = None,
        exclude_id: str | None = None,
        rerank: bool = False,
    ) -> list[SimilarFailure]:
        """Keyword (`$search`) and vector (`$vectorSearch`) hits fused by the server's `$rankFusion`.

        `$rankFusion` is reciprocal rank fusion, weight / (60 + rank) summed over both lists. With
        `rerank=True` the fused shortlist is reordered by Voyage's cross-encoder reranker and each score is
        its relevance score; otherwise the score is the fusion score.
        """
        self._ensure_search()
        vector = self.embedder.embed([text], "query")[0]
        filters = _filters(case_family, harness_version)
        pool = max(k * 4, 20)
        fusion: Doc = {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vec": [{"$vectorSearch": self._vector_stage(vector, pool, filters)}],
                        "txt": [{"$search": self._text_stage(text, filters)}, {"$limit": pool}],
                    }
                },
                "combination": {"weights": {"vec": 1, "txt": 1}},
                "scoreDetails": True,
            }
        }
        shortlist = self._hits(
            [
                fusion,
                {"$limit": pool + 1},
                {
                    "$addFields": {
                        "score": {"$getField": {"field": "value", "input": {"$meta": "scoreDetails"}}}
                    }
                },
            ],
            pool,
            exclude_id,
        )
        if not rerank:
            return shortlist[:k]
        ranked = self._rerank_model().rerank(text, [h.summary for h in shortlist], k)
        return [SimilarFailure(**{**asdict(shortlist[idx]), "score": score}) for idx, score in ranked]

    def _rerank_model(self) -> Reranker:
        if self._reranker is None:
            self._owned_reranker = VoyageReranker()
            self._reranker = self._owned_reranker
        return self._reranker

    @staticmethod
    def _vector_stage(vector: list[float], limit: int, filters: dict[str, str]) -> Doc:
        stage: Doc = {
            "index": atlas_setup.FAILURES_VECTOR_INDEX,
            "path": "embedding",
            "queryVector": vector,
            "numCandidates": max(limit * 20, 100),
            "limit": limit,
        }
        if filters:
            stage["filter"] = {"$and": [{f: v} for f, v in filters.items()]} if len(filters) > 1 else filters
        return stage

    @staticmethod
    def _text_stage(text: str, filters: dict[str, str]) -> Doc:
        compound: Doc = {"must": [{"text": {"query": text, "path": "summary"}}]}
        if filters:
            compound["filter"] = [{"equals": {"path": f, "value": v}} for f, v in filters.items()]
        return {"index": atlas_setup.FAILURES_TEXT_INDEX, "compound": compound}

    def _vector_hits(
        self, vector: list[float], k: int, filters: dict[str, str], exclude_id: str | None
    ) -> list[SimilarFailure]:
        stage = self._vector_stage(vector, k + (1 if exclude_id else 0), filters)
        return self._hits(
            [{"$vectorSearch": stage}, {"$addFields": {"score": {"$meta": "vectorSearchScore"}}}],
            k,
            exclude_id,
        )

    def wait_indexed(self, timeout_s: float = 60.0, poll_s: float = 0.25) -> int:
        """Block until both search indexes have caught up with every stored failure; returns that count.

        Atlas search indexes are eventually consistent: a document is queryable by `$vectorSearch` and by
        `$search` at different moments, so a hybrid query right after a write can see only one list.
        """
        self._ensure_search()
        coll = self._r("failures")
        want = coll.count_documents({})
        deadline = time.monotonic() + timeout_s
        while True:
            seen_text = next(
                coll.aggregate(
                    [
                        {
                            "$searchMeta": {
                                "index": atlas_setup.FAILURES_TEXT_INDEX,
                                "exists": {"path": "summary"},
                                "count": {"type": "total"},
                            }
                        }
                    ]
                ),
                {"count": {"total": 0}},
            )["count"]["total"]
            probe = coll.find_one({}, {"embedding": 1})
            seen_vec = 0
            if probe is not None:
                seen_vec = len(
                    list(
                        coll.aggregate(
                            [
                                {"$vectorSearch": self._vector_stage(probe["embedding"], want, {})},
                                {"$project": {"_id": 1}},
                            ]
                        )
                    )
                )
            if seen_text >= want and seen_vec >= want:
                return want
            if time.monotonic() > deadline:
                raise TimeoutError(f"search indexes saw text={seen_text} vector={seen_vec} of {want}")
            time.sleep(poll_s)

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
            for d in self._r("failures").aggregate(pipeline)
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
        doc = self._r("checkpoints").find_one({"_id": run_id})
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
        with self._r(collection).watch(
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


__all__ = [
    "ChangeEvent",
    "IntegrityError",
    "LineageNode",
    "NewFailure",
    "SimilarFailure",
    "Store",
]
