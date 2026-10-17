"""The core store: a signed, append-only log in one directory, needing no database.

Every write appends one line to `records.jsonl`: a DSSE envelope (payload type
`application/vnd.ratchet.store-record+json`), signed by the store's Ed25519 key, over the RFC 8785 JSON of

    {"seq", "prev", "collection", "id", "digest", "canonicalization", "at", "doc"}

`prev` is the SHA-256 of the previous line's bytes (64 zeros for the first), so a line edited, removed or
moved breaks every signature or link after it, and `digest` is the record's content digest exactly as the
Atlas store computes it. Opening the store checks every line against the trusted key and the chain and
refuses a log that fails (`IntegrityError`); a reader that only holds the public key opens it read-only.

Records are never changed in place. Writing an id again with the same content is a no-op and with other
content is refused, as in the Atlas store; a checkpoint is a new record and the latest one wins. Writes take
an exclusive lock on the log, so several processes can share one store; each re-reads what the others
appended before it links its own line.

It serves the loop (`ImproveStore`), the views (`ViewStore`) and runs (`WriteStore`). Atlas-only features
(vector search over failure memory, change streams, transactions for activation) stay in
`ratchet.store.atlas`, in the `atlas` extra.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from ratchet_cases.types import Cost, Event, Outcome, Trace, sha256_hex, to_jsonable

from ratchet.canonical import CANONICALIZATION, canonical_json, canonicalizer
from ratchet.improve.activate import Activation, ActivationError, check_binding
from ratchet.store.types import WATCHABLE, ChangeEvent, Doc, IntegrityError, LineageNode
from ratchet.types import Guard, Harness, Procedure
from ratchet.verify import dsse

RECORD_TYPE = "application/vnd.ratchet.store-record+json"
LOG = "records.jsonl"
GENESIS = "0" * 64
COLLECTIONS = ("harness_versions", "traces", "evals", "decisions", "checkpoints", "activations")


def _match(doc: Any, path: list[str], want: Any) -> bool:
    """MongoDB's equality and `$in` on a dotted path, where a list on the path matches through any element."""
    if not path:
        values = doc if isinstance(doc, list) else [doc]
        if isinstance(want, Mapping) and set(want) == {"$in"}:
            return any(v in want["$in"] for v in values)
        return doc == want or (isinstance(doc, list) and want in doc)
    if isinstance(doc, list):
        return any(_match(item, path, want) for item in doc)
    if not isinstance(doc, Mapping) or path[0] not in doc:
        return False
    return _match(doc[path[0]], path[1:], want)


def matches(doc: Mapping[str, Any], filter: Mapping[str, Any] | None) -> bool:
    return all(_match(doc, key.split("."), want) for key, want in (filter or {}).items())


def _digest(body: Doc, canonicalization: str | None) -> str:
    return sha256_hex(canonicalizer(canonicalization)(body))


def _same_content(body: Doc) -> Callable[[Doc], bool]:
    """Whether a stored record holds `body`, under the canonicaliser the stored record was digested with."""
    return lambda stored: bool(stored["digest"] == _digest(body, stored["digest_canonicalization"]))


def _harness_doc(h: Harness) -> Doc:
    return {"_id": h.version, **{k: v for k, v in asdict(h).items() if k != "version"}}


def _harness(doc: Mapping[str, Any]) -> Harness:
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


def _trace(doc: Mapping[str, Any]) -> Trace:
    o = doc["outcome"]
    return Trace(
        run_id=doc["run_id"],
        case_id=doc["case_id"],
        harness_version=doc["harness_version"],
        model=doc["model"],
        events=[Event(**e) for e in doc["events"]],
        outcome=Outcome(o["passed"], list(o["reasons"]), o["final_state_digest"], Cost(**o["cost"])),
    )


class FileStore:
    """A signed append-only store in `root`. Pass the private key to write, or only the public key to read."""

    def __init__(
        self, root: Path, *, key: Ed25519PrivateKey | None = None, trusted: Ed25519PublicKey | None = None
    ) -> None:
        public = key.public_key() if key is not None else trusted
        if public is None:
            raise ValueError("a file store needs its signing key, or the public key to read it with")
        if trusted is not None and dsse.keyid_of(trusted) != dsse.keyid_of(public):
            raise ValueError("the signing key is not the trusted key")
        self.root = Path(root)
        self._key = key
        self._public = public
        self._trusted = {dsse.keyid_of(public): public}
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Doc]] = {c: {} for c in COLLECTIONS}
        self._order: list[tuple[str, str]] = []
        self._seq = 0
        self._prev = GENESIS
        self._offset = 0
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / LOG).touch(exist_ok=True)
        with self._lock:
            self._catch_up()

    @property
    def keyid(self) -> str:
        return dsse.keyid_of(self._public)

    # -- the log -----------------------------------------------------------------------------------------

    def _catch_up(self) -> None:
        """Read and check every line past the last one this process has seen."""
        with (self.root / LOG).open("rb") as fh:
            fh.seek(self._offset)
            for line in fh:
                if not line.endswith(b"\n"):
                    raise IntegrityError(f"{self.root / LOG}: a partial line at byte {self._offset}")
                self._admit(line.rstrip(b"\n"))
                self._offset += len(line)

    def _admit(self, line: bytes) -> None:
        try:
            env = dsse.parse(line)
        except dsse.EnvelopeError as exc:
            raise IntegrityError(f"record {self._seq + 1}: {exc}") from exc
        if env.payload_type != RECORD_TYPE or dsse.verify_signature(env, self._trusted).keyid is None:
            raise IntegrityError(f"record {self._seq + 1} is not signed by the store's key")
        rec = json.loads(env.payload)
        if rec["seq"] != self._seq + 1 or rec["prev"] != self._prev:
            raise IntegrityError(f"record {self._seq + 1} does not follow record {self._seq}")
        self._seq, self._prev = rec["seq"], hashlib.sha256(line).hexdigest()
        doc = {**rec["doc"], "_id": rec["id"], "digest": rec["digest"],
               "digest_canonicalization": rec["canonicalization"], "created_at": rec["at"]}  # fmt: skip
        if rec["collection"] == "checkpoints" or rec["id"] not in self._records[rec["collection"]]:
            self._records[rec["collection"]][rec["id"]] = doc
            self._order.append((rec["collection"], rec["id"]))

    def _append(
        self,
        collection: str,
        rid: str,
        body: Doc,
        digest: str,
        same: Callable[[Doc], bool],
        precondition: Callable[[], None] | None = None,
    ) -> bool:
        """Append a record unless `rid` is already stored; `same(stored)` says whether the stored one holds
        this content (a no-op) or other content (refused). Checkpoints always append. `precondition` runs
        under the log's lock after every other writer's lines are read, so a check and its write cannot
        interleave."""
        if self._key is None:
            raise PermissionError("this file store was opened read-only")
        with self._lock, (self.root / LOG).open("ab") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                self._catch_up()
                stored = self._records[collection].get(rid)
                if stored is not None and collection != "checkpoints":
                    if not same(stored):
                        raise IntegrityError(f"{collection} record {rid} already holds different content")
                    return False
                if precondition is not None:
                    precondition()
                rec = {"seq": self._seq + 1, "prev": self._prev, "collection": collection, "id": rid,
                       "digest": digest, "canonicalization": CANONICALIZATION,
                       "at": datetime.now(UTC).isoformat(), "doc": body}  # fmt: skip
                line = dsse.sign(RECORD_TYPE, canonical_json(rec), [self._key])
                fh.write(line + b"\n")
                fh.flush()
                self._catch_up()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        return True

    def _all(self, collection: str, filter: Mapping[str, Any] | None = None) -> list[Doc]:
        with self._lock:
            self._catch_up()
            return [dict(d) for d in self._records[collection].values() if matches(d, filter)]

    # -- harness versions --------------------------------------------------------------------------------

    def put_harness(self, harness: Harness) -> bool:
        if not harness.verify_digest():
            raise IntegrityError(f"harness {harness.version} does not match its digest")
        body = {k: v for k, v in _harness_doc(harness).items() if k != "_id"}
        return self._append("harness_versions", harness.version, body, harness.digest,
                            lambda stored: stored["digest"] == harness.digest)  # fmt: skip

    def get_harness(self, version: str) -> Harness | None:
        doc = next((d for d in self._all("harness_versions") if d["_id"] == version), None)
        if doc is None:
            return None
        harness = _harness(doc)
        if not harness.verify_digest():
            raise IntegrityError(f"stored harness {version} fails its digest check")
        return harness

    def lineage(self, root: str | None = None) -> list[LineageNode]:
        docs = self._all("harness_versions")
        kids: dict[str | None, list[Doc]] = {}
        for d in docs:
            kids.setdefault(d["parent"], []).append(d)

        def node(d: Doc) -> LineageNode:
            children = sorted(kids.get(d["_id"], []), key=lambda c: (c["created_at"], c["_id"]))
            return LineageNode(d["_id"], d["parent"], d["created_by"], d["digest"],
                               datetime.fromisoformat(d["created_at"]),
                               [node(c) for c in children])  # fmt: skip

        if root is not None:
            return [node(d) for d in docs if d["_id"] == root]
        return [node(d) for d in sorted(kids.get(None, []), key=lambda c: (c["created_at"], c["_id"]))]

    # -- traces ------------------------------------------------------------------------------------------

    def put_trace(self, trace: Trace) -> bool:
        body = asdict(trace)
        tid = f"{trace.run_id}/{trace.case_id}"
        return self._append("traces", tid, body, _digest(body, CANONICALIZATION), _same_content(body))

    def _load_trace(self, doc: Doc) -> Trace:
        trace = _trace(doc)
        if _digest(asdict(trace), doc["digest_canonicalization"]) != doc["digest"]:
            raise IntegrityError(f"trace {doc['_id']} does not match its stored digest")
        return trace

    def get_trace(self, run_id: str, case_id: str) -> Trace | None:
        doc = next((d for d in self._all("traces") if d["_id"] == f"{run_id}/{case_id}"), None)
        return None if doc is None else self._load_trace(doc)

    def traces(
        self, filter: Mapping[str, Any] | None = None, *, limit: int = 0, newest_first: bool = False
    ) -> list[Trace]:
        docs = [d for d in self._all("traces") if matches({**d, "passed": d["outcome"]["passed"]}, filter)]
        if newest_first:
            docs.reverse()
        return [self._load_trace(d) for d in (docs[:limit] if limit else docs)]

    # -- evaluations, decisions, checkpoints ------------------------------------------------------------

    def _put_record(self, collection: str, record: Any, record_id: str | None) -> str:
        body = to_jsonable(record)
        if not isinstance(body, Mapping):
            raise TypeError(
                f"{collection} record must be a dataclass or mapping, got {type(record).__name__}"
            )
        body = dict(body)
        digest = _digest(body, CANONICALIZATION)
        rid = record_id or digest
        self._append(collection, rid, body, digest, _same_content(body))
        return rid

    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str:
        return self._put_record("evals", report, eval_id)

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        return self._put_record("decisions", decision, decision_id)

    def evals(self, filter: Mapping[str, Any] | None = None) -> list[Doc]:
        return self._all("evals", filter)

    def decisions(self, filter: Mapping[str, Any] | None = None) -> list[Doc]:
        return self._all("decisions", filter)

    def checkpoint(self, run_id: str, state: Mapping[str, Any]) -> int:
        """Append a checkpoint for a run; the latest one is what `resume` returns. Returns its sequence."""
        seq = int(self._records["checkpoints"].get(run_id, {}).get("seq", 0)) + 1
        body = {"state": dict(state), "seq": seq}
        self._append("checkpoints", run_id, body, _digest(body, CANONICALIZATION), lambda _stored: False)
        return seq

    def resume(self, run_id: str) -> Doc | None:
        doc = next((d for d in self._all("checkpoints") if d["_id"] == run_id), None)
        return None if doc is None else dict(doc["state"])

    def watch(
        self,
        collection: str,
        *,
        timeout_s: float | None = None,
        resume_after: Doc | None = None,
        poll_ms: int = 250,
    ) -> Iterator[ChangeEvent]:
        """Yield an insert event for every record appended to `collection` after this call returns (or after
        the record whose `{"seq": n}` token is `resume_after`), until `timeout_s` passes, or forever."""
        if collection not in WATCHABLE:
            raise ValueError(f"cannot watch {collection!r}; choose one of {', '.join(WATCHABLE)}")
        with self._lock:
            self._catch_up()
            start = int(resume_after["seq"]) if resume_after else len(self._order)
        return self._follow(collection, start, timeout_s, poll_ms)

    def _follow(
        self, collection: str, seen: int, timeout_s: float | None, poll_ms: int
    ) -> Iterator[ChangeEvent]:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while deadline is None or time.monotonic() < deadline:
            with self._lock:
                self._catch_up()
                fresh = self._order[seen:]
            for i, (coll, rid) in enumerate(fresh, start=seen + 1):
                if coll == collection:
                    doc = {k: v for k, v in self._records[coll][rid].items() if k != "events"}
                    yield ChangeEvent("insert", coll, rid, doc, {"seq": i})
            seen += len(fresh)
            time.sleep(poll_ms / 1000)

    def close(self) -> None:
        """Nothing is held open between calls."""

    def __enter__(self) -> FileStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FileActivator:
    """Activation over the file store: the active pointer is the latest activation record in the log.

    The same rules as the Atlas activator: the stored decision must admit this candidate against the expected
    parent with the named receipt, the pointer moves only from the expected version and revision (checked and
    written under the log's lock), and an idempotency key used again returns the earlier result.
    """

    def __init__(self, store: FileStore) -> None:
        self.store = store

    def _latest(self) -> tuple[str | None, int]:
        acts = self.store._all("activations")
        if not acts:
            return None, 0
        last = acts[-1]
        return last["version"], int(last["revision"])

    def active(self) -> tuple[str | None, int]:
        return self._latest()

    def bootstrap(self, version: str) -> tuple[str | None, int]:
        if self._latest()[0] is None:
            body = {
                "version": version,
                "revision": 1,
                "previous": None,
                "decision_id": "",
                "receipt_digest": "",
            }

            def unset() -> None:
                if self.store._records["activations"]:
                    raise ActivationError("another writer activated a version first")

            self.store._append("activations", "bootstrap", body, _digest(body, CANONICALIZATION),
                               lambda stored: stored["version"] == version, unset)  # fmt: skip
        return self._latest()

    def activate(
        self,
        *,
        candidate: str,
        expected: str | None,
        expected_revision: int,
        decision_id: str,
        receipt_digest: str,
        idempotency_key: str,
    ) -> Activation:
        done = next((a for a in self.store._all("activations") if a["_id"] == idempotency_key), None)
        if done is not None:
            if done["version"] != candidate:
                raise ActivationError("idempotency key already used for a different candidate")
            return Activation(done["version"], int(done["revision"]), done["previous"], done["decision_id"],
                              done["receipt_digest"], idempotency_key, replayed=True)  # fmt: skip
        body = {"version": candidate, "revision": expected_revision + 1, "previous": expected,
                "decision_id": decision_id, "receipt_digest": receipt_digest}  # fmt: skip

        def guarded() -> None:
            decision = self.store._records["decisions"].get(decision_id)
            check_binding(decision, candidate, expected, receipt_digest)
            acts = list(self.store._records["activations"].values())
            pointer = (acts[-1]["version"], int(acts[-1]["revision"])) if acts else (None, 0)
            if pointer != (expected, expected_revision):
                raise ActivationError("stale active reference: the active version or revision moved")

        self.store._append("activations", idempotency_key, body, _digest(body, CANONICALIZATION),
                           lambda stored: stored["version"] == candidate, guarded)  # fmt: skip
        return Activation(
            candidate, expected_revision + 1, expected, decision_id, receipt_digest, idempotency_key
        )
