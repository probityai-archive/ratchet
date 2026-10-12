"""Activation: make an admitted candidate the active worker harness, separately from admitting it.

Order is fixed: the decision and its signed receipt are stored first; activation then runs in one MongoDB
transaction that (1) returns the earlier result when the idempotency key was already used, (2) checks that
the stored decision admits exactly this candidate against exactly the expected active parent and that the
stored envelope has the expected digest, and (3) moves the active pointer only if it still names the
expected version and revision (compare and swap). A missing receipt, a rejected decision, or a stale active
reference leaves the active harness untouched.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from pymongo import MongoClient
from pymongo.client_session import ClientSession
from pymongo.errors import DuplicateKeyError
from ratchet_cases.types import sha256_hex

ACTIVE_ID = "worker"


class ActivationError(RuntimeError):
    """The activation was refused; the active harness did not change."""


@dataclass(frozen=True)
class Activation:
    version: str
    revision: int
    previous: str | None
    decision_id: str
    receipt_digest: str
    idempotency_key: str
    replayed: bool = False


class Activator(Protocol):
    def active(self) -> tuple[str | None, int]: ...

    def bootstrap(self, version: str) -> tuple[str | None, int]: ...

    def activate(
        self,
        *,
        candidate: str,
        expected: str | None,
        expected_revision: int,
        decision_id: str,
        receipt_digest: str,
        idempotency_key: str,
    ) -> Activation: ...


def envelope_digest(envelope_b64: str) -> str:
    return sha256_hex(base64.b64decode(envelope_b64))


def check_binding(decision: dict[str, Any] | None, candidate: str, expected: str | None, digest: str) -> None:
    """The stored decision must exist, admit this candidate against the expected parent, and carry the
    receipt whose digest the caller names."""
    if decision is None:
        raise ActivationError("no stored decision: the receipt must be persisted before activation")
    if not decision.get("admitted"):
        raise ActivationError("the stored decision did not admit this candidate")
    if decision.get("candidate") != candidate or decision.get("parent") != expected:
        raise ActivationError("the stored decision names a different candidate or parent")
    env = decision.get("envelope")
    if not isinstance(env, str) or envelope_digest(env) != digest:
        raise ActivationError("the stored receipt is missing or its digest differs")


class MongoActivator:
    """The active pointer, the activation log and the decisions live in one database."""

    def __init__(self, client: MongoClient[dict[str, Any]], db: str) -> None:
        self.client = client
        self.db = client[db]

    def active(self) -> tuple[str | None, int]:
        doc = self.db["active"].find_one({"_id": ACTIVE_ID})
        return (None, 0) if doc is None else (doc["version"], int(doc["revision"]))

    def bootstrap(self, version: str) -> tuple[str | None, int]:
        """Make the seed active when nothing is; a no-op when a harness already is."""
        doc = {"version": version, "revision": 1, "decision_id": "seed", "receipt_digest": None}
        self.db["active"].update_one({"_id": ACTIVE_ID}, {"$setOnInsert": doc}, upsert=True)
        return self.active()

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
        with self.client.start_session() as session:
            result: Activation = session.with_transaction(
                lambda s: self._activate(
                    s, candidate, expected, expected_revision, decision_id, receipt_digest, idempotency_key
                )
            )
            return result

    def _activate(
        self,
        s: ClientSession,
        candidate: str,
        expected: str | None,
        revision: int,
        decision_id: str,
        digest: str,
        key: str,
    ) -> Activation:
        log = self.db["activations"]
        done = log.find_one({"_id": key}, session=s)
        if done is not None:
            if done["version"] != candidate:
                raise ActivationError("idempotency key already used for a different candidate")
            return Activation(
                done["version"],
                done["revision"],
                done["previous"],
                done["decision_id"],
                done["receipt_digest"],
                key,
                replayed=True,
            )
        check_binding(
            self.db["decisions"].find_one({"_id": decision_id}, session=s), candidate, expected, digest
        )
        new = {
            "version": candidate,
            "revision": revision + 1,
            "decision_id": decision_id,
            "receipt_digest": digest,
        }
        active = self.db["active"]
        if expected is None:
            if revision != 0:
                raise ActivationError("first activation must expect revision 0")
            try:
                active.insert_one({"_id": ACTIVE_ID, **new}, session=s)
            except DuplicateKeyError as exc:
                raise ActivationError("stale active reference: a harness is already active") from exc
        else:
            res = active.update_one(
                {"_id": ACTIVE_ID, "version": expected, "revision": revision}, {"$set": new}, session=s
            )
            if res.matched_count != 1:
                raise ActivationError("stale active reference: the active version or revision moved")
        at = datetime.now(UTC).isoformat(timespec="seconds")
        log.insert_one({"_id": key, **new, "previous": expected, "at": at}, session=s)
        return Activation(candidate, revision + 1, expected, decision_id, digest, key)


class MemoryActivator:
    """The same rules over an in-memory decision map, for tests and dry runs."""

    def __init__(self, lookup: Callable[[str], dict[str, Any] | None]) -> None:
        self.lookup = lookup
        self.pointer: tuple[str | None, int] = (None, 0)
        self.log: dict[str, Activation] = {}

    def active(self) -> tuple[str | None, int]:
        return self.pointer

    def bootstrap(self, version: str) -> tuple[str | None, int]:
        if self.pointer[0] is None:
            self.pointer = (version, 1)
        return self.pointer

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
        done = self.log.get(idempotency_key)
        if done is not None:
            if done.version != candidate:
                raise ActivationError("idempotency key already used for a different candidate")
            return Activation(
                done.version,
                done.revision,
                done.previous,
                done.decision_id,
                done.receipt_digest,
                idempotency_key,
                replayed=True,
            )
        check_binding(self.lookup(decision_id), candidate, expected, receipt_digest)
        if self.pointer != (expected, expected_revision):
            raise ActivationError("stale active reference: the active version or revision moved")
        act = Activation(
            candidate, expected_revision + 1, expected, decision_id, receipt_digest, idempotency_key
        )
        self.pointer = (candidate, expected_revision + 1)
        self.log[idempotency_key] = act
        return act
