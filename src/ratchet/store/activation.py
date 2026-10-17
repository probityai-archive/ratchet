"""Activation in MongoDB: the compare-and-swap transaction behind `ratchet.improve.activate.Activator`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pymongo import MongoClient
from pymongo.client_session import ClientSession
from pymongo.errors import DuplicateKeyError

from ratchet.improve.activate import ACTIVE_ID, Activation, ActivationError, check_binding


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
