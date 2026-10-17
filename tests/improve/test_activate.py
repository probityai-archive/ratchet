"""Activation by compare-and-swap, in memory and in a MongoDB transaction on the local Atlas container."""

from __future__ import annotations

import base64
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from pymongo import MongoClient

from ratchet.improve.activate import (
    ActivationError,
    Activator,
    MemoryActivator,
    envelope_digest,
)
from ratchet.store.activation import MongoActivator

LOCAL_URI = "mongodb://ratchet:ratchet-local-dev@127.0.0.1:27018/?directConnection=true"
ENV = base64.b64encode(b"signed-envelope").decode()
DIGEST = envelope_digest(ENV)


def _decision(parent: str | None, cand: str, admitted: bool = True) -> dict[str, Any]:
    return {"parent": parent, "candidate": cand, "admitted": admitted, "envelope": ENV}


DECISIONS = {
    "none->c1": _decision(None, "c1"),
    "s->c1": _decision("s", "c1"),
    "c1->c2": _decision("c1", "c2"),
    "s->bad": _decision("s", "bad", False),
}


@pytest.fixture(scope="module")
def mongo() -> Iterator[tuple[MongoClient[dict[str, Any]], str]]:
    client: MongoClient[dict[str, Any]] = MongoClient(LOCAL_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    db = f"improve_activate_{uuid.uuid4().hex[:8]}"
    yield client, db
    client.drop_database(db)
    client.close()


def _activators(mongo: tuple[MongoClient[dict[str, Any]], str]) -> list[Activator]:
    client, db = mongo
    client[db]["decisions"].delete_many({})
    client[db]["active"].delete_many({})
    client[db]["activations"].delete_many({})
    client[db]["decisions"].insert_many([{"_id": k, **v} for k, v in DECISIONS.items()])
    return [MemoryActivator(DECISIONS.get), MongoActivator(client, db)]


def test_activation_moves_the_pointer_once_and_is_idempotent(mongo: Any) -> None:
    for act in _activators(mongo):
        assert act.bootstrap("s") == ("s", 1) and act.bootstrap("other") == ("s", 1)
        a = act.activate(candidate="c1", expected="s", expected_revision=1, decision_id="s->c1",
                         receipt_digest=DIGEST, idempotency_key="k1")  # fmt: skip
        assert (a.version, a.revision, a.previous, a.replayed) == ("c1", 2, "s", False)
        again = act.activate(candidate="c1", expected="s", expected_revision=1, decision_id="s->c1",
                             receipt_digest=DIGEST, idempotency_key="k1")  # fmt: skip
        assert again.replayed and act.active() == ("c1", 2)
        with pytest.raises(ActivationError, match="different candidate"):
            act.activate(candidate="c2", expected="c1", expected_revision=2, decision_id="c1->c2",
                         receipt_digest=DIGEST, idempotency_key="k1")  # fmt: skip


def test_stale_references_missing_receipts_and_rejections_change_nothing(mongo: Any) -> None:
    for act in _activators(mongo):
        act.bootstrap("s")
        cases = [
            ({"candidate": "c2", "expected": "c1", "expected_revision": 2, "decision_id": "c1->c2",
              "receipt_digest": DIGEST}, "stale"),
            ({"candidate": "c1", "expected": "s", "expected_revision": 1, "decision_id": "nope",
              "receipt_digest": DIGEST}, "persisted before activation"),
            ({"candidate": "c1", "expected": "s", "expected_revision": 1, "decision_id": "s->c1",
              "receipt_digest": "0" * 64}, "digest differs"),
            ({"candidate": "bad", "expected": "s", "expected_revision": 1, "decision_id": "s->bad",
              "receipt_digest": DIGEST}, "did not admit"),
            ({"candidate": "c2", "expected": "s", "expected_revision": 1, "decision_id": "s->c1",
              "receipt_digest": DIGEST}, "different candidate or parent"),
            ({"candidate": "c1", "expected": "s", "expected_revision": 7, "decision_id": "s->c1",
              "receipt_digest": DIGEST}, "stale"),
        ]  # fmt: skip
        for i, (kw, match) in enumerate(cases):
            with pytest.raises(ActivationError, match=match):
                act.activate(idempotency_key=f"x{i}", **kw)
        assert act.active() == ("s", 1)


def test_first_activation_without_a_seed(mongo: Any) -> None:
    for act in _activators(mongo):
        with pytest.raises(ActivationError, match=r"revision 0|stale"):
            act.activate(candidate="c1", expected=None, expected_revision=3, decision_id="none->c1",
                         receipt_digest=DIGEST, idempotency_key="f")  # fmt: skip
        assert act.activate(candidate="c1", expected=None, expected_revision=0, decision_id="none->c1",
                            receipt_digest=DIGEST, idempotency_key="g").revision == 1  # fmt: skip
        with pytest.raises(ActivationError, match="stale"):
            act.activate(candidate="c1", expected=None, expected_revision=0, decision_id="none->c1",
                         receipt_digest=DIGEST, idempotency_key="h")  # fmt: skip
