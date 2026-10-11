"""In-memory stand-ins for the pymongo and Voyage objects the probes use, so unit tests need no network."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from pymongo.errors import OperationFailure

from ratchet.store.embed import InputType


class FakeCollection:
    """Records calls; `aggregate_fn` decides what an aggregation returns or raises."""

    def __init__(
        self, name: str, database: FakeDb, aggregate_fn: Callable[[list[dict[str, Any]]], Any] | None = None
    ):
        self.name = name
        self.database = database
        self.docs: dict[Any, dict[str, Any]] = {}
        self.search_indexes: list[dict[str, Any]] = []
        self.created_indexes: list[Any] = []
        self.pipelines: list[list[dict[str, Any]]] = []
        self.aggregate_fn = aggregate_fn or (lambda _p: [])

    def replace_one(self, flt: dict[str, Any], doc: dict[str, Any], upsert: bool = False) -> None:
        self.docs[flt["_id"]] = dict(doc)

    def insert_one(self, doc: dict[str, Any]) -> None:
        self.docs[doc["_id"] if "_id" in doc else f"auto{len(self.docs)}"] = dict(doc)

    def update_one(self, flt: dict[str, Any], upd: dict[str, Any]) -> None:
        self.docs[flt["_id"]].update(upd["$set"])

    def delete_one(self, flt: dict[str, Any]) -> None:
        self.docs.pop(flt["_id"], None)

    def delete_many(self, flt: dict[str, Any]) -> Any:
        n = len(self.docs)
        self.docs.clear()
        return type("R", (), {"deleted_count": n})()

    def count_documents(self, flt: dict[str, Any]) -> int:
        if "_id" in flt:
            return int(flt["_id"] in self.docs)
        return len(self.docs)

    def find_one(self, flt: dict[str, Any]) -> dict[str, Any] | None:
        return self.docs.get(flt.get("_id"))

    def list_search_indexes(self) -> list[dict[str, Any]]:
        return list(self.search_indexes)

    def create_search_index(self, model: Any) -> str:
        self.created_indexes.append(model)
        return str(model.document["name"])

    def aggregate(self, pipeline: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        self.pipelines.append(pipeline)
        out = self.aggregate_fn(pipeline)
        if isinstance(out, Exception):
            raise out
        return iter(out)


class FakeDb:
    def __init__(self, name: str = "ratchet_featureprobe") -> None:
        self.name = name
        self.colls: dict[str, FakeCollection] = {}
        self.created: list[str] = []

    def __getitem__(self, name: str) -> FakeCollection:
        return self.colls.setdefault(name, FakeCollection(name, self))

    def list_collection_names(self) -> list[str]:
        return list(self.colls)

    def create_collection(self, name: str) -> None:
        self.created.append(name)
        self.colls.setdefault(name, FakeCollection(name, self))


class FakeAdmin:
    def command(self, name: str) -> dict[str, Any]:
        return {"version": "9.0.4", "ok": 1.0}


class FakeClient:
    def __init__(self) -> None:
        self.admin = FakeAdmin()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ConstEmbedder:
    model = "const"

    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


@contextmanager
def const_embedder(_db: Any) -> Iterator[ConstEmbedder]:
    yield ConstEmbedder()


def op_failure(code: int, errmsg: str) -> OperationFailure:
    return OperationFailure(errmsg, code, {"errmsg": errmsg, "codeName": f"Code{code}"})
