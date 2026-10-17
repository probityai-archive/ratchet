"""The store's plain record types, shared by every backend and importable without a database client."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ratchet_cases.types import sha256_hex

from ratchet.canonical import canonical_json

Doc = dict[str, Any]
WATCHABLE = (
    "harness_versions",
    "runs",
    "traces",
    "evals",
    "decisions",
    "failures",
    "checkpoints",
    "activations",
)


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
