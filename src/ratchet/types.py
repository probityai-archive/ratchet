"""Shared value types. Every lane imports these; only the integrator changes them."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

EventKind = Literal["model", "tool_call", "tool_result", "guard_block", "procedure", "checkpoint", "final"]
Split = Literal["train", "heldout"]
Author = Literal["seed", "improver", "human"]


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace, UTF-8. The input to every digest."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    call_id: str


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class Event:
    seq: int
    kind: EventKind
    payload: dict[str, Any]
    ts: float


@dataclass(frozen=True)
class Cost:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    wall_s: float = 0.0


@dataclass(frozen=True)
class Outcome:
    passed: bool
    reasons: list[str]
    final_state_digest: str
    cost: Cost = field(default_factory=Cost)


@dataclass(frozen=True)
class Trace:
    run_id: str
    case_id: str
    harness_version: str
    model: str
    events: list[Event]
    outcome: Outcome


@dataclass(frozen=True)
class Guard:
    id: str
    description: str
    source: str
    origin_failure_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Procedure:
    id: str
    name: str
    when: str
    steps: list[str]
    origin_failure_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Harness:
    """A complete, content-addressed harness version. Build new ones with `Harness.build`."""

    version: str
    parent: str | None
    system_prompt: str
    procedures: list[Procedure]
    guards: list[Guard]
    tool_allowlist: list[str]
    created_by: Author
    digest: str

    @staticmethod
    def content_of(
        parent: str | None,
        system_prompt: str,
        procedures: list[Procedure],
        guards: list[Guard],
        tool_allowlist: list[str],
        created_by: Author,
    ) -> dict[str, Any]:
        return {
            "parent": parent,
            "system_prompt": system_prompt,
            "procedures": [asdict(p) for p in procedures],
            "guards": [asdict(g) for g in guards],
            "tool_allowlist": sorted(tool_allowlist),
            "created_by": created_by,
        }

    @classmethod
    def build(
        cls,
        *,
        parent: str | None,
        system_prompt: str,
        procedures: list[Procedure],
        guards: list[Guard],
        tool_allowlist: list[str],
        created_by: Author,
    ) -> Harness:
        content = cls.content_of(parent, system_prompt, procedures, guards, tool_allowlist, created_by)
        digest = sha256_hex(canonical_json(content))
        return cls(
            version=digest[:12],
            parent=parent,
            system_prompt=system_prompt,
            procedures=list(procedures),
            guards=list(guards),
            tool_allowlist=sorted(tool_allowlist),
            created_by=created_by,
            digest=digest,
        )

    def verify_digest(self) -> bool:
        content = self.content_of(
            self.parent,
            self.system_prompt,
            self.procedures,
            self.guards,
            self.tool_allowlist,
            self.created_by,
        )
        digest = sha256_hex(canonical_json(content))
        return digest == self.digest and self.version == digest[:12]

    def child(self, **changes: Any) -> Harness:
        """A new version whose parent is this one. Fields not named keep their values."""
        base = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("version", "digest")}
        base.update(changes)
        base["parent"] = self.version
        return Harness.build(**base)


def to_jsonable(obj: Any) -> Any:
    """Dataclass tree to plain JSON-compatible values."""
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
