"""Value types shared by the case suite and any harness that runs it: calls, results, events, traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EventKind = Literal["model", "tool_call", "tool_result", "guard_block", "procedure", "checkpoint", "final"]
Split = Literal["train", "heldout", "audit"]


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


def to_jsonable(obj: Any) -> Any:
    """Dataclass tree to plain JSON-compatible values."""
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
