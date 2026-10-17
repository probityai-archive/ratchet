"""Value types shared by the case suite and any harness that runs it: calls, results, events, traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from probityai import jcs

EventKind = Literal["model", "tool_call", "tool_result", "guard_block", "procedure", "checkpoint", "final"]
Split = Literal["train", "heldout", "audit"]


def canonical_json(obj: Any) -> bytes:
    """RFC 8785 canonical JSON bytes of `obj`, the input to every digest.

    The JSON text Python writes is admitted and canonicalised by `probityai.jcs` (the `jcs-admit` crate):
    members sorted by UTF-16 code units, numbers in ECMAScript form (`1.0` is `1`, `1e-07` is `1e-7`),
    and no whitespace. NaN and infinities are refused, as RFC 8785 requires. An integer beyond 2**53 is
    written as the nearest double, so a value that needs every digit is carried as a string.
    """
    text = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return jcs.admit_with(text, max_bytes=None)


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
