"""Declared, deterministic faults. Each fires at a fixed point in the agent's own call sequence.

Only the agent's calls advance the step counter and count towards triggers; the second worker and the fault
operations themselves never do. Nothing here is random.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FaultKind = Literal["timeout_after_commit", "timeout_before_commit", "mutate"]


@dataclass(frozen=True)
class Fault:
    """One scheduled fault.

    - `timeout_after_commit`: the agent's `nth` call to `tool` takes effect, then returns `error="timeout"`.
    - `timeout_before_commit`: the agent's `nth` call to `tool` returns `error="timeout"` and changes nothing.
    - `mutate`: apply `ops` after the agent's `after_step`-th call overall, or `delay` calls after the agent's
      `nth` call to `tool` (delay 0 means straight after that call). Ops:
      `{"op": "call", "actor": "worker:dana", "tool": ..., "args": {...}}` (the second worker acts),
      `{"op": "approval_revoke", "approval_id": ..., "by": ..., "reason": ...}`,
      `{"op": "policy_publish", "doc_id": ..., "version": {...}}`,
      `{"op": "customer_message", "ticket_id": ..., "body": ..., "requests": [...]}`.
    """

    id: str
    kind: FaultKind
    tool: str | None = None
    nth: int = 1
    after_step: int | None = None
    delay: int = 0
    ops: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
