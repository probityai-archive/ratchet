"""A local JSON Lines store for benchmark runs: harnesses, traces, evaluations and signed decisions.

It implements the store surface `ratchet.improve.loop.evolve` writes to, so a benchmark run keeps every
version, trace and proof on disk (under `private/bench/tau/`, which is never committed) without needing a
database. Each record is one line; nothing is ever rewritten.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ratchet_cases.types import Cost, Event, Outcome, Trace

from ratchet.types import Guard, Harness, Procedure


def _default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def harness_from(raw: dict[str, Any]) -> Harness:
    h = Harness(
        version=raw["version"],
        parent=raw["parent"],
        system_prompt=raw["system_prompt"],
        procedures=[Procedure(**p) for p in raw["procedures"]],
        guards=[Guard(**g) for g in raw["guards"]],
        tool_allowlist=list(raw["tool_allowlist"]),
        created_by=raw["created_by"],
        digest=raw["digest"],
    )
    if not h.verify_digest():
        raise ValueError(f"harness {h.version} does not match its digest")
    return h


def trace_from(raw: dict[str, Any]) -> Trace:
    o = raw["outcome"]
    return Trace(
        run_id=raw["run_id"],
        case_id=raw["case_id"],
        harness_version=raw["harness_version"],
        model=raw["model"],
        events=[Event(e["seq"], e["kind"], e["payload"], e["ts"]) for e in raw["events"]],
        outcome=Outcome(o["passed"], list(o["reasons"]), o["final_state_digest"], Cost(**o["cost"])),
    )


class JsonlStore:
    """Append-only JSON Lines files in one directory. Safe to call from several threads."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _append(self, name: str, record: Any) -> None:
        line = json.dumps(record, default=_default, sort_keys=True)
        with self._lock, (self.root / f"{name}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read(self, name: str) -> list[dict[str, Any]]:
        path = self.root / f"{name}.jsonl"
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def put_harness(self, harness: Harness) -> bool:
        if any(h["version"] == harness.version for h in self.read("harnesses")):
            return False
        self._append("harnesses", harness)
        return True

    def get_harness(self, version: str) -> Harness | None:
        found = [h for h in self.read("harnesses") if h["version"] == version]
        return harness_from(found[-1]) if found else None

    def put_trace(self, trace: Trace) -> bool:
        self._append("traces", trace)
        return True

    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str:
        rid = eval_id or uuid.uuid4().hex
        self._append("evals", {"eval_id": rid, "report": report})
        return rid

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        rid = decision_id or uuid.uuid4().hex
        self._append("decisions", {"decision_id": rid, "decision": decision})
        return rid

    def decision(self, decision_id: str) -> dict[str, Any] | None:
        """The last decision stored under `decision_id` (what an activator checks before it activates)."""
        found = [r["decision"] for r in self.read("decisions") if r["decision_id"] == decision_id]
        return found[-1] if found else None

    def traces(self) -> list[Trace]:
        return [trace_from(r) for r in self.read("traces")]
