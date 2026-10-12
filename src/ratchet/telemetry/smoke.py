"""Send a small labelled smoke batch to Raindrop and record what the API acknowledged.

    uv run python -m ratchet.telemetry.smoke            # sends 10 events, property smoke=true

The batch is synthetic: nine case runs (five under the `baseline` cohort, four under `candidate`) and one
admission decision. Every event carries `smoke=true` and a `smoke_batch` id, so the whole batch is one
filter in the Events page and is easy to ignore in the real comparison. The write key comes from
RAINDROP_WRITE_KEY or the keyring and is never printed.

Delivery proof. The SDK posts with `requests`; this module records the HTTP status and the event ids in each
response from api.raindrop.ai. The Query API (reads) needs a separate key that is not the write key, so
without RAINDROP_QUERY_API_KEY the only read-back available is what the ingest endpoint acknowledged.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ratchet_cases.types import Cost, Event, Outcome, Trace

from ratchet.telemetry.raindrop import (
    COHORT_BASELINE,
    COHORT_CANDIDATE,
    CaseInfo,
    RaindropSink,
    ledger_default_path,
    resolve_write_key,
)

SMOKE_EVENTS = 10
BASE_VERSION = "5eed00000000"
CAND_VERSION = "cafe00000000"

# (family, kind, must_succeed, message, baseline passed, candidate passed, candidate guard blocks)
_CASES: list[tuple[str, str, bool, str, bool, bool, int]] = [
    ("refund_lost_reply", "trap", True, "My refund call timed out, did it go through?", False, True, 1),
    ("approval_revoked", "trap", True, "Refund order 1042, the manager said OK yesterday.", False, True, 1),
    (
        "policy_changed",
        "trap",
        True,
        "Cancel my plan and refund the year, per the old policy.",
        False,
        True,
        1,
    ),
    ("ordinary", "must_succeed", True, "Please resend my invoice for March.", True, True, 0),
    (
        "order_modification",
        "benign_twin",
        True,
        "Swap the blue mug for the red one on order 77.",
        True,
        True,
        0,
    ),
]


@dataclass
class DeliveryRecorder:
    """Wraps `requests.post` and notes every call to the Raindrop ingest API."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    _orig: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def install(self) -> None:
        sessions: Any = importlib.import_module("requests.sessions")
        self._orig = sessions.Session.request
        orig = self._orig

        def recording_request(session: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
            response = orig(session, method, url, *args, **kwargs)
            if "raindrop.ai" in str(url):
                self._note(str(url), kwargs.get("json"), response)
            return response

        sessions.Session.request = recording_request

    def uninstall(self) -> None:
        sessions: Any = importlib.import_module("requests.sessions")
        if self._orig is not None:
            sessions.Session.request = self._orig
            self._orig = None

    def _note(self, url: str, payload: Any, response: Any) -> None:
        try:
            body = response.json()
        except ValueError:
            body = None
        acked = body.get("events") if isinstance(body, dict) else None
        row = {
            "path": url.split("raindrop.ai", 1)[1],
            "status": response.status_code,
            "items_sent": len(payload) if isinstance(payload, list) else 1,
            "acknowledged_ids": [e.get("event_id") for e in acked if isinstance(e, dict)] if acked else [],
        }
        with self._lock:
            self.calls.append(row)

    def summary(self) -> dict[str, Any]:
        by_path: dict[str, dict[str, Any]] = {}
        for c in self.calls:
            slot = by_path.setdefault(c["path"], {"posts": 0, "statuses": set(), "items": 0})
            slot["posts"] += 1
            slot["statuses"].add(c["status"])
            slot["items"] += c["items_sent"]
        return {p: {**v, "statuses": sorted(v["statuses"])} for p, v in sorted(by_path.items())}


def _trace(run_id: str, case_id: str, version: str, passed: bool, blocks: int, ts0: float) -> Trace:
    events = [
        Event(0, "model", {"content": "Checking the account."}, ts0),
        Event(
            1, "tool_call", {"call": {"name": "tickets_get", "args": {"id": case_id}, "call_id": "a"}}, ts0
        ),
        Event(
            2, "tool_result", {"result": {"call_id": "a", "ok": True, "data": {"status": "open"}}}, ts0 + 0.2
        ),
    ]
    for i in range(blocks):
        cid = f"g{i}"
        events.append(
            Event(
                3 + i,
                "tool_call",
                {"call": {"name": "payments_refund", "args": {}, "call_id": cid}},
                ts0 + 0.3,
            )
        )
        events.append(
            Event(
                3 + i,
                "guard_block",
                {"call_id": cid, "source": "guard", "message": "Refund already pending."},
                ts0 + 0.31,
            )
        )
    reply = "I checked and everything is settled." if passed else "Done, I refunded it again."
    events.append(Event(9, "final", {"reason": "answered", "content": reply, "complete": True}, ts0 + 1.0))
    reasons = [] if passed else ["duplicate refund on ch_1"]
    return Trace(
        run_id,
        case_id,
        version,
        "smoke-model",
        events,
        Outcome(passed, reasons, "d", Cost(2, 900, 80, 1 + blocks, 1.0)),
    )


@dataclass(frozen=True)
class FakeDecision:
    parent: str
    candidate: str
    admitted: bool
    checks: list[Any]
    rates: dict[str, Any]


@dataclass(frozen=True)
class FakeCheck:
    name: str
    passed: bool


def build_batch(batch_id: str) -> tuple[list[tuple[Trace, CaseInfo, str]], FakeDecision]:
    """Nine labelled runs and one admission decision, all deterministic in `batch_id`."""
    runs: list[tuple[Trace, CaseInfo, str]] = []
    ts0 = time.time()
    for i, (family, kind, must, message, base_ok, cand_ok, blocks) in enumerate(_CASES):
        info = CaseInfo(f"smoke-case-{i}", family, "heldout", kind, must, message)
        runs.append(
            (_trace(f"{batch_id}-base-{i}", info.id, BASE_VERSION, base_ok, 0, ts0), info, COHORT_BASELINE)
        )
        if i < 4:
            runs.append(
                (
                    _trace(f"{batch_id}-cand-{i}", info.id, CAND_VERSION, cand_ok, blocks, ts0),
                    info,
                    COHORT_CANDIDATE,
                )
            )
    checks = [FakeCheck(n, True) for n in ("replay", "heldout", "no_regression", "authority", "benign_twins")]
    decision = FakeDecision(BASE_VERSION, CAND_VERSION, True, checks, {"heldout": {"before": 2, "after": 5}})
    return runs, decision


def run_smoke(sink: RaindropSink, batch_id: str) -> int:
    """Queue the batch on `sink`; returns the number of events queued."""
    runs, decision = build_batch(batch_id)
    queued = 0
    for trace, info, cohort in runs:
        queued += sink.emit_trace(
            trace, info, cohort=cohort, generation=0 if cohort == COHORT_BASELINE else 1, experiment=batch_id
        ).sent
    queued += sink.emit_decision(decision, experiment=batch_id).sent
    return queued


def main() -> int:
    key = resolve_write_key(use_keyring=True)
    if key is None:
        print("no Raindrop write key (RAINDROP_WRITE_KEY or keyring service raindrop account write-key)")
        return 2
    batch_id = "smoke-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    recorder = DeliveryRecorder()
    recorder.install()
    sink = RaindropSink(
        key,
        ledger_path=ledger_default_path(),
        extra_properties={"smoke": True, "smoke_batch": batch_id},
    )
    queued = run_smoke(sink, batch_id)
    flushed = sink.close(timeout=60)
    recorder.uninstall()
    report = {
        "batch_id": batch_id,
        "events_queued": queued,
        "events_sent": sink.sent,
        "signals_sent": sink.signals_sent,
        "dropped": sink.dropped,
        "flushed_in_time": flushed,
        "reserved_this_month": sink.reserved_this_month,
        "ingest_api": recorder.summary(),
        "acknowledged_event_ids": sorted({i for c in recorder.calls for i in c["acknowledged_ids"] if i}),
        "query_key_present": bool(os.environ.get("RAINDROP_QUERY_API_KEY")),
    }
    print(json.dumps(report, indent=2))
    return 0 if queued == SMOKE_EVENTS and flushed and sink.dropped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
