"""Pieces the referee's admission policies share with their drivers: the ruling-2 policy record, the
normalized trace a repeatability probe compares, the rebuild of a stored candidate, and a decision's
C1 flips.

None of these reads a held-out case; the referee applies them to the runs it holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ratchet_cases.types import Trace, sha256_hex

from ratchet.canonical import canonical_json
from ratchet.types import Guard, Harness, Procedure

RULING_2 = {"name": "ruling-2", "evaluator": "deterministic"}


def normalized(trace: Trace) -> str:
    """What must repeat: every event's kind and content, without timestamps, run ids or tool-call ids."""
    rows = []
    for e in trace.events:
        p = dict(e.payload)
        p.pop("latency_s", None)
        call = p.get("call")
        if isinstance(call, dict):
            p["call"] = {k: v for k, v in call.items() if k != "call_id"}
        res = p.get("result")
        if isinstance(res, dict):
            p["result"] = {k: v for k, v in res.items() if k != "call_id"}
        p.pop("call_id", None)
        if "tool_calls" in p:
            p["tool_calls"] = [{k: v for k, v in c.items() if k != "call_id"} for c in p["tool_calls"] or []]
        rows.append([e.kind, p])
    return sha256_hex(canonical_json([rows, trace.outcome.passed, trace.outcome.reasons]))


class RebuildError(ValueError):
    """The rebuilt candidate is not the one the stored decision judged."""


def rebuild(parent: Harness, decision: Mapping[str, Any]) -> Harness:
    """The exact candidate a stored decision judged, rebuilt from its recorded change, checked by digest."""
    ctx = decision["context"]
    change, kind = ctx["change"], ctx["kind"]
    if kind == "guard":
        child = parent.child(guards=[*parent.guards, Guard(**change)], created_by="improver")
    elif kind == "procedure":
        child = parent.child(procedures=[*parent.procedures, Procedure(**change)], created_by="improver")
    else:
        child = parent.child(
            system_prompt=f"{parent.system_prompt}\n\n{change['append']}", created_by="improver"
        )
    if child.version != decision["candidate"]:
        raise RebuildError(f"rebuilt {child.version}, decision names {decision['candidate']}")
    return child


def c1_flips(evidence: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Held-out cases a candidate's first fresh run flipped: (failed in both parent runs and passed in C1,
    passed in either parent run and failed in C1). Informative even when the candidate was not admitted."""
    up: list[str] = []
    down: list[str] = []
    for case, v in (evidence.get("outcomes") or {}).items():
        p1, p2, c1 = v[0], v[1], v[2]
        if p1 == p2 == "fail" and c1 == "pass":
            up.append(case)
        elif "pass" in (p1, p2) and c1 == "fail":
            down.append(case)
    return sorted(up), sorted(down)
