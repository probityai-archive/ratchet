"""Plain-JSON forms of the values that cross the improver-referee boundary, and back.

The referee runs in another process and never unpickles anything the improver sends: every request is JSON
built from these functions, and each reader rebuilds the dataclass field by field, so a request can carry
data and nothing that runs.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ratchet_cases.types import Cost, Event, Outcome, Trace

from ratchet.improve.diagnose import FailureCluster, FailurePoint
from ratchet.improve.evaluate import CaseResult, EvalReport
from ratchet.improve.overfit import Literals
from ratchet.types import Guard, Harness, Procedure

Doc = dict[str, Any]


def harness_doc(h: Harness) -> Doc:
    return asdict(h)


def harness(doc: Doc) -> Harness:
    h = Harness(
        version=str(doc["version"]),
        parent=doc["parent"],
        system_prompt=str(doc["system_prompt"]),
        procedures=[Procedure(**p) for p in doc["procedures"]],
        guards=[Guard(**g) for g in doc["guards"]],
        tool_allowlist=[str(t) for t in doc["tool_allowlist"]],
        created_by=doc["created_by"],
        digest=str(doc["digest"]),
    )
    if not h.verify_digest():
        raise ValueError(f"harness {h.version} does not match its digest")
    return h


def trace_doc(t: Trace) -> Doc:
    return asdict(t)


def trace(doc: Doc) -> Trace:
    o = doc["outcome"]
    return Trace(
        run_id=str(doc["run_id"]),
        case_id=str(doc["case_id"]),
        harness_version=str(doc["harness_version"]),
        model=str(doc["model"]),
        events=[Event(int(e["seq"]), e["kind"], dict(e["payload"]), float(e["ts"])) for e in doc["events"]],
        outcome=Outcome(
            bool(o["passed"]), [str(r) for r in o["reasons"]], str(o["final_state_digest"]), Cost(**o["cost"])
        ),
    )


def report_doc(r: EvalReport) -> Doc:
    return {
        "harness_version": r.harness_version,
        "wall_s": r.wall_s,
        "results": [asdict(x) for x in r.results],
        "traces": [trace_doc(t) for t in r.traces],
    }


def report(doc: Doc) -> EvalReport:
    results = [CaseResult(**{**x, "cost": Cost(**x["cost"])}) for x in doc["results"]]
    return EvalReport(
        str(doc["harness_version"]), results, float(doc["wall_s"]), [trace(t) for t in doc["traces"]]
    )


def cluster_doc(c: FailureCluster) -> Doc:
    return {
        "id": c.id,
        "key": dict(c.key),
        "summary": c.summary,
        "members": [asdict(m) for m in c.members],
        "traces": {k: trace_doc(t) for k, t in c.traces.items()},
    }


def cluster(doc: Doc) -> FailureCluster:
    return FailureCluster(
        id=str(doc["id"]),
        key={str(k): str(v) for k, v in doc["key"].items()},
        members=[FailurePoint(**m) for m in doc["members"]],
        summary=str(doc["summary"]),
        traces={str(k): trace(t) for k, t in doc["traces"].items()},
    )


def literals_doc(lit: Literals) -> Doc:
    return {"ids": sorted(lit.ids), "amounts": sorted(lit.amounts), "names": sorted(lit.names)}


def literals(doc: Doc) -> Literals:
    return Literals(
        {str(x) for x in doc["ids"]}, {int(x) for x in doc["amounts"]}, {str(x) for x in doc["names"]}
    )
