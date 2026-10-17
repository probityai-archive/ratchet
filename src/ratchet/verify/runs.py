"""Run records: what an evaluation report commits about the runs it rests on, and the checks a verifier runs
over those commitments and the traces behind them.

An evaluation report (`EvalReport.to_payload`) lists every run under `runs`: its run id, its case, the
SHA-256 of its trace document and the SHA-256 of each model event of that trace, in order. A decision signs
the digest of each report it names, so these commitments are signed with it, and the referee writes them
from the traces it holds. A digest here is SHA-256 over the RFC 8785 canonical JSON of the document, so a
third party recomputes it from the trace JSON alone.

Every check is a pure function of decoded JSON and raises `RecordError` with the verdict and a reason code
when the records disagree. Nothing here reads a file, a clock or the network.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ratchet_cases.types import sha256_hex

from ratchet.canonical import canonical_json

HEX64 = re.compile(r"^[0-9a-f]{64}$")
MODEL = "model"


class RecordError(Exception):
    """The records disagree. `decision` is `contradicted` or `not_established`."""

    def __init__(self, decision: str, reason: str) -> None:
        super().__init__(reason)
        self.decision, self.reason = decision, reason


def digest(doc: Any) -> str:
    """SHA-256 of the RFC 8785 canonical JSON of `doc`."""
    return sha256_hex(canonical_json(doc))


def _events(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = trace.get("events")
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        raise RecordError("contradicted", f"trace_malformed:{trace.get('run_id')}")
    return events


def model_call_digests(trace: Mapping[str, Any]) -> list[str]:
    """The digest of each model event of a trace document (`dataclasses.asdict(Trace)`), in order."""
    return [digest(e) for e in _events(trace) if e.get("kind") == MODEL]


def cached_calls(trace: Mapping[str, Any]) -> int:
    return sum(
        1 for e in _events(trace) if e.get("kind") == MODEL and (e.get("payload") or {}).get("cached") is True
    )


def run_record(trace: Mapping[str, Any]) -> dict[str, Any]:
    """What a report commits about one run, from its trace document."""
    return {
        "run_id": trace["run_id"],
        "case_id": trace["case_id"],
        "trace_digest": digest(trace),
        "model_calls": model_call_digests(trace),
    }


# ---------------------------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    """One evaluation report the decision names, with the traces supplied for it (None: not supplied)."""

    name: str
    doc: Mapping[str, Any]
    traces: Mapping[str, Mapping[str, Any]] | None = None


def has_runs(report: Report) -> bool:
    return "runs" in report.doc


def runs_of(report: Report) -> dict[str, dict[str, Any]]:
    """The report's run records by run id. A malformed record list is a contradiction: the report's digest is
    signed, so the signer committed to it."""
    runs = report.doc.get("runs")
    bad = RecordError("contradicted", f"run_records_malformed:{report.name}")
    if not isinstance(runs, list):
        raise bad
    out: dict[str, dict[str, Any]] = {}
    for r in runs:
        ok = (
            isinstance(r, dict)
            and isinstance(r.get("run_id"), str)
            and isinstance(r.get("case_id"), str)
            and isinstance(r.get("trace_digest"), str)
            and HEX64.fullmatch(r["trace_digest"]) is not None
            and isinstance(r.get("model_calls"), list)
            and all(isinstance(d, str) and HEX64.fullmatch(d) for d in r["model_calls"])
        )
        if not ok or r["run_id"] in out:
            raise bad
        out[r["run_id"]] = r
    return out


def rows_of(report: Report) -> list[dict[str, Any]]:
    rows = report.doc.get("results")
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise RecordError("contradicted", f"report_malformed:{report.name}")
    return rows


def _count(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def check_model_calls(reports: Sequence[Report]) -> dict[str, Any]:
    """Every counted model call is committed, once, in the run that made it.

    For each result row with a run: the run is committed with the row's case, its committed model calls
    number exactly the row's fresh plus cached calls, and a FRESH row has no cached call. Across every report
    of the decision, no run id and no model-call digest appears twice, so no run and no answer is counted in
    two places. Where a run's trace is supplied, its model events digest to exactly the committed list and
    its cached calls equal the row's. Returns the counts it checked."""
    seen_runs: dict[str, str] = {}
    seen_calls: dict[str, str] = {}
    runs_total = traced = 0
    for report in reports:
        runs = runs_of(report)
        for row in rows_of(report):
            run_id, case_id = row.get("run_id"), row.get("case_id")
            if run_id is None:
                continue
            entry = runs.get(str(run_id))
            if entry is None:
                raise RecordError("not_established", f"run_record_missing:{report.name}:{case_id}")
            if entry["case_id"] != case_id:
                raise RecordError("contradicted", f"run_record_case_mismatch:{report.name}:{case_id}")
            fresh, cached = _count(row, "fresh_calls"), _count(row, "cached_calls")
            if fresh < 0 or cached < 0 or fresh + cached != len(entry["model_calls"]):
                raise RecordError("contradicted", f"model_call_count_mismatch:{report.name}:{case_id}")
            if row.get("mode") == "FRESH" and cached:
                raise RecordError("contradicted", f"fresh_row_with_cached_calls:{report.name}:{case_id}")
        for run_id, entry in runs.items():
            if run_id in seen_runs:
                raise RecordError("contradicted", f"run_reused:{seen_runs[run_id]}:{report.name}:{run_id}")
            seen_runs[run_id] = report.name
            for call in entry["model_calls"]:
                if call in seen_calls:
                    raise RecordError("contradicted", f"model_call_reused:{report.name}:{run_id}")
                seen_calls[call] = run_id
            runs_total += 1
            trace = (report.traces or {}).get(run_id)
            if trace is None:
                continue
            if model_call_digests(trace) != entry["model_calls"]:
                raise RecordError("contradicted", f"model_calls_not_the_trace:{report.name}:{run_id}")
            rows = [r for r in rows_of(report) if r.get("run_id") == run_id]
            if rows and cached_calls(trace) != _count(rows[0], "cached_calls"):
                raise RecordError("contradicted", f"cached_calls_not_the_trace:{report.name}:{run_id}")
            traced += 1
    return {"runs": runs_total, "model_calls": len(seen_calls), "traces_checked": traced}


def _payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _requested(events: Sequence[Mapping[str, Any]]) -> bool:
    """Every call the run acted on (a tool call, or a procedure the model selected) was asked for by the
    model turn before it."""
    asked: set[str] = set()
    for e in events:
        p = _payload(e)
        if e.get("kind") == MODEL:
            asked = {str(c.get("call_id")) for c in p.get("tool_calls") or [] if isinstance(c, dict)}
        elif e.get("kind") == "tool_call":
            if str((p.get("call") or {}).get("call_id")) not in asked:
                return False
        elif (
            e.get("kind") == "procedure" and p.get("how") == "selected" and str(p.get("call_id")) not in asked
        ):
            return False
    return True


def _chain_problem(trace: Mapping[str, Any], entry: Mapping[str, Any], harness_version: Any) -> str | None:
    """The first way the committed trace is not one well-formed run of this report, or None."""
    events = _events(trace)
    if trace.get("case_id") != entry["case_id"]:
        return "case_id"
    if trace.get("harness_version") != harness_version:
        return "harness_version"
    finals = [i for i, e in enumerate(events) if e.get("kind") == "final"]
    if any(e.get("seq") != i for i, e in enumerate(events)) or finals != [len(events) - 1]:
        return "events"
    if not _requested(events):
        return "unrequested_call"
    return None


def _row_problem(row: Mapping[str, Any], trace: Mapping[str, Any]) -> str | None:
    """The first field of a result row that its trace does not yield, or None."""
    events = _events(trace)
    raw = trace.get("outcome")
    outcome: Mapping[str, Any] = raw if isinstance(raw, dict) else {}
    hits = sum(
        1 for e in events if e.get("kind") == "guard_block" and _payload(e).get("source") != "arguments"
    )
    calls = sum(1 for e in events if e.get("kind") == MODEL)
    derived = {
        "passed": outcome.get("passed"),
        "reasons": outcome.get("reasons"),
        "interventions": hits,
        "fresh_calls": calls - cached_calls(trace),
        "cached_calls": cached_calls(trace),
    }
    return next((k for k, v in derived.items() if row.get(k, 0 if k.endswith("calls") else None) != v), None)


def check_record_chain(reports: Sequence[Report]) -> dict[str, Any]:
    """The traces are the committed records of the runs, and the reports are derived from them.

    Each supplied trace digests to the trace its report commits; it is a run of the report's harness on the
    committed case; its events are numbered in order and end with one final event; every tool call and
    selected procedure answers a call the model turn before it asked for; and the result row of that run
    is what the trace yields: outcome, reasons, harness interventions, fresh and cached calls. A trace for an
    uncommitted run is a contradiction. Returns the counts it checked."""
    total = checked = 0
    for report in reports:
        runs = runs_of(report)
        total += len(runs)
        rows = {r.get("run_id"): r for r in rows_of(report) if r.get("run_id") is not None}
        for run_id, trace in (report.traces or {}).items():
            entry = runs.get(run_id)
            if entry is None:
                raise RecordError("contradicted", f"trace_not_committed:{report.name}:{run_id}")
            if digest(trace) != entry["trace_digest"]:
                raise RecordError("contradicted", f"trace_digest_mismatch:{report.name}:{run_id}")
            case_id = entry["case_id"]
            problem = _chain_problem(trace, entry, report.doc.get("harness_version"))
            if problem is None and run_id in rows:
                problem = _row_problem(rows[run_id], trace)
            if problem is not None:
                raise RecordError("contradicted", f"record_chain_mismatch:{report.name}:{case_id}:{problem}")
            checked += 1
    return {"runs": total, "traces_checked": checked}
