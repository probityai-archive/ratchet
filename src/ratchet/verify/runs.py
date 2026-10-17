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
from dataclasses import asdict, dataclass, is_dataclass
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


def case_document(case: Any) -> dict[str, Any]:
    """The JSON document of a case: every field of a dataclass case (for the case suite's own cases, what
    `ratchet_cases.types.to_jsonable` gives, so anyone holding the published suite can rebuild it), else the
    fields every case carries."""
    if is_dataclass(case) and not isinstance(case, type):
        return asdict(case)
    return {"id": case.id, "split": case.split, "kind": case.kind, "must_succeed": case.must_succeed}


def case_digest(case: Any) -> str:
    """What a result row commits about the case it graded."""
    return digest(case_document(case))


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


def _check_rows(report: Report, runs: Mapping[str, Mapping[str, Any]]) -> None:
    """Each result row with a run names a committed run of its case, with as many committed model calls as
    it counts fresh and cached, and no cached call when it says FRESH."""
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


def _check_trace_calls(report: Report, run_id: str, entry: Mapping[str, Any]) -> bool:
    """When the run's trace is supplied, its model events are exactly the committed calls and its cached
    calls are the row's. False when no trace was supplied."""
    trace = (report.traces or {}).get(run_id)
    if trace is None:
        return False
    if model_call_digests(trace) != entry["model_calls"]:
        raise RecordError("contradicted", f"model_calls_not_the_trace:{report.name}:{run_id}")
    rows = [r for r in rows_of(report) if r.get("run_id") == run_id]
    if rows and cached_calls(trace) != _count(rows[0], "cached_calls"):
        raise RecordError("contradicted", f"cached_calls_not_the_trace:{report.name}:{run_id}")
    return True


def check_model_calls(reports: Sequence[Report]) -> dict[str, Any]:
    """Every counted model call is committed, once, in the run that made it.

    Each report's rows agree with its run records (`_check_rows`). Across every report of the decision, no
    run id and no model-call digest appears twice, so no run and no recorded turn is counted in two places.
    Where a run's trace is supplied, its model events digest to exactly the committed list (and so carry the
    run's own sequence numbers and timestamps) and its cached calls equal the row's. Returns the counts it
    checked."""
    seen_runs: dict[str, str] = {}
    seen_calls: dict[str, str] = {}
    runs_total = traced = 0
    for report in reports:
        runs = runs_of(report)
        _check_rows(report, runs)
        for run_id, entry in runs.items():
            if run_id in seen_runs:
                raise RecordError("contradicted", f"run_reused:{seen_runs[run_id]}:{report.name}:{run_id}")
            seen_runs[run_id] = report.name
            for call in entry["model_calls"]:
                if call in seen_calls:
                    raise RecordError("contradicted", f"model_call_reused:{report.name}:{run_id}")
                seen_calls[call] = run_id
            runs_total += 1
            traced += _check_trace_calls(report, run_id, entry)
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


def _answer_problem(
    kind: str, p: Mapping[str, Any], name: str, allowed: set[str], guards: list[str]
) -> str | None:
    """The rule an answer to a call of tool `name` breaks, or None."""
    decisions = [d for d in p.get("decisions") or [] if isinstance(d, dict)]
    ids, blocked = [str(d.get("guard_id")) for d in decisions], any(d.get("blocked") for d in decisions)
    source = p.get("source")
    problems = {
        "tool_outside_allowlist": kind == "tool_result" and name not in allowed,
        "guards_not_consulted": kind == "tool_result" and ids != guards,
        "blocked_call_executed": kind == "tool_result" and blocked,
        "allowed_tool_blocked_as_off_the_allowlist": source == "allowlist" and name in allowed,
        "guard_block_without_a_blocking_guard": source == "guard" and (ids != guards or not blocked),
        "block_of_unknown_source": kind == "guard_block"
        and source not in ("allowlist", "guard", "arguments"),
    }
    return next((k for k, broken in problems.items() if broken), None)


def _policy_problem(events: Sequence[Mapping[str, Any]], harness: Mapping[str, Any]) -> str | None:
    """The first rule of the harness's tool policy that the recorded run breaks, or None.

    Each tool call is answered by exactly one event naming it, before the next call: a `tool_result` when it
    ran, a `guard_block` when it did not. A call that ran names a tool on the allowlist and carries one
    decision from every guard of the harness, in order, none of them blocking. A block names its source: the
    allowlist only for a tool off it, a guard only with a blocking decision from the full guard list."""
    allowed = {str(t) for t in harness.get("tool_allowlist") or []}
    guards = [str((g or {}).get("id")) for g in harness.get("guards") or []]
    pending: Mapping[str, Any] | None = None
    for e in events:
        kind, p = str(e.get("kind")), _payload(e)
        if kind == "tool_call":
            if pending is not None:
                return "tool_call_unanswered"
            call = p.get("call")
            pending = call if isinstance(call, dict) else {}
        elif kind in ("tool_result", "guard_block"):
            answered = (p.get("result") or {}).get("call_id") if kind == "tool_result" else p.get("call_id")
            if pending is None or answered != pending.get("call_id"):
                return "answer_without_call"
            if problem := _answer_problem(kind, p, str(pending.get("name")), allowed, guards):
                return problem
            pending = None
    return "tool_call_unanswered" if pending is not None else None


def check_tool_policy(
    reports: Sequence[Report], harnesses: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Every supplied trace obeys the tool policy of the harness its report ran (`harnesses`, by version):
    see `_policy_problem`. The traces are bound to the reports by `check_record_chain`, which runs first.
    Returns the counts it checked."""
    total = checked = calls = 0
    for report in reports:
        harness = harnesses.get(str(report.doc.get("harness_version")))
        if harness is None:
            raise RecordError("contradicted", f"report_of_unknown_harness:{report.name}")
        runs = runs_of(report)
        total += len(runs)
        for run_id, trace in (report.traces or {}).items():
            events = _events(trace)
            if problem := _policy_problem(events, harness):
                case_id = runs[run_id]["case_id"] if run_id in runs else trace.get("case_id")
                raise RecordError("contradicted", f"tool_policy_violated:{report.name}:{case_id}:{problem}")
            calls += sum(1 for e in events if e.get("kind") == "tool_call")
            checked += 1
    return {"runs": total, "traces_checked": checked, "tool_calls": calls}


# ---------------------------------------------------------------------------------------------------------
# Held-out grades
# ---------------------------------------------------------------------------------------------------------


def _suite_case(doc: Mapping[str, Any]) -> Any:
    """The case suite's own `Case` rebuilt from its document, or None for a case of another world (a
    benchmark adapter's), which this verifier cannot rebuild."""
    from ratchet_cases.cases.model import Case, Expect
    from ratchet_cases.world.faults import Fault

    try:
        fields = {**doc, "faults": [Fault(**f) for f in doc["faults"]], "grader": Expect(**doc["grader"])}
        return Case(**fields)
    except (KeyError, TypeError):
        return None


def _trace_of(doc: Mapping[str, Any]) -> Any:
    from ratchet_cases.types import Cost, Event, Outcome, Trace

    o = doc["outcome"]
    return Trace(
        str(doc["run_id"]),
        str(doc["case_id"]),
        str(doc["harness_version"]),
        str(doc["model"]),
        [Event(int(e["seq"]), e["kind"], dict(e["payload"]), float(e["ts"])) for e in _events(doc)],
        Outcome(bool(o["passed"]), list(o["reasons"]), str(o["final_state_digest"]), Cost(**o["cost"])),
    )


def _replay(world: Any, trace: Mapping[str, Any]) -> bool:
    """Apply every call the trace records as executed to `world`, in order; False at the first recorded
    result the world does not give back."""
    from ratchet_cases.types import ToolCall

    calls: dict[str, Mapping[str, Any]] = {}
    for e in _events(trace):
        p = _payload(e)
        if e.get("kind") == "tool_call":
            call = p.get("call") or {}
            calls[str(call.get("call_id"))] = call
        elif e.get("kind") == "tool_result":
            recorded = p.get("result") or {}
            call = calls.get(str(recorded.get("call_id"))) or {}
            done = world.call(
                ToolCall(str(call.get("name")), dict(call.get("args") or {}), str(call.get("call_id")))
            )
            if canonical_json(asdict(done)) != canonical_json(recorded):
                return False
    return True


def regrade(case_doc: Mapping[str, Any], trace: Mapping[str, Any]) -> str | None:
    """Grade a run again from the case document and its trace, as the worker graded it: rebuild the world
    from the case, apply every call the trace records as executed, in order, comparing each recorded result
    with the world's; then the final state digest, and the grader's verdict on that world (a run that did
    not end with the model's answer fails as incomplete). Returns the first field that disagrees, or None.
    Raises `LookupError` for a case of a world this verifier cannot rebuild."""
    from ratchet_cases.cases.grade import grade
    from ratchet_cases.world import World

    case = _suite_case(case_doc)
    if case is None:
        raise LookupError(str(case_doc.get("id")))
    world = World.from_case(case)
    if not _replay(world, trace):
        return "tool_result"
    outcome = trace.get("outcome") or {}
    if world.digest() != outcome.get("final_state_digest"):
        return "final_state"
    graded = grade(case, world, _trace_of(trace))
    events = _events(trace)
    final = _payload(events[-1]) if events else {}
    complete = final.get("complete") is True
    reasons = [*graded.reasons, *([] if complete else [f"incomplete: {final.get('reason')}"])]
    if (graded.passed and complete) != outcome.get("passed"):
        return "passed"
    return "reasons" if reasons != outcome.get("reasons") else None


def _committed_cases(reports: Sequence[Report]) -> dict[str, str]:
    """Held-out case id to the case digest every report commits for it. A held-out row without a digest is
    not established; two digests for one case are a contradiction."""
    committed: dict[str, str] = {}
    for report in reports:
        for row in rows_of(report):
            if row.get("split") != "heldout":
                continue
            case_id, d = str(row.get("case_id")), row.get("case_digest")
            if not isinstance(d, str) or not HEX64.fullmatch(d):
                raise RecordError("not_established", f"case_digest_missing:{report.name}:{case_id}")
            if committed.setdefault(case_id, d) != d:
                raise RecordError("contradicted", f"heldout_case_changed:{case_id}")
    return committed


def commits_cases(reports: Sequence[Report]) -> bool:
    """Whether the reports were written with case digests at all (reports written before them carry none)."""
    return any("case_digest" in row for report in reports for row in rows_of(report))


def check_heldout_grades(reports: Sequence[Report], cases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The held-out grades, recomputed. Every report commits one case digest for each held-out case, the
    same in every run; a disclosed case document (`cases`, by id) must be the committed one; and every
    held-out run whose case and trace are supplied is graded again (`regrade`) and must agree with its
    trace and its row. A case of a world this verifier cannot rebuild is counted as not regraded. Returns
    the counts it checked."""
    committed = _committed_cases(reports)
    disclosed = {k: v for k, v in cases.items() if k in committed}
    for case_id, doc in disclosed.items():
        if doc.get("id") != case_id:
            raise RecordError("contradicted", f"case_document_names_another_case:{case_id}")
        if digest(doc) != committed[case_id]:
            raise RecordError("contradicted", f"heldout_case_digest_mismatch:{case_id}")
    rows = regraded = 0
    other_world: set[str] = set()
    for report in reports:
        for row in rows_of(report):
            if row.get("split") != "heldout":
                continue
            rows += 1
            case_id, trace = str(row.get("case_id")), (report.traces or {}).get(str(row.get("run_id")))
            if case_id not in disclosed or trace is None:
                continue
            try:
                problem = regrade(disclosed[case_id], trace)
            except LookupError:
                other_world.add(case_id)
                continue
            if problem is None and row.get("passed") != trace["outcome"]["passed"]:
                problem = "row"
            if problem is not None:
                raise RecordError("contradicted", f"grade_mismatch:{report.name}:{case_id}:{problem}")
            regraded += 1
    return {
        "heldout_rows": rows,
        "cases_disclosed": len(disclosed),
        "regraded": regraded,
        "not_rebuildable": sorted(other_world),
    }
