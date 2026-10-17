"""A decision over real runs: the worker, the simulated company, its grader and the guard sandbox, run as the
referee runs them (P1 and C1 over a training case and two held-out cases, P2 and C2 over the held-out cases),
so every run record, trace and case digest in it is the one the code writes.

The model is scripted. Each run reads the ticket, tries a refund the harness does not allow (blocked by the
allowlist), adds a note and answers. Both harnesses carry one guard that never blocks; the candidate appends
a sentence to the prompt, so it adds no guard and replays nothing. The receipt claims what the reports show
for every gate, so the gates recompute and the run records are what is under test.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any

from ratchet_cases.cases import get

from ratchet.agent.model import ModelTurn, ScriptedModel, scripted_turn
from ratchet.agent.worker import run_case
from ratchet.improve.evaluate import EvalReport, evaluate
from ratchet.improve.receipt import Signer
from ratchet.types import Guard, Harness
from ratchet.verify import Options, VerifyReport, report_digest, verify_decision
from ratchet.verify import recompute as rc
from ratchet.verify.runs import case_document

from .helpers import NOOP, payload, sign

OPTIONS = Options()
TRAIN = "refund_lost_reply.v1"
HELDOUT = ("refund_lost_reply.v3", "refund_lost_reply.v4")
ROLES = ("parent_run_1", "parent_run_2", "candidate_run", "candidate_confirm_run")

PARENT = Harness.build(
    parent=None,
    system_prompt="Resolve the customer's ticket.",
    procedures=[],
    guards=[Guard("g-noop", "never blocks", NOOP, [])],
    tool_allowlist=["tickets_get", "tickets_add_note"],
    created_by="seed",
)
CANDIDATE = PARENT.child(system_prompt=f"{PARENT.system_prompt}\nBe brief.", created_by="improver")
RECOMPUTED = {"overfit_lint": {"hits": []}}
# The held-out case documents a third party holds: the published case suite's own cases.
CASES = {i: case_document(get(i)) for i in HELDOUT}


def _script(ticket: str) -> list[ModelTurn]:
    return [
        scripted_turn(
            tool_calls=[
                ("tickets_get", {"ticket_id": ticket}),
                ("payments_refund", {"charge_id": "ch_none", "amount_cents": 1, "idempotency_key": "k"}),
            ],
            id_prefix="a",
        ),
        scripted_turn(
            tool_calls=[("tickets_add_note", {"ticket_id": ticket, "body": "Looked."})], id_prefix="b"
        ),
        scripted_turn("Done."),
    ]


def _run(harness: Harness, case: Any, model: Any, store: Any, *, run_id: str) -> Any:
    return run_case(harness, case, ScriptedModel(_script(case.ticket_id)), store, run_id=run_id)


def _evaluate(harness: Harness, ids: tuple[str, ...]) -> EvalReport:
    return evaluate(harness, [get(i) for i in ids], _run, None, workers=1)


@dataclass
class Scenario:
    reports: dict[str, dict[str, Any]]  # by role name, the report payloads the referee stores
    traces: dict[str, list[dict[str, Any]]]  # by role name, the trace documents of its runs
    harnesses: dict[str, dict[str, Any]]

    def copy(self) -> Scenario:
        return copy.deepcopy(self)


@cache
def _built() -> Scenario:
    runs = [
        _evaluate(PARENT, (TRAIN, *HELDOUT)),
        _evaluate(PARENT, HELDOUT),
        _evaluate(CANDIDATE, (TRAIN, *HELDOUT)),
        _evaluate(CANDIDATE, HELDOUT),
    ]
    # The stored form: what a store or a JSON file gives back.
    reports = {n: json.loads(json.dumps(r.to_payload())) for n, r in zip(ROLES, runs, strict=True)}
    traces = {
        n: [json.loads(json.dumps(asdict(t))) for t in r.traces] for n, r in zip(ROLES, runs, strict=True)
    }
    return Scenario(reports, traces, {"parent": asdict(PARENT), "candidate": asdict(CANDIDATE)})


def scenario() -> Scenario:
    return _built().copy()


def body_for(s: Scenario) -> dict[str, Any]:
    """A receipt that claims exactly what the reports and harnesses show for every gate."""
    p1, p2, c1, c2 = (s.reports[n] for n in ROLES)
    gates = {
        "replay": rc.Gate(True, {"new_guards": [], "reports": []}),
        "heldout": rc.heldout(p1, p2, c1, c2),
        "no_regression": rc.no_regression([p1, p2], [c1, c2]),
        "authority": rc.authority(s.harnesses["parent"], s.harnesses["candidate"]),
        "benign_twins": rc.benign_twins([c1, c2]),
        "overfit_lint": rc.Gate(True, {"hits": 0}),
    }
    return payload(
        parent=PARENT.version,
        candidate=CANDIDATE.version,
        candidate_digest=CANDIDATE.digest,
        admitted=all(g.passed for g in gates.values()),
        checks=[{"name": n, "passed": g.passed, "evidence": g.evidence} for n, g in gates.items()],
        corpus_digests={},
        report_digests={n: report_digest(s.reports[n]) for n in ROLES},
    )


def verify(
    signer: Signer,
    s: Scenario,
    *,
    traces: bool = True,
    harnesses: bool = True,
    cases: dict[str, Any] | None = None,
    options: Options = OPTIONS,
    **kw: Any,
) -> VerifyReport:
    """Sign a receipt over the scenario's reports (as they now stand) and verify it with what is supplied."""
    return verify_decision(
        sign(body_for(s), signer),
        signer.public_key,
        reports=s.reports,
        harnesses=s.harnesses if harnesses else {},
        recomputed=RECOMPUTED,
        traces=s.traces if traces else {},
        cases=CASES if cases is None else cases,
        options=options,
        **kw,
    )


def status(r: VerifyReport) -> dict[str, str]:
    return {i["id"]: i["status"] for i in r.recomputed}


def item(r: VerifyReport, name: str) -> dict[str, Any]:
    return next(i for i in r.recomputed if i["id"] == name)


def run_of(s: Scenario, role: str, case_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(result row, run record, trace document) of one case in one report."""
    row = next(r for r in s.reports[role]["results"] if r["case_id"] == case_id)
    record = next(r for r in s.reports[role]["runs"] if r["run_id"] == row["run_id"])
    trace = next(t for t in s.traces[role] if t["run_id"] == row["run_id"])
    return row, record, trace


def recommit(
    s: Scenario, role: str, edit: Callable[[dict[str, Any]], None] | None = None, *, rows: bool = False
) -> None:
    """After a trace was edited, rewrite the report's run records from the traces, as a signer who
    re-signs would; with `rows`, also rewrite each result row from its trace. `edit` then changes the
    report itself."""
    from ratchet.verify.runs import cached_calls, run_record

    s.reports[role]["runs"] = sorted((run_record(t) for t in s.traces[role]), key=lambda r: r["run_id"])
    if rows:
        by_run = {t["run_id"]: t for t in s.traces[role]}
        for row in s.reports[role]["results"]:
            t = by_run[row["run_id"]]
            models = by_kind(t, "model")
            row["passed"], row["reasons"] = t["outcome"]["passed"], t["outcome"]["reasons"]
            blocks = [e for e in by_kind(t, "guard_block") if e["payload"]["source"] != "arguments"]
            row["interventions"] = len(blocks)
            row["cached_calls"] = cached_calls(t)
            row["fresh_calls"] = len(models) - row["cached_calls"]
    if edit is not None:
        edit(s.reports[role])


def first(events: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    return next(e for e in events if e["kind"] == kind)


def by_kind(trace: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in trace["events"] if e["kind"] == kind]
