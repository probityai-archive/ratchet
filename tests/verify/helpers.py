"""Builders shared by the verify tests: a coherent admission scenario and receipts signed with the IMPROVE
lane's signer.

The scenario is small and satisfies the contract's rule: two stable recoveries (h1, h2), a protected case
(h3) passing in every run, a must-succeed case (m1), two benign twins (b1, b2) never intervened on, and one
added guard whose replay clears.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
from typing import Any

from ratchet.canonical import CANONICALIZATION, canonical_json
from ratchet.improve.receipt import DECISION_TYPE, Signer
from ratchet.types import Guard, Harness
from ratchet.verify import DEFAULT_GATES, PREDICATE_URI, report_digest

SUITE = "a" * 64
SNAPSHOT = "c" * 64
NOOP = "def check(call, view, history):\n    return None\n"
GUARD_ID = "g-new"


def _row(
    case: str, split: str, kind: str, passed: bool, *, must: bool = False, hits: int = 0
) -> dict[str, Any]:
    return {
        "case_id": case,
        "split": split,
        "kind": kind,
        "must_succeed": must,
        "passed": passed,
        "reasons": [],
        "run_id": f"run-{case}",
        "interventions": hits,
        "mode": "FRESH",
        "error": None,
    }


def eval_report(version: str, passes: set[str], *, hits: dict[str, int] | None = None) -> dict[str, Any]:
    hits = hits or {}
    specs = [
        ("h1", "heldout", "trap", False),
        ("h2", "heldout", "trap", False),
        ("h3", "heldout", "trap", False),
        ("b1", "heldout", "benign_twin", True),
        ("b2", "heldout", "benign_twin", True),
        ("m1", "train", "ordinary", True),
    ]
    rows = [
        _row(c, s, k, c in passes, must=c in {"m1", "b1", "b2"}, hits=hits.get(c, 0)) for c, s, k, _ in specs
    ]
    return {"type": "eval", "harness_version": version, "results": rows}


ALWAYS = {"h3", "b1", "b2", "m1"}
PARENT = lambda v: eval_report(v, ALWAYS)  # noqa: E731
EVAL_REPORTS = {
    "parent_run_1": eval_report("p", ALWAYS),
    "parent_run_2": eval_report("p", ALWAYS),
    "candidate_run": eval_report("c", ALWAYS | {"h1", "h2"}),
    "candidate_confirm_run": eval_report("c", ALWAYS | {"h1", "h2"}),
}
REPLAY_REPORT: dict[str, Any] = {
    "guard_id": GUARD_ID,
    "passed": True,
    "verdict": "CLEARED",
    "snapshot_digest": SNAPSHOT,
    "targets": [{"failure_id": "f1", "hit": True}, {"failure_id": "f2", "hit": True}],
    "false_positives": [],
    "errors": [],
    "traces": 9,
    "check_s": 0.12,
}
REPORTS: dict[str, dict[str, Any]] = {**EVAL_REPORTS, f"replay:{GUARD_ID}": REPLAY_REPORT}

PARENT_HARNESS = Harness.build(
    parent=None,
    system_prompt="Be careful with refunds.",
    procedures=[],
    guards=[Guard("g-old", "old", NOOP, [])],
    tool_allowlist=["refund", "lookup"],
    created_by="seed",
)
CANDIDATE_HARNESS = Harness.build(
    parent=PARENT_HARNESS.version,
    system_prompt="Be careful with refunds.\nNever refund twice.",
    procedures=[],
    guards=[
        *PARENT_HARNESS.guards,
        Guard(GUARD_ID, "new", "def check(call, view, history):\n    return None\n", []),
    ],
    tool_allowlist=["refund"],
    created_by="improver",
)
HARNESSES = {"parent": asdict(PARENT_HARNESS), "candidate": asdict(CANDIDATE_HARNESS)}
RECOMPUTED: dict[str, dict[str, Any]] = {f"replay:{GUARD_ID}": REPLAY_REPORT, "overfit_lint": {"hits": []}}


def replay_row() -> dict[str, Any]:
    return {
        "guard_id": GUARD_ID,
        "passed": True,
        "snapshot_digest": SNAPSHOT,
        "targets_hit": 2,
        "targets_total": 2,
        "false_fires": [],
        "errors": 0,
    }


def payload(**overrides: Any) -> dict[str, Any]:
    evidence: dict[str, dict[str, Any]] = {
        "replay": {"new_guards": [GUARD_ID], "reports": [replay_row()]},
    }
    body: dict[str, Any] = {
        "predicateType": PREDICATE_URI,
        "canonicalization": CANONICALIZATION,
        "type": "decision",
        "parent": PARENT_HARNESS.version,
        "candidate": CANDIDATE_HARNESS.version,
        "candidate_digest": CANDIDATE_HARNESS.digest,
        "admitted": True,
        "checks": [{"name": g, "passed": True, "evidence": evidence.get(g, {"n": 3})} for g in DEFAULT_GATES]
        + [{"name": "cost", "passed": True, "evidence": {}}],
        "evaluator": {"version": "e1", "components": {"suite_file": SUITE}},
        "corpus_digests": {GUARD_ID: SNAPSHOT},
        "report_digests": {name: report_digest(r) for name, r in REPORTS.items()},
        "exceptions": [],
        "decided_at": "2026-10-11T12:00:00+00:00",
    }
    body.update(overrides)
    return body


def sign(body: dict[str, Any], signer: Signer, payload_type: str = DECISION_TYPE) -> bytes:
    return signer.sign(payload_type, canonical_json(body))


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
