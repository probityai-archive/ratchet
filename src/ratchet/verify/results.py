"""Verify a referee results file: the per-case held-out tables a referee signed, against the bundles behind
them.

    python -m ratchet.verify.results RESULTS.dsse [--key PEM] [--case-suite | --case ID=FILE ...] [--json]

The results file is a DSSE envelope (payload type `application/vnd.ratchet.referee-results+json`) signed by
the decision key. Its directory holds one verification bundle per decision. For each decision this checks:

- the bundle's receipt is the one the results file names (its SHA-256), and it verifies (`verify_decision`)
  against the bundled reports, harnesses and traces, re-grading every held-out run from the disclosed case
  documents (`heldout_grades`); the receipt's admission flag is the one the results file lists;
- ruling 2: the per-case outcome table, the stable recoveries, the protected cases broken and the C1 flips
  are recomputed from the bundled evaluation reports;
- stochastic v3: each arm's execution records (shared by a cohort's decisions, named relative to the
  results directory) digest to the receipt's `arm:<name>` digest, every per-case
  count (in total and per panel) is recounted from them and must equal both the results file and the
  receipt's signed comparison, and every execution with a trace is graded again from its case document.

Run-level tables: the repeatability probe is recomputed from its traces (the normalized trace digests of
both runs); the parent discordance from the bundled parent runs. Exit status: 0 verified, 1 contradicted
or not established, 2 the input was malformed and no verdict was reached.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from ratchet_cases.types import sha256_hex

from ratchet.verify import dsse, recompute, runs
from ratchet.verify.keys import DEFAULT_KEY_PATH, TrustedKeyError, load_trusted_key
from ratchet.verify.receipt import Options, named_suite_digest, verify_decision

RESULTS_TYPE = "application/vnd.ratchet.referee-results+json"
V3 = "stochastic-v3"
ROLES = ("parent_run_1", "parent_run_2", "candidate_run", "candidate_confirm_run")


class ResultsError(Exception):
    """The results disagree with their evidence. `decision` is `contradicted` or `not_established`."""

    def __init__(self, decision: str, reason: str) -> None:
        super().__init__(reason)
        self.decision, self.reason = decision, reason


@dataclass
class ResultsReport:
    decision: str
    reason: str
    decisions: int = 0
    verified: list[str] = field(default_factory=list)
    recomputed: list[dict[str, Any]] = field(default_factory=list)
    authenticated_only: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.decision == "supported"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "decision": self.decision,
            "reason": self.reason,
            "decisions": self.decisions,
            "verified": self.verified,
            "recomputed": self.recomputed,
            "authenticated_only": self.authenticated_only,
        }


def _inside(root: Path, rel: Any) -> Path:
    if not isinstance(rel, str):
        raise ResultsError("not_established", "bundle_missing")
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ResultsError("contradicted", f"bundle_outside_results:{rel}")
    return path


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultsError("not_established", f"unreadable:{path.name}") from exc


def _lines(path: Path) -> list[Any]:
    try:
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultsError("not_established", f"unreadable:{path.name}") from exc


def _cell(row: Mapping[str, Any] | None) -> str:
    """A held-out cell as admission writes it: a FRESH, error-free, infrastructure-sound result passes or
    fails; anything else is invalid evidence."""
    if row is None or row.get("mode") != "FRESH" or row.get("error") or row.get("infra_failed"):
        return "invalid"
    return "pass" if row.get("passed") is True else "fail"


def _held(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(r["case_id"]): r for r in report.get("results", []) if r.get("split") == "heldout"}


def _table(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    p1 = _held(reports["parent_run_1"])
    others = [_held(reports[r]) if r in reports else None for r in ROLES[1:]]
    out: dict[str, list[str]] = {}
    for case_id in p1:
        cells = [_cell(p1[case_id])] + [_cell(o.get(case_id)) if o is not None else "" for o in others]
        out[case_id] = cells
    return out


def _check_ruling2(entry: Mapping[str, Any], reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The per-case table, recoveries, broken protected cases and C1 flips, from the bundled reports."""
    signed = entry.get("heldout") or {}
    table = _table(reports)
    outcomes = signed.get("outcomes") or {}
    if set(outcomes) != set(table):
        raise ResultsError("contradicted", f"heldout_cases_differ:{entry['decision_id']}")
    for case_id, cells in table.items():
        claimed = list(outcomes[case_id])
        has_c2 = "candidate_confirm_run" in reports
        if claimed[:3] != cells[:3] or (has_c2 and claimed[3] != cells[3]):
            raise ResultsError("contradicted", f"heldout_table_mismatch:{entry['decision_id']}:{case_id}")
        if not has_c2 and not str(claimed[3]).startswith("not run"):
            raise ResultsError("contradicted", f"confirm_cell_without_run:{entry['decision_id']}:{case_id}")
    c2 = reports.get("candidate_confirm_run")
    gate = recompute.heldout(
        dict(reports["parent_run_1"]), dict(reports["parent_run_2"]), dict(reports["candidate_run"]),
        None if c2 is None else dict(c2),
    )  # fmt: skip
    up = sorted(k for k, v in table.items() if v[0] == v[1] == "fail" and v[2] == "pass")
    down = sorted(k for k, v in table.items() if "pass" in v[:2] and v[2] == "fail")
    expect = {
        "stable_recoveries": sorted(gate.evidence.get("stable_recoveries", [])),
        "protected_broken": sorted(gate.evidence.get("protected_broken", [])),
        "c1_fail_to_pass": up,
        "c1_pass_to_fail": down,
    }
    for name, value in expect.items():
        if sorted(signed.get(name) or []) != value:
            raise ResultsError("contradicted", f"{name}_mismatch:{entry['decision_id']}")
    return {
        "cases": len(table),
        "c1_fail_to_pass": len(up),
        "stable_recoveries": len(expect["stable_recoveries"]),
    }


def _counts(records: Sequence[Mapping[str, Any]], case_id: str, panels: int) -> tuple[int, list[int]]:
    mine = [r for r in records if r.get("case_id") == case_id and r.get("passed") is True]
    return len(mine), [sum(1 for r in mine if r.get("panel") == j) for j in range(panels)]


def _regrade_executions(
    records: Sequence[Mapping[str, Any]], traces: Sequence[Mapping[str, Any]], cases: Mapping[str, Any]
) -> tuple[int, set[str]]:
    by_run = {str(t.get("run_id")): t for t in traces}
    regraded, other_world = 0, set[str]()
    for r in records:
        trace = by_run.get(str(r.get("run_id")))
        if r.get("passed") is None or trace is None or r.get("case_id") not in cases:
            continue
        if trace.get("case_id") != r["case_id"]:
            raise ResultsError("contradicted", f"execution_trace_of_another_case:{r['run_id']}")
        try:
            problem = runs.regrade(cases[r["case_id"]], trace)
        except LookupError:
            other_world.add(str(r["case_id"]))
            continue
        if problem is None and trace["outcome"]["passed"] is not r["passed"]:
            problem = "record"
        if problem is not None:
            raise ResultsError(
                "contradicted", f"execution_grade_mismatch:{r['case_id']}:{r['run_id']}:{problem}"
            )
        regraded += 1
    return regraded, other_world


def _check_v3(
    entry: Mapping[str, Any],
    payload: Mapping[str, Any],
    root: Path,
    manifest: Mapping[str, Any],
    cases: Mapping[str, Any],
) -> dict[str, Any]:
    digests = payload.get("report_digests") or {}
    panels = int((payload.get("policy") or {}).get("panels", 2))
    arms = {"parent": "parent", "arm": str(entry.get("arm"))}
    records: dict[str, list[Any]] = {}
    traces: dict[str, list[Any]] = {}
    for key, name in arms.items():
        rel = (manifest.get("arms") or {}).get(name)
        records[key] = _json(_inside(root, rel)) if isinstance(rel, str) else []
        if runs.digest(records[key]) != digests.get(f"arm:{name}"):
            raise ResultsError("contradicted", f"arm_records_not_signed:{entry['decision_id']}:{name}")
        trel = (manifest.get("arm_traces") or {}).get(name)
        path = _inside(root, trel) if isinstance(trel, str) else None
        traces[key] = _lines(path) if path is not None and path.is_file() else []
    gate: Mapping[str, Any] = next(
        (c for c in payload.get("checks", []) if c.get("name") == "stochastic_v3"), {}
    )
    signed_rows = {
        r["case_id"]: r for r in (gate.get("evidence") or {}).get("comparison", {}).get("cases", [])
    }
    rows = entry.get("per_case") or []
    if {r["case_id"] for r in rows} != set(signed_rows):
        raise ResultsError("contradicted", f"v3_cases_differ:{entry['decision_id']}")
    for row in rows:
        cid = row["case_id"]
        for key, side in (("parent", "parent"), ("arm", "arm")):
            total, per = _counts(records[key], cid, panels)
            if row[side] != total or list(row[f"{side}_panels"]) != per:
                raise ResultsError("contradicted", f"v3_count_mismatch:{entry['decision_id']}:{cid}:{side}")
        if any(row[k] != signed_rows[cid].get(k) for k in row):
            raise ResultsError("contradicted", f"v3_row_not_the_signed_one:{entry['decision_id']}:{cid}")
    regraded, other = 0, set[str]()
    for key in arms:
        n, o = _regrade_executions(records[key], traces[key], cases)
        regraded, other = regraded + n, other | o
    executions = sum(1 for k in arms for r in records[k] if r.get("passed") is not None)
    return {
        "cases": len(rows),
        "executions": executions,
        "regraded": regraded,
        "not_rebuildable": sorted(other),
    }


def _verify_one(
    root: Path, entry: Mapping[str, Any], keys: Sequence[Ed25519PublicKey], cases: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    base = _inside(root, entry.get("bundle"))
    env = (base / "receipt.dsse").read_bytes() if (base / "receipt.dsse").is_file() else b""
    if sha256_hex(env) != entry.get("receipt_sha256"):
        raise ResultsError("contradicted", f"receipt_not_the_listed_one:{entry.get('decision_id')}")
    manifest = _json(base / "manifest.json")
    reports = {k: _json(_inside(base, v)) for k, v in (manifest.get("reports") or {}).items()}
    harnesses = {k: _json(_inside(base, v)) for k, v in (manifest.get("harnesses") or {}).items()}
    traces = {k: _lines(_inside(base, v)) for k, v in (manifest.get("traces") or {}).items()}
    v3 = entry.get("policy") == V3
    result = verify_decision(
        env,
        list(keys),
        reports=reports,
        harnesses=harnesses,
        traces=traces,
        cases={} if v3 else cases,
        options=Options(require_reports=not v3),
    )
    if not result.ok:
        raise ResultsError(
            "contradicted" if result.decision == "contradicted" else "not_established",
            f"receipt_does_not_verify:{entry.get('decision_id')}:{result.reason}",
        )
    if result.admitted is not entry.get("admitted") or result.candidate != entry.get("candidate"):
        raise ResultsError("contradicted", f"listed_verdict_not_the_receipts:{entry.get('decision_id')}")
    detail: dict[str, Any] = {"decision_id": entry["decision_id"], "policy": entry.get("policy")}
    if v3:
        payload = json.loads(dsse.parse(env).payload)
        detail |= _check_v3(entry, payload, root, manifest, cases)
    else:
        detail |= _check_ruling2(entry, reports)
        grades = next((i for i in result.recomputed if i["id"] == "heldout_grades"), {})
        detail["heldout_grades"] = {k: grades.get(k) for k in ("status", "heldout_rows", "regraded")}
    return detail, reports


def _check_repeatability(root: Path, signed: Mapping[str, Any]) -> str:
    from ratchet.improve.policies import normalized

    files = [root / "runs" / f"{r}.jsonl" for r in ("repeat_a", "repeat_b")]
    if not all(f.is_file() for f in files):
        return "authenticated_only"
    for f, key in zip(files, ("digests_a", "digests_b"), strict=True):
        recomputed = {t["case_id"]: normalized(runs._trace_of(t)) for t in _lines(f)}
        if recomputed != signed.get(key):
            raise ResultsError("contradicted", f"repeatability_{key}_mismatch")
    same = {k: signed["digests_a"].get(k) == signed["digests_b"].get(k) for k in signed.get("cases", {})}
    if same != signed.get("cases"):
        raise ResultsError("contradicted", "repeatability_verdict_mismatch")
    return "recomputed"


def verify_results(
    envelope: bytes,
    key: Ed25519PublicKey | Sequence[Ed25519PublicKey],
    root: Path,
    cases: Mapping[str, Any] | None = None,
) -> ResultsReport:
    """Verify a results envelope whose bundles live under `root`. Never raises on bad evidence."""
    keys = [key] if isinstance(key, Ed25519PublicKey) else list(key)
    try:
        env = dsse.parse(envelope)
    except dsse.EnvelopeError as exc:
        return ResultsReport("no_verdict", f"malformed_envelope:{exc}")
    if env.payload_type != RESULTS_TYPE:
        return ResultsReport("not_established", "unsupported_payload_type")
    if dsse.verify_signature(env, {dsse.keyid_of(k): k for k in keys}).keyid is None:
        return ResultsReport("contradicted", "signature_invalid")
    try:
        doc = json.loads(env.payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ResultsReport("not_established", "payload_not_json_object")
    if not isinstance(doc, dict) or doc.get("type") != "ratchet-referee-results" or doc.get("version") != 1:
        return ResultsReport("not_established", "unsupported_results_version")
    out = ResultsReport("supported", "verified", decisions=len(doc.get("decisions", [])))
    try:
        parents: dict[str, Mapping[str, Any]] = {}
        for entry in doc.get("decisions", []):
            detail, reports = _verify_one(root, entry, keys, cases or {})
            out.verified.append(str(entry["decision_id"]))
            out.recomputed.append(detail)
            parents = parents or {k: v for k, v in reports.items() if k in ROLES[:2]}
        notes = doc.get("runs") or {}
        if "repeatability" in notes:
            if _check_repeatability(root, notes["repeatability"]) == "recomputed":
                out.recomputed.append({"id": "repeatability", "cases": len(notes["repeatability"]["cases"])})
            else:
                out.authenticated_only.append("repeatability")
        if "parent_discordance" in notes and len(parents) == 2:
            a, b = (_held(parents[r]) for r in ROLES[:2])
            differ = sorted(k for k in a if a[k].get("passed") != (b.get(k) or {}).get("passed"))
            if differ != sorted(notes["parent_discordance"]):
                raise ResultsError("contradicted", "parent_discordance_mismatch")
            out.recomputed.append({"id": "parent_discordance", "cases": len(a)})
        if "v3_identity_controls" in notes:
            out.authenticated_only.append("v3_identity_controls")
    except ResultsError as exc:
        return ResultsReport(exc.decision, exc.reason, out.decisions, out.verified, out.recomputed)
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ratchet-verify-results", description=__doc__.split("\n\n")[0])
    p.add_argument("results", type=Path, help="the results envelope; its directory holds the bundles")
    p.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH, help="pinned Ed25519 public key (PEM)")
    p.add_argument("--case", action="append", default=[], metavar="ID=FILE", help="a disclosed case document")
    p.add_argument(
        "--case-suite",
        action="store_true",
        help="disclose every case of the installed probity-ratchet-cases package to regrade held-out runs",
    )
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    return p


def _suite(root: Path, envelope: bytes) -> dict[str, Any]:
    """Every case document of the installed suite, which must be the suite every listed receipt names."""
    from ratchet_cases.cases import all_cases, suite_digest

    installed = suite_digest()
    doc = json.loads(dsse.parse(envelope).payload)
    for entry in doc.get("decisions", []):
        receipt = _inside(root, entry.get("bundle")) / "receipt.dsse"
        named = named_suite_digest(receipt.read_bytes())
        if named != installed:
            raise ValueError(
                f"--case-suite: the installed case suite is {installed}, a receipt names {named}"
            )
    return {c.id: runs.case_document(c) for c in all_cases()}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        key = load_trusted_key(args.key)
        envelope = args.results.read_bytes()
        root = args.results.resolve().parent
        cases: dict[str, Any] = _suite(root, envelope) if args.case_suite else {}
        for spec in args.case:
            name, sep, file = spec.partition("=")
            if not sep or not name or not file:
                raise ValueError(f"--case expects ID=FILE, got {spec!r}")
            cases[name] = json.loads(Path(file).read_bytes())
    except (TrustedKeyError, OSError, ValueError, ResultsError, dsse.EnvelopeError) as exc:
        print(f"ratchet-verify-results: {exc}", file=sys.stderr)
        return 2
    report = verify_results(envelope, key, root, cases)
    if args.json:
        print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
    else:
        done = f"{len(report.verified)} of {report.decisions} decisions verified"
        print(f"{report.decision} {report.reason}: {done}")
        for d in report.recomputed:
            print(f"  recomputed {json.dumps(d, sort_keys=True)}")
        for name in report.authenticated_only:
            print(f"  authenticated only: {name}")
    if report.decision == "no_verdict":
        return 2
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
