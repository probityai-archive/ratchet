"""Command line: `python -m ratchet.verify RECEIPT [--key PEM] [--report NAME=FILE ...] ...`.

Exit status: 0 the receipt verifies (and was admitted when `--require-admitted` is given), 1 it does not
verify, 2 the input was malformed or unreadable and no verdict was reached.

The output keeps RECOMPUTED apart from AUTHENTICATED ONLY and names every item left unsupplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ratchet.verify.keys import DEFAULT_KEY_PATH, TrustedKeyError, load_trusted_key
from ratchet.verify.receipt import NO_VERDICT, Options, verify_decision


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ratchet-verify", description=__doc__)
    p.add_argument("receipt", type=Path, help="DSSE envelope file")
    p.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH, help="pinned Ed25519 public key (PEM)")
    add = p.add_argument
    add("--report", action="append", default=[], metavar="NAME=FILE", help="a stored report to check")
    add("--harness", action="append", default=[], metavar="parent|candidate=FILE", help="a harness document")
    add(
        "--recomputed",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="replay:<guard> or overfit_lint",
    )
    add(
        "--traces",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="JSON lines of the trace documents of the evaluation report NAME",
    )
    add("--approval", type=Path, help="human approval envelope for an authority exception")
    add(
        "--sessions",
        type=Path,
        help="the k-session bundle directory a tool addition or procedure edit was admitted on",
    )
    add("--human-key", type=Path, action="append", default=[], help="pinned key of a human approver")
    add("--require-reports", action="store_true", help="every named report must be supplied")
    add("--require-complete", action="store_true", help="every recomputable item must be recomputed")
    add("--require-admitted", action="store_true", help="a verified rejection exits 1")
    add("--json", action="store_true", help="print the full report as JSON")
    return p


def _load(specs: list[str], flag: str) -> dict[str, Any]:
    docs: dict[str, Any] = {}
    for spec in specs:
        name, sep, file = spec.partition("=")
        if not sep or not name or not file:
            raise ValueError(f"{flag} expects NAME=FILE, got {spec!r}")
        docs[name] = json.loads(Path(file).read_bytes())
    return docs


def _files(specs: list[str], flag: str) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for spec in specs:
        name, sep, file = spec.partition("=")
        if not sep or not name or not file:
            raise ValueError(f"{flag} expects NAME=FILE, got {spec!r}")
        out[name] = Path(file).read_bytes()
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        key = load_trusted_key(args.key)
        humans = tuple(load_trusted_key(k) for k in args.human_key)
        reports = _load(args.report, "--report")
        harnesses = _load(args.harness, "--harness")
        recomputed = _load(args.recomputed, "--recomputed")
        traces = _files(args.traces, "--traces")
        approval = args.approval.read_bytes() if args.approval else None
        envelope = args.receipt.read_bytes()
    except (TrustedKeyError, OSError, ValueError) as exc:
        print(f"ratchet-verify: {exc}", file=sys.stderr)
        return 2
    options = Options(
        require_reports=args.require_reports,
        require_complete=args.require_complete,
        trusted_human_keys=humans,
    )
    result = verify_decision(
        envelope,
        key,
        reports=reports,
        harnesses=harnesses,
        recomputed=recomputed,
        approval=approval,
        options=options,
        sessions=args.sessions,
        traces=traces,
    )
    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
    else:
        verdict = "admitted" if result.admitted else "rejected" if result.admitted is False else "unknown"
        print(f"{verdict}; {result.reports_summary}")
        print(result.summary)
    if result.decision == NO_VERDICT:
        return 2
    if not result.ok or (args.require_admitted and not result.admitted):
        return 1
    return 0
