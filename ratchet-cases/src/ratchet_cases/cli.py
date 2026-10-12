"""`ratchet-cases`: list cases, check the digest, prove the suite with scripted solvers, run and grade.

Exit status: 0 success; 1 a check failed (selftest, digest, regrade disagreement); 2 a usage or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ratchet_cases import __version__, released_digest
from ratchet_cases.adapters.openai_compat import EndpointError
from ratchet_cases.adapters.python_callable import AdapterError
from ratchet_cases.cases import CASE_KINDS, SPLITS, Case, all_cases, get, grade, suite_digest
from ratchet_cases.cases.solvers import NAIVE, REFERENCE, REFUSER, Habits, solve
from ratchet_cases.run import Act, Regraded, TraceFileError, read_traces, regrade, run_case, trace_line

METRICS = (
    "pass_rate",
    "trap_pass_rate",
    "benign_twin_pass_rate",
    "must_succeed_pass_rate",
    "false_intervention_rate",
)


def _select(args: argparse.Namespace) -> list[Case]:
    if getattr(args, "case", None):
        return [get(case_id) for case_id in args.case]
    cases = all_cases()
    if args.split:
        cases = [c for c in cases if c.split == args.split]
    if args.kind:
        cases = [c for c in cases if c.kind == args.kind]
    return cases


def cmd_list(args: argparse.Namespace) -> int:
    cases = _select(args)
    if args.json:
        rows = [
            {"id": c.id, "family": c.family, "split": c.split, "kind": c.kind, "trap": c.trap} for c in cases
        ]
        print(json.dumps(rows, indent=2))
        return 0
    for c in cases:
        print(f"{c.id:40} {c.split:8} {c.kind:13} {c.trap or ''}")
    print(f"{len(cases)} case(s)")
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    computed = suite_digest()
    try:
        recorded = released_digest()
    except KeyError:
        print(f"{computed}\nDIGESTS.json has no entry for {__version__}", file=sys.stderr)
        return 1
    print(computed)
    if computed != recorded:
        print(f"MISMATCH: DIGESTS.json records {recorded} for {__version__}", file=sys.stderr)
        return 1
    print(f"matches DIGESTS.json for {__version__}", file=sys.stderr)
    return 0


def _expected(habits: Habits, case: Case) -> bool:
    if habits is REFUSER:
        return False
    if habits is NAIVE:
        return case.kind != "trap"
    return True


def cmd_selftest(args: argparse.Namespace) -> int:
    cases = _select(args)
    failures: list[str] = []
    for name, habits in (("reference", REFERENCE), ("naive", NAIVE), ("refuser", REFUSER)):
        tally: Counter[tuple[str, bool]] = Counter()
        for case in cases:
            passed = grade(case, solve(case, habits)).passed
            tally[(case.kind, passed)] += 1
            if passed != _expected(habits, case):
                failures.append(f"{name} {'passed' if passed else 'failed'} {case.id} ({case.kind})")
        parts = [f"{k} {tally[(k, True)]}/{tally[(k, True)] + tally[(k, False)]}" for k in CASE_KINDS]
        print(f"{name:9} " + ", ".join(parts))
    if failures:
        print("SELFTEST FAILED:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    print(f"selftest passed on {len(cases)} case(s); suite digest {suite_digest()}")
    return 0


def _act(args: argparse.Namespace) -> Act:
    if args.adapter == "openai":
        from ratchet_cases.adapters.openai_compat import Endpoint

        return Endpoint.from_env(base_url=args.base_url, model=args.model)
    if args.adapter.startswith("python:"):
        from ratchet_cases.adapters.python_callable import load

        return load(args.adapter.removeprefix("python:"))
    raise ValueError(f"unknown adapter {args.adapter!r}: use openai or python:module:function")


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    path = out / "traces.jsonl"
    if path.exists():
        print(f"{path} exists; choose a new --out so runs never mix", file=sys.stderr)
        return 2
    act = _act(args)
    model = args.model or args.adapter
    out.mkdir(parents=True, exist_ok=True)
    passed = 0
    cases = _select(args)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            trace = run_case(case, act, model=model, max_steps=args.max_steps)
            handle.write(trace_line(trace, __version__) + "\n")
            handle.flush()
            passed += trace.outcome.passed
            print(f"{case.id:40} {'pass' if trace.outcome.passed else 'FAIL'}")
    print(f"{passed}/{len(cases)} passed; traces in {path}")
    return 0


def _regraded(paths: Sequence[Path]) -> tuple[list[Regraded], list[str]]:
    results: list[Regraded] = []
    problems: list[str] = []
    for path in paths:
        for record, trace in read_traces(path):
            if record.get("suite_digest") != suite_digest():
                problems.append(
                    f"{trace.case_id}: run under suite {record.get('suite_digest')}, not this one"
                )
            result = regrade(trace)
            if not result.digest_matches:
                problems.append(f"{trace.case_id}: replay does not reach the recorded final state")
            if result.outcome.passed != result.recorded_passed:
                problems.append(
                    f"{trace.case_id}: recorded {result.recorded_passed}, regraded {result.outcome.passed}"
                )
            results.append(result)
    return results, problems


def metrics(results: Sequence[Regraded]) -> dict[str, Any]:
    """Pass rates overall and per kind, and the false intervention rate (benign twins failed / run)."""

    def rate(rows: Sequence[Regraded]) -> float | None:
        return sum(r.outcome.passed for r in rows) / len(rows) if rows else None

    by_kind = {k: [r for r in results if r.kind == k] for k in CASE_KINDS}
    twins = rate(by_kind["benign_twin"])
    return {
        "runs": len(results),
        "pass_rate": rate(results),
        "trap_pass_rate": rate(by_kind["trap"]),
        "benign_twin_pass_rate": twins,
        "must_succeed_pass_rate": rate(by_kind["must_succeed"]),
        "false_intervention_rate": None if twins is None else 1 - twins,
        "counts": {k: len(v) for k, v in by_kind.items()},
    }


def _trace_files(paths: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        found.extend(sorted(p.rglob("traces.jsonl")) if p.is_dir() else [p])
    if not found:
        raise FileNotFoundError(f"no traces.jsonl under {', '.join(paths)}")
    return found


def cmd_grade(args: argparse.Namespace) -> int:
    results, problems = _regraded(_trace_files(args.paths))
    for r in results:
        print(f"{r.case_id:40} {'pass' if r.outcome.passed else 'FAIL'}")
    print(json.dumps(metrics(results), indent=2))
    for problem in problems:
        print("DISAGREES:", problem, file=sys.stderr)
    return 1 if problems else 0


def cmd_report(args: argparse.Namespace) -> int:
    results, problems = _regraded(_trace_files(args.paths))
    table = metrics(results)
    if args.metric:
        print(json.dumps({args.metric: table[args.metric], "runs": table["runs"]}))
    else:
        print(json.dumps(table, indent=2))
    for problem in problems:
        print("DISAGREES:", problem, file=sys.stderr)
    return 1 if problems else 0


def _filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--split", choices=SPLITS)
    p.add_argument("--kind", choices=CASE_KINDS)
    p.add_argument("--case", action="append", help="a case id; repeatable; overrides --split and --kind")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ratchet-cases", description="List, check, selftest, run and grade the case suite."
    )
    ap.add_argument("--version", action="version", version=f"ratchet-cases {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list", help="list cases")
    _filters(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("digest", help="print the suite digest and check it against DIGESTS.json")
    p.set_defaults(func=cmd_digest)
    p = sub.add_parser("selftest", help="run the scripted solvers; exit 1 unless each scores as designed")
    _filters(p)
    p.set_defaults(func=cmd_selftest)
    p = sub.add_parser("run", help="run an agent over the cases and write traces.jsonl")
    _filters(p)
    p.add_argument("--adapter", required=True, help="openai, or python:module:function")
    p.add_argument("--model", help="model name (openai adapter; RATCHET_MODEL otherwise)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (RATCHET_BASE_URL otherwise)")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--out", required=True, help="a new directory for traces.jsonl")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("grade", help="regrade recorded traces from their replayed final state")
    p.add_argument("paths", nargs="+", help="traces.jsonl files or directories holding them")
    p.set_defaults(func=cmd_grade)
    p = sub.add_parser("report", help="metrics over one or more run directories")
    p.add_argument("paths", nargs="+")
    p.add_argument("--metric", choices=METRICS)
    p.set_defaults(func=cmd_report)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except (KeyError, ValueError, FileNotFoundError, TraceFileError, EndpointError, AdapterError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
