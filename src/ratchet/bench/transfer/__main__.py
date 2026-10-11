"""Command line: `python -m ratchet.bench.transfer probe|run|render ...`.

probe   one tool-calling request and one full `run_case` of the seed harness on one case; prints JSON
run     v0 and an admitted harness, FRESH and paired, on the chosen splits; writes one result file
render  the markdown report from one or more result files

The endpoint key comes from `RATCHET_API_KEY` (any value for an unauthenticated server) and is never written
to a record. The store URI comes from `RATCHET_MONGODB_URI`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ratchet.bench.transfer.render import render
from ratchet.bench.transfer.run import (
    default_client,
    endpoint_info,
    harness_from,
    now_iso,
    persist,
    probe_tool_call,
    result_payload,
    run_transfer,
    trace_summary,
)
from ratchet.types import Harness


def _log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", file=sys.stderr, flush=True)


def _commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _key() -> str:
    return os.environ.get("RATCHET_API_KEY") or "unused"


def cmd_probe(args: argparse.Namespace) -> int:
    from ratchet.agent.seed import seed_harness
    from ratchet.agent.worker import run_case
    from ratchet.cases.registry import get

    info = endpoint_info(args.base_url, _key())
    with default_client(args.base_url, _key(), args.model) as model:
        tool = probe_tool_call(model)
        case = get(args.case)
        started = time.monotonic()
        trace = run_case(seed_harness(), case, model)
        run = trace_summary(trace, time.monotonic() - started)
    out = {"at": now_iso(), "model": args.model, "endpoint": info, "tool_call": tool, "run_case": run}
    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if tool["tool_call_ok"] else 1


def _candidate(args: argparse.Namespace, store: Any) -> tuple[Harness, dict[str, Any]]:
    if args.candidate_file:
        raw = json.loads(Path(args.candidate_file).read_text(encoding="utf-8"))
        return harness_from(raw.get("harness", raw)), {"source": args.candidate_file}
    if store is None:
        raise SystemExit("--candidate needs RATCHET_MONGODB_URI (or pass --candidate-file)")
    admitted = store.decisions({"admitted": True})
    if args.candidate != "latest":
        admitted = [d for d in admitted if d.get("candidate") == args.candidate]
    if not admitted:
        raise SystemExit(f"no admitted decision for candidate {args.candidate!r} in the store")
    decision = admitted[-1]
    harness = store.get_harness(decision["candidate"])
    if harness is None:
        raise SystemExit(f"admitted version {decision['candidate']} is not in harness_versions")
    return harness, {
        "decision_id": str(decision.get("_id")),
        "decided_at": decision.get("decided_at"),
        "derived_on": derived_on(decision),
        "evaluator_version": (decision.get("evaluator") or {}).get("version"),
        "admitted_decisions_in_store": len(store.decisions({"admitted": True})),
    }


def derived_on(decision: dict[str, Any]) -> Any:
    """The model the admission evidence was produced with: the frozen evaluator's model configuration."""
    cfg = (decision.get("evaluator") or {}).get("model_config") or {}
    return cfg.get("model") or (decision.get("context") or {}).get("model")


def _harness_meta(h: Harness, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": h.version,
        "parent": h.parent,
        "digest": h.digest,
        "guards": len(h.guards),
        "procedures": len(h.procedures),
        "created_by": h.created_by,
        **(extra or {}),
    }


def cmd_run(args: argparse.Namespace) -> int:
    from ratchet.agent.seed import seed_harness
    from ratchet.agent.worker import run_case
    from ratchet.cases.registry import by_split, suite_digest

    uri = os.environ.get("RATCHET_MONGODB_URI")
    source = None
    if uri:
        from ratchet.store import Store

        source = Store(uri, args.store_db)
    base = seed_harness()
    cand, cand_extra = _candidate(args, source)
    splits = [s for s in args.splits.split(",") if s]
    cases = [c for s in splits for c in by_split(s)]
    info = endpoint_info(args.base_url, _key())
    _log(f"{args.model}: base {base.version}, cand {cand.version}, {len(cases)} cases, splits {splits}")
    started = now_iso()
    with default_client(args.base_url, _key(), args.model) as model:
        base_reports, cand_reports = run_transfer(
            base,
            cand,
            cases,
            model,
            run_case,
            repeats=args.repeats,
            workers=args.workers,
            store=None,
            log=_log,
        )
    meta = {
        "model": args.model,
        "endpoint": info,
        "base": _harness_meta(base),
        "cand": _harness_meta(cand, cand_extra),
        "suite_digest": suite_digest(),
        "splits": splits,
        "case_counts": {s: sum(1 for c in cases if c.split == s) for s in splits},
        "repeats": args.repeats,
        "workers": args.workers,
        "started_at": started,
        "finished_at": now_iso(),
        "commit": _commit(),
        "trace_db": args.trace_db if uri and args.trace_db else None,
    }
    payload = result_payload(meta, base_reports, cand_reports, splits)
    if uri and args.trace_db:
        from pymongo import MongoClient

        client: MongoClient[dict[str, Any]] = MongoClient(uri, serverSelectionTimeoutMS=10_000)
        try:
            written = persist(client[args.trace_db], payload, [*base_reports, *cand_reports])
        finally:
            client.close()
        _log(f"stored {written['traces']} traces and eval {written['eval_id']} in {args.trace_db}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    for split, s in payload["summaries"].items():
        _log(
            f"{args.model} {split}: v0 {s['base']['passes']} cand {s['cand']['passes']} b={s['b']} c={s['c']}"
        )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    results = []
    for path in args.results:
        data = Path(path).read_bytes()
        results.append((json.loads(data), hashlib.sha256(data).hexdigest()))
    header = dict(item.split("=", 1) for item in args.note)
    Path(args.out).write_text(render(results, header), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ratchet.bench.transfer")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--base-url", required=True)
    pr.add_argument("--model", required=True)
    pr.add_argument("--case", required=True)
    pr.add_argument("--out")
    rn = sub.add_parser("run")
    rn.add_argument("--base-url", required=True)
    rn.add_argument("--model", required=True)
    rn.add_argument("--candidate", default="latest")
    rn.add_argument("--candidate-file")
    rn.add_argument("--splits", default="heldout,audit")
    rn.add_argument("--repeats", type=int, default=2)
    rn.add_argument("--workers", type=int, default=8)
    rn.add_argument("--store-db", default=os.environ.get("RATCHET_DB", "ratchet"))
    rn.add_argument("--trace-db", default="ratchet_heldout", help="evaluator-only database; empty to skip")
    rn.add_argument("--out", required=True)
    rd = sub.add_parser("render")
    rd.add_argument("results", nargs="+")
    rd.add_argument("--out", required=True)
    rd.add_argument("--note", action="append", default=[], help="KEY=VALUE line for the setup section")
    args = p.parse_args(argv)
    handlers = {"probe": cmd_probe, "run": cmd_run, "render": cmd_render}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
