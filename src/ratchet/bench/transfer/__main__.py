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
    SingleCallModel,
    default_client,
    determinism_probe,
    endpoint_info,
    harness_from,
    now_iso,
    persist,
    probe_tool_call,
    result_payload,
    run_transfer,
    sampling_for,
    trace_docs,
    trace_summary,
)
from ratchet.types import Harness

STORE_ATTEMPTS = 5
STORE_BACKOFF_S = 15.0


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
        extra: dict[str, Any] = {"source": args.candidate_file}
        decision = raw.get("decision")
        if decision:
            extra |= {
                "decision_id": str(decision.get("_id")),
                "decided_at": decision.get("decided_at"),
                "derived_on": derived_on(decision),
                "evaluator_version": (decision.get("evaluator") or {}).get("version"),
                "store": raw.get("store"),
                "decision": decision,
            }
        return harness_from(raw.get("harness", raw)), extra
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
        "decision": decision,
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
    try:
        cand, cand_extra = _candidate(args, source)
    finally:
        if source is not None:
            source.close()
    sampling = sampling_for(args.temperature, args.seed, cand_extra.pop("decision", None))
    splits = [s for s in args.splits.split(",") if s]
    cases = [c for s in splits for c in by_split(s)]
    info = endpoint_info(args.base_url, _key())
    _log(f"{args.model}: base {base.version}, cand {cand.version}, {len(cases)} cases, splits {splits}")
    started = now_iso()
    with default_client(
        args.base_url, _key(), args.model, temperature=sampling["temperature"], seed=sampling["seed"]
    ) as client:
        model: Any = SingleCallModel(client) if args.single_tool_call else client
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
        "sampling": sampling,
        "started_at": started,
        "finished_at": now_iso(),
        "commit": _commit(),
        "trace_db": args.trace_db if uri and args.trace_db else None,
        "single_tool_call": args.single_tool_call,
        "truncated_turns": model.truncated_turns if args.single_tool_call else 0,
    }
    payload = result_payload(meta, base_reports, cand_reports, splits)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    traces = trace_docs([*base_reports, *cand_reports])
    traces_path = out.with_suffix(".traces.jsonl")
    with traces_path.open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, sort_keys=True, default=str) + "\n")
    for split, s in payload["summaries"].items():
        _log(
            f"{args.model} {split}: v0 {s['base']['passes']} cand {s['cand']['passes']} b={s['b']} c={s['c']}"
        )
    if uri and args.trace_db:
        return _store(uri, args.trace_db, payload, traces)
    return 0


def _store(uri: str, db_name: str, payload: dict[str, Any], traces: list[dict[str, Any]]) -> int:
    """Store with a bounded retry (a replica-set election makes writes fail for a while); the result and
    trace files are already on disk, so a failure here loses nothing and `store` can be rerun later."""
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    for attempt in range(1, STORE_ATTEMPTS + 1):
        mongo: MongoClient[dict[str, Any]] = MongoClient(uri, serverSelectionTimeoutMS=10_000)
        try:
            written = persist(mongo[db_name], payload, traces)
        except PyMongoError as exc:
            _log(f"store attempt {attempt}/{STORE_ATTEMPTS} failed: {type(exc).__name__}: {str(exc)[:200]}")
            time.sleep(STORE_BACKOFF_S * attempt)
        else:
            _log(f"stored {written['traces']} traces and eval {written['eval_id']} in {db_name}")
            return 0
        finally:
            mongo.close()
    _log(f"STORE_FAILED: results and traces kept on disk; rerun `store` to write them to {db_name}")
    return 3


def cmd_store(args: argparse.Namespace) -> int:
    uri = os.environ.get("RATCHET_MONGODB_URI")
    if not uri:
        raise SystemExit("store needs RATCHET_MONGODB_URI")
    out = Path(args.result)
    payload = json.loads(out.read_text(encoding="utf-8"))
    traces_path = out.with_suffix(".traces.jsonl")
    traces = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines() if line]
    return _store(uri, args.trace_db, payload, traces)


def case_first_request(case_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The first request the worker sends for a case under the seed harness: system prompt, ticket, tools."""
    from ratchet.agent.seed import seed_harness
    from ratchet.agent.worker import read_procedure_spec, system_prompt
    from ratchet.cases.registry import get
    from ratchet.world import World

    case, harness = get(case_id), seed_harness()
    allowed = set(harness.tool_allowlist)
    tools = [t for t in World.from_case(case).tools() if t["function"]["name"] in allowed]
    messages = [
        {"role": "system", "content": system_prompt(harness, [])},
        {"role": "user", "content": f"Ticket {case.ticket_id}\n\n{case.customer_message}"},
    ]
    return messages, [*tools, read_procedure_spec(harness)]


def cmd_determinism(args: argparse.Namespace) -> int:
    messages, tools = case_first_request(args.case) if args.case else (None, None)
    rounds = [
        determinism_probe(
            args.base_url,
            _key(),
            args.model,
            repeats=args.repeats,
            background=args.background,
            temperature=args.temperature,
            seed=args.seed,
            messages=messages,
            tools=tools,
            max_tokens=args.max_tokens,
        )
        for _ in range(args.rounds)
    ]
    across = {json.dumps(r["outputs"][0], sort_keys=True) for r in rounds}
    out = {"at": now_iso(), "rounds": rounds, "identical_across_rounds": len(across) == 1}
    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    ok = all(r["identical"] for r in rounds) and out["identical_across_rounds"]
    print(
        f"determinism: rounds={len(rounds)} within-round identical={[r['identical'] for r in rounds]} "
        f"across-rounds identical={out['identical_across_rounds']}"
    )
    return 0 if ok else 1


def cmd_render(args: argparse.Namespace) -> int:
    def load(paths: list[str]) -> list[tuple[dict[str, Any], str]]:
        out = []
        for path in paths:
            data = Path(path).read_bytes()
            out.append((json.loads(data), hashlib.sha256(data).hexdigest()))
        return out

    header = dict(item.split("=", 1) for item in args.note)
    text = render(load(args.results), header, load(args.control))
    Path(args.out).write_text(text, encoding="utf-8")
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
    rn.add_argument("--temperature", type=float, help="default: the derivation run's, else 0")
    rn.add_argument("--seed", type=int, help="default: the derivation run's, else none")
    rn.add_argument("--workers", type=int, default=8)
    rn.add_argument("--store-db", default=os.environ.get("RATCHET_DB", "ratchet"))
    rn.add_argument("--trace-db", default="ratchet_heldout", help="evaluator-only database; empty to skip")
    rn.add_argument(
        "--single-tool-call",
        action="store_true",
        help="keep only the first tool call per turn (for checkpoints whose template forbids several)",
    )
    rn.add_argument("--out", required=True)
    dt = sub.add_parser("determinism")
    dt.add_argument("--base-url", required=True)
    dt.add_argument("--model", required=True)
    dt.add_argument("--repeats", type=int, default=5)
    dt.add_argument("--background", type=int, default=8)
    dt.add_argument("--rounds", type=int, default=2)
    dt.add_argument("--temperature", type=float, default=0.0)
    dt.add_argument("--seed", type=int, default=7)
    dt.add_argument("--case", help="use this case's first worker request as the repeated request")
    dt.add_argument("--max-tokens", type=int, default=2048)
    dt.add_argument("--out")
    st = sub.add_parser("store")
    st.add_argument("result")
    st.add_argument("--trace-db", default="ratchet_heldout")
    rd = sub.add_parser("render")
    rd.add_argument("results", nargs="*")
    rd.add_argument("--out", required=True)
    rd.add_argument("--control", action="append", default=[], help="an A/A result file (v0 against v0)")
    rd.add_argument("--note", action="append", default=[], help="KEY=VALUE line for the setup section")
    args = p.parse_args(argv)
    handlers = {
        "probe": cmd_probe,
        "run": cmd_run,
        "store": cmd_store,
        "determinism": cmd_determinism,
        "render": cmd_render,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
