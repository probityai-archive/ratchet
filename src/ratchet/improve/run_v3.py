"""Evaluate candidates under stochastic policy v3, live: a parent reference arm, two identity-control arms
and each candidate arm, 40 fresh executions per case per arm, interleaved by a predeclared schedule.

    uv run python -m ratchet.improve.run_v3 --cases a,b --candidate db:ad50dac8194c --out x.json

Every execution is stored in Atlas as it completes (`v3_executions`), so a stopped run keeps its evidence.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ratchet_cases.cases import all_cases
from ratchet_cases.types import canonical_json, sha256_hex

from ratchet.agent.model import ModelClient, ModelConfig
from ratchet.agent.seed import seed_harness
from ratchet.agent.worker import run_case
from ratchet.improve.admit import Check, admit_v3
from ratchet.improve.evaluate import _one
from ratchet.improve.evaluator import evaluator_identity
from ratchet.improve.loop import frozen_suite_digest
from ratchet.improve.receipt import Signer, receipt
from ratchet.improve.run import SeededTransport
from ratchet.improve.stochastic import (
    POLICY,
    RUNS,
    Arm,
    CaseInfo,
    Execution,
    compare,
    execution_record,
    panel_of,
    qualify,
    schedule,
    schedule_digest,
)
from ratchet.store import Store
from ratchet.types import Guard, Harness, Procedure


def rebuild(parent: Harness, decision: dict[str, Any]) -> Harness:
    """The exact candidate a stored decision judged, rebuilt from its recorded change, checked by digest."""
    ctx = decision["context"]
    change, kind = ctx["change"], ctx["kind"]
    if kind == "guard":
        child = parent.child(guards=[*parent.guards, Guard(**change)], created_by="improver")
    elif kind == "procedure":
        child = parent.child(procedures=[*parent.procedures, Procedure(**change)], created_by="improver")
    else:
        child = parent.child(
            system_prompt=f"{parent.system_prompt}\n\n{change['append']}", created_by="improver"
        )
    if child.version != decision["candidate"]:
        raise SystemExit(f"rebuilt {child.version}, decision names {decision['candidate']}")
    return child


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ratchet-improve-run-v3")
    p.add_argument("--endpoints", default="http://localhost:8801/v1")
    p.add_argument(
        "--endpoints-file",
        type=Path,
        default=None,
        help="re-read before every execution; replicas of the same served model join as they answer",
    )
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--workers", type=int, default=128)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--schedule-seed", type=int, default=20261011)
    p.add_argument("--db", default="ratchet_v3")
    p.add_argument("--cases", required=True)
    p.add_argument(
        "--candidate", action="append", default=[], help="db:candidate_version of a stored decision"
    )
    p.add_argument("--deadline-s", type=float, default=float("inf"))
    p.add_argument("--scope-note", default="")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = _args(argv)
    uri = os.environ.get("RATCHET_MONGODB_URI", "")
    if not uri:
        print("error: set RATCHET_MONGODB_URI", file=sys.stderr)
        return 2
    every = {c.id: c for c in all_cases()}
    case_ids = [c.strip() for c in a.cases.split(",") if c.strip()]
    cases = [every[c] for c in case_ids]
    if any(c.split != "heldout" for c in cases):
        print("error: v3 evaluates declared held-out cases only", file=sys.stderr)
        return 2
    infos = [CaseInfo(c.id, c.kind, c.must_succeed) for c in cases]
    parent = seed_harness()
    signer = Signer.for_decisions()
    originals: dict[str, dict[str, Any]] = {}
    arms: dict[str, Harness] = {"parent": parent, "identity_1": parent, "identity_2": parent}
    for spec in a.candidate:
        db, version = spec.split(":")
        with Store(uri, db) as s:
            d = s.db.decisions.find_one({"candidate": version}, {"envelope": 0})
        if d is None:
            print(f"error: no decision for {spec}", file=sys.stderr)
            return 2
        originals[version] = {**d, "source_db": db}
        arms[f"candidate:{version}"] = rebuild(parent, d)
    endpoints = [e.strip() for e in a.endpoints.split(",") if e.strip()]
    model_config = {
        "endpoints": endpoints,
        "model": a.model,
        "temperature": 0.0,
        "seed_base": a.seed_base,
        "seed_schedule": "seed_base + repetition, shared across arms",
        "workers": a.workers,
    }
    evaluator = evaluator_identity(cases, model_config, frozen_suite_digest())
    order = schedule(list(arms), case_ids, a.schedule_seed)
    frozen = {
        "policy": POLICY,
        "schedule_seed": a.schedule_seed,
        "schedule_digest": schedule_digest(order),
        "cases": case_ids,
        "arms": {k: h.version for k, h in arms.items()},
        "evaluator": evaluator["version"],
        "model_config": model_config,
        "infra_retries": 2,
        "scope_note": a.scope_note,
    }
    print("FROZEN", json.dumps(frozen), flush=True)
    clients: dict[tuple[str, int], ModelClient] = {}
    client_lock = threading.Lock()

    def current_endpoints() -> list[str]:
        if a.endpoints_file is not None and a.endpoints_file.is_file():
            listed = [x.strip() for x in a.endpoints_file.read_text().split() if x.strip()]
            if listed:
                return listed
        return endpoints

    def client_for(index: int, rep: int) -> tuple[str, ModelClient]:
        live = current_endpoints()
        url = live[index % len(live)]
        with client_lock:
            if (url, rep) not in clients:
                clients[(url, rep)] = ModelClient(
                    ModelConfig(base_url=url, model=a.model, api_key="unused", temperature=0.0),
                    transport=SeededTransport(a.seed_base + rep),
                )
            return url, clients[(url, rep)]

    results: dict[str, Arm] = {k: Arm(k, h.version) for k, h in arms.items()}
    lock = threading.Lock()
    started = time.perf_counter()
    done = [0]
    store = Store(uri, a.db)
    store.put_eval({"type": "v3_frozen", **frozen}, eval_id=f"v3-frozen-{frozen['schedule_digest'][:16]}")

    def one(index: int, arm: str, case_id: str, rep: int) -> None:
        try:
            run_one(index, arm, case_id, rep)
        except Exception as exc:  # recorded as an invalid execution, and shown at once
            print(f"DRIVER_ERROR {arm} {case_id} {rep}: {type(exc).__name__}: {exc}", flush=True)
            ex = Execution(
                arm, case_id, rep, panel_of(rep), a.seed_base + rep, None, note=f"driver error: {exc}"[:300]
            )
            with lock:
                results[arm].executions.append(ex)

    def run_one(index: int, arm: str, case_id: str, rep: int) -> None:
        if time.perf_counter() - started > a.deadline_s:
            ex = Execution(
                arm, case_id, rep, panel_of(rep), a.seed_base + rep, None, note="not run: deadline"
            )
        else:
            url, client = client_for(index, rep)
            res, trace = _one(run_case, arms[arm], every[case_id], client, None)
            valid = res.mode == "FRESH" and not res.infra_failed and not res.error
            ex = Execution(
                arm,
                case_id,
                rep,
                panel_of(rep),
                a.seed_base + rep,
                res.passed if valid else None,
                res.run_id,
                res.interventions,
                res.attempts,
                "" if valid else (res.error or "invalid"),
                url,
            )
            if trace is not None:
                store.put_trace(trace)
        store.put_eval(
            {"type": "v3_execution", "schedule_digest": frozen["schedule_digest"], **execution_record(ex)}
        )
        with lock:
            results[arm].executions.append(ex)
            done[0] += 1
            if done[0] % 50 == 0:
                rate = 60 * done[0] / (time.perf_counter() - started)
                print(f"progress {done[0]}/{len(order)} runs/min={rate:.1f}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures: list[Future[None]] = [pool.submit(one, i, arm, c, r) for i, (arm, c, r) in enumerate(order)]
        for f in futures:
            f.result()
    wall = time.perf_counter() - started
    cohort = qualify(results["parent"], [results["identity_1"], results["identity_2"]], infos)
    arm_digests = {
        k: sha256_hex(
            canonical_json(
                [execution_record(e) for e in sorted(v.executions, key=lambda e: (e.case_id, e.rep))]
            )
        )
        for k, v in results.items()
    }
    summary: dict[str, Any] = {
        "frozen": frozen,
        "wall_s": round(wall, 1),
        "cohort": cohort["status"],
        "controls": [
            {
                k: c[k]
                for k in (
                    "arm",
                    "qualifies",
                    "stable_recoveries",
                    "stable_regressions",
                    "case_declines_over_limit",
                    "aggregate",
                    "incomplete_cases",
                )
            }
            for c in cohort["controls"]
        ],
        "decisions": [],
    }
    for version, d in originals.items():
        name = f"candidate:{version}"
        cmp = compare(results["parent"], results[name], infos)
        benign = sum(
            e.interventions for e in results[name].executions if every[e.case_id].kind == "benign_twin"
        )
        checks = {c["name"]: c for c in d["checks"]}
        dec = admit_v3(
            parent,
            arms[name],
            cohort=cohort,
            comparison=cmp,
            benign_interventions=benign,
            replay_check=Check(
                "replay",
                checks["replay"]["passed"],
                {
                    **checks["replay"]["evidence"],
                    "from_decision": f"{d['source_db']}:{d['parent']}->{d['candidate']}",
                },
            ),
            overfit_check=Check(
                "overfit_lint", checks["overfit_lint"]["passed"], checks["overfit_lint"]["evidence"]
            ),
            evaluator=evaluator,
            visible_case_ids=d.get("visible_case_ids", []),
            report_digests={f"arm:{k}": v for k, v in arm_digests.items()}
            | {"schedule": frozen["schedule_digest"]},
            corpus_digests=d.get("corpus_digests", {}),
            scope={"cases": case_ids, "note": a.scope_note, "runs_per_arm_per_case": RUNS},
            exceptions=[
                f"{e.arm} {e.case_id} rep {e.rep}: {e.note}"
                for v in results.values()
                for e in v.executions
                if e.passed is None
            ],
            context={
                "policy": "stochastic-v3",
                "original_decision": f"{d['parent']}->{d['candidate']}",
                "kind": d["context"]["kind"],
            },
        )
        env = receipt(dec, signer)
        decision_id = f"v3:{dec.parent}->{dec.candidate}:{frozen['schedule_digest'][:12]}"
        store.put_decision(
            {**dec.to_payload(), "envelope": base64.b64encode(env).decode("ascii"), "keyid": signer.keyid},
            decision_id=decision_id,
        )
        summary["decisions"].append(
            {
                "decision_id": decision_id,
                "candidate": version,
                "admitted": dec.admitted,
                "receipt_sha256": sha256_hex(env),
                "failed": [c.name for c in dec.checks if not c.passed],
                "stable_recoveries": cmp["stable_recoveries"],
                "stable_regressions": cmp["stable_regressions"],
                "declines_over_limit": cmp["case_declines_over_limit"],
                "aggregate": cmp["aggregate"],
                "per_case": [
                    {
                        k: r[k]
                        for k in (
                            "case_id",
                            "parent",
                            "parent_panels",
                            "arm",
                            "arm_panels",
                            "parent_class",
                            "arm_class",
                            "decline",
                        )
                    }
                    for r in cmp["cases"]
                ],
                "benign_interventions": benign,
            }
        )
    summary["identity_per_case"] = {
        c["arm"]: [
            {k: r[k] for k in ("case_id", "parent", "arm", "parent_panels", "arm_panels")} for r in c["cases"]
        ]
        for c in cohort["controls"]
    }
    summary["signer_keyid"] = signer.keyid
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    store.put_eval({"type": "v3_summary", **summary})
    for c in clients.values():
        c.close()
    store.close()
    print(
        "DONE",
        json.dumps(
            {
                "cohort": cohort["status"],
                "decisions": [(x["candidate"], x["admitted"], x["failed"]) for x in summary["decisions"]],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
