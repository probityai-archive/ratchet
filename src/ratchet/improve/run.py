"""Run the ratchet for real: the case suite, the worker, the sandboxed guard runtime, Atlas and a live model.

    uv run python -m ratchet.improve.run --generations 1 --out private/improve/live.json

The agent model, the improver model, the endpoint and the store come from flags or the environment
(`RATCHET_BASE_URL`, `RATCHET_API_KEY`, `RATCHET_MODEL`, `RATCHET_IMPROVER_MODEL`, `RATCHET_MONGODB_URI`,
`RATCHET_DB`, `RATCHET_SIGNING_KEY`). No key is written to the summary, a log or the store.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pymongo import MongoClient

from ratchet.agent.guards import GuardRuntime
from ratchet.agent.model import ModelClient, ModelConfig
from ratchet.agent.seed import seed_harness
from ratchet.agent.worker import run_case
from ratchet.cases import all_cases
from ratchet.improve.activate import MongoActivator
from ratchet.improve.loop import EvolveResult, Lab, evolve
from ratchet.improve.receipt import Signer, public_key_bytes
from ratchet.improve.sandbox import SandboxRunner
from ratchet.improve.worldsim import WorldSimulator
from ratchet.store import Store


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    env = os.environ
    p = argparse.ArgumentParser(prog="ratchet-improve-run")
    p.add_argument("--generations", type=int, default=1)
    p.add_argument("--base-url", default=env.get("RATCHET_BASE_URL", "http://localhost:8787/v1"))
    p.add_argument("--model", default=env.get("RATCHET_MODEL", "gpt-5.6-luna"))
    p.add_argument("--improver-model", default=env.get("RATCHET_IMPROVER_MODEL", "gpt-5.6-terra"))
    p.add_argument("--improver-base-url", default=env.get("RATCHET_IMPROVER_BASE_URL", ""))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--uri", default=env.get("RATCHET_MONGODB_URI", ""))
    p.add_argument("--db", default=env.get("RATCHET_DB", "ratchet"))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max-clusters", type=int, default=4)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--max-evals", type=int, default=3)
    p.add_argument("--family", action="append", default=[], help="limit to these case families")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def _mongo(uri: str) -> MongoClient[dict[str, Any]]:
    return MongoClient(uri, tz_aware=True)


def summary(result: EvolveResult, signer: Signer, wall_s: float, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": config,
        "signer_keyid": signer.keyid,
        "signer_public_key_hex": public_key_bytes(signer.public_key).hex(),
        "final_harness": result.harness.version,
        "final_instructions": result.instructions.version,
        "wall_s": round(wall_s, 1),
        "generations": [g.to_payload() for g in result.generations],
    }


def main(argv: Sequence[str] | None = None) -> int:
    a = _args(argv)
    key = os.environ.get("RATCHET_API_KEY", "")
    if not a.uri or not key:
        print("error: set RATCHET_MONGODB_URI (or --uri) and RATCHET_API_KEY", file=sys.stderr)
        return 2
    cases = [c for c in all_cases() if not a.family or c.family in a.family]
    signer = Signer.from_env() if os.environ.get("RATCHET_SIGNING_KEY") else Signer.generate()
    agent_cfg = ModelConfig(base_url=a.base_url, model=a.model, api_key=key, temperature=a.temperature)
    improver_url = a.improver_base_url or a.base_url
    improver_key = os.environ.get("RATCHET_IMPROVER_API_KEY", key)
    improver_cfg = ModelConfig(
        base_url=improver_url, model=a.improver_model, api_key=improver_key, temperature=a.temperature
    )
    model_config = {
        "base_url": a.base_url,
        "model": a.model,
        "temperature": a.temperature,
        "improver_model": a.improver_model,
        "improver_base_url": improver_url,
    }
    config = {
        **model_config,
        "workers": a.workers,
        "max_clusters": a.max_clusters,
        "candidates": a.candidates,
        "max_evals": a.max_evals,
        "families": a.family,
        "cases": len(cases),
        "db": a.db,
    }
    started = time.perf_counter()
    with (
        Store(a.uri, a.db) as store,
        ModelClient(agent_cfg) as model,
        ModelClient(improver_cfg) as improver,
        GuardRuntime() as runtime,
        _mongo(a.uri) as client,
    ):
        store.setup(wait_search=False)
        lab = Lab(
            cases=cases,
            run_case=run_case,
            simulator=WorldSimulator(cases),
            runner=SandboxRunner(runtime),
            signer=signer,
            activator=MongoActivator(client, a.db),
            improver=improver,
            workers=a.workers,
            max_clusters=a.max_clusters,
            candidates_per_cluster=a.candidates,
            max_evals=a.max_evals,
            model_config=model_config,
            log=lambda msg: print(msg, flush=True),
        )
        result = evolve(store, model, a.generations, lab=lab, seed=seed_harness())
        report = summary(result, signer, time.perf_counter() - started, config)
        report["agent_model_cost"] = model.cost().__dict__
        report["improver_model_cost"] = improver.cost().__dict__
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
