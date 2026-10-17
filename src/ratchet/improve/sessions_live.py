"""Run the k-session protocol (`ratchet.improve.sessions`) against live OpenAI-compatible endpoints.

One host, or the servers of one Slurm allocation: the endpoints are read from a file before every session
(a job that starts its own servers writes their URLs there), and `--session-start-cmd` runs before each
session so that a deployment can be restarted fresh. Each execution runs the case once with the slot's own
seed on the slot's own endpoint; a lost execution is reported as lost, and the protocol, not this module,
reruns it once.

    python -m ratchet.improve.sessions_live --label my-run --cases a,b,c \\
        --harness parent=parent.json --harness candidate=candidate.json \\
        --endpoints-file endpoints.txt --model gpt-oss-120b --workers 32 --out bundle/
    python -m ratchet.improve.sessions verify bundle/

Harness files are the JSON documents `ratchet show` and the exporters write; each is checked against its
own digest before anything runs. The identity arms run the parent's bytes under their own seed streams.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ratchet_cases.cases import all_cases

from ratchet.agent.model import ModelClient, ModelConfig, SeededTransport
from ratchet.agent.worker import run_case
from ratchet.export.claude_code import harness_from_json
from ratchet.improve.evaluate import RunCase, is_infra_failure, model_calls
from ratchet.improve.sessions import ARMS, Executor, Outcome, Protocol, Slot, run_protocol
from ratchet.types import Harness


def seeded_client(endpoint: str, seed: int, model: str, temperature: float = 0.0) -> ModelClient:
    """A chat client for one endpoint that sends `seed` with every completion request."""
    config = ModelConfig(base_url=endpoint, model=model, api_key="unused", temperature=temperature)
    return ModelClient(config, transport=SeededTransport(seed))


ClientFor = Callable[[str, int], ModelClient]


def live_executor(
    harnesses: Mapping[str, Harness],
    cases: Mapping[str, Any],
    client_for: ClientFor,
    *,
    runner: RunCase | None = None,
) -> Executor:
    """An executor that runs each slot once on its endpoint with its seed. Lost: an exception, an endpoint
    failure inside the run, or any cached model call (only fresh executions count)."""
    run = runner or run_case

    def execute(slot: Slot) -> Outcome:
        harness = harnesses[slot.arm]
        client = client_for(slot.endpoint, int(slot.seed))
        run_id = f"run-{harness.version}-{uuid.uuid4().hex[:12]}"
        try:
            trace = run(harness, cases[slot.case_id], client, None, run_id=run_id)
        except Exception as exc:
            return Outcome(None, run_id, f"{type(exc).__name__}: {exc}"[:300])
        finally:
            client.close()
        if is_infra_failure(trace):
            return Outcome(None, run_id, "endpoint failed during the run")
        if model_calls(trace)[1]:
            return Outcome(None, run_id, "a cached model call: not a fresh execution")
        return Outcome(bool(trace.outcome.passed), run_id)

    return execute


def load_harness(path: Path) -> Harness:
    harness = harness_from_json(json.loads(path.read_text(encoding="utf-8")))
    if not harness.verify_digest():
        raise SystemExit(f"{path}: harness does not match its own digest")
    return harness


def _endpoints(args: argparse.Namespace) -> list[str]:
    if args.endpoints_file is not None:
        listed = [x.strip() for x in args.endpoints_file.read_text(encoding="utf-8").split() if x.strip()]
    else:
        listed = [x.strip() for x in args.endpoints.split(",") if x.strip()]
    if not listed:
        raise SystemExit("no endpoints: give --endpoints or a nonempty --endpoints-file")
    return listed


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m ratchet.improve.sessions_live")
    p.add_argument("--label", required=True, help="names the run; every seed and schedule derives from it")
    p.add_argument("--cases", required=True, help="comma-separated case ids from the installed case suite")
    p.add_argument("--harness", action="append", default=[], metavar="ARM=FILE",
                   help="parent=FILE and candidate=FILE; the identity arms run the parent")  # fmt: skip
    p.add_argument("-k", type=int, default=8)
    p.add_argument("-n", type=int, default=5)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--endpoints", default="")
    src.add_argument("--endpoints-file", type=Path, default=None, help="re-read before every session")
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--session-start-cmd", default="", help="run (and waited for) before every session")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = _parse(argv)
    files = dict(h.split("=", 1) for h in a.harness)
    if set(files) != {"parent", "candidate"}:
        print("error: give exactly --harness parent=FILE and --harness candidate=FILE", file=sys.stderr)
        return 2
    parent, candidate = load_harness(Path(files["parent"])), load_harness(Path(files["candidate"]))
    harnesses = {"parent": parent, "identity_1": parent, "identity_2": parent, "candidate": candidate}
    every = {c.id: c for c in all_cases()}
    case_ids = [c.strip() for c in a.cases.split(",") if c.strip()]
    unknown = [c for c in case_ids if c not in every]
    if unknown:
        print(f"error: unknown case ids {unknown}", file=sys.stderr)
        return 2
    protocol = Protocol.for_k(a.label, case_ids, {arm: harnesses[arm].version for arm in ARMS}, k=a.k, n=a.n)

    def start(session: int) -> None:
        if a.session_start_cmd:
            subprocess.run([*shlex.split(a.session_start_cmd), str(session)], check=True)

    verdict = run_protocol(
        protocol,
        live_executor(harnesses, every, lambda url, seed: seeded_client(url, seed, a.model)),
        a.out,
        endpoints_for=lambda _s: _endpoints(a),
        workers=a.workers,
        on_session_start=start,
    )
    print(json.dumps({"bundle": str(a.out), "verdict": verdict["verdict"], "reasons": verdict["reasons"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
