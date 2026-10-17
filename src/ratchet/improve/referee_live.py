"""The referee factories for the case suite.

They run in the referee's own process (`ratchet.improve.referee.ProcessReferee`). The decision key comes from
RATCHET_SIGNING_KEY or the keyring there, and the records go to the referee's own store: a signed file store
at the configured directory, signed with the decision key, or an Atlas held-out database, which the
improver's database role cannot read.

`case_suite_parts` (`ratchet evolve`, `ratchet.improve.run`) takes the agent model from the RATCHET_* model
variables:

    config = {"cases": [held-out case ids], "store": {"kind": "file", "dir": ...}
                                                    | {"kind": "atlas", "db": "ratchet_heldout"},
              "workers": 8, "repeats": 1, "suite_digest": "...", "model_config": {...}}

`policy_parts` (`run_det`, `run_v3`, `instruction_search`) builds the worker from an endpoint configuration
the driver names by file and digest (`ratchet.improve.endpoints`), so the model configuration lives here and
the evaluator names its digest:

    config = {"cases": "heldout" | [held-out case ids], "store": {...}, "results_dir": ...,
              "endpoints": {"file": ..., "digest": ...},
              "worker": "round_robin" (seeded) | "spread" (unseeded) | "per_repetition",
              "replicas_file": ... (per_repetition only), "workers": ..., "suite_digest": ...}

With an Atlas store it also reads the stored decisions a policy re-judges, by `db:version`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ratchet.improve.referee import RefereeError, RefereeParts

REFEREE_URI_ENV = "RATCHET_REFEREE_MONGODB_URI"


def case_suite_parts(config: Mapping[str, Any]) -> RefereeParts:
    """The factory `ProcessReferee` names: the agent model from the RATCHET_* model variables."""
    from ratchet.agent.model import ModelClient

    stack = ExitStack()
    try:
        return case_suite_parts_with(config, stack.enter_context(ModelClient.from_env()), stack)
    except BaseException:
        stack.close()
        raise


def _held_out(selection: Any) -> list[Any]:
    from ratchet_cases.cases import all_cases

    every = {c.id: c for c in all_cases()}
    if selection == "heldout":
        return [c for c in every.values() if c.split == "heldout"]
    missing = [i for i in selection if i not in every]
    if missing:
        raise RefereeError(f"unknown held-out case ids {missing}")
    return [every[i] for i in selection]


def case_suite_parts_with(
    config: Mapping[str, Any],
    model: Any,
    stack: ExitStack | None = None,
    **overrides: Any,
) -> RefereeParts:
    """The referee's parts around a given agent model; `stack` (closed with the parts) owns what it opened.
    `overrides` replace parts (the worker, the model configuration, the results directory)."""
    from ratchet_cases.cases import all_cases

    from ratchet.agent.guards import GuardRuntime
    from ratchet.agent.worker import run_case
    from ratchet.improve.receipt import Signer
    from ratchet.improve.sandbox import SandboxRunner
    from ratchet.improve.worldsim import WorldSimulator

    cases = _held_out(config["cases"])
    stack = ExitStack() if stack is None else stack
    signer = Signer.for_decisions()
    store, activator = _store(config["store"], signer, stack)
    runtime = stack.enter_context(GuardRuntime())
    fields: dict[str, Any] = dict(
        cases=cases,
        run_case=run_case,
        model=model,
        simulator=WorldSimulator(all_cases()),
        runner=SandboxRunner(runtime),
        signer=signer,
        store=store,
        activator=activator,
        model_config=dict(config.get("model_config", {})),
        suite_digest=config.get("suite_digest"),
        workers=int(config.get("workers", 8)),
        repeats=int(config.get("repeats", 1)),
        session_bundles={str(k): Path(v) for k, v in config.get("session_bundles", {}).items()},
        results_dir=Path(config["results_dir"]) if config.get("results_dir") else None,
        close=stack.close,
    )
    return RefereeParts(**(fields | overrides))


def policy_parts(config: Mapping[str, Any]) -> RefereeParts:
    """The factory the policy drivers name: the worker from the endpoint configuration the driver passed by
    reference, refused when the file is not the one its digest names."""
    from ratchet.agent.worker import run_case
    from ratchet.improve import endpoints as ep

    stack = ExitStack()
    try:
        endpoints = ep.resolve(config["endpoints"])
        overrides: dict[str, Any] = {"model_config": endpoints.model_config()}
        if config.get("worker") == "per_repetition":
            replicas = Path(config["replicas_file"]) if config.get("replicas_file") else None
            reps = ep.PerRepetition(endpoints, replicas)
            stack.callback(reps.close)
            overrides["repetitions"] = reps
        else:
            spread = ep.RoundRobin(endpoints, run_case, seeded=config.get("worker") == "round_robin")
            stack.callback(spread.close)
            overrides |= {"run_case": spread, "workers": int(config.get("workers") or len(endpoints.urls))}
        if config["store"]["kind"] == "atlas":
            overrides["decisions"] = _stored_decision
        return case_suite_parts_with(config, None, stack, **overrides)
    except BaseException:
        stack.close()
        raise


def _stored_decision(db: str, version: str) -> Any:
    """A stored decision, without its envelope, from the named database of the referee's Atlas cluster."""
    from ratchet.store.atlas import Store

    with Store(_uri(), db) as s:
        return s.db.decisions.find_one({"candidate": version}, {"envelope": 0, "_id": 0})


def _uri() -> str:
    uri = os.environ.get(REFEREE_URI_ENV) or os.environ.get("RATCHET_MONGODB_URI", "")
    if not uri:
        raise RefereeError(f"the referee needs {REFEREE_URI_ENV} (or RATCHET_MONGODB_URI)")
    return uri


def _store(spec: Mapping[str, Any], signer: Any, stack: ExitStack) -> tuple[Any, Any]:
    if spec["kind"] == "file":
        from ratchet.store.file_store import FileActivator, FileStore

        store = FileStore(Path(spec["dir"]), key=signer.private_key)
        return store, FileActivator(store)
    if spec["kind"] == "atlas":
        from ratchet.store.activation import MongoActivator
        from ratchet.store.atlas import Store

        atlas = stack.enter_context(Store(_uri(), str(spec["db"])))
        return atlas, MongoActivator(atlas.db.client, atlas.db.name)
    raise RefereeError(f"unknown referee store kind {spec['kind']!r}")
