"""The referee factory for the case suite, which `ratchet evolve` and `ratchet.improve.run` start the referee
with.

It runs in the referee's own process (`ratchet.improve.referee.ProcessReferee`). The decision key comes from
RATCHET_SIGNING_KEY or the keyring there, the agent model from the RATCHET_* model variables, and the records
go to the referee's own store: a signed file store at the configured directory, signed with the decision
key, or the Atlas held-out database `<db>_heldout`, which the improver's database role cannot read.

    config = {"cases": [held-out case ids], "store": {"kind": "file", "dir": ...}
                                                    | {"kind": "atlas", "db": "ratchet_heldout"},
              "workers": 8, "repeats": 1, "suite_digest": "...", "model_config": {...}}
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


def case_suite_parts_with(
    config: Mapping[str, Any], model: Any, stack: ExitStack | None = None
) -> RefereeParts:
    """The referee's parts around a given agent model; `stack` (closed with the parts) owns what it opened."""
    from ratchet_cases.cases import all_cases

    from ratchet.agent.guards import GuardRuntime
    from ratchet.agent.worker import run_case
    from ratchet.improve.receipt import Signer
    from ratchet.improve.sandbox import SandboxRunner
    from ratchet.improve.worldsim import WorldSimulator

    every = {c.id: c for c in all_cases()}
    missing = [i for i in config["cases"] if i not in every]
    if missing:
        raise RefereeError(f"unknown held-out case ids {missing}")
    stack = ExitStack() if stack is None else stack
    signer = Signer.for_decisions()
    store, activator = _store(config["store"], signer, stack)
    runtime = stack.enter_context(GuardRuntime())
    return RefereeParts(
        cases=[every[i] for i in config["cases"]],
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
        close=stack.close,
    )


def _store(spec: Mapping[str, Any], signer: Any, stack: ExitStack) -> tuple[Any, Any]:
    if spec["kind"] == "file":
        from ratchet.store.file_store import FileActivator, FileStore

        store = FileStore(Path(spec["dir"]), key=signer.private_key)
        return store, FileActivator(store)
    if spec["kind"] == "atlas":
        from ratchet.store.activation import MongoActivator
        from ratchet.store.atlas import Store

        uri = os.environ.get(REFEREE_URI_ENV) or os.environ.get("RATCHET_MONGODB_URI", "")
        if not uri:
            raise RefereeError(f"the referee needs {REFEREE_URI_ENV} (or RATCHET_MONGODB_URI)")
        atlas = stack.enter_context(Store(uri, str(spec["db"])))
        return atlas, MongoActivator(atlas.db.client, atlas.db.name)
    raise RefereeError(f"unknown referee store kind {spec['kind']!r}")
