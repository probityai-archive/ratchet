"""The referee as its own process: the loop runs against it, holds no held-out data, and gets totals only."""

from __future__ import annotations

import base64
import dataclasses
import os
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ratchet.improve.loop import Lab, evolve
from ratchet.improve.receipt import Signer, verify
from ratchet.improve.referee import ProcessReferee, RefereeError, Totals, Verdict, combine_identity
from ratchet.store.file_store import FileActivator, FileStore

from .fakes import FakeRunCase, FakeSimulator, LocalRunner
from .test_loop import CASES, HELDOUT_IDS, SEED, MemStore, RoutingChat

TESTS = Path(__file__).resolve().parents[1]


def _referee(tmp_path: Path, decision_key: Signer) -> tuple[ProcessReferee, Ed25519PrivateKey]:
    store_key = Ed25519PrivateKey.generate()
    config = {
        "store_dir": str(tmp_path / "referee"),
        "store_key": store_key.private_bytes_raw().hex(),
        "decision_key": decision_key.private_key.private_bytes_raw().hex(),
    }
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")])}
    return ProcessReferee("improve.referee_factory:parts", config, env=env), store_key


def test_the_loop_runs_against_a_referee_process_and_never_holds_held_out_data(tmp_path: Path) -> None:
    signer, chat, store = Signer.generate(), RoutingChat(), MemStore()
    referee, store_key = _referee(tmp_path, signer)
    with referee:
        lab = Lab(
            cases=[c for c in CASES if c.split != "heldout"],
            run_case=FakeRunCase(),
            simulator=FakeSimulator(),
            runner=LocalRunner(),
            referee=referee,
            improver=chat,
            workers=4,
            max_evals=4,
            suite_digest="suite",
            max_clusters=1,
        )
        result = evolve(store, None, 2, lab=lab, seed=SEED)
    g1 = result.generations[0]
    assert g1.promoted is not None and g1.level1["child_score"]["heldout_passed"] >= 2
    # The improver's own records: training runs only, no decision, no held-out case anywhere.
    assert not store.decisions
    assert all(t.case_id not in HELDOUT_IDS for t in store.traces.values())
    assert all(r["heldout"]["total"] == 0 for r in store.evals if r.get("type") == "eval")
    for case_id in HELDOUT_IDS:
        assert case_id not in chat.prompts()
    # The referee's records: every decision, signed by the key the improver never held, and verifiable.
    held = FileStore(tmp_path / "referee", trusted=store_key.public_key())
    decisions = held.decisions()
    assert decisions and any(d["admitted"] for d in decisions)
    for d in decisions:
        assert (
            verify(base64.b64decode(d["envelope"]), {signer.keyid: signer.public_key})["candidate"]
            == d["candidate"]
        )
    assert {e["role"] for e in held.evals({"type": "eval"})} >= {"parent_run_1", "parent_run_2", "candidate"}
    assert FileActivator(held).active()[0] == result.harness.version


def test_what_the_improver_receives_is_counts_and_names_never_case_ids() -> None:
    for kind in (Totals, Verdict):
        for f in dataclasses.fields(kind):
            assert f.type in {"str", "int", "bool", "tuple[str, ...]", "dict[str, Any]"}, (kind, f.name)
    assert {f.name for f in dataclasses.fields(Verdict)} == {
        "decision_id",
        "receipt_digest",
        "candidate",
        "admitted",
        "failed",
        "heldout_passed",
        "heldout_total",
        "stable_recoveries",
        "protected_broken",
        "c1_fail_to_pass",
        "c1_pass_to_fail",
        "rates",
    }


def test_the_referee_refuses_held_out_results_from_the_improver(tmp_path: Path) -> None:
    from ratchet.improve.evaluate import evaluate_many
    from ratchet.improve.evaluator import evaluator_identity

    referee, _ = _referee(tmp_path, Signer.generate())
    with referee:
        train = [c for c in CASES if c.split != "heldout"]
        referee.start(evaluator_identity(train, {}, "suite"))
        smuggled = evaluate_many([SEED], CASES, FakeRunCase(), None, workers=2)[0]
        with pytest.raises(RefereeError, match="held-out results"):
            referee.parent_runs(1, SEED, smuggled)


def test_a_referee_that_cannot_start_says_so(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(TESTS)}
    with pytest.raises(RefereeError, match=r"could not start: AttributeError: .*missing"):
        ProcessReferee("improve.referee_factory:missing", {}, env=env)
    with pytest.raises(RefereeError, match="could not start: KeyError"):
        ProcessReferee("improve.referee_factory:parts", {}, env=env)  # a config the factory cannot read


def test_the_two_halves_of_the_evaluator_must_name_one_suite_and_one_model() -> None:
    def ident(suite: str, model: dict[str, Any]) -> dict[str, Any]:
        return {"components": {"cases": "a" * 64, "suite_file": suite}, "model_config": model, "cases": 1}

    both = combine_identity(ident("s", {"m": 1}), ident("s", {"m": 1}))
    assert set(both["components"]) == {"train_cases", "heldout_cases", "suite_file"} and both["cases"] == 2
    with pytest.raises(RefereeError, match="case suites"):
        combine_identity(ident("s", {}), ident("t", {}))
    with pytest.raises(RefereeError, match="model configurations"):
        combine_identity(ident("s", {"m": 1}), ident("s", {"m": 2}))
