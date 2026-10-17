"""The referee's policy methods: several candidates at once, ruling 2 on a deterministic endpoint, stochastic
v3, and the signed results file that carries every per-case held-out table the improver never receives."""

from __future__ import annotations

import base64
import dataclasses
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from ratchet.improve.activate import MemoryActivator
from ratchet.improve.evaluate import EvalReport, evaluate_many
from ratchet.improve.evaluator import evaluator_identity
from ratchet.improve.overfit import Literals
from ratchet.improve.receipt import Signer, verify
from ratchet.improve.referee import (
    ProcessReferee,
    RefereeCore,
    RefereeError,
    RefereeParts,
    Request,
    Verdict,
)
from ratchet.improve.results import RESULTS_TYPE
from ratchet.types import Harness, Procedure
from ratchet.verify import dsse

from .fakes import FakeRunCase, FakeSimulator, LocalRunner, make_trace
from .test_loop import CASES, HELDOUT_IDS, SEED, MemStore

TESTS = Path(__file__).resolve().parents[1]
TRAIN = [c for c in CASES if c.split == "train"]
HELD = [c for c in CASES if c.split == "heldout"]
PROC = Procedure(
    "p-refund", "finish the refund", "a refund is owed", ["payments_get_charge", "payments_refund"], []
)
FIX = SEED.child(procedures=[PROC], created_by="improver")
EDIT = SEED.child(system_prompt=f"{SEED.system_prompt}\n\nBe careful.", created_by="improver")
MODEL = {"endpoints": ["http://a/v1"], "model": "m", "temperature": 0.0, "seed": 7,
         "endpoint_config_digest": "e" * 64}  # fmt: skip


def _core(tmp_path: Path, signer: Signer, store: MemStore, **kw: Any) -> RefereeCore:
    args: dict[str, Any] = {
        "cases": HELD,
        "run_case": FakeRunCase(),
        "model": None,
        "simulator": FakeSimulator(),
        "runner": LocalRunner(),
        "signer": signer,
        "store": store,
        "activator": MemoryActivator(store.decision),
        "suite_digest": "suite",
        "workers": 4,
        "model_config": MODEL,
        "results_dir": tmp_path / "results",
    } | kw
    return RefereeCore(RefereeParts(**args))


def _train(h: Harness) -> EvalReport:
    return evaluate_many([h], TRAIN, FakeRunCase(), None, workers=2)[0]


def _request(h: Harness, decision_id: str | None = None) -> Request:
    return Request(h, _train(h), [], {}, Literals(set(), set(), set()), [c.id for c in TRAIN], {"arm": "a"},
                   decision_id)  # fmt: skip


def _started(core: RefereeCore) -> None:
    core.start(evaluator_identity(TRAIN, MODEL, "suite"))


def _open(path: Path, signer: Signer) -> dict[str, Any]:
    env = dsse.parse(path.read_bytes())
    assert env.payload_type == RESULTS_TYPE
    assert dsse.verify_signature(env, {signer.keyid: signer.public_key}).keyid == signer.keyid
    doc: dict[str, Any] = json.loads(env.payload)
    return doc


def _no_case_ids(value: Any) -> None:
    text = json.dumps(value, default=str)
    for case_id in HELDOUT_IDS:
        assert case_id not in text, case_id


def test_decide_many_signs_each_candidate_and_keeps_the_per_case_tables_for_the_results_file(
    tmp_path: Path,
) -> None:
    signer, store = Signer.generate(), MemStore()
    core = _core(tmp_path, signer, store)
    _started(core)
    core.parent_runs(1, SEED, _train(SEED))
    fix, edit = core.decide_many(1, [_request(FIX, "exp:fix"), _request(EDIT)])
    assert fix.decision_id == "exp:fix" and edit.decision_id == f"{SEED.version}->{EDIT.version}"
    # The procedure turns both omit cases from failing to passing in C1; the prompt edit changes nothing.
    assert (fix.stable_recoveries, fix.c1_fail_to_pass, fix.c1_pass_to_fail, fix.protected_broken) == (
        2,
        2,
        0,
        0,
    )
    assert (edit.c1_fail_to_pass, edit.c1_pass_to_fail) == (0, 0) and not edit.admitted
    _no_case_ids([asdict(fix), asdict(edit)])
    decisions = {d["candidate"]: d for d in store.decisions}
    for v in (fix, edit):
        env = base64.b64decode(decisions[v.candidate]["envelope"])
        assert verify(env, {signer.keyid: signer.public_key})["candidate"] == v.candidate
    published = core.results("run")
    assert published["decisions"] == 2 and published["admitted"] == int(fix.admitted)
    doc = _open(Path(published["path"]), signer)
    rows = {e["decision_id"]: e for e in doc["decisions"]}
    assert rows["exp:fix"]["heldout"]["c1_fail_to_pass"] == ["omit-10", "omit-9"]
    assert rows["exp:fix"]["heldout"]["outcomes"]["omit-9"][:3] == ["fail", "fail", "pass"]
    bundle = tmp_path / "results" / rows["exp:fix"]["bundle"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert set(manifest["reports"]) >= {"parent_run_1", "parent_run_2", "candidate_run"}
    assert (bundle / "receipt.dsse").read_bytes() == base64.b64decode(decisions[FIX.version]["envelope"])
    lines = (tmp_path / "results" / "entries.jsonl").read_text().splitlines()
    assert len(lines) == 2  # written as each decision was signed


def test_a_candidate_of_another_parent_and_an_unstarted_referee_are_refused(tmp_path: Path) -> None:
    core = _core(tmp_path, Signer.generate(), MemStore())
    with pytest.raises(RefereeError, match="no parent runs"):
        core.decide_many(1, [_request(FIX)])
    _started(core)
    core.parent_runs(1, SEED, _train(SEED))
    stranger = FIX.child(system_prompt="x", created_by="improver")
    with pytest.raises(RefereeError, match="is not a child"):
        core.decide_many(1, [_request(stranger)])
    with pytest.raises(RefereeError, match="not a plain file name"):
        core.results("../x")
    without = _core(tmp_path, Signer.generate(), MemStore(), results_dir=None)
    with pytest.raises(RefereeError, match="no results directory"):
        without.results("run")


def test_ruling_two_on_a_repeatable_endpoint_decides_and_names_the_endpoint_digest(tmp_path: Path) -> None:
    signer, store = Signer.generate(), MemStore()
    core = _core(tmp_path, signer, store)
    _started(core)
    out = core.deterministic(SEED, _request(FIX, "det:fix"), _train(SEED), repeat_heldout=3)
    assert out["repeatability"] == {"probed": 3, "repeatable": True, "differing": 0}
    assert out["parent_discordance"] == 0 and out["result"] == "decided"
    assert out["verdict"]["decision_id"] == "det:fix"
    assert {t["role"] for t in out["throughput"]} >= {"repeat_a", "repeat_b", "parent_run_1"}
    _no_case_ids(out)
    stored = store.decisions[-1]
    assert stored["policy"] == {
        "name": "ruling-2",
        "evaluator": "deterministic",
        "endpoint_config_digest": "e" * 64,
    }
    assert stored["context"]["endpoint_config_digest"] == "e" * 64 and stored["context"]["arm"] == "a"
    doc = _open(Path(core.results("det")["path"]), signer)
    assert set(doc["runs"]["repeatability"]["cases"]) == set(HELDOUT_IDS[:3])
    assert doc["runs"]["parent_discordance"] == [] and doc["decisions"][0]["policy"] == "ruling-2"


class Flaky(FakeRunCase):
    """An endpoint that answers the same request differently on alternate calls."""

    def __call__(self, harness: Any, case: Any, model: Any, store: Any, *, run_id: str) -> Any:
        self.calls += 1
        ch = {"charge_id": "ch_" + case.id.rsplit("-", 1)[-1]}
        script = [("payments_get_charge", ch)] * (1 + self.calls % 2)
        return make_trace(case.id, script, f"{run_id}-{case.id}", harness.version, harness.guards)


def test_ruling_two_stops_on_an_endpoint_that_does_not_repeat(tmp_path: Path) -> None:
    store = MemStore()
    core = _core(tmp_path, Signer.generate(), store, run_case=Flaky(), workers=1)
    _started(core)
    out = core.deterministic(SEED, _request(FIX), _train(SEED), repeat_heldout=0)
    assert out["result"].startswith("NOT REPEATABLE: two identical runs") and "verdict" not in out
    assert out["repeatability"]["probed"] == len(HELD) and not store.decisions
    assert store.evals[-1]["type"] == "endpoint_verdict"


class Reps:
    """Per-repetition clients for the fake world, which ignores the model."""

    seed_base = 1000

    def client_for(self, index: int, rep: int) -> tuple[str, Any]:
        return f"http://r{index % 2}/v1", None


def _original(h: Harness) -> dict[str, Any]:
    return {
        "parent": SEED.version,
        "candidate": h.version,
        "context": {"kind": "procedure", "change": asdict(PROC), "cluster_id": "c1", "generation": 1},
        "checks": [
            {"name": "replay", "passed": True, "evidence": {"rows": []}},
            {"name": "overfit_lint", "passed": True, "evidence": {"hits": []}},
            {"name": "heldout", "passed": False, "evidence": {"outcomes": {"omit-9": ["fail"]}}},
        ],
        "visible_case_ids": [c.id for c in TRAIN],
    }


def test_stochastic_v3_runs_every_arm_on_one_schedule_and_returns_counts_only(tmp_path: Path) -> None:
    signer, store = Signer.generate(), MemStore()
    originals = {("src", FIX.version): _original(FIX)}
    core = _core(tmp_path, signer, store, repetitions=Reps(), decisions=lambda db, v: originals.get((db, v)))
    out = core.stochastic(SEED, [f"src:{FIX.version}"], HELDOUT_IDS, 20261011, workers=8)
    assert out["executions"] == 4 * len(HELDOUT_IDS) * 40 and out["invalid"] == 0
    # 5 held-out cases is below the minimum cohort size, so the run is a path check and nothing is admitted
    assert out["cohort"] == "TOO_SMALL" and out["frozen"]["cases"] == len(HELDOUT_IDS)
    (dec,) = out["decisions"]
    assert (
        dec["candidate"] == FIX.version and dec["stable_recoveries"] == 2 and dec["stable_regressions"] == 0
    )
    _no_case_ids(out)
    stored = store.decisions[-1]
    assert stored["policy"]["version"] == "v3.1" and stored["context"]["policy"] == "stochastic-v3"
    assert stored["checks"][0]["evidence"]["from_decision"] == f"src:{SEED.version}->{FIX.version}"
    doc = _open(Path(core.results("v3")["path"]), signer)
    entry = doc["decisions"][0]
    assert sorted(entry["stable_recoveries"]) == ["omit-10", "omit-9"]
    assert {r["case_id"]: r["arm"] for r in entry["per_case"]}["omit-9"] == 40
    assert set(doc["runs"]["v3_identity_controls"]) == {"identity_1", "identity_2"}
    manifest = json.loads((tmp_path / "results" / entry["bundle"] / "manifest.json").read_text())
    arms = tmp_path / "results" / manifest["arms"]["parent"]
    assert len(json.loads(arms.read_text())) == len(HELDOUT_IDS) * 40
    traces = tmp_path / "results" / manifest["arm_traces"][f"candidate:{FIX.version}"]
    assert len(traces.read_text().splitlines()) == len(HELDOUT_IDS) * 40


def test_stochastic_v3_refuses_cases_it_does_not_hold_and_unknown_candidates(tmp_path: Path) -> None:
    core = _core(tmp_path, Signer.generate(), MemStore(), repetitions=Reps(), decisions=lambda db, v: None)
    with pytest.raises(RefereeError, match="held-out cases only"):
        core.stochastic(SEED, [], ["timeout-1"], 1)
    with pytest.raises(RefereeError, match="no decision for src:abc"):
        core.stochastic(SEED, ["src:abc"], HELDOUT_IDS, 1)
    with pytest.raises(RefereeError, match="db:version"):
        core.candidate("nocolon", SEED)
    bare = _core(tmp_path, Signer.generate(), MemStore())
    with pytest.raises(RefereeError, match="per-repetition"):
        bare.stochastic(SEED, [], HELDOUT_IDS, 1)
    with pytest.raises(RefereeError, match="no source of stored decisions"):
        bare.candidate("src:abc", SEED)


def test_the_candidate_lookup_returns_the_improvers_part_of_a_decision_and_never_its_checks(
    tmp_path: Path,
) -> None:
    originals = {("src", FIX.version): _original(FIX)}
    core = _core(tmp_path, Signer.generate(), MemStore(), decisions=lambda db, v: originals.get((db, v)))
    found = core.candidate(f"src:{FIX.version}", SEED)
    assert found["harness"]["version"] == FIX.version and found["context"]["cluster_id"] == "c1"
    assert "checks" not in json.dumps(found) and "omit-9" not in json.dumps(found)
    wrong = {("src", FIX.version): {**_original(FIX), "candidate": "0" * 12}}
    core = _core(tmp_path, Signer.generate(), MemStore(), decisions=lambda db, v: wrong.get((db, v)))
    with pytest.raises(ValueError, match="rebuilt"):
        core.candidate(f"src:{FIX.version}", SEED)


def test_the_verdict_carries_counts_and_names_never_case_ids() -> None:
    for f in dataclasses.fields(Verdict):
        assert f.type in {"str", "int", "bool", "tuple[str, ...]", "dict[str, Any]"}, f.name


def test_the_policy_methods_cross_the_process_boundary(tmp_path: Path) -> None:
    signer = Signer.generate()
    store_key = Signer.generate()
    originals = tmp_path / "originals.json"
    originals.write_text(json.dumps({f"src:{FIX.version}": _original(FIX)}))
    config = {
        "store_dir": str(tmp_path / "referee"),
        "store_key": store_key.private_key.private_bytes_raw().hex(),
        "decision_key": signer.private_key.private_bytes_raw().hex(),
        "results_dir": str(tmp_path / "results"),
        "decisions_file": str(originals),
    }
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")])}
    with ProcessReferee("improve.referee_factory:parts", config, env=env) as referee:
        found = referee.candidate(f"src:{FIX.version}", SEED)
        assert found["harness"]["version"] == FIX.version
        referee.start(evaluator_identity(TRAIN, {}, "suite"))
        referee.parent_runs(1, SEED, _train(SEED))
        (v,) = referee.decide_many(1, [_request(FIX, "p:fix")])
        assert isinstance(v, Verdict) and v.decision_id == "p:fix" and v.failed == tuple(v.failed)
        published = referee.results("proc")
    doc = _open(Path(published["path"]), signer)
    assert doc["decisions"][0]["decision_id"] == "p:fix"


def test_the_dispatcher_reads_the_policy_requests(tmp_path: Path) -> None:
    from ratchet.improve import wire
    from ratchet.improve.referee import _dispatch, request_doc

    originals = {("src", FIX.version): _original(FIX)}
    core = _core(
        tmp_path,
        Signer.generate(),
        MemStore(),
        repetitions=Reps(),
        decisions=lambda db, v: originals.get((db, v)),
    )
    _started(core)
    det = _dispatch(core, {"method": "deterministic", "params": {
        "parent": wire.harness_doc(SEED), "request": request_doc(_request(FIX, "d:fix")),
        "parent_train": wire.report_doc(_train(SEED)), "repeat_heldout": 2}})  # fmt: skip
    assert det["verdict"]["decision_id"] == "d:fix"
    v3 = _dispatch(core, {"method": "stochastic", "params": {
        "parent": wire.harness_doc(SEED), "candidates": [f"src:{FIX.version}"], "case_ids": HELDOUT_IDS[:2],
        "schedule_seed": 3, "deadline_s": 1e9, "scope_note": "n", "workers": 4}})  # fmt: skip
    assert v3["executions"] == 4 * 2 * 40
    with pytest.raises(RefereeError, match="unknown method"):
        _dispatch(core, {"method": "nope"})
