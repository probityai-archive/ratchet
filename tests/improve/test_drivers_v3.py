"""The ruling-2 and v3 drivers: each runs its held-out side through the referee factory it names, holds no
held-out case or decision key, and records only counts, signed verdicts and the results file's digest."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from ratchet.improve import run_det, run_v3
from ratchet.improve.activate import MemoryActivator
from ratchet.improve.policies import RebuildError, normalized, rebuild
from ratchet.types import Guard, Harness, Procedure
from ratchet.verify.results import verify_results

from .fakes import NO_DUP_GUARD, FakeRunCase, FakeSimulator, LocalRunner, bad_timeout
from .test_loop import CASES, HELDOUT_IDS
from .test_loop import SEED as LOOP_SEED
from .test_referee_policies import FIX, _original

SEED = Harness.build(parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=["t"],
                     created_by="seed")  # fmt: skip


def test_rebuild_reproduces_the_judged_candidate_by_digest() -> None:
    g = Guard("g1", "d", NO_DUP_GUARD, ["r@1"])
    cand = SEED.child(guards=[g], created_by="improver")
    assert (
        rebuild(SEED, {"candidate": cand.version, "context": {"kind": "guard", "change": asdict(g)}}) == cand
    )
    proc = Procedure("p1", "n", "w", ["s"], [])
    pc = SEED.child(procedures=[proc], created_by="improver")
    assert (
        rebuild(SEED, {"candidate": pc.version, "context": {"kind": "procedure", "change": asdict(proc)}})
        == pc
    )
    edit = SEED.child(system_prompt="p\n\nMore.", created_by="improver")
    ctx = {"kind": "prompt_edit", "change": {"append": "More."}}
    assert rebuild(SEED, {"candidate": edit.version, "context": ctx}) == edit
    with pytest.raises(RebuildError, match="rebuilt"):
        rebuild(SEED, {"candidate": "0" * 12, "context": ctx})


def test_normalized_traces_ignore_times_and_ids_but_not_behaviour() -> None:
    a = bad_timeout(1, run="r1")
    b = bad_timeout(1, run="r2")  # different run and call ids
    later = replace(b, events=[replace(e, ts=e.ts + 99) for e in b.events])
    assert normalized(a) == normalized(later)
    other = bad_timeout(2, run="r1")
    assert normalized(a) != normalized(other)


class Coll:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def find_one(self, query: dict[str, Any], projection: Any = None) -> dict[str, Any] | None:
        for r in self.rows:
            if all(r.get(k) == v for k, v in query.items()):
                return {k: v for k, v in r.items() if k != "envelope"}
        return None


class FakeAtlas:
    """Atlas databases in memory, shared by every client of one test."""

    dbs: ClassVar[dict[str, dict[str, list[Any]]]] = {}

    def __init__(self, uri: str, db: str) -> None:
        self.mem = FakeAtlas.dbs.setdefault(db, {"evals": [], "decisions": [], "traces": []})
        self.db = SimpleNamespace(
            evals=Coll(self.mem["evals"]), decisions=Coll(self.mem["decisions"]), name=db, client=None
        )

    def __enter__(self) -> FakeAtlas:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def put_trace(self, trace: Any) -> bool:
        self.mem["traces"].append(trace)
        return True

    def get_trace(self, run_id: str, case_id: str) -> Any:
        return None

    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str:
        self.mem["evals"].append(dict(report))
        return eval_id or "e"

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        self.mem["decisions"].append({**dict(decision), "_decision_id": decision_id})
        return decision_id or "d"


class _Ctx:
    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, inprocess_referee: Callable[..., Any]) -> dict[str, Any]:
    """The fake billing world behind the real referee factory: suite, worker, store, sandbox."""
    import ratchet_cases.cases as cases_mod

    import ratchet.agent.guards as guards_mod
    import ratchet.agent.seed as seed_mod
    import ratchet.agent.worker as worker_mod
    import ratchet.improve.sandbox as sandbox_mod
    import ratchet.improve.worldsim as worldsim_mod
    import ratchet.store.activation as activation_mod
    import ratchet.store.atlas as atlas_mod

    FakeAtlas.dbs = {}
    for mod in (run_det, run_v3):
        monkeypatch.setattr(mod, "ProcessReferee", inprocess_referee)
    monkeypatch.setattr(cases_mod, "all_cases", lambda: CASES)
    monkeypatch.setattr(worker_mod, "run_case", FakeRunCase())
    monkeypatch.setattr(worldsim_mod, "WorldSimulator", lambda cases: FakeSimulator())
    monkeypatch.setattr(sandbox_mod, "SandboxRunner", lambda rt: LocalRunner())
    monkeypatch.setattr(guards_mod, "GuardRuntime", _Ctx)
    monkeypatch.setattr(seed_mod, "seed_harness", lambda: LOOP_SEED)
    monkeypatch.setattr(atlas_mod, "Store", FakeAtlas)
    monkeypatch.setattr(
        activation_mod, "MongoActivator", lambda client, name: MemoryActivator(lambda _i: None)
    )
    monkeypatch.setenv("RATCHET_MONGODB_URI", "mongodb://fake")
    src = FakeAtlas("u", "src")
    src.mem["decisions"].append(_original(FIX))
    src.mem["evals"].append({"type": "generation", "generation": 1, "clusters": [{"id": "c1", "members": [
        {"run_id": "r-omit-7", "case_id": "omit-7", "seq": 1, "reason": "x", "mode": "omission",
         "tool": "payments_refund"}]}]})  # fmt: skip
    return FakeAtlas.dbs


def _endpoints(tmp_path: Path, **extra: Any) -> Path:
    path = tmp_path / "endpoints.json"
    path.write_text(
        json.dumps({"endpoints": ["http://127.0.0.1:9/v1"], "model": "m", "temperature": 0.0, **extra})
    )
    return path


def _no_case_ids(doc: Any) -> None:
    text = json.dumps(doc, default=str)
    for case_id in HELDOUT_IDS:
        assert case_id not in text, case_id


def test_run_det_decides_through_the_referee_and_writes_counts_and_the_results_digest(
    world: dict[str, Any], tmp_path: Path, decision_key: Any
) -> None:
    out = tmp_path / "det.json"
    argv = ["--endpoints-file", str(_endpoints(tmp_path, seed=7)), "--candidate", f"src:{FIX.version}",
            "--out", str(out), "--repeat-heldout", "2"]  # fmt: skip
    assert run_det.main(argv) == 0
    doc = json.loads(out.read_text())
    _no_case_ids(doc)
    assert doc["result"] == "decided" and doc["repeatability"]["repeatable"] is True
    assert doc["verdict"]["decision_id"].startswith(f"det:{LOOP_SEED.version}->{FIX.version}:")
    assert doc["verdict"]["stable_recoveries"] == 2
    roles = [t["role"] for t in doc["throughput"]]
    assert roles[:2] == ["parent_train", "candidate_train"] and "repeat_a" in roles
    # The driver's own database holds training runs only; the decision lives in the referee's.
    assert all(t.case_id not in HELDOUT_IDS for t in world["ratchet_det"]["traces"])
    held = world["ratchet_det_heldout"]["decisions"]
    assert held and held[-1]["policy"]["endpoint_config_digest"] == doc["endpoint_config_digest"]
    results = Path(doc["results"]["path"])
    assert results.parent == tmp_path / "det-results"
    assert verify_results(results.read_bytes(), decision_key.public_key, results.parent).ok


def test_run_det_refuses_without_a_store_or_a_seed_and_a_cluster_outside_training(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "x.json"
    argv = ["--candidate", f"src:{FIX.version}", "--out", str(out)]
    assert run_det.main(["--endpoints-file", str(_endpoints(tmp_path)), *argv]) == 2
    world["src"]["evals"][0]["clusters"][0]["members"][0]["case_id"] = "omit-9"
    with pytest.raises(SystemExit, match="non-training cases"):
        run_det.main(["--endpoints-file", str(_endpoints(tmp_path, seed=7)), *argv])
    monkeypatch.delenv("RATCHET_MONGODB_URI")
    assert run_det.main(["--endpoints-file", str(_endpoints(tmp_path, seed=7)), *argv]) == 2


def test_run_v3_runs_the_schedule_in_the_referee_and_writes_counts_only(
    world: dict[str, Any], tmp_path: Path, decision_key: Any
) -> None:
    out = tmp_path / "v3.json"
    replicas = tmp_path / "replicas"
    replicas.write_text("http://127.0.0.1:9/v1\nhttp://127.0.0.1:10/v1\n")
    argv = ["--endpoints-file", str(_endpoints(tmp_path, seed_base=1000)), "--replicas-file", str(replicas),
            "--cases", ",".join(HELDOUT_IDS[:2]), "--candidate", f"src:{FIX.version}", "--out", str(out),
            "--workers", "8"]  # fmt: skip
    assert run_v3.main(argv) == 0
    doc = json.loads(out.read_text())
    _no_case_ids({k: v for k, v in doc.items()})
    assert doc["cohort"] == "TOO_SMALL" and doc["executions"] == 4 * 2 * 40  # 2 cases: a path check
    (dec,) = doc["decisions"]
    assert dec["candidate"] == FIX.version and isinstance(dec["stable_recoveries"], int)
    executions = [e for e in world["ratchet_v3"]["evals"] if e.get("type") == "v3_execution"]
    assert len(executions) == 4 * 2 * 40 and {e["endpoint"] for e in executions} == set(
        replicas.read_text().split()
    )
    results = Path(doc["results"]["path"])
    assert verify_results(results.read_bytes(), decision_key.public_key, results.parent).ok


def test_run_v3_refuses_without_a_store_or_a_seed_schedule(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = ["--cases", HELDOUT_IDS[0], "--out", str(tmp_path / "x.json")]
    assert run_v3.main(["--endpoints-file", str(_endpoints(tmp_path, seed=7)), *argv]) == 2
    monkeypatch.delenv("RATCHET_MONGODB_URI")
    assert run_v3.main(["--endpoints-file", str(_endpoints(tmp_path, seed_base=1)), *argv]) == 2
