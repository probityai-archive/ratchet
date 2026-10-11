"""The demo's id derivation reads the store and the result files, and fails loudly on a missing value."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ratchet.types import Event
from ratchet.view.admission import fresh_view
from ratchet.view.data import resolve_candidates
from ratchet.view.matrix import pick_rows

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


derive = load("ratchet_demo_derive_ids", ROOT / "docs" / "demo" / "derive_ids.py")
fakes = load("ratchet_view_fakes_for_demo", ROOT / "tests" / "view" / "fakes.py")


class CheckpointCursor:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def sort(self, *_a: Any) -> CheckpointCursor:
        return self

    def limit(self, _n: int) -> list[dict[str, Any]]:
        return self.docs[:1]


class CheckpointCollection:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def find(self, flt: dict[str, Any]) -> CheckpointCursor:
        want = flt["state.status"]
        return CheckpointCursor([d for d in self.docs if d["state"]["status"] == want])


class Db:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def __getitem__(self, name: str) -> CheckpointCollection:
        assert name == "checkpoints"
        return CheckpointCollection(self.docs)


class WithCheckpoints:
    """The fake store with a `checkpoints` collection the fake does not have."""

    def __init__(self, inner: Any, checkpoints: list[dict[str, Any]]) -> None:
        self.inner = inner
        self.db = Db(checkpoints)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def store_with(checkpoints: list[dict[str, Any]]) -> Any:
    store, _seed, _good = fakes.scenario()
    return WithCheckpoints(store, checkpoints)


def derive_store(store: Any, verifies: Any = lambda _id: None) -> Any:
    out = derive.Derived()
    derive.derive_from_store(
        store,
        out,
        case_info=fakes.info,
        verifies=verifies,
        resolve=resolve_candidates,
        pick_rows=pick_rows,
        fresh_view=fresh_view,
    )
    return out


def test_every_store_value_comes_from_the_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(derive, "BAD_TRACE_FAMILY", "refund")
    store = store_with([{"_id": "run-9", "state": {"status": "running"}}])
    admitted = next(d for d in store.inner.decision_docs if d["admitted"])
    out = derive_store(store)
    assert out.missing.get("ADMITTED_DECISION") is None
    got = {k: v for k, (v, _src) in out.values.items()}
    assert got["ADMITTED_DECISION"] == admitted["_id"]
    assert got["CLEARED_CAND"] == admitted["candidate"]
    assert got["REJECTED_CAND"] == "rej00000001"  # the guard that blocked recorded successes
    assert got["N_DECISIONS"] == str(len(store.inner.decision_docs))
    assert got["N_REJECTED"] == str(sum(1 for d in store.inner.decision_docs if not d["admitted"]))
    assert got["STABLE_RECOVERIES"] == "2"
    assert got["PROTECTED_N"] == "2"
    assert got["HELDOUT_N"] == "5"
    assert got["REPLAY_TRACES"] == str(len({t.case_id for t in store.inner.trace_list}))
    assert got["RESUME_CMD"] == "ratchet resume run-9"
    assert all(src for _v, src in out.values.values())


def test_missing_values_are_named_not_guessed() -> None:
    out = derive_store(store_with([]))
    assert "no checkpoint is still `running`" in out.missing["RESUME_RUN"]
    assert "AUTHORITY_DECISION" in out.missing  # the scenario has no authority rejection
    assert "BAD_TRACE" in out.missing  # no refund_lost_reply trace in the fakes
    assert "RESUME_CMD" not in out.values


def test_admitted_needs_a_fresh_gate_a_receipt_and_a_verifying_signature() -> None:
    good = fakes.decision_doc("p", "c1", True, checks=[{"name": "fresh", "passed": True, "evidence": {}}])
    no_fresh = fakes.decision_doc("p", "c2", True)
    no_receipt = fakes.decision_doc(
        "p", "c3", True, checks=[{"name": "fresh", "passed": True, "evidence": {}}], envelope=None
    )
    bad_sig = fakes.decision_doc("p", "c4", True, checks=[{"name": "fresh", "passed": True, "evidence": {}}])
    bad_sig["decided_at"] = "2026-10-11T11:00:00+00:00"
    docs = [good, no_fresh, no_receipt, bad_sig]
    picked, _why = derive.pick_admitted(docs, lambda i: "INVALID" if i.endswith("c4") else None)
    assert picked is good
    none, why = derive.pick_admitted([no_fresh, no_receipt], lambda _i: None)
    assert none is None
    assert "no `fresh` gate" in why
    assert "no signed receipt" in why
    assert derive.pick_admitted([], lambda _i: None)[1] == "no admitted decision is stored yet"


def test_pair_needs_the_same_parent() -> None:
    store, _seed, _good = fakes.scenario()
    docs = derive.decision_docs(store)
    pair = derive.pick_pair(store, docs, resolve_candidates)
    assert pair is not None
    assert pair[0]["parent"] == pair[1]["parent"]
    other = [{**d, "parent": "elsewhere"} if not d["admitted"] else d for d in docs]
    assert derive.pick_pair(store, other, resolve_candidates) is None


def test_timeout_then_retry_and_the_bad_trace_pick() -> None:
    def ev(seq: int, kind: str, payload: dict[str, Any]) -> Event:
        return Event(seq, kind, payload, float(seq))  # type: ignore[arg-type]

    retry = [
        ev(1, "tool_call", {"call": {"name": "refund", "args": {}, "call_id": "a"}}),
        ev(2, "tool_result", {"result": {"call_id": "a", "ok": False, "error": "timeout"}}),
        ev(3, "tool_call", {"call": {"name": "refund", "args": {}, "call_id": "b"}}),
    ]
    other_tool = [*retry[:2], ev(3, "tool_call", {"call": {"name": "mail", "args": {}, "call_id": "b"}})]
    no_timeout = [
        ev(1, "tool_call", {"call": {"name": "refund", "args": {}, "call_id": "a"}}),
        ev(2, "tool_result", {"result": {"call_id": "a", "ok": True, "data": {}}}),
        ev(3, "tool_call", {"call": {"name": "refund", "args": {}, "call_id": "b"}}),
    ]
    assert derive.timeout_then_retry(retry)
    assert not derive.timeout_then_retry(other_tool)
    assert not derive.timeout_then_retry(no_timeout)

    store = store_with([])
    base = fakes.make_trace("r-x", "refund.t1", False, [("refund", {})], version="parent000001")
    store.inner.trace_list = [
        type(base)(base.run_id, base.case_id, base.harness_version, base.model, retry, base.outcome)
    ]
    assert derive.pick_bad_trace(store, "parent000001", fakes.info, family="refund") == "r-x/refund.t1"
    assert derive.pick_bad_trace(store, "parent000001", fakes.info, family="mail") is None


RESULT_AGENTDOJO = """# AgentDojo results

## Overview (held-out cases only)

| Suite | Agent model | Status | v0 utility under attack | v0 targeted ASR |
|---|---|---|---|---|
| banking | `m1` | in progress, 1 gen | 90.6% (29/32) | 0.0% (0/32) |
| travel | `m1` | done | 85.0% (34/40) | 0.0% (0/40) |

## Runs
### banking (v1.2.2), run `a`, in progress

- Cases: 48 training, 40 held-out (the improver reads training traces only)

### travel (v1.2.2), run `b`, done

- Cases: 30 training, 25 held-out (the improver reads training traces only)

### banking (v1.2.2), run `c`, done

- Cases: 48 training, 40 held-out (the improver reads training traces only)
"""

RESULT_TAU = """# tau-bench retail: Ratchet results

- split seed `1`, 40 train / 115 held-out, digest `d`
- split seed `2`, 40 train / 20 held-out, digest `e`

| harness | split | tasks | trials/task | runs | successes | crashed | pass^1 | pass^2 |
|---|---|---|---|---|---|---|---|---|
| `h1` | heldout | 115 | 1 | 115 | 70 | 7 | 60.9% |  |

| source | role | gen | harness | held-out passed | train passed | crashed |
|---|---|---|---|---|---|---|
| `evolve/r` | parent_run_1 | 1 | `h1` | 31/40 | 21/30 | 0 |
| `evolve/r` | parent_run_2 | 1 | `h1` | 26/40 | 0/0 | 0 |
"""


def test_results_counts_are_parsed_from_the_pages(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "agentdojo.md").write_text(RESULT_AGENTDOJO, encoding="utf-8")
    (results / "tau-retail.md").write_text(RESULT_TAU, encoding="utf-8")
    out = derive.Derived()
    derive.derive_from_results(out, results, tmp_path / "out")
    got = {k: v for k, (v, _s) in out.values.items()}
    assert got["AGENTDOJO_CASES"] == "65"  # banking 40 (the larger of its two runs) + travel 25
    assert got["AGENTDOJO_SUITES"] == "2"
    assert got["TAU_CASES"] == "115"
    assert (got["TAU_SELF_SCORES"], got["TAU_SELF_TOTAL"]) == ("31 and 26", "40")
    summary = (tmp_path / "out" / "results-summary.txt").read_text(encoding="utf-8")
    assert "banking" in summary
    assert "60.9%" in summary
    assert not out.missing


def test_results_missing_is_reported(tmp_path: Path) -> None:
    out = derive.Derived()
    derive.derive_from_results(out, tmp_path, tmp_path / "out")
    assert set(out.missing) == {"AGENTDOJO_CASES", "TAU_CASES"}
    assert not (tmp_path / "out" / "results-summary.txt").exists()


def arm(version: str, admitted: int) -> dict[str, Any]:
    return {
        "instructions_version": version,
        "slots": 12,
        "proposed": 12,
        "invalid": 1,
        "duplicates": 2,
        "replay_cleared": 6,
        "evaluated_fresh": 5,
        "admitted": admitted,
        "admission_yield": f"{admitted}/12",
    }


def test_level2_reads_the_recorded_arms(tmp_path: Path) -> None:
    src = tmp_path / "l2.json"
    src.write_text(
        json.dumps({"claim": True, "arms": {"seed": arm("i0", 1), "search": arm("i1", 3)}}), encoding="utf-8"
    )
    out = derive.Derived()
    derive.derive_from_level2(out, [src], tmp_path / "out")
    got = {k: v for k, (v, _s) in out.values.items()}
    assert (got["LEVEL2_SEARCH_YIELD"], got["LEVEL2_SEED_YIELD"]) == ("3/12", "1/12")
    assert "i1" in (tmp_path / "out" / "level2-summary.txt").read_text(encoding="utf-8")


def test_level2_without_a_claim_or_a_file_fails(tmp_path: Path) -> None:
    src = tmp_path / "l2.json"
    src.write_text(json.dumps({"claim": False, "reason": "need clusters on both sides"}), encoding="utf-8")
    out = derive.Derived()
    derive.derive_from_level2(out, [src], tmp_path / "out")
    assert "need clusters on both sides" in out.missing["LEVEL2_SEARCH_YIELD"]
    gone = derive.Derived()
    derive.derive_from_level2(gone, [tmp_path / "none.json"], tmp_path / "out")
    assert set(gone.missing) == {"LEVEL2_FILE", "LEVEL2_SEARCH_YIELD", "LEVEL2_SEED_YIELD"}


def test_a_missing_value_deletes_the_stale_file_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = tmp_path / "ids.env"
    ids.write_text("OLD=1\n", encoding="utf-8")
    out = derive.Derived()
    out.set("A", "1", "somewhere")
    out.fail("B", "does not exist yet")
    assert derive.write_or_fail(out, ids) == 1
    assert not ids.exists()
    err = capsys.readouterr().err
    assert "NOT DERIVED  B: does not exist yet" in err


def test_complete_values_are_written_with_their_sources(tmp_path: Path) -> None:
    ids = tmp_path / "ids.env"
    out = derive.Derived()
    out.set("A", "1", "the store")
    out.set("B", "two words", "a file")
    assert derive.write_or_fail(out, ids) == 0
    text = ids.read_text(encoding="utf-8")
    assert "# source: the store\nA=1\n" in text
    assert "# source: a file\nB=two words\n" in text
    assert text.startswith("# Generated by docs/demo/derive_ids.py")


# -- the two outcomes -------------------------------------------------------------------------------------


def test_admitted_outcome_is_named_and_keeps_every_admitted_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(derive, "BAD_TRACE_FAMILY", "refund")
    out = derive_store(store_with([{"_id": "run-9", "state": {"status": "running"}}]))
    got = {k: v for k, (v, _src) in out.values.items()}
    assert got["DEMO_OUTCOME"] == "admitted"
    assert set(derive.ADMITTED_ONLY) <= set(got)


def test_refused_outcome_leaves_the_admitted_keys_absent_and_still_requires_the_rest() -> None:
    store = store_with([{"_id": "run-9", "state": {"status": "running"}}])
    store.inner.decision_docs = [d for d in store.inner.decision_docs if not d["admitted"]]
    parent = store.inner.decision_docs[0]["parent"]
    out = Derived = derive.Derived()
    derive.derive_from_store(
        store,
        Derived,
        case_info=fakes.info,
        verifies=lambda _id: None,
        resolve=resolve_candidates,
        pick_rows=pick_rows,
        fresh_view=fresh_view,
        refused_parent=parent,
    )
    got = {k: v for k, (v, _src) in out.values.items()}
    assert got["DEMO_OUTCOME"] == "refused"
    assert not set(derive.ADMITTED_ONLY) & (set(got) | set(out.missing))
    assert got["REJECTED_CAND"] == "rej00000001"
    assert got["REPLAY_TRACES"] == str(len({t.case_id for t in store.inner.trace_list}))
    assert "BAD_TRACE" in out.missing  # still required: the fakes hold no refund_lost_reply trap
    assert "AUTHORITY_DECISION" in out.missing


def fresh_doc(
    cand: str, recov: list[str], c1: dict[str, str], regressed: list[str], c2: str
) -> dict[str, Any]:
    cases = ["a", "b", "p1", "p2"]
    outcomes = {c: ["pass" if c.startswith("p") else "fail", "fail", c1.get(c, "fail"), c2] for c in cases}
    outcomes["p1"][1] = "pass"
    ev = {
        "manifest": 4,
        "order": ["P1", "P2", "C1", "C2"],
        "outcomes": outcomes,
        "stable_recoveries": [],
        "stable_recoveries_c1": recov,
        "protected": ["p1", "p2"],
        "discordant": ["p2"],
    }
    return fakes.decision_doc(
        "par",
        cand,
        False,
        checks=[
            {"name": "replay", "passed": True, "evidence": {}},
            {"name": "heldout", "passed": False, "evidence": ev},
            {"name": "no_regression", "passed": False, "evidence": {"regressed": regressed}},
        ],
    )


def test_the_closest_refusal_ranks_recoveries_then_kept_protection_then_fewest_regressions() -> None:
    keeps = {"a": "pass", "b": "pass", "p1": "pass", "p2": "pass"}
    best = fresh_doc("best", ["a", "b"], keeps, ["x"], "invalid")
    breaks = fresh_doc("breaks", ["a", "b"], {**keeps, "p2": "fail"}, [], "invalid")
    worse = fresh_doc("worse", ["a", "b"], keeps, ["x", "y"], "invalid")
    fewer = fresh_doc("fewer", ["a"], keeps, [], "invalid")
    db, picked = derive.pick_closest({"live1": [breaks, worse], "live2": [best, fewer]}) or ("", None)
    assert (db, picked.doc["candidate"] if picked else None) == ("live2", "best")
    out = derive.Derived()
    derive.derive_refusal(out, (db, picked))  # type: ignore[arg-type]
    got = {k: v for k, (v, _src) in out.values.items()}
    assert got["REFUSED_RECOVERIES"] == "2"
    assert got["REFUSED_RECOVERY_RUNS"] == "its first fresh run"
    assert (got["REFUSED_KEPT"], got["REFUSED_PROTECTED_N"]) == ("2", "2")
    assert got["REFUSED_WHY"] == "1 must-succeed case failed (x) and its confirmation run was invalid"
    assert (got["PARENT_DISCORDANT"], got["PARENT_DISCORDANT_CASES"]) == ("1", "p2")
    none = derive.Derived()
    derive.derive_refusal(none, derive.pick_closest({"live": []}))
    assert "REFUSED_CAND" in none.missing


def test_the_demo_database_prefers_admissions_then_volume() -> None:
    assert derive.rank_databases({"a": (0, 51), "b": (0, 3)}) == "a"
    assert derive.rank_databases({"a": (0, 51), "demo": (1, 2)}) == "demo"
    assert derive.rank_databases({}) is None


def test_level2_is_read_from_the_stored_arms(tmp_path: Path) -> None:
    def doc(run: str, arm: str, cand: str, replay: bool) -> dict[str, Any]:
        d = fakes.decision_doc("par", cand, False, checks=[{"name": "replay", "passed": replay}])
        return {**d, "_id": f"level2-{run}:{arm}:par->{cand}"}

    docs = [
        doc("run1", "seed", "s1", True),
        doc("run1", "seed", "s2", False),
        doc("run1", "search", "q1", True),
    ]
    docs.append(doc("run2", "seed", "only", True))  # no search arm: not a comparison
    out = derive.Derived()
    derive.derive_level2_from_store(out, docs, "l2db", tmp_path)
    got = {k: v for k, (v, _src) in out.values.items()}
    assert (got["LEVEL2_SEED_YIELD"], got["LEVEL2_SEARCH_YIELD"]) == ("0/2", "0/1")
    assert "run level2-run1" in (tmp_path / "level2-summary.txt").read_text(encoding="utf-8")
    empty = derive.Derived()
    derive.derive_level2_from_store(empty, docs[3:], "l2db", tmp_path)
    assert set(empty.missing) == {"LEVEL2_FILE", "LEVEL2_SEARCH_YIELD", "LEVEL2_SEED_YIELD"}


def test_tau_self_runs_come_from_the_parent_rows() -> None:
    page = """| source | role | gen | harness | held-out passed | train passed | crashed |
|---|---|---|---|---|---|---|
| `evolve/r` | parent_run_1 | 1 | `h1` | 31/40 | 21/30 | 0 |
| `evolve/r` | parent_run_2 | 1 | `h1` | 26/40 | 0/0 | 0 |
| `runs/x` | trial |  | `h1` | 70/115 | 32/40 | 7 |
"""
    assert derive.tau_self_runs(page) == ("h1", "31 and 26", "40")
    assert derive.tau_self_runs(page.replace("parent_run_2", "trial")) is None
