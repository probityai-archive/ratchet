"""The k-session rule: its statistics against the reference implementation, the runner and the bundle."""

from __future__ import annotations

import json
import math
import random
import threading
from pathlib import Path
from typing import Any

import pytest

from ratchet.improve import _dist, sessions
from ratchet.improve.sessions import ARMS, UNCHANGED, Outcome, Protocol, Slot

ORACLE = Path(__file__).with_name("data") / "d37_oracle.json"


def _close(got: float, want: float) -> bool:
    return abs(got - want) <= 1e-12 + 1e-9 * abs(want)


def _oracle_data(fixture: dict[str, Any], cohort: dict[str, Any]) -> sessions.Data:
    cases = [f"c{i}" for i in range(fixture["cases"])]
    protocol = Protocol.for_k("oracle", cases, {a: a for a in ARMS}, k=fixture["k"], n=fixture["n"])
    data = sessions.Data(protocol)
    for ai, arm in enumerate(ARMS):
        for s in range(protocol.k):
            for ci, case in enumerate(cases):
                for r in range(protocol.n):
                    t = cohort["times"][ai][s][ci][r]
                    data.cells[(s, arm, case, r)] = sessions._Cell(
                        bool(cohort["passed"][ai][s][ci][r]), t, t, f"e{cohort['endpoints'][ai][s][ci][r]}"
                    )
    return data


def test_every_statistic_matches_the_reference_implementation() -> None:
    """24 cohorts drawn and judged by the reference code: admissions, log-check fires and harm failures."""
    fixture = json.loads(ORACLE.read_text(encoding="utf-8"))
    verdicts = {"admitted": 0, "logs_fire": 0, "harm_fail": 0}
    for i, cohort in enumerate(fixture["cohorts"]):
        want = cohort["expect"]
        got = sessions.evaluate(_oracle_data(fixture, cohort))
        logs = got["log_checks"]
        pairs = [(logs["E"]["p"], want["p_endpoint"]), (logs["T*"]["p_time"], want["p_time"]),
                 (logs["T*"]["p_trend"], want["p_trend"]), (logs["P"]["z"], want["z_pairing"])]  # fmt: skip
        assert all(_close(g, w) for g, w in pairs), (i, pairs)
        for g, w in zip((got["t"][a] for a in UNCHANGED), want["t"], strict=True):
            assert (g is None) == (w is None) and (g is None or _close(g, w)), (i, g, w)
        assert (not got["harm_screen"]["harmed_cases"]) == want["harm_ok"], i
        assert (got["verdict"] == "admitted") == want["admitted"], (i, got["verdict"], got["reasons"])
        verdicts["admitted"] += got["verdict"] == "admitted"
        verdicts["logs_fire"] += not want["logs_ok"]
        verdicts["harm_fail"] += not want["harm_ok"]
    assert verdicts == {"admitted": 9, "logs_fire": 5, "harm_fail": 2}


def test_the_frozen_constants_are_the_tests_own_quantiles() -> None:
    assert abs(_dist.t_isf(0.05, 7) - sessions.D37_T_CRITICAL_K8) < 1e-9
    assert abs(_dist.norm_isf(0.01 / 3) - sessions.D37_Z_PAIRING) < 1e-12
    assert _close(_dist.chi2_sf(10.0, 4), 0.04042768199451279)
    for t in (0.3, 1.0, 1.9, 4.2):  # Student t with 7 df has a closed form
        th = math.atan(t / math.sqrt(7))
        c = math.cos(th)
        closed = 0.5 - (th + math.sin(th) * c * (1 + 2 / 3 * c * c + 8 / 15 * c**4)) / math.pi
        assert _close(_dist.t_sf(t, 7), closed)


def test_a_protocol_cannot_declare_a_laxer_threshold() -> None:
    with pytest.raises(ValueError, match="t_critical"):
        Protocol("p", ("a",), {a: a for a in ARMS}, 8, 5, 1.5, sessions.D37_Z_PAIRING)
    with pytest.raises(ValueError, match="z_pairing"):
        Protocol("p", ("a",), {a: a for a in ARMS}, 8, 5, sessions.D37_T_CRITICAL_K8, 3.5)
    assert Protocol.for_k("p", ["a"], {a: a for a in ARMS}, k=12).t_critical == _dist.t_isf(0.05, 11)


def test_seeds_are_per_arm_streams_written_as_strings() -> None:
    p = Protocol.for_k("seeds", ["a", "b"], {a: a for a in ARMS})
    seeds = {arm: [sessions.arm_seed(p, 0, arm, r) for r in range(p.n)] for arm in ARMS}
    flat = [s for v in seeds.values() for s in v]
    assert len(set(flat)) == len(flat)
    assert all(isinstance(s, str) and 0 <= int(s) < 2**64 for s in flat)
    assert any(int(s) > 2**53 for s in flat)


def _protocol(cases: int = 6) -> Protocol:
    return Protocol.for_k("test-run", [f"case{i}" for i in range(cases)], {a: f"h-{a}" for a in ARMS})


def _executor(gain: float = 0.0, lose: set[tuple[int, str, str, int]] | None = None,
              lose_twice: bool = False) -> sessions.Executor:  # fmt: skip
    """Outcomes drawn from each slot's own seed: a fixed per-case rate, plus `gain` for the candidate."""
    lose = lose or set()
    lock = threading.Lock()
    calls: dict[tuple[int, str, str, int], int] = {}

    def run(slot: Slot) -> Outcome:
        key = (slot.session, slot.arm, slot.case_id, slot.rep)
        with lock:
            calls[key] = calls.get(key, 0) + 1
            n = calls[key]
        if key in lose and (n == 1 or lose_twice):
            return Outcome(None, note="endpoint lost")
        rate = 0.2 + 0.1 * int(slot.case_id[-1]) + (gain if slot.arm == "candidate" else 0.0)
        draw = random.Random(f"{slot.seed}/{slot.case_id}/{slot.attempt}").random()
        return Outcome(draw < rate, run_id=f"run-{slot.session}-{slot.position}")

    return run


@pytest.mark.parametrize("workers", [1, 4])
def test_a_run_writes_a_bundle_that_recomputes_offline(tmp_path: Path, workers: int) -> None:
    p = _protocol()
    lost = {(2, "candidate", "case3", 1), (5, "parent", "case0", 4)}
    verdict = sessions.run_protocol(p, _executor(gain=0.35, lose=lost), tmp_path, workers=workers,
                                    endpoints_for=lambda s: [f"http://e{i}" for i in range(3)])  # fmt: skip
    assert verdict["reruns"] == 2 and verdict["attempts"] == len(ARMS) * 6 * p.n * p.k + 2
    same, fresh = sessions.verify_bundle(tmp_path)
    assert same and fresh == verdict
    assert verdict["verdict"] in {"admitted", "not_admitted", "inconclusive"}
    assert sessions.main(["verify", str(tmp_path)]) == 0


def test_a_clear_gain_is_admitted_and_no_effect_is_not(tmp_path: Path) -> None:
    pool = lambda s: ["http://a", "http://b"]  # noqa: E731
    big = sessions.run_protocol(_protocol(), _executor(gain=0.6), tmp_path / "gain", endpoints_for=pool)
    assert big["verdict"] == "admitted", big["reasons"]
    null = sessions.run_protocol(_protocol(), _executor(gain=0.0), tmp_path / "null", endpoints_for=pool)
    assert null["verdict"] != "admitted"


def test_an_execution_lost_twice_makes_the_comparison_inconclusive(tmp_path: Path) -> None:
    lost = {(0, "identity_1", "case2", 0)}
    v = sessions.run_protocol(_protocol(), _executor(gain=0.6, lose=lost, lose_twice=True), tmp_path,
                              endpoints_for=lambda s: ["http://a"])  # fmt: skip
    assert v["verdict"] == "inconclusive" and v["reasons"] == ["invalid_execution"]
    assert v["invalid"] == ["(0, 'identity_1', 'case2', 0): lost twice"]


def test_an_edited_outcome_no_longer_matches_the_stored_verdict(tmp_path: Path) -> None:
    sessions.run_protocol(_protocol(), _executor(gain=0.6), tmp_path, endpoints_for=lambda s: ["http://a"])
    log = tmp_path / "executions.jsonl"
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    for row in rows:
        if row["arm"] == "candidate" and row["passed"] is True:
            row["passed"] = False
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    same, fresh = sessions.verify_bundle(tmp_path)
    assert not same and fresh["verdict"] == "not_admitted"
    assert sessions.main(["verify", str(tmp_path)]) == 1


@pytest.mark.parametrize("field", ["endpoint", "seed", "position"])
def test_a_bundle_whose_log_departs_from_the_schedule_is_refused(tmp_path: Path, field: str) -> None:
    sessions.run_protocol(
        _protocol(), _executor(), tmp_path, endpoints_for=lambda s: ["http://a", "http://b"]
    )
    log = tmp_path / "executions.jsonl"
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    rows[0][field] = {"endpoint": "http://elsewhere", "seed": "1", "position": 10**6}[field]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(sessions.BundleError):
        sessions.load_bundle(tmp_path)
    assert sessions.main(["verify", str(tmp_path)]) == 2


def test_a_manifest_that_claims_another_schedule_is_refused(tmp_path: Path) -> None:
    sessions.run_protocol(
        _protocol(), _executor(), tmp_path, endpoints_for=lambda s: ["http://a", "http://b"]
    )
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    manifest["sessions"][3]["endpoints"] = ["http://b", "http://a"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(sessions.BundleError, match="session 3"):
        sessions.load_bundle(tmp_path)


def test_a_zero_sd_is_inconclusive() -> None:
    p = _protocol(cases=1)
    data = sessions.Data(p)
    for s in range(p.k):
        for arm in ARMS:
            for r in range(p.n):
                data.cells[(s, arm, "case0", r)] = sessions._Cell(arm == "candidate", r + s, r + s, "e")
    v = sessions.evaluate(data)
    assert v["verdict"] == "inconclusive" and "zero_sd" in v["reasons"]


def test_session_start_hook_runs_before_each_session(tmp_path: Path) -> None:
    started: list[int] = []
    sessions.run_protocol(_protocol(cases=2), _executor(), tmp_path, endpoints_for=lambda s: ["http://a"],
                          on_session_start=started.append)  # fmt: skip
    assert started == list(range(8))


def _fake_trace(passed: bool, reasons: list[str] | None = None, cached: bool = False) -> Any:
    from ratchet_cases.types import Cost, Event, Trace
    from ratchet_cases.types import Outcome as CaseOutcome

    events = [Event(0, "model", {"cached": cached}, 0.0)]
    return Trace("r", "c", "h", "m", events, CaseOutcome(passed, reasons or [], "d", Cost()))


def test_the_live_executor_runs_each_slot_once_with_its_seed_and_endpoint() -> None:
    from ratchet.agent.seed import seed_harness
    from ratchet.improve.evaluate import INFRA_REASON
    from ratchet.improve.sessions_live import live_executor, seeded_client

    seen: list[tuple[str, int]] = []

    class Client:
        def close(self) -> None:
            pass

    def client_for(url: str, seed: int) -> Any:
        seen.append((url, seed))
        return Client()

    def runner(harness: Any, case: Any, model: Any, store: Any, *, run_id: str) -> Any:
        if case == "lost":
            raise RuntimeError("endpoint refused")
        if case == "infra":
            return _fake_trace(False, [f"{INFRA_REASON}: timeout"])
        if case == "cached":
            return _fake_trace(True, cached=True)
        return _fake_trace(case == "pass")

    h = seed_harness()
    cases = {c: c for c in ("pass", "fail", "lost", "infra", "cached")}
    execute = live_executor({a: h for a in ARMS}, cases, client_for, runner=runner)
    assert execute(Slot(0, 0, "candidate", "pass", 0, str(2**60 + 7), "http://a/v1")).passed is True
    assert execute(Slot(0, 1, "parent", "fail", 0, "5", "http://b/v1")).passed is False
    for case in ("lost", "infra", "cached"):
        assert execute(Slot(0, 2, "parent", case, 0, "5", "http://b/v1")).passed is None
    assert seen[:2] == [("http://a/v1", 2**60 + 7), ("http://b/v1", 5)]
    client = seeded_client("http://c/v1", 9, "m")
    client.close()
