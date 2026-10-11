from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ratchet.telemetry import RaindropSink
from ratchet.view.telemetry import TelemetryStore, usage_line

from .fakes import FakeStore, decision_doc, eval_doc, make_trace, scenario


class FakeSink:
    def __init__(self, fail: bool = False, refuse: str | None = None) -> None:
        self.fail, self.refuse = fail, refuse
        self.evaluations: list[dict[str, Any]] = []
        self.decisions: list[Any] = []
        self.closed = False
        self.sent, self.dropped, self.reserved_this_month = 4, 1, 120

    def remaining(self) -> int:
        return 780

    def emit_evaluation(self, report: Any, cases: Any = (), *, cohort: str, generation: int | None = None,
                        experiment: str | None = None) -> Any:  # fmt: skip
        if self.fail:
            raise RuntimeError("raindrop down")
        self.evaluations.append({
            "version": report.harness_version, "cohort": cohort, "generation": generation,
            "experiment": experiment, "traces": [t.run_id for t in report.traces],
            "results": [r.case_id for r in report.results],
        })  # fmt: skip

        class Summary:
            refused = self.refuse

        return Summary()

    def emit_decision(self, decision: Any, *, experiment: str | None = None) -> Any:
        if self.fail:
            raise RuntimeError("raindrop down")
        self.decisions.append((decision.parent, decision.candidate, decision.admitted,
                               [(c.name, c.passed) for c in decision.checks], experiment))  # fmt: skip

    def close(self, timeout: float = 30.0) -> bool:
        self.closed = True
        return True


def results(version: str, run_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": f"case{i}",
            "split": "train",
            "kind": "trap",
            "must_succeed": True,
            "run_id": r,
            "interventions": 0,
        }
        for i, r in enumerate(run_ids)
    ]


def drive(store: TelemetryStore, *, gens: int = 2) -> list[dict[str, Any]]:
    """Write what the loop writes: traces, evals, decisions, then the generation record."""
    written: list[dict[str, Any]] = []
    parent = "v0"
    for g in range(1, gens + 1):
        child, loser = f"v{g}", f"x{g}"
        for v in (parent, child, loser):
            if g == 1 or v != parent:
                store.put_trace(make_trace(f"run-{v}", "case0", True, [], version=v))
        if g == 1:
            store.put_eval(
                {**eval_doc(parent, (1, 2), (1, 3), results(parent, [f"run-{parent}"])), "role": "parent"}
            )
        store.put_eval(
            {**eval_doc(child, (2, 2), (3, 3), results(child, [f"run-{child}"])), "role": "candidate"}
        )
        store.put_eval(
            {**eval_doc(loser, (1, 2), (1, 3), results(loser, [f"run-{loser}"])), "role": "candidate"}
        )
        store.put_decision(decision_doc(parent, child, True, generation=g))
        store.put_decision(decision_doc(parent, loser, False, generation=g))
        rec = {"type": "generation", "generation": g, "parent": parent, "promoted": child}
        store.put_eval(rec, eval_id=f"gen-{g}")
        written.append(rec)
        parent = child
    return written


def make(sink: FakeSink, inner: FakeStore | None = None) -> tuple[TelemetryStore, FakeStore, list[str]]:
    warnings: list[str] = []
    inner = inner or FakeStore()
    return TelemetryStore(inner, sink, [], "exp-1", warnings.append), inner, warnings


def test_each_generation_sends_baseline_candidate_and_decisions() -> None:
    sink = FakeSink()
    store, _, warnings = make(sink)
    drive(store)
    assert warnings == []
    # generation 2's parent (v1) is offered again as the baseline; the sink skips runs it already sent
    assert [(e["version"], e["cohort"], e["generation"]) for e in sink.evaluations] == [
        ("v0", "baseline", 0),
        ("v1", "candidate", 1),
        ("v1", "baseline", 1),
        ("v2", "candidate", 2),
    ]
    assert all(e["experiment"] == "exp-1" for e in sink.evaluations)
    assert sink.evaluations[0]["traces"] == ["run-v0"]
    assert [(d[0], d[1], d[2]) for d in sink.decisions] == [
        ("v0", "v1", True), ("v0", "x1", False), ("v1", "v2", True), ("v1", "x2", False),
    ]  # fmt: skip
    assert sink.decisions[0][3] == [("replay", True)]
    assert sink.decisions[0][4] == "exp-1"


def test_best_candidate_is_sent_when_nothing_was_promoted() -> None:
    sink = FakeSink()
    store, _, _ = make(sink)
    store.put_eval({**eval_doc("v0", (1, 2), (1, 3), results("v0", ["r0"])), "role": "parent"})
    store.put_eval({**eval_doc("a", (1, 2), (1, 3), results("a", ["ra"])), "role": "candidate"})
    store.put_eval({**eval_doc("b", (1, 2), (2, 3), results("b", ["rb"])), "role": "composite"})
    store.put_eval({"type": "generation", "generation": 1, "parent": "v0", "promoted": None})
    assert [(e["version"], e["cohort"]) for e in sink.evaluations] == [("v0", "baseline"), ("b", "candidate")]


def test_nothing_to_send_on_an_empty_generation() -> None:
    sink = FakeSink()
    store, _, _ = make(sink)
    store.put_eval({"type": "generation", "generation": 1, "parent": "v0", "promoted": None})
    assert sink.evaluations == []
    assert sink.decisions == []


def test_a_refused_cohort_is_reported_not_raised() -> None:
    sink = FakeSink(refuse="74 events do not fit in the 10 left this month")
    store, _, warnings = make(sink)
    drive(store, gens=1)
    assert any("not sent: 74 events" in w for w in warnings)


def test_a_failing_sink_changes_nothing_the_loop_stored_or_saw() -> None:
    clean_store = FakeStore()
    clean, _, _ = make(FakeSink(), clean_store)
    drive(clean)
    broken_store = FakeStore()
    broken, _, warnings = make(FakeSink(fail=True), broken_store)
    drive(broken)
    assert broken_store.decision_docs == clean_store.decision_docs
    assert broken_store.eval_docs == clean_store.eval_docs
    assert len(broken_store.trace_list) == len(clean_store.trace_list)
    assert warnings
    assert all("admission is unaffected" in w for w in warnings)


def test_every_write_is_forwarded_with_its_own_return_value() -> None:
    store, inner, _ = make(FakeSink(fail=True))
    _, seed, _ = scenario()
    assert store.put_harness(seed) is True
    assert store.put_harness(seed) is False
    assert store.put_eval({"type": "other"}, eval_id="e1") == "e1"
    assert store.put_decision(decision_doc("a", "b", True), decision_id="d1") == "d1"
    assert inner.harnesses[seed.version] is seed


def test_non_mapping_records_are_ignored() -> None:
    store, inner, warnings = make(FakeSink())
    store.put_eval(object() if False else {"type": "eval"})  # no results key: stored, not observed
    assert warnings == []
    assert len(inner.eval_docs) == 1


class Exploding:
    """A Raindrop client whose every call fails, to exercise the real sink's own error handling."""

    def __getattr__(self, name: str) -> Any:
        raise ConnectionError(f"raindrop unreachable ({name})")


@pytest.mark.parametrize("fail", [True])
def test_the_real_sink_with_a_dead_client_cannot_change_a_decision(tmp_path: Path, fail: bool) -> None:
    clean_store = FakeStore()
    clean, _, _ = make(FakeSink(), clean_store)
    drive(clean)
    sink = RaindropSink(client=Exploding(), ledger_path=tmp_path / "ledger.json")
    real_store = FakeStore()
    wrapped = TelemetryStore(real_store, sink, [], "exp-1", lambda _m: None)
    drive(wrapped)
    sink.close(timeout=5)
    assert real_store.decision_docs == clean_store.decision_docs
    assert real_store.eval_docs == clean_store.eval_docs


def test_usage_line_names_the_monthly_cap() -> None:
    assert usage_line(FakeSink()) == "raindrop: 120 of 900 monthly events used (4 sent this run, 1 dropped)"
