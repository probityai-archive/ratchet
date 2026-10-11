from __future__ import annotations

from pathlib import Path

import pytest

from ratchet.bench.tau.runner import seed_harness
from ratchet.bench.tau.store import JsonlStore, _default
from ratchet.types import Cost, Event, Outcome, Trace


def test_store_round_trips_traces_and_dedupes_harnesses(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "s")
    assert store.read("traces") == [] and store.traces() == []
    h = seed_harness()
    assert store.put_harness(h) is True and store.put_harness(h) is False
    assert store.get_harness(h.version) == h and store.get_harness("nope") is None
    trace = Trace(
        "r", "c", h.version, "m", [Event(0, "final", {"x": 1}, 1.5)], Outcome(False, ["why"], "d", Cost(1))
    )
    assert store.put_trace(trace)
    assert store.traces() == [trace]
    assert store.put_eval({"a": {1, 2}}, eval_id="e1") == "e1"
    assert len(store.put_decision({"b": b"\x01"})) == 32
    assert store.read("evals") == [{"eval_id": "e1", "report": {"a": [1, 2]}}]
    assert store.read("decisions")[0]["decision"] == {"b": "01"}
    store.put_decision({"v": 1}, decision_id="d")
    store.put_decision({"v": 2}, decision_id="d")
    assert store.decision("d") == {"v": 2} and store.decision("missing") is None


def test_unserialisable_values_are_refused() -> None:
    with pytest.raises(TypeError):
        _default(object())
