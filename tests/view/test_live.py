from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

import pytest
from rich.console import Console

from ratchet.store import ChangeEvent
from ratchet.view.live import follow, format_change

from .fakes import FakeStore, decision_doc

NOW = 0.0


def line(collection: str, doc: dict[str, Any] | None, op: str = "insert") -> str:
    t = format_change(collection, op, doc, NOW)
    return "" if t is None else t.plain


def test_trace_lines() -> None:
    ok = line("traces", {"case_id": "refund.t1", "harness_version": "abc", "passed": True})
    assert "PASS" in ok
    assert "refund.t1" in ok
    bad = line(
        "traces",
        {
            "case_id": "x",
            "harness_version": "abc",
            "passed": False,
            "outcome": {"reasons": ["refunded twice"]},
        },
    )
    assert "FAIL" in bad
    assert "refunded twice" in bad


def test_candidate_version_line() -> None:
    doc = {
        "_id": "abc123", "parent": "p0", "created_by": "improver", "procedures": [],
        "guards": [{"id": "g1", "description": "block retried refund"}],
    }  # fmt: skip
    out = line("harness_versions", doc)
    assert "abc123" in out
    assert "parent p0" in out
    assert "block retried refund" in out
    seed = line(
        "harness_versions", {"_id": "s", "parent": None, "created_by": "seed", "guards": [], "procedures": []}
    )
    assert "parent -" in seed


def test_decision_lines() -> None:
    ok = line("decisions", decision_doc("p", "c", True, change={"description": "g"}))
    assert "ADMITTED" in ok
    assert "p -> c" in ok
    assert "rcpt:" in ok
    no = line("decisions", decision_doc("p", "c", False, envelope=None))
    assert "REJECTED" in no
    assert "rcpt" not in no


def test_eval_and_generation_lines() -> None:
    gen = line(
        "evals",
        {"type": "generation", "generation": 2, "parent": "p", "promoted": "c",
         "parent_score": {"heldout_passed": 3}, "child_score": {"heldout_passed": 6, "heldout_total": 8}},
    )  # fmt: skip
    assert "generation 2" in gen
    assert "promoted c" in gen
    assert "3 -> 6/8" in gen
    assert "nothing admitted" in line("evals", {"type": "generation", "generation": 1, "parent": "p"})
    assert "held-out 1/2" in line(
        "evals",
        {
            "type": "eval",
            "harness_version": "v",
            "train": {"passed": 3, "total": 4},
            "heldout": {"passed": 1, "total": 2},
        },
    )
    assert "improver instructions v9" in line("evals", {"type": "improver_instructions", "version": "v9"})
    assert line("evals", {"type": "mystery"}) == ""


def test_failures_updates_and_unknown_collections() -> None:
    assert "stale read" in line("failures", {"summary": "stale read"})
    assert line("traces", {"case_id": "x"}, op="update") == ""
    assert line("traces", None) == ""
    assert line("checkpoints", {"a": 1}) == ""


class Scripted(FakeStore):
    """`script[collection]` is a list of attempts; each attempt is a list of events, ending in an exception
    to simulate a dropped stream. Every `watch` call consumes one attempt and records its resume token."""

    def __init__(self, script: dict[str, list[Any]]) -> None:
        super().__init__()
        self.script = {k: list(v) for k, v in script.items()}
        self.resumes: list[Any] = []

    def watch(
        self, collection: str, *, timeout_s: float | None = None, resume_after: dict[str, Any] | None = None
    ) -> Iterator[Any]:
        self.resumes.append(resume_after)
        attempts = self.script.get(collection, [])
        if not attempts:
            return
        for item in attempts.pop(0):
            if isinstance(item, Exception):
                raise item
            yield item


def change(collection: str, doc: dict[str, Any], op: str = "insert") -> ChangeEvent:
    return ChangeEvent(op, collection, doc.get("_id", "id"), doc, {"_data": "t"})


def run(store: FakeStore, **kw: Any) -> tuple[int, str]:
    buf = io.StringIO()
    n = follow(store, Console(file=buf, width=140), **kw)
    return n, buf.getvalue()


def test_follow_prints_every_stream_and_stops_when_they_end() -> None:
    version = {"_id": "c", "parent": "p", "created_by": "improver", "guards": []}
    store = Scripted({
        "traces": [[change("traces", {"case_id": "a", "passed": True, "harness_version": "h"})]],
        "decisions": [[change("decisions", decision_doc("p", "c", True))]],
        "harness_versions": [[change("harness_versions", version)]],
    })  # fmt: skip
    n, out = run(store, collections=("traces", "decisions", "harness_versions"))
    assert n == 3
    assert "TRACE" in out
    assert "DECISION" in out
    assert "VERSION" in out


def test_follow_skips_uninteresting_changes_and_reports_a_dead_stream() -> None:
    store = Scripted({
        "traces": [[change("traces", {"case_id": "a"}, op="update")]],
        "evals": [[RuntimeError("stream closed")]] * 3,
    })  # fmt: skip
    n, out = run(store, collections=("traces", "evals"), retries=2, backoff_s=0)
    assert n == 0
    assert out.count("reconnecting") == 2
    assert "evals: change stream stopped: stream closed" in out


def test_a_dropped_stream_resumes_from_its_last_token() -> None:
    first = change("traces", {"case_id": "a", "passed": True, "harness_version": "h"})
    second = change("traces", {"case_id": "b", "passed": False, "harness_version": "h"})
    first = ChangeEvent(first.operation, first.collection, first.doc_id, first.document, {"_data": "token-1"})
    store = Scripted({"traces": [[first, ConnectionError("reset")], [second]]})
    n, out = run(store, collections=("traces",), backoff_s=0)
    assert n == 2
    assert out.count("TRACE") == 2
    assert store.resumes == [None, {"_data": "token-1"}]
    assert "stream dropped (reset)" in out


def test_follow_stops_at_max_events() -> None:
    docs = [change("traces", {"case_id": f"c{i}", "passed": True, "harness_version": "h"}) for i in range(5)]
    n, out = run(Scripted({"traces": [docs]}), collections=("traces",), max_events=2)
    assert n == 2
    assert out.count("TRACE") == 2


def test_follow_survives_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    import queue

    def boom(self: Any, *a: Any, **k: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(queue.Queue, "get", boom)
    n, out = run(Scripted({"traces": [[]]}), collections=("traces",))
    assert n == 0
    assert "stopped" in out
