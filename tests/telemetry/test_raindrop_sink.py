from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ratchet.telemetry import raindrop as rd
from ratchet.telemetry.raindrop import CaseInfo, RaindropSink
from ratchet.types import Cost, Event, Outcome, Trace


class FakeInteraction:
    def __init__(self, client: FakeClient, kwargs: dict[str, Any]) -> None:
        self.client = client
        self.kwargs = kwargs
        self.tools: list[dict[str, Any]] = []
        self.output: str | None = None

    def track_tool(self, **kwargs: Any) -> None:
        self.tools.append(kwargs)

    def finish(self, *, output: str | None = None) -> None:
        self.output = output


class FakeClient:
    def __init__(self, fail_on: str | None = None) -> None:
        self.interactions: list[FakeInteraction] = []
        self.signals: list[dict[str, Any]] = []
        self.flushed = 0
        self.shut = 0
        self.fail_on = fail_on
        self.gate: threading.Event | None = None

    def begin(self, **kwargs: Any) -> FakeInteraction:
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail_on == "begin":
            raise RuntimeError("boom")
        it = FakeInteraction(self, kwargs)
        self.interactions.append(it)
        return it

    def track_signal(self, **kwargs: Any) -> None:
        self.signals.append(kwargs)

    def flush(self) -> None:
        if self.fail_on == "flush":
            raise RuntimeError("flush")
        self.flushed += 1

    def shutdown(self) -> None:
        if self.fail_on == "shutdown":
            raise RuntimeError("shutdown")
        self.shut += 1


@dataclass
class Case:
    id: str
    family: str = "refund_lost_reply"
    split: str = "train"
    kind: str = "trap"
    must_succeed: bool = True
    customer_message: str = "Please refund my order."


@dataclass
class Result:
    case_id: str
    split: str = "heldout"
    kind: str = "trap"
    must_succeed: bool = True


@dataclass
class Report:
    harness_version: str
    traces: list[Trace]
    results: list[Result]


@dataclass
class Check:
    name: str
    passed: bool


@dataclass
class Decision:
    parent: str
    candidate: str
    admitted: bool
    checks: list[Check]
    rates: dict[str, Any]


def ev(seq: int, kind: str, payload: dict[str, Any], ts: float = 0.0) -> Event:
    return Event(seq, kind, payload, ts)  # type: ignore[arg-type]


def make_trace(
    run_id: str = "r1",
    case_id: str = "c1",
    passed: bool = True,
    blocks: tuple[str, ...] = (),
    version: str = "v0v0v0v0v0v0",
) -> Trace:
    events = [
        ev(0, "model", {"content": "thinking"}),
        ev(1, "tool_call", {"call": {"name": "tickets_get", "args": {"id": "T1"}, "call_id": "a"}}, 1.0),
        ev(2, "tool_result", {"result": {"call_id": "a", "ok": True, "data": {"status": "open"}}}, 1.25),
        ev(3, "tool_call", {"call": {"name": "payments_refund", "args": {"c": "ch1"}, "call_id": "b"}}, 2.0),
        ev(
            4, "tool_result", {"result": {"call_id": "b", "ok": False, "data": None, "error": "timeout"}}, 2.5
        ),
    ]
    for i, source in enumerate(blocks):
        events.append(
            ev(5 + i, "guard_block", {"call_id": f"g{i}", "source": source, "message": f"no ({source})"})
        )
        events.append(
            ev(5 + i, "tool_call", {"call": {"name": "mail_send", "args": {}, "call_id": f"g{i}"}}, 3.0)
        )
    events.append(ev(20, "final", {"reason": "answered", "content": "All done.", "complete": True}))
    outcome = Outcome(passed, [] if passed else ["refund twice"], "d", Cost(3, 100, 20, 2, 1.23456))
    return Trace(run_id, case_id, version, "model-x", events, outcome)


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


def sink_for(client: FakeClient, tmp_path: Path, **kw: Any) -> RaindropSink:
    return RaindropSink(client=client, ledger_path=tmp_path / "ledger.json", **kw)


def test_emit_trace_sends_one_event_with_cohort_properties(client: FakeClient, tmp_path: Path) -> None:
    sink = sink_for(client, tmp_path, extra_properties={"smoke": True})
    out = sink.emit_trace(make_trace(), Case("c1"), cohort="baseline", generation=0, experiment="exp1")
    assert sink.flush() is True
    assert out.sent == 1 and out.event_ids == ["r1"] and sink.sent == 1
    (it,) = client.interactions
    k = it.kwargs
    assert k["event"] == "support_case" and k["event_id"] == "r1" and k["convo_id"] == "c1"
    assert k["input"] == "Please refund my order." and it.output == "All done." and k["model"] == "model-x"
    p = k["properties"]
    assert p["harness_version"] == "v0v0v0v0v0v0" and p["harness_cohort"] == "baseline"
    assert p["harness_generation"] == 0 and p["experiment"] == "exp1" and p["smoke"] is True
    assert p["case_family"] == "refund_lost_reply" and p["split"] == "train" and p["kind"] == "trap"
    assert p["passed"] is True and p["model_calls"] == 3 and p["wall_s"] == 1.235
    assert client.flushed == 1


def test_tool_calls_become_tool_spans_with_results_and_errors(client: FakeClient, tmp_path: Path) -> None:
    sink = sink_for(client, tmp_path)
    sink.emit_trace(make_trace(blocks=("guard",)), Case("c1"))
    sink.flush()
    tools = client.interactions[0].tools
    assert [t["name"] for t in tools] == ["tickets_get", "payments_refund", "mail_send"]
    assert tools[0]["output"] == {"status": "open"} and tools[0]["error"] is None
    assert tools[0]["duration_ms"] == pytest.approx(250.0)
    assert tools[1]["error"] == "timeout"
    assert tools[2]["error"] == "guard blocked (guard): no (guard)"


def test_signals_for_success_failure_guard_and_unwarranted(client: FakeClient, tmp_path: Path) -> None:
    sink = sink_for(client, tmp_path)
    sink.emit_trace(make_trace("ok"), Case("c1"))
    sink.emit_trace(
        make_trace("bad", passed=False, blocks=("guard", "guard")),
        Case("c2", kind="trap", must_succeed=False),
    )
    sink.emit_trace(make_trace("twin", blocks=("guard",)), Case("c3", kind="benign_twin"))
    sink.emit_trace(make_trace("args", blocks=("arguments",)), Case("c4", kind="benign_twin"))
    sink.flush()
    by_event: dict[str, list[tuple[str, str | None]]] = {}
    for s in client.signals:
        by_event.setdefault(s["event_id"], []).append((s["name"], s["sentiment"]))
    assert by_event["ok"] == [("task_success", "POSITIVE")]
    assert by_event["bad"] == [("task_failure", "NEGATIVE"), ("guard_block", None)]
    assert by_event["twin"] == [
        ("task_success", "POSITIVE"),
        ("guard_block", None),
        ("unwarranted_intervention", "NEGATIVE"),
    ]
    assert by_event["args"] == [("task_success", "POSITIVE"), ("guard_block", None)]
    guard_props = next(s for s in client.signals if s["event_id"] == "bad" and s["name"] == "guard_block")
    assert guard_props["properties"] == {"source": "guard", "count": 2}
    assert sink.signals_sent == 1 + 2 + 3 + 2


def test_no_key_is_a_noop() -> None:
    sink = RaindropSink(write_key=None)
    assert sink.enabled is False
    assert sink.emit_trace(make_trace(), Case("c1")).sent == 0
    d = Decision("p", "c", True, [], {})
    assert sink.emit_decision(d).sent == 0
    rep = sink.emit_evaluation(Report("v", [make_trace()], []), cohort="baseline")
    assert rep.refused == "no write key" and sink.flush() is True and sink.close() is True


def test_duplicate_run_is_sent_once(client: FakeClient, tmp_path: Path) -> None:
    sink = sink_for(client, tmp_path)
    assert sink.emit_trace(make_trace(), Case("c1")).sent == 1
    again = sink.emit_trace(make_trace(), Case("c1"))
    sink.flush()
    assert again.sent == 0 and again.duplicates == 1 and len(client.interactions) == 1


def test_budget_refuses_past_max_and_persists_across_sinks(client: FakeClient, tmp_path: Path) -> None:
    sink = sink_for(client, tmp_path, max_events=2)
    assert [sink.emit_trace(make_trace(f"r{i}"), Case("c")).sent for i in range(3)] == [1, 1, 0]
    sink.flush()
    assert sink.reserved_this_month == 2 and sink.remaining() == 0
    second = sink_for(client, tmp_path, max_events=2)
    assert second.reserved_this_month == 2
    refused = second.emit_trace(make_trace("r9"), Case("c"))
    assert refused.over_budget == 1 and refused.sent == 0


def test_ledger_resets_in_a_new_month(client: FakeClient, tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"month": "2026-09", "events": 500}))
    now = datetime(2026, 10, 11, tzinfo=UTC)
    sink = RaindropSink(client=client, ledger_path=path, now=lambda: now)
    assert sink.reserved_this_month == 0
    sink.emit_trace(make_trace(), Case("c"))
    assert json.loads(path.read_text()) == {"month": "2026-10", "events": 1}


def test_ledger_rolls_over_while_running(client: FakeClient, tmp_path: Path) -> None:
    clock = [datetime(2026, 10, 31, tzinfo=UTC)]
    sink = RaindropSink(client=client, ledger_path=tmp_path / "l.json", now=lambda: clock[0])
    sink.emit_trace(make_trace("a"), Case("c"))
    assert sink.reserved_this_month == 1
    clock[0] = datetime(2026, 11, 1, tzinfo=UTC)
    assert sink.reserved_this_month == 0


@pytest.mark.parametrize("content", ["not json", json.dumps({"month": "2026-10"}), json.dumps([1])])
def test_corrupt_ledger_counts_as_empty(client: FakeClient, tmp_path: Path, content: str) -> None:
    path = tmp_path / "l.json"
    path.write_text(content)
    now = datetime(2026, 10, 11, tzinfo=UTC)
    assert RaindropSink(client=client, ledger_path=path, now=lambda: now).reserved_this_month == 0


def test_unwritable_ledger_does_not_raise(client: FakeClient, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    sink = RaindropSink(client=client, ledger_path=blocker / "sub" / "l.json")
    assert sink.emit_trace(make_trace(), Case("c")).sent == 1


def test_evaluation_tags_every_run_with_the_cohort(client: FakeClient, tmp_path: Path) -> None:
    sink = sink_for(client, tmp_path)
    traces = [make_trace(f"r{i}", case_id=f"c{i}", version="aaaaaaaaaaaa") for i in range(3)]
    report = Report("aaaaaaaaaaaa", traces, [Result("c0"), Result("c1", kind="benign_twin"), Result("c2")])
    out = sink.emit_evaluation(
        report, {"c0": Case("c0", family="fam0")}, cohort="candidate", generation=2, experiment="e"
    )
    sink.flush()
    assert out.sent == 3 and out.refused is None
    props = {it.kwargs["event_id"]: it.kwargs["properties"] for it in client.interactions}
    assert {p["harness_cohort"] for p in props.values()} == {"candidate"}
    assert props["r0"]["case_family"] == "fam0" and props["r0"]["split"] == "train"
    assert props["r1"]["kind"] == "benign_twin" and props["r1"]["split"] == "heldout"
    assert props["r2"]["case_family"] == "unknown"


def test_evaluation_accepts_a_case_list_and_unknown_case(client: FakeClient, tmp_path: Path) -> None:
    sink = sink_for(client, tmp_path)
    report = Report("v", [make_trace("r0", case_id="zz")], [])
    out = sink.emit_evaluation(report, [Case("other")], cohort="baseline")
    sink.flush()
    assert out.sent == 1 and client.interactions[0].kwargs["properties"]["case_id"] == "zz"


def test_evaluation_is_all_or_nothing_for_the_budget(client: FakeClient, tmp_path: Path) -> None:
    sink = sink_for(client, tmp_path, max_events=2)
    report = Report("v", [make_trace(f"r{i}") for i in range(3)], [])
    out = sink.emit_evaluation(report, cohort="baseline")
    sink.flush()
    assert out.sent == 0 and out.refused is not None and "2 left" in out.refused
    assert client.interactions == [] and sink.reserved_this_month == 0


def test_decision_event_has_gates_rates_and_signal(client: FakeClient, tmp_path: Path) -> None:
    sink = sink_for(client, tmp_path)
    rejected = Decision(
        "p1",
        "c1",
        False,
        [Check("replay", True), Check("heldout", False)],
        {"heldout": {"before": 3, "after": 3}, "note": "x"},
    )
    admitted = Decision("p1", "c2", True, [Check("replay", True)], {})
    assert sink.emit_decision(rejected, experiment="e").sent == 1
    assert sink.emit_decision(admitted).sent == 1
    assert sink.emit_decision(admitted).duplicates == 1
    sink.flush()
    rej, adm = client.interactions
    assert rej.kwargs["event"] == "ratchet_admission" and rej.kwargs["user_id"] == "ratchet-improver"
    p = rej.kwargs["properties"]
    assert p["admitted"] is False and p["failed_checks"] == "heldout" and p["check_replay"] is True
    assert p["check_heldout"] is False and p["rate_heldout_after"] == 3 and p["experiment"] == "e"
    assert "rate_note" not in p
    assert rej.output == "REJECTED: heldout" and adm.output == "ADMITTED"
    assert adm.kwargs["properties"]["failed_checks"] == "none"
    assert [(s["name"], s["sentiment"]) for s in client.signals] == [
        ("rejected", "NEGATIVE"),
        ("admitted", "POSITIVE"),
    ]


def test_rejected_without_failed_check_names_no_reason(client: FakeClient, tmp_path: Path) -> None:
    sink = sink_for(client, tmp_path)
    sink.emit_decision(Decision("p", "c", False, [], {}))
    sink.flush()
    assert client.interactions[0].output == "REJECTED: no reason"


def test_send_failure_is_counted_not_raised(tmp_path: Path) -> None:
    bad = FakeClient(fail_on="begin")
    sink = sink_for(bad, tmp_path)
    sink.emit_trace(make_trace(), Case("c"))
    sink.emit_decision(Decision("p", "c", True, [], {}))
    assert sink.flush() is True
    assert sink.sent == 0 and sink.dropped == 2


def test_flush_and_close_survive_sdk_errors(tmp_path: Path) -> None:
    assert sink_for(FakeClient(fail_on="flush"), tmp_path).flush() is False
    assert sink_for(FakeClient(fail_on="flush"), tmp_path).close() is False
    sink = sink_for(FakeClient(fail_on="shutdown"), tmp_path)
    assert sink.close() is False
    ok = FakeClient()
    assert sink_for(ok, tmp_path).close() is True and ok.shut == 1


def test_full_queue_drops_without_blocking(tmp_path: Path) -> None:
    client = FakeClient()
    client.gate = threading.Event()
    sink = sink_for(client, tmp_path, queue_size=1)
    outs = [sink.emit_trace(make_trace(f"r{i}"), Case("c")) for i in range(6)]
    client.gate.set()
    sink.flush()
    assert sum(o.dropped for o in outs) >= 1
    assert sink.dropped >= 1


def test_flush_times_out_while_work_is_stuck(tmp_path: Path) -> None:
    client = FakeClient()
    client.gate = threading.Event()
    sink = sink_for(client, tmp_path)
    sink.emit_trace(make_trace(), Case("c"))
    assert sink.flush(timeout=0.05) is False
    client.gate.set()
    assert sink.flush() is True


def test_final_reply_falls_back_to_last_model_message() -> None:
    t = make_trace()
    no_final = Trace(t.run_id, t.case_id, t.harness_version, t.model, t.events[:-1], t.outcome)
    assert rd.final_reply(no_final) == "thinking"
    empty = Trace("r", "c", "v", "m", [], t.outcome)
    assert rd.final_reply(empty) == ""


def test_empty_reply_is_sent_as_placeholder(client: FakeClient, tmp_path: Path) -> None:
    t = make_trace()
    empty = Trace("r", "c", "v", "m", [], t.outcome)
    sink = sink_for(client, tmp_path)
    sink.emit_trace(empty, CaseInfo("c"))
    sink.flush()
    assert client.interactions[0].output == "(no reply)"


def test_tool_call_without_result_has_no_output(client: FakeClient, tmp_path: Path) -> None:
    t = Trace(
        "r",
        "c",
        "v",
        "m",
        [ev(0, "tool_call", {"call": {"name": "x", "args": {}, "call_id": "q"}}, 1.0)],
        make_trace().outcome,
    )
    (step,) = rd.tool_steps(t)
    assert (step.output, step.error, step.duration_ms) == (None, None, 0.0)


def test_reasons_are_capped() -> None:
    long = Trace("r", "c", "v", "m", [], Outcome(False, ["x" * 900], "d"))
    props = RaindropSink(client=FakeClient()).run_properties(long, CaseInfo("c"))
    assert len(str(props["reasons"])) == rd.MAX_REASON_CHARS and str(props["reasons"]).endswith("...")


def test_write_key_from_env_then_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAINDROP_WRITE_KEY", " abc ")
    assert rd.resolve_write_key() == "abc"
    monkeypatch.delenv("RAINDROP_WRITE_KEY")
    assert rd.resolve_write_key() is None

    def run_ok(*a: Any, **k: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(a, 0, stdout="fromring\n", stderr="")

    monkeypatch.setattr("subprocess.run", run_ok)
    assert rd.resolve_write_key(use_keyring=True) == "fromring"

    def run_missing(*a: Any, **k: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(a, 1, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", run_missing)
    assert rd.resolve_write_key(use_keyring=True) is None

    def run_boom(*a: Any, **k: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("secret-tool")

    monkeypatch.setattr("subprocess.run", run_boom)
    assert rd.resolve_write_key(use_keyring=True) is None


def test_ledger_default_path_follows_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert rd.ledger_default_path() == tmp_path / "ratchet" / "raindrop-ledger.json"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert rd.ledger_default_path().name == "raindrop-ledger.json"


def test_key_builds_a_real_sdk_client_lazily(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class FakeModule:
        @staticmethod
        def Raindrop(**kwargs: Any) -> FakeClient:
            seen.update(kwargs)
            return FakeClient()

    monkeypatch.setattr("importlib.import_module", lambda name: FakeModule)
    sink = RaindropSink("secret-key", project_id="demo", ledger_path=tmp_path / "l.json", tracing=True)
    assert sink.enabled and seen == {"api_key": "secret-key", "project_id": "demo", "tracing_enabled": True}


def test_client_construction_failure_disables_the_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", boom)
    assert RaindropSink("k").enabled is False


def test_key_never_appears_in_properties(client: FakeClient, tmp_path: Path) -> None:
    sink = sink_for(client, tmp_path)
    sink.emit_trace(make_trace(), Case("c"))
    sink.flush()
    assert "secret-key" not in json.dumps(client.interactions[0].kwargs, default=str)
