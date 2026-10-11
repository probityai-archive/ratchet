from __future__ import annotations

import importlib
import json
import types
from pathlib import Path
from typing import Any

import pytest

from ratchet.telemetry import smoke
from ratchet.telemetry.raindrop import RaindropSink


class Interaction:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs

    def track_tool(self, **kwargs: Any) -> None:
        pass

    def finish(self, *, output: str | None = None) -> None:
        pass


class Client:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.signals: list[dict[str, Any]] = []

    def begin(self, **kwargs: Any) -> Interaction:
        self.events.append(kwargs)
        return Interaction(kwargs)

    def track_signal(self, **kwargs: Any) -> None:
        self.signals.append(kwargs)

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


def test_batch_is_ten_events_in_two_cohorts_and_one_decision() -> None:
    runs, decision = smoke.build_batch("b1")
    assert len(runs) == 9 and len(runs) + 1 == smoke.SMOKE_EVENTS
    cohorts = [c for _, _, c in runs]
    assert cohorts.count("baseline") == 5 and cohorts.count("candidate") == 4
    assert len({t.run_id for t, _, _ in runs}) == 9
    assert decision.admitted and len(decision.checks) == 5
    base_fail = sum(not t.outcome.passed for t, _, c in runs if c == "baseline")
    cand_fail = sum(not t.outcome.passed for t, _, c in runs if c == "candidate")
    assert (base_fail, cand_fail) == (3, 0)


def test_run_smoke_sends_labelled_events(tmp_path: Path) -> None:
    client = Client()
    sink = RaindropSink(
        client=client,
        ledger_path=tmp_path / "l.json",
        extra_properties={"smoke": True, "smoke_batch": "b1"},
    )
    assert smoke.run_smoke(sink, "b1") == 10
    assert sink.flush() is True
    assert len(client.events) == 10 and all(e["properties"]["smoke"] is True for e in client.events)
    assert {e["properties"]["experiment"] for e in client.events} == {"b1"}
    names = {s["name"] for s in client.signals}
    assert {"task_success", "task_failure", "guard_block", "admitted"} <= names


def test_delivery_recorder_notes_raindrop_posts_only(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resp:
        status_code = 200

        def __init__(self, body: Any) -> None:
            self.body = body

        def json(self) -> Any:
            if isinstance(self.body, Exception):
                raise self.body
            return self.body

    class Session:
        calls: list[str] = []

        def request(self, method: str, url: str, **kwargs: Any) -> Resp:
            Session.calls.append(url)
            if "other" in url:
                return Resp({})
            if url.endswith("/bad"):
                return Resp(ValueError("no json"))
            return Resp({"events": [{"event_id": "e1"}, {"event_id": "e2"}]})

    original = Session.request
    fake = types.SimpleNamespace(Session=Session)
    real = importlib.import_module
    monkeypatch.setattr("importlib.import_module", lambda n: fake if n == "requests.sessions" else real(n))
    rec = smoke.DeliveryRecorder()
    rec.install()
    session = Session()
    session.request(method="post", url="https://api.raindrop.ai/v1/events/track", json=[{"a": 1}, {"a": 2}])
    session.request("post", "https://api.raindrop.ai/v1/signals/track", json={"x": 1})
    session.request("post", "https://api.raindrop.ai/v1/bad", json=[])
    session.request("post", "https://other.example/other", json=[])
    rec.uninstall()
    rec.uninstall()
    assert Session.request is original
    assert len(rec.calls) == 3 and len(Session.calls) == 4
    summary = rec.summary()
    assert summary["/v1/events/track"] == {"posts": 1, "statuses": [200], "items": 2}
    assert rec.calls[0]["acknowledged_ids"] == ["e1", "e2"] and rec.calls[2]["acknowledged_ids"] == []


def test_main_without_a_key_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(smoke, "resolve_write_key", lambda use_keyring=False: None)
    assert smoke.main() == 2
    assert "no Raindrop write key" in capsys.readouterr().out


def test_main_reports_and_never_prints_the_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    client = Client()
    real_sink = RaindropSink

    def sink_factory(key: str, **kw: Any) -> RaindropSink:
        assert key == "SECRET-KEY-VALUE"
        return real_sink(
            client=client, ledger_path=tmp_path / "l.json", extra_properties=kw["extra_properties"]
        )

    monkeypatch.setattr(smoke, "resolve_write_key", lambda use_keyring=False: "SECRET-KEY-VALUE")
    monkeypatch.setattr(smoke, "RaindropSink", sink_factory)
    monkeypatch.setattr(smoke.DeliveryRecorder, "install", lambda self: None)
    monkeypatch.setattr(smoke.DeliveryRecorder, "uninstall", lambda self: None)
    assert smoke.main() == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["events_queued"] == 10 and report["events_sent"] == 10 and report["dropped"] == 0
    assert "SECRET-KEY-VALUE" not in out
    assert len(client.events) == 10 and client.events[0]["properties"]["smoke_batch"] == report["batch_id"]
