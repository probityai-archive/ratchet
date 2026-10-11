"""Trigger probe: Admin API client and orchestration against a fake App Services API and fake collections."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from atlas_fakes import FakeClient, FakeCollection, FakeDb

from ratchet.atlas_features import triggers
from ratchet.atlas_features.common import ProbeResult


class FakeAdminApi:
    """Stores apps, services, functions and triggers; answers like the Admin API."""

    def __init__(self, existing: bool = False) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.store: dict[str, list[dict[str, Any]]] = {
            "apps": [{"_id": "app1", "name": triggers.APP_NAME, "client_app_id": "c"}] if existing else [],
            "services": [{"_id": "svc1", "name": "mongodb-atlas"}] if existing else [],
            "functions": [{"_id": "fn1", "name": triggers.FUNCTION_NAME}] if existing else [],
            "triggers": [{"_id": "tr1", "name": triggers.TRIGGER_NAME}] if existing else [],
        }
        self.fail_login = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.calls.append((request.method, f"{path}?{request.url.query.decode()}", body))
        if path.endswith("/auth/providers/mongodb-cloud/login"):
            if self.fail_login:
                return httpx.Response(401, text="bad key")
            return httpx.Response(200, json={"access_token": "tok"})
        assert request.headers["Authorization"] == "Bearer tok"
        if path.endswith("/debug/execute_function"):
            return httpx.Response(200, json={"result": {"id": "fm:x", "inserted": False}})
        if path.endswith("/boom"):
            return httpx.Response(500, text="server error")
        kind = [seg for seg in path.split("/") if seg in self.store][-1]
        if request.method == "GET":
            return httpx.Response(200, json=self.store[kind])
        if request.method == "POST":
            doc = {"_id": f"{kind}-new", **(body or {})}
            self.store[kind].append(doc)
            return httpx.Response(201, json=doc)
        return httpx.Response(204)


def make(api: FakeAdminApi, monkeypatch: pytest.MonkeyPatch) -> triggers.AppServices:
    monkeypatch.setenv("ATLAS_PUBLIC_KEY", "pub")
    monkeypatch.setenv("ATLAS_PRIVATE_KEY", "pri")
    return triggers.AppServices(httpx.Client(transport=httpx.MockTransport(api.handle)))


def test_login_failure_names_the_status_but_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAdminApi()
    api.fail_login = True
    with pytest.raises(triggers.AppServicesError) as err:
        make(api, monkeypatch)
    assert "401" in str(err.value)
    assert "pri" not in str(err.value)


def test_ensure_creates_each_resource_once_and_reuses_it_after(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAdminApi()
    svc = make(api, monkeypatch)
    app = triggers.ensure_app(svc)
    service_id = triggers.ensure_service(svc, app["_id"])
    function_id = triggers.ensure_function(svc, app["_id"])
    triggers.ensure_trigger(svc, app["_id"], service_id, function_id)
    posts = [c for c in api.calls if c[0] == "POST" and "login" not in c[1]]
    assert len(posts) == 4
    again = FakeAdminApi(existing=True)
    svc2 = make(again, monkeypatch)
    assert triggers.ensure_app(svc2)["_id"] == "app1"
    assert triggers.ensure_service(svc2, "app1") == "svc1"
    assert triggers.ensure_function(svc2, "app1") == "fn1"
    assert triggers.ensure_trigger(svc2, "app1", "svc1", "fn1") == "tr1"
    assert [c for c in again.calls if c[0] == "POST" and "login" not in c[1]] == []
    assert [c[0] for c in again.calls if c[0] == "PUT"] == ["PUT", "PUT"]


def test_a_refused_request_carries_method_path_status_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = make(FakeAdminApi(), monkeypatch)
    with pytest.raises(triggers.AppServicesError, match=r"GET /boom -> HTTP 500: server error"):
        svc.call("GET", "/boom")
    svc.close()


def test_execute_function_asks_to_run_as_system(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAdminApi()
    out = triggers.execute_function(make(api, monkeypatch), "app1", {"fullDocument": {}})
    assert out["result"]["inserted"] is False
    assert "run_as_system=true" in api.calls[-1][1]


def test_ensure_memory_index_creates_the_collection_and_the_auto_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(triggers, "wait_queryable", lambda *_a, **_k: 0.0)
    db = FakeDb()
    triggers.ensure_memory_index(db)  # type: ignore[arg-type]
    assert db.created == [triggers.MEMORY]
    assert db[triggers.MEMORY].created_indexes[0].document["name"] == triggers.MEMORY_INDEX


def test_wait_for_finds_a_document_and_times_out() -> None:
    coll = FakeCollection("c", FakeDb())
    coll.docs["a"] = {}
    assert triggers.wait_for(coll, "a", 5) is not None  # type: ignore[arg-type]
    assert triggers.wait_for(coll, "missing", 0.0) is None  # type: ignore[arg-type]


def mirrored_db() -> FakeDb:
    """A db whose source inserts are mirrored into the memory collection, as the trigger would."""
    db = FakeDb()
    src = db[triggers.SOURCE]
    mem = db[triggers.MEMORY]
    orig = src.insert_one

    def insert(doc: dict[str, Any]) -> None:
        orig(doc)
        key = f"fm:{doc['failure_id']}"
        if key not in mem.docs:
            text = doc["summary"]
            for raw, clean in (
                ("ch_204", "<id>"),
                ("7781", "<id>"),
                ("$1,250.00", "<amount>"),
                ("bob@example.com", "<email>"),
            ):
                text = text.replace(raw, clean)
            mem.docs[key] = {"failure_id": doc["failure_id"], "summary": text}

    src.insert_one = insert  # type: ignore[method-assign]
    mem.aggregate_fn = lambda _p: [dict(next(iter(mem.docs.values())))] if mem.docs else []
    return db


def test_prove_checks_latency_sanitization_dedupe_and_redelivery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr(triggers, "execute_function", lambda *_a, **_k: {"result": {"inserted": False}})
    proof = triggers.prove(mirrored_db(), None, "app1")  # type: ignore[arg-type]
    assert proof["sanitized"] is True
    assert proof["count_after_second_insert_of_same_failure"] == 1
    assert proof["count_after_duplicate_delivery"] == 1
    assert proof["memory_appeared_s"] is not None


def test_retrieval_check_finds_the_inserted_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    out = triggers.retrieval_check(mirrored_db())  # type: ignore[arg-type]
    assert out["searchable_by_meaning_s"] is not None


def run_with(monkeypatch: pytest.MonkeyPatch, retrieval_ok: bool) -> ProbeResult:
    svc = make(FakeAdminApi(), monkeypatch)
    monkeypatch.setattr(triggers, "connect", lambda: FakeClient())
    monkeypatch.setattr(triggers, "probe_db", lambda _c: mirrored_db())
    monkeypatch.setattr(triggers, "AppServices", lambda: svc)
    monkeypatch.setattr(triggers, "wait_queryable", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(triggers, "execute_function", lambda *_a, **_k: {"result": {"inserted": False}})
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        triggers, "retrieval_check", lambda _db: {"searchable_by_meaning_s": 1.0 if retrieval_ok else None}
    )
    return triggers.run()


def test_run_reports_success_when_every_check_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    result = run_with(monkeypatch, True)
    assert result.works is True, result.data
    assert result.pipeline["trigger"]["config"]["operation_types"] == ["INSERT"]


def test_run_reports_failure_when_retrieval_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    result = run_with(monkeypatch, False)
    assert result.works is False
    assert result.error == "see data.proof"


def test_run_turns_an_admin_api_refusal_into_a_recorded_result(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAdminApi()
    api.fail_login = True
    monkeypatch.setenv("ATLAS_PUBLIC_KEY", "pub")
    monkeypatch.setenv("ATLAS_PRIVATE_KEY", "pri")
    monkeypatch.setattr(triggers, "connect", lambda: FakeClient())
    client = httpx.Client(transport=httpx.MockTransport(api.handle))
    real = triggers.AppServices
    monkeypatch.setattr(triggers, "AppServices", lambda: real(client))
    result = triggers.run()
    assert result.works is False
    assert result.error is not None
    assert "401" in result.error


def test_main_returns_the_exit_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = ProbeResult(probe="triggers", works=True, summary="fine")
    monkeypatch.setattr(triggers, "run", lambda: ok)
    monkeypatch.setattr(ProbeResult, "save", lambda self, *_a, **_k: tmp_path / "x.json")
    assert triggers.main() == 0
    bad = ProbeResult(probe="triggers", works=False, summary="no", error="why")
    monkeypatch.setattr(triggers, "run", lambda: bad)
    assert triggers.main() == 1
    assert "why" in capsys.readouterr().out
