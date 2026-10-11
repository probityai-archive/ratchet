"""The improver's database role can work in the main database and cannot read held-out cases."""

from __future__ import annotations

import secrets
import statistics
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
import pytest
from conftest import HashEmbedder
from pymongo.errors import OperationFailure

from ratchet.store import Store, VoyageEmbedder, atlas_admin
from ratchet.types import Cost, Event, Harness, Outcome, Trace

PRIVATE = Path(__file__).resolve().parents[2] / "private"
UNAUTHORIZED = 13
_REAL_CLIENT = httpx.Client


def _case(case_id: str, split: str) -> dict[str, object]:
    return {"id": case_id, "split": split, "family": "refund", "customer_message": "refund me"}


def _seed() -> Harness:
    return Harness.build(
        parent=None, system_prompt="s", procedures=[], guards=[], tool_allowlist=[], created_by="seed"
    )


def _trace() -> Trace:
    out = Outcome(passed=True, reasons=[], final_state_digest="d", cost=Cost())
    return Trace("r", "c", "v", "m", [Event(0, "final", {}, 0.0)], out)


def _with_user(uri: str, user: str, password: str) -> str:
    host = urlsplit(uri).hostname
    port = urlsplit(uri).port
    return (
        f"mongodb://{user}:{quote(password, safe='')}@{host}:{port}/?directConnection=true&authSource=admin"
    )


@pytest.fixture
def improver_uri(atlas_uri: str, store: Store) -> Iterator[str]:
    """A local user whose role mirrors ratchetImprover: readWrite on the main database only."""
    admin = store._client.admin
    role, user, password = (
        f"improver_{uuid.uuid4().hex[:8]}",
        f"u_{uuid.uuid4().hex[:8]}",
        secrets.token_hex(12),
    )
    admin.command("createRole", role, privileges=[], roles=[{"role": "readWrite", "db": store.db.name}])
    admin.command("createUser", user, pwd=password, roles=[{"role": role, "db": "admin"}])
    yield _with_user(atlas_uri, user, password)
    admin.command("dropUser", user)
    admin.command("dropRole", role)


def test_improver_works_in_the_main_database_and_is_denied_held_out_cases(
    store: Store, improver_uri: str, hash_embedder: HashEmbedder
) -> None:
    store.put_case(_case("train-1", "train"))
    store.put_case(_case("held-1", "heldout"))
    assert [c["_id"] for c in store.cases("heldout")] == ["held-1"]
    with Store(improver_uri, store.db.name, embedder=hash_embedder) as improver:
        seed = _seed()
        assert improver.put_harness(seed) and improver.put_trace(_trace())
        assert [c["_id"] for c in improver.cases("train")] == ["train-1"]
        with pytest.raises(OperationFailure) as read_denied:
            improver.cases("heldout")
        assert read_denied.value.code == UNAUTHORIZED
        with pytest.raises(OperationFailure) as write_denied:
            improver.put_case(_case("held-2", "heldout"))
        assert write_denied.value.code == UNAUTHORIZED
    assert store.heldout["cases"].count_documents({}) == 1


def test_put_case_validates_and_is_idempotent(store: Store) -> None:
    assert store.put_case(_case("a", "train")) == "a"
    assert store.put_case(_case("a", "train")) == "a"
    assert len(store.cases("train")) == 1
    with pytest.raises(TypeError, match="split"):
        store.put_case({"id": "x", "split": "dev"})
    with pytest.raises(Exception, match="different content"):
        store.put_case({**_case("a", "train"), "family": "other"})


def test_admin_request_bodies_grant_only_the_main_database() -> None:
    role = atlas_admin.improver_role_body("ratchet")
    assert role["inheritedRoles"] == [{"db": "ratchet", "role": "readWrite"}] and role["actions"] == []
    user = atlas_admin.improver_user_body("g1", "pw")
    assert user["roles"] == [{"databaseName": "admin", "roleName": atlas_admin.IMPROVER_ROLE}]


def test_admin_main_requires_its_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for key in ("ATLAS_PUBLIC_KEY", "ATLAS_PRIVATE_KEY", "ATLAS_GROUP_ID", "RATCHET_IMPROVER_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    assert atlas_admin.main() == 2
    assert "ATLAS_GROUP_ID" in capsys.readouterr().err


def _keyring(account: str) -> str:
    out = subprocess.run(
        ["secret-tool", "lookup", "service", "mongodb", "account", account], capture_output=True, text=True
    ).stdout.strip()
    assert out, f"keyring entry {account} is missing"
    return out


@pytest.mark.sandbox
def test_sandbox_improver_cannot_read_held_out_cases_on_atlas() -> None:
    admin_uri = _keyring("hackathon-uri")
    host = urlsplit(admin_uri).hostname
    improver_uri = (
        f"mongodb+srv://{atlas_admin.IMPROVER_USER}:{quote(_keyring('hackathon-db-improver'), safe='')}"
        f"@{host}/?retryWrites=true&w=majority"
    )
    probe = f"sandbox-probe-{uuid.uuid4().hex[:8]}"
    with Store(admin_uri, "ratchet", embedder=HashEmbedder()) as admin:
        admin.put_case(_case(probe, "heldout"))
        try:
            with Store(improver_uri, "ratchet", embedder=HashEmbedder()) as improver:
                improver.put_case(_case(probe + "-train", "train"))
                with pytest.raises(OperationFailure) as denied:
                    improver.cases("heldout")
                assert denied.value.code == UNAUTHORIZED
                assert improver.cases("train")
                improver.db["cases"].delete_one({"_id": probe + "-train"})
        finally:
            admin.heldout["cases"].delete_one({"_id": probe})


@pytest.mark.sandbox
def test_sandbox_vector_search_round_trip_and_latency() -> None:
    uri = _keyring("hackathon-uri")
    name = f"ratchet_probe_{uuid.uuid4().hex[:8]}"
    texts = [
        ("refund timed out after the provider committed and the agent refunded again", "refund", "f1"),
        ("agent emailed the customer before the approval was granted", "approval", "f2"),
        ("ticket closed while the payment dispute was still open", "dispute", "f3"),
    ]
    import os

    key = os.environ.get("VOYAGE_API_KEY")
    embedder = VoyageEmbedder() if key else HashEmbedder()
    closer = getattr(embedder, "close", lambda: None)
    with Store(uri, name, embedder=embedder) as s:
        try:
            for text, family, fid in texts:
                s.put_failure(text, case_family=family, failure_id=fid)
            s.setup()
            query = "a payment was refunded twice after a timeout"
            hits = []
            for _ in range(120):
                hits = s.similar_failures(query, 3)
                if hits and hits[0].id == "f1":
                    break
                time.sleep(1)
            assert hits[0].id == "f1"
            vector = s.embedder.embed([query], "query")[0]
            raw: list[float] = []
            for _ in range(30):
                t0 = time.perf_counter()
                s._vector_hits(vector, 3, {}, None)
                raw.append((time.perf_counter() - t0) * 1000)
            e2e: list[float] = []
            for i in range(10):
                t0 = time.perf_counter()
                s.similar_failures(f"{query} {i}", 3)
                e2e.append((time.perf_counter() - t0) * 1000)
            hybrid = s.similar_failures_hybrid(query, 3)
            assert hybrid[0].id == "f1"
            PRIVATE.mkdir(exist_ok=True)
            (PRIVATE / "sandbox-latency.txt").write_text(
                f"embedder={type(embedder).__name__}\n"
                f"vector_search_ms median={statistics.median(raw):.1f} p95={sorted(raw)[28]:.1f} n=30\n"
                f"e2e_ms median={statistics.median(e2e):.1f} max={max(e2e):.1f} n=10 (embed+search)\n"
            )
        finally:
            for coll in s.db.list_collection_names():
                s.db.drop_collection(coll)
            closer()


def _api(handler: Any) -> httpx.Client:
    return _REAL_CLIENT(transport=httpx.MockTransport(handler))


def test_create_improver_reports_created_and_exists_and_hides_error_bodies() -> None:
    codes = iter([201, 201, 409, 409])
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(next(codes), json={})

    with _api(handler) as client:
        assert atlas_admin.create_improver(client, "g1", "ratchet", "pw") == ("created", "created")
        assert atlas_admin.create_improver(client, "g1", "ratchet", "pw") == ("exists", "exists")
    assert paths[0].endswith("/customDBRoles/roles") and paths[1].endswith("/databaseUsers")

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"errorCode": "INVALID", "detail": "echo pw-secret"})

    with _api(failing) as client, pytest.raises(RuntimeError, match="INVALID") as err:
        atlas_admin.create_improver(client, "g1", "ratchet", "pw-secret")
    assert "pw-secret" not in str(err.value)


def test_admin_main_runs_against_the_api(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for key, value in {
        "ATLAS_PUBLIC_KEY": "a",
        "ATLAS_PRIVATE_KEY": "b",
        "ATLAS_GROUP_ID": "g",
        "RATCHET_IMPROVER_PASSWORD": "p",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(httpx, "Client", lambda **kw: _api(lambda r: httpx.Response(201, json={})))
    assert atlas_admin.main() == 0
    assert "created" in capsys.readouterr().out


def test_module_entry_point_requires_a_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    monkeypatch.delenv("RATCHET_MONGODB_URI", raising=False)
    monkeypatch.setattr(sys, "argv", ["ratchet.store"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("ratchet.store", run_name="__main__")
    assert exit_info.value.code == 2


@pytest.mark.sandbox
def test_sandbox_change_stream_delivers_inserts_on_atlas() -> None:
    import threading

    name = f"ratchet_probe_{uuid.uuid4().hex[:8]}"
    seen: list[str] = []
    with Store(_keyring("hackathon-uri"), name, embedder=HashEmbedder()) as s:
        s.setup(wait_search=False)
        try:
            reader = threading.Thread(
                target=lambda: seen.extend(e.doc_id for e in s.watch("harness_versions", timeout_s=15))
            )
            reader.start()
            time.sleep(3)
            seed = _seed()
            s.put_harness(seed)
            reader.join()
            assert seen == [seed.version]
        finally:
            for coll in s.db.list_collection_names():
                s.db.drop_collection(coll)
