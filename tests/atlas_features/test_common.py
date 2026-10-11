"""Shared plumbing: secrets, URI construction, connection failure, result files, index waits."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from atlas_fakes import FakeDb

from ratchet.atlas_features import common


def test_keyring_returns_stdout_and_empty_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = subprocess.CompletedProcess([], 0, stdout="value\n", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: ok)
    assert common.keyring("svc", "acct") == "value"

    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("no secret-tool")

    monkeypatch.setattr(subprocess, "run", boom)
    assert common.keyring("svc", "acct") == ""


def test_secret_prefers_the_environment_and_names_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_SECRET", "from-env")
    assert common.secret("X_SECRET", "svc", "acct") == "from-env"
    monkeypatch.delenv("X_SECRET")
    monkeypatch.setattr(common, "keyring", lambda _s, _a: "")
    with pytest.raises(common.ProbeUnavailableError, match=r"X_SECRET.*service=svc account=acct"):
        common.secret("X_SECRET", "svc", "acct")


def test_probe_uri_uses_an_explicit_uri_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATCHET_PROBES_URI", "mongodb://explicit")
    assert common.probe_uri() == "mongodb://explicit"


def test_probe_uri_builds_an_escaped_srv_uri_from_the_cluster_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RATCHET_PROBES_URI", raising=False)
    monkeypatch.setenv("RATCHET_DB_PASSWORD", "p@ss/word")
    monkeypatch.setattr(common, "keyring", lambda _s, _a: "")
    monkeypatch.setattr(common, "cluster_srv_host", lambda: "cluster0.example.mongodb.net")
    uri = common.probe_uri()
    assert uri.startswith("mongodb+srv://ratchet:p%40ss%2Fword@cluster0.example.mongodb.net/")


def test_cluster_srv_host_strips_the_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"connectionStrings": {"standardSrv": "mongodb+srv://cluster0.abc.mongodb.net"}}
    resp = httpx.Response(200, json=body, request=httpx.Request("GET", "https://x"))
    monkeypatch.setattr(common, "atlas_get", lambda _p: resp)
    assert common.cluster_srv_host() == "cluster0.abc.mongodb.net"


def test_atlas_admin_client_uses_digest_auth_and_the_pinned_accept_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATLAS_PUBLIC_KEY", "pub")
    monkeypatch.setenv("ATLAS_PRIVATE_KEY", "pri")
    with common.atlas_admin_client() as client:
        assert isinstance(client.auth, httpx.DigestAuth)
        assert client.headers["Accept"] == common.ATLAS_ACCEPT


def test_atlas_get_sends_the_requested_accept_header(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["accept"] = request.headers["Accept"]
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x")
    monkeypatch.setattr(common, "atlas_admin_client", lambda: client)
    common.atlas_get("/p", accept="application/json")
    assert seen["accept"] == "application/json"


def test_connect_fails_fast_with_a_message_that_holds_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATCHET_PROBES_URI", "mongodb://user:topsecret@127.0.0.1:1/?directConnection=true")
    with pytest.raises(common.ProbeUnavailableError) as err:
        common.connect(timeout_ms=200)
    assert "topsecret" not in str(err.value)
    assert "unreachable" in str(err.value)


def test_probe_result_saves_json_with_an_optional_suffix(tmp_path: Path) -> None:
    result = common.ProbeResult(probe="p", works=True, summary="s", data={"n": 1})
    path = result.save(tmp_path, suffix="-9.0.4")
    assert path.name == "p-9.0.4.json"
    assert json.loads(path.read_text())["data"] == {"n": 1}


def test_wait_queryable_returns_when_ready_and_times_out_with_the_states() -> None:
    db = FakeDb()
    coll = db["c"]
    coll.search_indexes = [{"name": "a", "queryable": True}]
    assert common.wait_queryable(db, "c", ["a"], timeout_s=5) >= 0.0  # type: ignore[arg-type]
    coll.search_indexes = [{"name": "a", "queryable": False, "status": "BUILDING"}]
    with pytest.raises(TimeoutError, match="BUILDING"):
        common.wait_queryable(db, "c", ["a", "b"], timeout_s=0.0)  # type: ignore[arg-type]


def test_embedder_closes_the_voyage_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    closed: list[bool] = []

    class V:
        model = "m"

        def __init__(self, api_key: str) -> None:
            assert api_key == "k"

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(common, "VoyageEmbedder", V)
    with common.embedder(FakeDb()) as emb:  # type: ignore[arg-type]
        assert emb.model == "m"
    assert closed == [True]


def test_probe_db_selects_the_featureprobe_database() -> None:
    class C:
        def __getitem__(self, name: str) -> str:
            return name

    assert common.probe_db(C()) == "ratchet_featureprobe"  # type: ignore[arg-type]
