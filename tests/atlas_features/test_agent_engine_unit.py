"""Agent Engine probe: client routes, credential discovery, evidence capture and the full run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ratchet.atlas_features import agent_engine as ae
from ratchet.atlas_features.common import ProbeResult


def recording_engine(
    responses: dict[tuple[str, str], Any],
) -> tuple[ae.AgentEngine, list[tuple[str, str, Any]]]:
    seen: list[tuple[str, str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        out = responses.get((request.method, request.url.path), {})
        return httpx.Response(200, json=out)

    client = httpx.Client(base_url=ae.GATEWAY, transport=httpx.MockTransport(handler))
    return ae.AgentEngine("tok", client=client), seen


def test_client_routes_and_bodies_match_the_documented_endpoints() -> None:
    engine, seen = recording_engine({})
    engine.list_policies("projects", "p1")
    engine.update_policy("projects", "p1", "pol", {"enabled": True})
    engine.delete_policy("organizations", "o1", "pol")
    engine.preview_org_allowlist("o1", ["b", "a", "a"])
    engine.mcp("p1", "tools/list")
    engine.memory_search("p1", {"type": "procedural", "query": "q", "user_id": "u"})
    assert seen == [
        ("GET", "/api/v1/projects/p1/policies", None),
        ("PUT", "/api/v1/projects/p1/policies/pol", {"enabled": True}),
        ("DELETE", "/api/v1/organizations/o1/policies/pol", None),
        (
            "POST",
            "/api/v1/organizations/o1/policies/preview",
            {"type": "AUTHORIZED_TOOLS", "string_list": ["a", "b"]},
        ),
        (
            "POST",
            "/api/v1/projects/p1/mcp",
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        ),
        ("POST", "/api/v1/projects/p1/memory/search", {"type": "procedural", "query": "q", "user_id": "u"}),
    ]


def test_an_empty_success_body_returns_none() -> None:
    client = httpx.Client(base_url=ae.GATEWAY, transport=httpx.MockTransport(lambda _r: httpx.Response(204)))
    assert ae.AgentEngine("tok", client=client).delete_policy("projects", "p", "x") is None


def test_install_updates_the_existing_authorized_tools_policy() -> None:
    existing = {"policies": [{"id": "pol9", "type": "AUTHORIZED_TOOLS"}]}
    engine, seen = recording_engine({("GET", "/api/v1/projects/p1/policies"): existing})
    ae.install_tool_allowlist(engine, "p1", ["b", "a"])
    assert seen[-1] == (
        "PUT",
        "/api/v1/projects/p1/policies/pol9",
        {"string_list": ["a", "b"], "enabled": True},
    )


def test_procedural_roundtrip_writes_through_the_tool_and_searches_back() -> None:
    tools = {
        "result": {
            "tools": [{"name": "save_procedure", "inputSchema": {"properties": {"name": {}, "user_id": {}}}}]
        }
    }
    engine, seen = recording_engine({("POST", "/api/v1/projects/p1/mcp"): tools})
    out = ae.procedural_roundtrip(engine, "p1", ae.PROBE_PROCEDURE, "u1", "s1")
    assert out["write_tool"] == "save_procedure"
    assert out["write_arguments"] == {"name": "verify_refund_before_retry", "user_id": "u1"}
    assert seen[-1][1] == "/api/v1/projects/p1/memory/search"
    assert seen[-1][2]["type"] == "procedural"
    assert seen[-1][2]["user_id"] == "u1"


def test_procedural_roundtrip_reports_a_server_without_a_write_tool() -> None:
    engine, _ = recording_engine({("POST", "/api/v1/projects/p1/mcp"): {"result": {"tools": []}}})
    out = ae.procedural_roundtrip(engine, "p1", ae.PROBE_PROCEDURE, "u", "s")
    assert out["write"] == "no procedure-writing tool in tools/list"


def test_configured_engine_prefers_the_token_then_exchanges_keyring_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTENGINE_TOKEN", "t")
    assert isinstance(ae.configured_engine(), ae.AgentEngine)
    monkeypatch.delenv("AGENTENGINE_TOKEN")
    monkeypatch.setattr(ae, "keyring", lambda _s, _a: "")
    assert ae.configured_engine() is None
    monkeypatch.setattr(ae, "keyring", lambda _s, account: f"value-{account}")
    exchanged: list[tuple[str, str]] = []

    def fake_exchange(cid: str, secret: str, **_k: Any) -> str:
        exchanged.append((cid, secret))
        return "engine"

    monkeypatch.setattr(ae.AgentEngine, "from_service_account", staticmethod(fake_exchange))
    assert ae.configured_engine() == "engine"  # type: ignore[comparison-overlap]
    assert exchanged == [("value-sa-client-id", "value-sa-client-secret")]


def test_gateway_evidence_records_each_unauthenticated_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401 if "/api/" in request.url.path else 200, text=request.url.path)

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **k: real(transport=httpx.MockTransport(handler), **k))
    out = ae.gateway_evidence()
    assert out["GET /health"]["status"] == 200
    assert out["GET /api/v1/organizations"]["status"] == 401
    assert "POST /api/v1/oauth/token (no client)" in out


class StubEngine:
    """Records the calls `run()` makes and refuses a widening like the platform does."""

    def __init__(self) -> None:
        self.installed: list[list[str]] = []

    def list_policies(self, *_a: Any) -> dict[str, Any]:
        return {"policies": []}

    def create_policy(self, _scope: str, _sid: str, body: dict[str, Any]) -> dict[str, Any]:
        tools = body["string_list"]
        if "tool_not_in_org_policy" in tools:
            raise ae.AgentEngineError(400, "less restrictive than the organization policy", "POST x")
        self.installed.append(tools)
        return {"id": "pol1", **body}

    def effective(self, _pid: str) -> dict[str, Any]:
        return {"policies": []}

    def mcp(self, *_a: Any, **_k: Any) -> dict[str, Any]:
        raise ae.AgentEngineError(403, "forbidden", "POST mcp")


def test_run_without_a_credential_is_blocked_with_the_gateway_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ae, "configured_engine", lambda: None)
    monkeypatch.setattr(ae, "gateway_evidence", lambda: {"GET /health": {"status": 200}})
    result = ae.run()
    assert result.works is False
    assert result.data["gateway_without_credential"]["GET /health"]["status"] == 200


def test_run_with_a_credential_but_no_project_names_the_missing_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ae, "configured_engine", lambda: StubEngine())
    monkeypatch.setattr(ae, "gateway_evidence", lambda: {})
    monkeypatch.delenv("AGENTENGINE_PROJECT_ID", raising=False)
    result = ae.run()
    assert result.works is False
    assert result.error is not None
    assert "AGENTENGINE_PROJECT_ID" in result.error


def test_run_proves_the_widening_refusal_and_records_a_failed_memory_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ae, "configured_engine", lambda: StubEngine())
    monkeypatch.setattr(ae, "gateway_evidence", lambda: {})
    monkeypatch.setenv("AGENTENGINE_PROJECT_ID", "p1")
    monkeypatch.setenv("AGENTENGINE_ORG_TOOLS", "a,b,c")
    result = ae.run()
    assert result.works is True
    assert result.data["widen_refused"] is True
    assert "forbidden" in result.data["procedural_memory"]["error"]


def test_main_prints_the_blocker_and_returns_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ae, "configured_engine", lambda: None)
    monkeypatch.setattr(ae, "gateway_evidence", lambda: {})
    monkeypatch.setattr(ProbeResult, "save", lambda self, *_a, **_k: tmp_path / "x.json")
    assert ae.main() == 1
    assert "PROJECT_OWNER" in capsys.readouterr().out


def test_session_engine_reads_the_cli_auth_file_and_tolerates_its_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"credentials": {ae.GATEWAY: {"access_token": "t"}}}))
    monkeypatch.setenv("AGENTENGINE_AUTH_FILE", str(auth))
    assert isinstance(ae.session_engine(), ae.AgentEngine)
    auth.write_text("not json")
    assert ae.session_engine() is None
    monkeypatch.setenv("AGENTENGINE_AUTH_FILE", str(tmp_path / "missing.json"))
    assert ae.session_engine() is None


def test_install_can_target_the_organization_scope() -> None:
    engine, seen = recording_engine({})
    ae.install_tool_allowlist(engine, "o1", ["a"], "organizations")
    assert seen[-1][:2] == ("POST", "/api/v1/organizations/o1/policies")


def test_run_records_the_organization_policy_step_and_its_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ae, "configured_engine", lambda: StubEngine())
    monkeypatch.setattr(ae, "gateway_evidence", lambda: {})
    monkeypatch.setenv("AGENTENGINE_PROJECT_ID", "p1")
    monkeypatch.setenv("AGENTENGINE_ORG_TOOLS", "a,b,c")
    monkeypatch.setenv("AGENTENGINE_ORG_ID", "o1")
    monkeypatch.setattr(ae, "session_engine", lambda: None)
    result = ae.run()
    assert result.data["org_policy"]["status"] == 0
    assert "no CLI session" in result.data["org_policy"]["refused"]
    session = StubEngine()
    monkeypatch.setattr(ae, "session_engine", lambda: session)
    result = ae.run()
    assert "installed" in result.data["org_policy"]
