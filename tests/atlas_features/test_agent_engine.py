"""Policy Engine client: request shapes and the widening refusal, against a faithful in-memory gateway."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ratchet.atlas_features import agent_engine as ae


class FakeGateway:
    """Implements the documented merge rule: a project allowlist must be a subset of the org allowlist."""

    def __init__(self, org_tools: list[str]) -> None:
        self.org_tools = org_tools
        self.project: dict[str, dict[str, Any]] = {}
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        assert request.headers["Authorization"] == "Bearer tok"
        path = request.url.path
        if path == "/api/v1/projects/p1/policies" and request.method == "GET":
            return httpx.Response(200, json={"policies": list(self.project.values())})
        if path == "/api/v1/projects/p1/policies" and request.method == "POST":
            return self._save("pol1", body or {})
        if path.startswith("/api/v1/projects/p1/policies/pol1") and request.method == "PUT":
            return self._save("pol1", {**self.project["pol1"], **(body or {})})
        if path == "/api/v1/projects/p1/policies/effective":
            mine = self.project.get("pol1", {}).get("string_list")
            eff = None if mine is None else sorted(set(mine) & set(self.org_tools))
            return httpx.Response(
                200, json={"policies": [{"type": "AUTHORIZED_TOOLS", "effective_string_list": eff}]}
            )
        return httpx.Response(404, json={"error": "no route"})

    def _save(self, pid: str, body: dict[str, Any]) -> httpx.Response:
        extra = ae.widening(self.org_tools, body["string_list"])
        if extra:
            return httpx.Response(400, json={"error": f"project policy is less restrictive: {extra}"})
        self.project[pid] = {"id": pid, "type": "AUTHORIZED_TOOLS", **body}
        return httpx.Response(201, json=self.project[pid])


def engine_for(gateway: FakeGateway) -> ae.AgentEngine:
    client = httpx.Client(base_url=ae.GATEWAY, transport=httpx.MockTransport(gateway.handle))
    return ae.AgentEngine("tok", client=client)


def test_allowlist_policy_body_is_sorted_and_deduplicated() -> None:
    body = ae.tool_allowlist_policy(["b", "a", "b"])
    assert body == {"type": "AUTHORIZED_TOOLS", "string_list": ["a", "b"], "enabled": True}


def test_widening_names_exactly_the_added_tools() -> None:
    assert ae.widening(["a", "b"], ["a", "c"]) == ["c"]
    assert ae.widening(["a", "b"], ["b"]) == []


def test_install_creates_then_updates_one_policy() -> None:
    gw = FakeGateway(["a", "b", "c"])
    engine = engine_for(gw)
    ae.install_tool_allowlist(engine, "p1", ["a", "b"])
    ae.install_tool_allowlist(engine, "p1", ["a"])
    methods = [m for m, _p, _b in gw.requests if m in ("POST", "PUT")]
    assert methods == ["POST", "PUT"]
    assert gw.project["pol1"]["string_list"] == ["a"]


def test_widening_is_refused_with_the_server_body_and_effective_list_is_unchanged() -> None:
    gw = FakeGateway(["a", "b", "c"])
    steps = ae.prove_authority_cannot_grow(engine_for(gw), "p1", ["a", "b", "c"], ["a", "b"])
    assert steps["widen_refused"] is True
    assert steps["widen_status"] == 400
    assert "tool_not_in_org_policy" in steps["widen_body"]
    assert steps["effective_after_widen_attempt"]["policies"][0]["effective_string_list"] == ["a", "b"]


def test_a_gateway_that_accepts_widening_is_reported_as_failure() -> None:
    gw = FakeGateway(["a", "b", "c", "tool_not_in_org_policy"])
    steps = ae.prove_authority_cannot_grow(engine_for(gw), "p1", ["a", "b", "c"], ["a", "b"])
    assert steps["widen_refused"] is False


def test_service_account_exchange_uses_basic_auth_and_client_credentials() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"].startswith("Basic ")
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"access_token": "tok", "token_type": "Bearer"})

    http = httpx.Client(base_url=ae.GATEWAY, transport=httpx.MockTransport(handler))
    engine = ae.AgentEngine.from_service_account("ae_sa_id_x", "ae_sa_sk_y", client=http)
    assert seen == {"auth": True, "body": "grant_type=client_credentials"}
    assert isinstance(engine, ae.AgentEngine)


def test_service_account_exchange_failure_carries_the_server_response() -> None:
    http = httpx.Client(
        base_url=ae.GATEWAY,
        transport=httpx.MockTransport(lambda _r: httpx.Response(401, json={"error": "invalid_client"})),
    )
    with pytest.raises(ae.AgentEngineError) as err:
        ae.AgentEngine.from_service_account("id", "secret", client=http)
    assert err.value.status == 401
    assert "invalid_client" in err.value.body
    assert "secret" not in str(err.value)


def test_run_without_credentials_records_the_exact_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTENGINE_TOKEN", raising=False)
    monkeypatch.setattr(ae, "keyring", lambda _s, _a: "")
    monkeypatch.setattr(ae, "gateway_evidence", lambda: {"GET /health": {"status": 200}})
    result = ae.run()
    assert result.works is False
    assert result.error is not None
    assert "PROJECT_OWNER" in result.error
    assert result.pipeline["project_policy_body"]["type"] == "AUTHORIZED_TOOLS"


def test_revision_within_org_is_recorded_and_the_harness_list_is_restored() -> None:
    gw = FakeGateway(["a", "b", "c"])
    steps = ae.prove_authority_cannot_grow(engine_for(gw), "p1", ["a", "b", "c"], ["a", "b"])
    assert steps["revision_wider_within_org_accepted"] is True
    assert gw.project["pol1"]["string_list"] == ["a", "b"]


def test_procedure_write_tool_is_found_by_name_and_arguments_follow_the_schema() -> None:
    tools: list[dict[str, Any]] = [
        {"name": "search_semantic"},
        {
            "name": "save_procedure",
            "inputSchema": {"properties": {"name": {}, "steps": {}, "user_id": {}, "unrelated": {}}},
        },
    ]
    tool = ae.procedure_write_tool(tools)
    assert tool is not None
    args = ae.procedure_arguments(tool, ae.PROBE_PROCEDURE, "u1", "s1")
    assert args == {
        "name": "verify_refund_before_retry",
        "steps": ae.PROBE_PROCEDURE["steps"],
        "user_id": "u1",
    }
    assert ae.procedure_write_tool([{"name": "search_semantic"}]) is None
