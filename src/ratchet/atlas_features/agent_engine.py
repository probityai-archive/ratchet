"""Probe 4: the Atlas Agent Engine Policy Engine as a policy integration demonstration.

    uv run python -m ratchet.atlas_features.agent_engine

A harness's `tool_allowlist` becomes a project `AUTHORIZED_TOOLS` policy. The platform merges it with the
organization policy by intersection and rejects a project policy that is less restrictive than the
organization's, so MongoDB itself refuses a widening past the organization.

Scope of the claim: the platform enforces policies on executions that run on its Orchestration Engine.
Ratchet's worker runs outside it, so this proves the platform refuses a widening and mirrors an admitted
allowlist; it does not enforce Ratchet's own worker. The probe also tests temporal monotonicity: whether a
revision of a project policy must be stricter than its predecessor, which is a different rule from
stricter-than-organization.

The gateway accepts only a bearer token: a service account access token (`ae_sa_id_...` / `ae_sa_sk_...`
exchanged at `/api/v1/oauth/token`) or a user session token. Neither can be minted from the Atlas Admin API
key. Credentials are read from `AGENTENGINE_TOKEN`, or from the keyring entries `service=agentengine
account=sa-client-id` and `account=sa-client-secret`. Without them `run()` records the blocker, with the
gateway's own responses, instead of pretending.

Docs relied on (captured under research/_raw/atlas-probes/): the Policy Engine page and the gateway OpenAPI
document `agentengine-openapi.json`.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

from ratchet.atlas_features.common import ProbeResult, keyring, timed

GATEWAY = "https://agentengine.mongodb.com"
PREVIEW_ACCEPT = "application/vnd.agent-engine-2026-09-20-preview+json"
AUTHORIZED_TOOLS = "AUTHORIZED_TOOLS"


class AgentEngineError(RuntimeError):
    """The gateway refused or failed a request. `body` is the server response, never a credential."""

    def __init__(self, status: int, body: str, request: str) -> None:
        super().__init__(f"{request} -> HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body
        self.request = request


def tool_allowlist_policy(tool_allowlist: Iterable[str], *, enabled: bool = True) -> dict[str, Any]:
    """The `PolicyCreateRequest` body that expresses a harness `tool_allowlist` on the platform."""
    return {"type": AUTHORIZED_TOOLS, "string_list": sorted(set(tool_allowlist)), "enabled": enabled}


def widening(org_list: Iterable[str], project_list: Iterable[str]) -> list[str]:
    """Tools the project list adds beyond the org list: what the platform rejects. Empty means narrower."""
    return sorted(set(project_list) - set(org_list))


class AgentEngine:
    """A thin client for the policy routes of the gateway."""

    def __init__(self, token: str, *, client: httpx.Client | None = None, base_url: str = GATEWAY) -> None:
        self._client = client or httpx.Client(base_url=base_url, timeout=30.0)
        self._headers = {"Authorization": f"Bearer {token}", "Accept": PREVIEW_ACCEPT}

    @classmethod
    def from_service_account(
        cls, client_id: str, client_secret: str, *, client: httpx.Client | None = None
    ) -> AgentEngine:
        """Exchange client credentials for a one-hour bearer token (`POST /api/v1/oauth/token`)."""
        http = client or httpx.Client(base_url=GATEWAY, timeout=30.0)
        resp = http.post(
            "/api/v1/oauth/token",
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
        )
        if resp.status_code != 200:
            raise AgentEngineError(resp.status_code, resp.text, "POST /api/v1/oauth/token")
        return cls(str(resp.json()["access_token"]), client=http)

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        resp = self._client.request(method, path, headers=self._headers, json=body)
        if resp.status_code >= 400:
            raise AgentEngineError(resp.status_code, resp.text, f"{method} {path}")
        return resp.json() if resp.content else None

    def effective(self, project_id: str) -> Any:
        return self._call("GET", f"/api/v1/projects/{project_id}/policies/effective")

    def list_policies(self, scope: str, scope_id: str) -> Any:
        return self._call("GET", f"/api/v1/{scope}/{scope_id}/policies")

    def create_policy(self, scope: str, scope_id: str, body: dict[str, Any]) -> Any:
        return self._call("POST", f"/api/v1/{scope}/{scope_id}/policies", body)

    def update_policy(self, scope: str, scope_id: str, policy_id: str, body: dict[str, Any]) -> Any:
        return self._call("PUT", f"/api/v1/{scope}/{scope_id}/policies/{policy_id}", body)

    def delete_policy(self, scope: str, scope_id: str, policy_id: str) -> Any:
        return self._call("DELETE", f"/api/v1/{scope}/{scope_id}/policies/{policy_id}")

    def mcp(self, project_id: str, method: str, params: dict[str, Any] | None = None) -> Any:
        """One JSON-RPC message to the project's memory tool server (`POST /api/v1/projects/{id}/mcp`)."""
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        return self._call("POST", f"/api/v1/projects/{project_id}/mcp", body)

    def memory_search(self, project_id: str, body: dict[str, Any]) -> Any:
        """Search by `type` (semantic, episodic, procedural, taxonomic); needs `user_id` and `query`."""
        return self._call("POST", f"/api/v1/projects/{project_id}/memory/search", body)

    def preview_org_allowlist(self, org_id: str, tools: Iterable[str]) -> Any:
        body = {"type": AUTHORIZED_TOOLS, "string_list": sorted(set(tools))}
        return self._call("POST", f"/api/v1/organizations/{org_id}/policies/preview", body)


def install_tool_allowlist(
    engine: AgentEngine, scope_id: str, tool_allowlist: Iterable[str], scope: str = "projects"
) -> Any:
    """Create or replace the `AUTHORIZED_TOOLS` policy of a project (or, with `scope`, an organization)."""
    body = tool_allowlist_policy(tool_allowlist)
    for policy in engine.list_policies(scope, scope_id).get("policies", []):
        if policy.get("type") == AUTHORIZED_TOOLS:
            return engine.update_policy(
                scope,
                scope_id,
                str(policy["id"]),
                {"string_list": body["string_list"], "enabled": True},
            )
    return engine.create_policy(scope, scope_id, body)


def prove_authority_cannot_grow(
    engine: AgentEngine, project_id: str, org_tools: list[str], harness_tools: list[str]
) -> dict[str, Any]:
    """The refusal sequence: install a narrower project list, then try to widen past the org list.

    The org policy must already hold `org_tools` (an organization owner sets it; a service account cannot).
    Returns each step's request and the gateway's answer, including the error body of the refusal.
    """
    steps: dict[str, Any] = {"org_tools": org_tools, "harness_tools": harness_tools}
    steps["installed"] = install_tool_allowlist(engine, project_id, harness_tools)
    steps["effective_after_install"] = engine.effective(project_id)
    too_wide = sorted({*org_tools, "tool_not_in_org_policy"})
    try:
        install_tool_allowlist(engine, project_id, too_wide)
        steps["widen_refused"] = False
    except AgentEngineError as exc:
        steps["widen_refused"] = True
        steps["widen_attempt"] = too_wide
        steps["widen_status"] = exc.status
        steps["widen_body"] = exc.body
    steps["effective_after_widen_attempt"] = engine.effective(project_id)
    steps.update(probe_revision_monotonicity(engine, project_id, org_tools, harness_tools))
    return steps


def probe_revision_monotonicity(
    engine: AgentEngine, project_id: str, org_tools: list[str], harness_tools: list[str]
) -> dict[str, Any]:
    """Revise the project policy to a superset that is still inside the org list, then back.

    Accepted means the platform checks a revision only against the organization, not against the previous
    revision, so Ratchet's own parent-to-candidate subset check is the only temporal guard.
    """
    wider_within_org = sorted(set(org_tools))
    out: dict[str, Any] = {"revision_attempt": wider_within_org, "prior_revision": sorted(set(harness_tools))}
    try:
        install_tool_allowlist(engine, project_id, wider_within_org)
        out["revision_wider_within_org_accepted"] = True
    except AgentEngineError as exc:
        out["revision_wider_within_org_accepted"] = False
        out["revision_status"] = exc.status
        out["revision_body"] = exc.body
    install_tool_allowlist(engine, project_id, harness_tools)
    return out


_WRITE_HINTS = ("save", "write", "store", "create", "add", "put", "upsert")


def procedure_write_tool(tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The memory tool whose name says it writes a procedure, found from `tools/list`, or None."""
    for tool in tools:
        name = str(tool.get("name", "")).lower()
        if "procedur" in name and any(h in name for h in _WRITE_HINTS):
            return tool
    return None


def procedure_arguments(
    tool: dict[str, Any], procedure: dict[str, Any], user_id: str, session_id: str
) -> dict[str, Any]:
    """Fill a tool's input schema from a procedure by property name; ids come from the explicit identity."""
    values = {
        "name": procedure["name"],
        "title": procedure["name"],
        "description": procedure["when"],
        "when": procedure["when"],
        "trigger": procedure["when"],
        "steps": procedure["steps"],
        "content": "\n".join(procedure["steps"]),
        "user_id": user_id,
        "session_id": session_id,
        "metadata": procedure.get("metadata", {}),
    }
    props = dict(tool.get("inputSchema", {}).get("properties", {}))
    return {k: values[k] for k in props if k in values}


def procedural_roundtrip(
    engine: AgentEngine, project_id: str, procedure: dict[str, Any], user_id: str, session_id: str
) -> dict[str, Any]:
    """Write one admitted procedure through the memory tool server, then read it back by search."""
    listed = engine.mcp(project_id, "tools/list")
    tools = list(listed.get("result", {}).get("tools", []))
    tool = procedure_write_tool(tools)
    out: dict[str, Any] = {"tools": [t.get("name") for t in tools]}
    if tool is None:
        out["write"] = "no procedure-writing tool in tools/list"
        return out
    args = procedure_arguments(tool, procedure, user_id, session_id)
    out["write_tool"] = tool["name"]
    out["write_arguments"] = args
    out["write_response"] = engine.mcp(project_id, "tools/call", {"name": tool["name"], "arguments": args})
    out["search_response"] = engine.memory_search(
        project_id,
        {
            "type": "procedural",
            "query": procedure["when"],
            "user_id": user_id,
            "session_id": session_id,
            "top_k": 5,
        },
    )
    return out


def configured_engine() -> AgentEngine | None:
    """An engine from env or keyring credentials, or None when no credential exists."""
    token = os.environ.get("AGENTENGINE_TOKEN", "")
    if token:
        return AgentEngine(token)
    cid = keyring("agentengine", "sa-client-id")
    secret = keyring("agentengine", "sa-client-secret")
    if cid and secret:
        return AgentEngine.from_service_account(cid, secret)
    return None


def session_engine() -> AgentEngine | None:
    """An engine on the signed-in CLI user's session token, for the organization-scope policy only.

    The organization policy needs an `ORG_ADMIN` user: a service account cannot hold that role. The token is
    read from the CLI's own auth file (`AGENTENGINE_AUTH_FILE` overrides the path) and is never printed.
    """
    path = Path(os.environ.get("AGENTENGINE_AUTH_FILE", str(Path.home() / ".agentengine" / "auth.json")))
    try:
        creds = json.loads(path.read_text())["credentials"][GATEWAY]
        return AgentEngine(str(creds["access_token"]))
    except (OSError, KeyError, ValueError):
        return None


def gateway_evidence() -> dict[str, Any]:
    """What the gateway answers with no credential: proof it is live and what it demands."""
    out: dict[str, Any] = {}
    with httpx.Client(base_url=GATEWAY, timeout=15.0, follow_redirects=False) as http:
        for path in ("/health", "/api/v1/organizations", "/api/v1/projects/x/policies/effective"):
            resp = http.get(path)
            out[f"GET {path}"] = {"status": resp.status_code, "body": resp.text[:200]}
        resp = http.post("/api/v1/oauth/token", data={"grant_type": "client_credentials"})
        out["POST /api/v1/oauth/token (no client)"] = {"status": resp.status_code, "body": resp.text[:200]}
    return out


PROBE_PROCEDURE: dict[str, Any] = {
    "name": "verify_refund_before_retry",
    "when": "a refund call times out or its reply is lost",
    "steps": ["read the provider ledger for the charge", "retry only if no refund is recorded"],
}

BLOCKER = (
    "No Agent Engine credential. The gateway accepts only a bearer token: a service account access token "
    "or a signed-in user's session token. The Atlas Admin API key (digest auth) is not accepted. An operator "
    "must (1) sign in at https://agentengine.mongodb.com with the Atlas account that owns the hackathon "
    "organization, or run `agentengine auth login`; (2) create a project service account with role "
    "PROJECT_OWNER (UI: Service Accounts, or POST /api/v1/projects/{id}/service-accounts with the session "
    "token); (3) store its client id and secret in the keyring as service=agentengine account=sa-client-id "
    "and account=sa-client-secret. The organization AUTHORIZED_TOOLS policy needs an ORG_OWNER user session: "
    "a service account holds only ORG_GROUP_CREATOR or ORG_READ_ONLY at organization scope."
)


def run() -> ProbeResult:
    with timed() as elapsed:
        evidence = gateway_evidence()
        engine = configured_engine()
        if engine is None:
            return ProbeResult(
                probe="agent_engine_policy",
                works=False,
                summary="blocked: no operator-issued Agent Engine credential",
                latency_s=elapsed(),
                pipeline={"project_policy_body": tool_allowlist_policy(["lookup_order", "issue_refund"])},
                data={"gateway_without_credential": evidence},
                error=BLOCKER,
            )
        project_id = os.environ.get("AGENTENGINE_PROJECT_ID", "")
        org_tools = [t for t in os.environ.get("AGENTENGINE_ORG_TOOLS", "").split(",") if t]
        if not project_id or not org_tools:
            return ProbeResult(
                probe="agent_engine_policy",
                works=False,
                summary="credential present but AGENTENGINE_PROJECT_ID / AGENTENGINE_ORG_TOOLS unset",
                latency_s=elapsed(),
                data={"gateway_without_credential": evidence},
                error="set AGENTENGINE_PROJECT_ID and AGENTENGINE_ORG_TOOLS (comma separated)",
            )
        harness = org_tools[:-1] or org_tools
        org_id = os.environ.get("AGENTENGINE_ORG_ID", "")
        org_step: dict[str, Any] = {"skipped": "AGENTENGINE_ORG_ID unset"}
        if org_id:
            session = session_engine()
            try:
                if session is None:
                    raise AgentEngineError(0, "no CLI session in the auth file", "org policy")
                org_step = {"installed": install_tool_allowlist(session, org_id, org_tools, "organizations")}
            except AgentEngineError as exc:
                org_step = {"refused": exc.body, "status": exc.status}
        steps = prove_authority_cannot_grow(engine, project_id, org_tools, harness)
        try:
            steps["procedural_memory"] = procedural_roundtrip(
                engine, project_id, PROBE_PROCEDURE, "ratchet-probe", "ratchet-probe-session"
            )
        except AgentEngineError as exc:
            steps["procedural_memory"] = {"error": str(exc)}
        steps["org_policy"] = org_step
        return ProbeResult(
            probe="agent_engine_policy",
            works=bool(steps["widen_refused"]),
            summary="project policy installed; widening past the org policy "
            + ("refused" if steps["widen_refused"] else "ACCEPTED"),
            latency_s=elapsed(),
            data=steps,
        )


def main() -> int:
    result = run()
    path = result.save()
    print(f"{result.probe}: works={result.works} {result.summary} -> {path}")
    if result.error:
        print(result.error)
    return 0 if result.works else 1


if __name__ == "__main__":
    sys.exit(main())
