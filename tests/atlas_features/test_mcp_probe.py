"""MCP probe: JSON-RPC plumbing against a stand-in server (unit), the real pinned server (sandbox)."""

from __future__ import annotations

import sys

import pytest

from ratchet.atlas_features import mcp_probe

FAKE_SERVER = """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    if msg["method"] == "boom":
        out = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -1, "message": "no"}}
    else:
        out = {"jsonrpc": "2.0", "id": msg["id"], "result": {"echo": msg["method"]}}
    print(json.dumps(out), flush=True)
"""


def test_server_env_is_read_only_and_drops_unrelated_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_SECRET", "x")
    env = mcp_probe.server_env("mongodb://u:p@h/")
    assert env["MDB_MCP_READ_ONLY"] == "true"
    assert env["MDB_MCP_TELEMETRY"] == "disabled"
    assert "SOME_SECRET" not in env


def test_client_returns_results_and_raises_on_errors() -> None:
    client = mcp_probe.StdioClient([sys.executable, "-I", "-c", FAKE_SERVER], {})
    try:
        assert client.request("ping") == {"echo": "ping"}
        with pytest.raises(mcp_probe.McpError, match="no"):
            client.request("boom")
    finally:
        client.close()


def test_pinned_server_version_is_exact() -> None:
    assert "@3.0.5" in mcp_probe.SERVER_PACKAGE


@pytest.mark.sandbox
def test_pinned_server_reads_one_document_and_exposes_no_write_tool(sandbox_client: object) -> None:
    result = mcp_probe.run()
    assert result.works, result.error
    assert result.data["write_tools_exposed"] == []
