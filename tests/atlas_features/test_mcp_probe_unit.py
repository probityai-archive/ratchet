"""MCP probe orchestration against a stand-in server process and a fake database."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from atlas_fakes import FakeClient, FakeDb

from ratchet.atlas_features import mcp_probe
from ratchet.atlas_features.common import ProbeResult

SERVER = """
import json, sys
SENTINEL = "featureprobe-mcp-sentinel"
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method, params = msg["method"], msg.get("params", {})
    if method == "initialize":
        result = {"serverInfo": {"name": "fake", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "find"}, {"name": "aggregate"}]}
    else:
        db = params["arguments"]["database"]
        if db == "ratchet_heldout":
            result = {"isError": True, "content": [{"type": "text", "text": "not authorized"}]}
        else:
            result = {"content": [{"type": "text", "text": "resulted in 1 documents " + SENTINEL}]}
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
"""


def fake_server(write_tool: bool = False, heldout_allowed: bool = False) -> list[str]:
    code = SERVER
    if write_tool:
        code = code.replace('{"name": "aggregate"}', '{"name": "insert-many"}')
    if heldout_allowed:
        code = code.replace('if db == "ratchet_heldout":', "if False:")
    return [sys.executable, "-I", "-c", code]


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> FakeDb:
    db = FakeDb()
    monkeypatch.setattr(mcp_probe, "connect", lambda: _Client(db))
    monkeypatch.setattr(mcp_probe, "improver_uri", lambda: "mongodb://improver")
    return db


class _Client(FakeClient):
    def __init__(self, db: FakeDb) -> None:
        super().__init__()
        self._db = db

    def __getitem__(self, name: str) -> FakeDb:
        return self._db


def test_improver_read_succeeds_and_the_heldout_read_is_refused(
    monkeypatch: pytest.MonkeyPatch, wired: FakeDb
) -> None:
    monkeypatch.setattr(mcp_probe, "server_argv", fake_server)
    result = mcp_probe.run()
    assert result.works is True, result.error
    assert result.data["heldout_read_isError"] is True
    assert result.data["write_tools_exposed"] == []
    assert wired["traces"].docs == {}, "the sentinel trace must be removed afterwards"


def test_a_write_tool_in_the_listing_fails_the_probe(monkeypatch: pytest.MonkeyPatch, wired: FakeDb) -> None:
    monkeypatch.setattr(mcp_probe, "server_argv", lambda: fake_server(write_tool=True))
    result = mcp_probe.run()
    assert result.works is False
    assert result.data["write_tools_exposed"] == ["insert-many"]


def test_a_heldout_read_that_succeeds_fails_the_probe(monkeypatch: pytest.MonkeyPatch, wired: FakeDb) -> None:
    monkeypatch.setattr(mcp_probe, "server_argv", lambda: fake_server(heldout_allowed=True))
    result = mcp_probe.run()
    assert result.works is False
    assert "ALLOWED" in result.summary


def test_a_server_that_exits_is_reported_as_a_failed_read(
    monkeypatch: pytest.MonkeyPatch, wired: FakeDb
) -> None:
    monkeypatch.setattr(mcp_probe, "server_argv", lambda: [sys.executable, "-I", "-c", "pass"])
    result = mcp_probe.run()
    assert result.works is False
    assert result.error is not None
    assert "server exited" in result.error


def test_improver_uri_escapes_the_password_and_names_the_restricted_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RATCHET_IMPROVER_PASSWORD", "a/b@c")
    monkeypatch.setattr(mcp_probe, "cluster_srv_host", lambda: "cluster0.example.net")
    uri = mcp_probe.improver_uri()
    assert uri == (
        "mongodb+srv://ratchet-improver:a%2Fb%40c@cluster0.example.net/?retryWrites=true&w=majority"
    )


def test_server_argv_points_at_the_private_install() -> None:
    argv = mcp_probe.server_argv()
    assert argv[0] == "node"
    assert Path(argv[1]).is_relative_to(mcp_probe.INSTALL_DIR)


def test_tool_text_joins_content_blocks() -> None:
    assert mcp_probe.tool_text({"content": [{"text": "a"}, {"text": "b"}]}) == "a b"
    assert mcp_probe.tool_text({}) == ""


def test_main_returns_the_exit_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad: Any = ProbeResult(probe="mcp_server", works=False, summary="no", error="why")
    monkeypatch.setattr(mcp_probe, "run", lambda: bad)
    monkeypatch.setattr(ProbeResult, "save", lambda self, *_a, **_k: tmp_path / "x.json")
    assert mcp_probe.main() == 1
    assert "why" in capsys.readouterr().out
