from __future__ import annotations

import io
import json
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from ratchet.agent.guards import GuardRuntime
from ratchet.export.mcp_gateway import Gateway, main
from ratchet.improve.admit import Check, Decision
from ratchet.improve.receipt import Signer, public_key_bytes, receipt
from ratchet.types import Guard, Harness

NO_DELETE = """
def check(call, view, history):
    if call.name == "delete_file":
        return "deleting files is not allowed"
    return None
"""
AFTER_FAILURE = """
def check(call, view, history):
    failed = [e for e in history if e.kind == "tool_result" and not e.payload["result"]["ok"]]
    if call.name == "write_file" and failed:
        return "a call failed earlier (" + failed[-1].payload["result"]["call_id"] + "); read before writing"
    return None
"""
CRASH = "def check(call, view, history):\n    raise RuntimeError('boom')\n"


def tools_call(rid: Any, name: str, args: Any = None) -> dict[str, Any]:
    params: dict[str, Any] = {"name": name}
    if args is not None:
        params["arguments"] = args
    return {"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": params}


def line(message: Any) -> bytes:
    return (json.dumps(message) + "\n").encode()


class Relay:
    def __init__(self, guards: list[Guard], runtime: GuardRuntime) -> None:
        self.client: list[Any] = []
        self.server: list[Any] = []
        self.gw = Gateway(guards, runtime, self._to_client, self._to_server)

    def _to_client(self, data: bytes) -> None:
        try:
            self.client.append(json.loads(data))
        except ValueError:
            self.client.append(data)

    def _to_server(self, data: bytes) -> None:
        self.server.append(data)


@pytest.fixture(scope="module")
def runtime() -> Iterator[GuardRuntime]:
    with GuardRuntime(workers=1) as rt:
        yield rt


def test_gateway_forwards_allowed_calls_and_answers_blocked_ones(runtime: GuardRuntime) -> None:
    r = Relay([Guard("no-delete", "", NO_DELETE), Guard("after-failure", "", AFTER_FAILURE)], runtime)
    init = line({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
    r.gw.from_client(init)
    r.gw.from_client(b"not json\n")
    assert r.server == [init, b"not json\n"]

    read = line(tools_call(1, "read_file", {"path": "a"}))
    r.gw.from_client(read)
    assert r.server[-1] == read and r.client == []
    failed = {"jsonrpc": "2.0", "id": 1, "result": {"content": [], "isError": True}}
    r.gw.from_server(line(failed))
    assert r.client == [failed]

    r.gw.from_client(line(tools_call(2, "delete_file", {"path": "a"})))
    assert r.client[-1] == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "content": [
                {"type": "text", "text": "blocked by guard no-delete: deleting files is not allowed"}
            ],
            "isError": True,
        },
    }
    r.gw.from_client(line(tools_call(3, "write_file", {"path": "a"})))
    assert "a call failed earlier (1)" in r.client[-1]["result"]["content"][0]["text"]
    assert len(r.server) == 3 and (r.gw.blocked, r.gw.forwarded) == (2, 1)
    assert [e.kind for e in r.gw.events] == [
        "tool_call", "tool_result", "tool_call", "guard_block", "tool_call", "guard_block",
    ]  # fmt: skip
    block = r.gw.events[3].payload
    assert block["call_id"] == "2" and [d["guard_id"] for d in block["decisions"]] == [
        "no-delete",
        "after-failure",
    ]


def test_gateway_records_server_results(runtime: GuardRuntime) -> None:
    r = Relay([Guard("no-delete", "", NO_DELETE)], runtime)
    r.gw.from_client(line(tools_call("a", "read_file", {})))
    r.gw.from_client(line(tools_call("b", "read_file")))
    r.gw.from_server(
        line({"jsonrpc": "2.0", "id": "a", "result": {"content": [{"type": "text", "text": "x"}]}})
    )
    r.gw.from_server(line({"jsonrpc": "2.0", "id": "b", "error": {"code": -32000, "message": "nope"}}))
    r.gw.from_server(line({"jsonrpc": "2.0", "id": "zzz", "result": {}}))
    r.gw.from_server(b"garbage\n")
    results = [e.payload["result"] for e in r.gw.events if e.kind == "tool_result"]
    assert results[0] == {
        "call_id": '"a"',
        "ok": True,
        "data": {"content": [{"type": "text", "text": "x"}]},
        "error": None,
    }
    assert results[1]["ok"] is False and "nope" in results[1]["error"]
    assert len(results) == 2 and len(r.client) == 4


def test_gateway_fails_closed(runtime: GuardRuntime) -> None:
    r = Relay([Guard("crash", "", CRASH)], runtime)
    r.gw.from_client(line(tools_call(1, "read_file", {})))
    assert r.client[-1]["result"]["content"][0]["text"] == "blocked by guard crash: guard error"
    r.gw.from_client(line(tools_call(2, "read_file", "not an object")))
    assert (
        r.client[-1]["result"]["content"][0]["text"]
        == "blocked by guard gateway: malformed tools/call params"
    )
    r.gw.from_client(line({"jsonrpc": "2.0", "id": 3, "method": "tools/call"}))
    assert r.client[-1]["id"] == 3 and r.server == []


def test_gateway_splits_batches(runtime: GuardRuntime) -> None:
    r = Relay([Guard("no-delete", "", NO_DELETE)], runtime)
    note = {"jsonrpc": "2.0", "method": "notifications/progress"}
    r.gw.from_client(line([tools_call(1, "delete_file", {}), tools_call(2, "read_file", {}), note]))
    assert json.loads(r.server[-1]) == [tools_call(2, "read_file", {}), note]
    assert [a["id"] for a in r.client[-1]] == [1]
    r.gw.from_client(line([tools_call(3, "delete_file", {})]))
    assert len(r.server) == 1 and r.client[-1][0]["id"] == 3
    r.gw.from_server(line([{"jsonrpc": "2.0", "id": 2, "result": {}}]))
    assert r.gw.events[-1].kind == "tool_result"


FAKE_SERVER = """
import json, sys
for raw in sys.stdin:
    message = json.loads(raw)
    if isinstance(message, dict) and "id" in message:
        name = message.get("params", {}).get("name", "")
        result = {"content": [{"type": "text", "text": "ran " + name}]}
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
"""


def admitted(tmp_path: Path, admitted: bool = True) -> list[str]:
    h = Harness.build(
        parent=None,
        system_prompt="",
        procedures=[],
        guards=[Guard("no-delete", "", NO_DELETE)],
        tool_allowlist=[],
        created_by="human",
    )
    signer = Signer.generate()
    decision = Decision(
        "root", h.version, h.digest, admitted, [Check("replay", admitted, {})], decided_at="t"
    )
    (tmp_path / "harness.json").write_text(json.dumps(asdict(h)), encoding="utf-8")
    (tmp_path / "receipt.json").write_bytes(receipt(decision, signer))
    key = public_key_bytes(signer.public_key).hex()
    return [
        "--harness",
        str(tmp_path / "harness.json"),
        "--receipt",
        str(tmp_path / "receipt.json"),
        "--trusted-key",
        key,
    ]


def test_gateway_end_to_end_over_stdio(tmp_path: Path) -> None:
    server = tmp_path / "server.py"
    server.write_text(FAKE_SERVER, encoding="utf-8")
    requests = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
        tools_call(1, "read_file", {"path": "a"}),
        tools_call(2, "delete_file", {"path": "a"}),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    stdin = io.BytesIO(b"".join(line(m) for m in requests))
    stdout = io.BytesIO()
    code = main([*admitted(tmp_path), "--", sys.executable, str(server)], stdin, stdout)
    assert code == 0
    replies = {m["id"]: m for m in map(json.loads, stdout.getvalue().splitlines())}
    assert replies[0]["result"]["content"][0]["text"] == "ran "
    assert replies[1]["result"]["content"][0]["text"] == "ran read_file"
    assert replies[2]["result"]["isError"] is True


def test_gateway_stops_when_the_server_is_gone(tmp_path: Path) -> None:
    stdin = io.BytesIO(b"".join(line(tools_call(i, "read_file", {"pad": "x" * 70000})) for i in range(4)))
    code = main([*admitted(tmp_path), "--", sys.executable, "-c", "pass"], stdin, io.BytesIO())
    assert code == 0


def test_gateway_refuses_an_unadmitted_harness(tmp_path: Path) -> None:
    assert main([*admitted(tmp_path, admitted=False), "--", sys.executable, "-c", "pass"]) == 1
