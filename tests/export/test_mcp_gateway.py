from __future__ import annotations

import io
import json
import sys
import threading
from collections.abc import Iterator
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from ratchet.agent.guards import GuardRuntime
from ratchet.export import mcp_gateway
from ratchet.export.mcp_gateway import NOT_SENT, Gateway, HttpGateway, http_server, main
from ratchet.improve.admit import Check, Decision
from ratchet.improve.receipt import Signer, receipt
from ratchet.types import Guard, Harness
from ratchet.verify.dsse import raw_public

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
    key = raw_public(signer.public_key).hex()
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


# --- Streamable HTTP ----------------------------------------------------------------------------------------


class Upstream(BaseHTTPRequestHandler):
    """A fake MCP server on Streamable HTTP: JSON for most requests, SSE for `stream_tool`."""

    seen: ClassVar[list[Any]] = []

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def do_POST(self) -> None:
        message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Upstream.seen.append(message)
        if isinstance(message, dict) and "id" not in message:
            self.send_response(202)
            self.end_headers()
            return
        items = message if isinstance(message, list) else [message]
        replies = [self.reply(m) for m in items if "id" in m]
        if items[0].get("params", {}).get("name") == "stream_tool":
            progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}
            body = "".join(f"event: message\ndata: {json.dumps(m)}\n\n" for m in [progress, *replies])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(body.encode())
            return
        data = json.dumps(replies if isinstance(message, list) else replies[0]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if items[0].get("method") == "initialize":
            self.send_header("Mcp-Session-Id", "sess-1")
        self.end_headers()
        self.wfile.write(data)

    def reply(self, message: dict[str, Any]) -> dict[str, Any]:
        name = message.get("params", {}).get("name", "")
        result = {"content": [{"type": "text", "text": "ran " + name}], "isError": name == "fail_tool"}
        return {"jsonrpc": "2.0", "id": message["id"], "result": result}

    def do_GET(self) -> None:
        self.send_response(405)
        self.end_headers()

    def do_DELETE(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", "6")
        self.end_headers()
        self.wfile.write(b"closed")


@pytest.fixture
def http_pair(runtime: GuardRuntime) -> Iterator[tuple[str, HttpGateway]]:
    Upstream.seen = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    guards = [Guard("no-delete", "", NO_DELETE), Guard("after-failure", "", AFTER_FAILURE)]
    with httpx.Client(timeout=30) as client:
        gateway = HttpGateway(guards, runtime, f"http://127.0.0.1:{upstream.server_address[1]}/mcp", client)
        server = http_server(gateway, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{server.server_address[1]}/mcp", gateway
        server.shutdown()
        server.server_close()
    upstream.shutdown()
    upstream.server_close()


def post(url: str, message: Any, session: str | None = None) -> httpx.Response:
    headers = {"Accept": "application/json, text/event-stream"}
    if session is not None:
        headers["Mcp-Session-Id"] = session
    return httpx.post(url, json=message, headers=headers, timeout=30)


def test_http_gateway_forwards_json_and_sse_and_answers_blocked_calls(
    http_pair: tuple[str, HttpGateway],
) -> None:
    url, gateway = http_pair
    init = post(url, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
    assert init.status_code == 200 and init.headers["mcp-session-id"] == "sess-1"
    assert post(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, "sess-1").status_code == 202
    assert (
        post(url, tools_call(1, "read_file", {}), "sess-1").json()["result"]["content"][0]["text"]
        == "ran read_file"
    )
    streamed = post(url, tools_call(2, "stream_tool", {}), "sess-1")
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert streamed.text.count("data: ") == 2 and "ran stream_tool" in streamed.text
    blocked = post(url, tools_call(3, "delete_file", {}), "sess-1").json()
    assert blocked["result"]["isError"] is True and "deleting files is not allowed" in str(blocked)
    assert post(url, tools_call(4, "fail_tool", {}), "sess-1").json()["result"]["isError"] is True
    after = post(url, tools_call(5, "write_file", {}), "sess-1").json()
    assert "a call failed earlier (4)" in after["result"]["content"][0]["text"]
    assert "ran write_file" in post(url, tools_call(6, "write_file", {}), "other").text
    forwarded_ids = [m.get("id") for m in Upstream.seen if isinstance(m, dict)]
    assert 3 not in forwarded_ids and 5 not in forwarded_ids
    kinds = [e.kind for e in gateway.sessions["sess-1"].events]
    assert kinds.count("tool_result") == 3 and kinds.count("guard_block") == 2


def test_http_gateway_holds_back_a_batch_with_a_blocked_call(http_pair: tuple[str, HttpGateway]) -> None:
    url, _ = http_pair
    note = {"jsonrpc": "2.0", "method": "notifications/progress"}
    replies = post(url, [tools_call(1, "delete_file", {}), tools_call(2, "read_file", {}), note]).json()
    assert [r["id"] for r in replies] == [1, 2]
    assert replies[1]["result"]["content"][0]["text"] == NOT_SENT
    assert Upstream.seen == []
    passed = post(url, [tools_call(3, "read_file", {}), tools_call(4, "read_file", {})]).json()
    assert [r["id"] for r in passed] == [3, 4]


def test_http_gateway_relays_other_methods_and_upstream_failures(
    http_pair: tuple[str, HttpGateway], runtime: GuardRuntime
) -> None:
    url, _ = http_pair
    assert httpx.get(url, headers={"Accept": "text/event-stream"}).status_code == 405
    closed = httpx.delete(url, headers={"Mcp-Session-Id": "sess-1"})
    assert (closed.status_code, closed.text) == (200, "closed")
    with httpx.Client(timeout=5) as client:
        down = http_server(HttpGateway([], runtime, "http://127.0.0.1:9/mcp", client), "127.0.0.1", 0)
        threading.Thread(target=down.serve_forever, daemon=True).start()
        response = post(f"http://127.0.0.1:{down.server_address[1]}/mcp", tools_call(1, "read_file", {}))
        down.shutdown()
        down.server_close()
    assert response.status_code == 502 and response.json()["error"]["code"] == -32000


def test_command_line_chooses_one_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = admitted(tmp_path)
    assert main(args) == 1
    assert main([*args, "--upstream", "http://127.0.0.1:9/mcp", "--", "server"]) == 1

    class Stops:
        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            return None

    monkeypatch.setattr(mcp_gateway, "http_server", lambda gateway, host, port: Stops())
    assert main([*args, "--upstream", "http://127.0.0.1:9/mcp", "--listen", "127.0.0.1:0"]) == 0
