"""Enforce an admitted harness's guards at an MCP gateway, before a `tools/call` reaches the server.

The gateway sits between an MCP client and an MCP server, over either transport: it starts a stdio server
and relays newline-delimited JSON-RPC both ways, or it listens on HTTP and forwards to a Streamable HTTP
endpoint, relaying JSON and SSE responses as they stream, with one history per `Mcp-Session-Id`.
Every `tools/call` request is shown to every guard of the harness, in the sandboxed guard runtime, as a
`ToolCall(name, arguments, call_id=<request id>)` with the session's history so far. A block, or a guard
error (fail closed), never reaches the server: the client gets a tool result with `isError: true` and the
reasons, the same text the Ratchet worker gives its model. Everything else passes unchanged.

History uses the worker's event shapes: `tool_call` {"call", "view"}, `tool_result` {"result",
"decisions"} and `guard_block` {"call_id", "source", "message", "decisions"}. The view is {}: an MCP server
exposes no state to the gateway beyond what its results say, and those are in the history.

Only an admitted harness is enforced: the receipt must verify under a trusted key and admit exactly this
harness, as for the Claude Code hook export.

    python -m ratchet.export.mcp_gateway --harness h.json --receipt r.json --trusted-key <hex> \
        -- <server command> [args...]
    python -m ratchet.export.mcp_gateway --harness h.json --receipt r.json --trusted-key <hex> \
        --upstream https://host/mcp --listen 127.0.0.1:8808

Over HTTP a batch (allowed before protocol version 2025-06-18) that holds a blocked call is answered
whole: the blocked calls carry their reasons and the other requests say they were not forwarded, because
one HTTP reply cannot merge a forwarded part, which may stream.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO, Any

import httpx
from ratchet_cases.types import Event, EventKind, ToolCall, sha256_hex

from ratchet.agent.guards import GuardDecision, GuardRuntime
from ratchet.export.claude_code import ExportError, harness_from_json, trusted_keys, verify_admission
from ratchet.types import Guard

Write = Callable[[bytes], None]


def _line(message: Any) -> bytes:
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


class Gateway:
    """The relay's decisions, independent of how the two streams are connected."""

    def __init__(
        self,
        guards: Sequence[Guard],
        runtime: GuardRuntime,
        to_client: Write,
        to_server: Write,
        lock: threading.Lock | None = None,
    ) -> None:
        self.guards = list(guards)
        self.runtime = runtime
        self.to_client = to_client
        self.to_server = to_server
        self.events: list[Event] = []
        self.pending: dict[str, str] = {}
        self.blocked = 0
        self.forwarded = 0
        self._lock = lock or threading.Lock()  # shared when several gateways share one guard runtime

    def record(self, kind: EventKind, payload: dict[str, Any]) -> None:
        self.events.append(Event(len(self.events), kind, payload, time.time()))

    def check(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """None to forward the request, or the error result that answers it instead."""
        raw = request.get("params")
        params: dict[str, Any] = raw if isinstance(raw, dict) else {}
        name, args = params.get("name"), params.get("arguments", {})
        call_id = json.dumps(request.get("id"))
        with self._lock:
            history = list(self.events)
            if not isinstance(name, str) or not isinstance(args, dict):
                decisions = [GuardDecision("gateway", True, "malformed tools/call params")]
            else:
                call = ToolCall(name, args, call_id)
                self.record("tool_call", {"call": asdict(call), "view": {}})
                decisions = self.runtime.check_all(self.guards, call, {}, history)
            blocks = [d for d in decisions if d.blocked]
            if not blocks:
                self.pending[call_id] = str(name)
                self.forwarded += 1
                return None
            message = "; ".join(f"blocked by guard {d.guard_id}: {d.reason}" for d in blocks)
            payload = [d.to_payload() for d in decisions]
            self.record(
                "guard_block",
                {"call_id": call_id, "source": "guard", "message": message, "decisions": payload},
            )
            self.blocked += 1
        result = {"content": [{"type": "text", "text": message}], "isError": True}
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}

    def from_client(self, raw: bytes) -> None:
        """One line from the client: forward it, or answer the tools/call requests the guards block."""
        try:
            message = json.loads(raw)
        except ValueError:
            self.to_server(raw)
            return
        batch = message if isinstance(message, list) else [message]
        forward: list[Any] = []
        answers: list[dict[str, Any]] = []
        for item in batch:
            answer = (
                self.check(item) if isinstance(item, dict) and item.get("method") == "tools/call" else None
            )
            if answer is None:
                forward.append(item)
            else:
                answers.append(answer)
        if not answers:
            self.to_server(raw)
            return
        if forward:
            self.to_server(_line(forward if isinstance(message, list) else forward[0]))
        self.to_client(_line(answers if isinstance(message, list) else answers[0]))

    def from_server(self, raw: bytes) -> None:
        """One line from the server: record the results of forwarded calls, then pass it on."""
        try:
            message = json.loads(raw)
        except ValueError:
            message = None
        for item in message if isinstance(message, list) else [message]:
            if isinstance(item, dict) and "id" in item:
                self.record_response(item)
        self.to_client(raw)

    def record_response(self, response: dict[str, Any]) -> None:
        """Record the result of a forwarded tools/call; anything else is ignored."""
        call_id = json.dumps(response.get("id"))
        with self._lock:
            if self.pending.pop(call_id, None) is None:
                return
            result = response.get("result")
            error = response.get("error")
            failed = error is not None or (isinstance(result, dict) and bool(result.get("isError")))
            text = None if not failed else json.dumps(error if error is not None else result)
            outcome = {
                "call_id": call_id,
                "ok": not failed,
                "data": None if failed else result,
                "error": text,
            }
            self.record("tool_result", {"result": outcome, "decisions": []})


class _Pipe:
    """A serialised byte writer: both relay threads may answer the client."""

    def __init__(self, stream: IO[bytes]) -> None:
        self.stream = stream
        self.lock = threading.Lock()

    def __call__(self, data: bytes) -> None:
        with self.lock:
            self.stream.write(data)
            self.stream.flush()


def serve(
    gateway_for: Callable[[Write, Write], Gateway],
    command: Sequence[str],
    stdin: IO[bytes],
    stdout: IO[bytes],
) -> int:
    """Start the server, relay both directions until the client closes stdin, return the server's status."""
    server = subprocess.Popen(list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert server.stdin is not None and server.stdout is not None
    gateway = gateway_for(_Pipe(stdout), _Pipe(server.stdin))
    server_out = server.stdout

    def pump_server() -> None:
        for line in server_out:
            gateway.from_server(line)

    thread = threading.Thread(target=pump_server, daemon=True)
    thread.start()
    for line in stdin:
        try:
            gateway.from_client(line)
        except BrokenPipeError:
            break
    with contextlib.suppress(BrokenPipeError):
        server.stdin.close()
    code = server.wait()
    thread.join()
    server_out.close()
    return code


# --- Streamable HTTP --------------------------------------------------------------------------------------

HOP_BY_HOP = frozenset(
    {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
     "proxy-authorization", "proxy-authenticate", "accept-encoding", "content-encoding"}
)  # fmt: skip
NOT_SENT = "not forwarded: another call in this batch was blocked"


class HttpGateway:
    """The gateway for an MCP server reached over Streamable HTTP. Each `Mcp-Session-Id` has its own
    history; all sessions share one guard runtime under one lock."""

    def __init__(
        self, guards: Sequence[Guard], runtime: GuardRuntime, upstream: str, client: httpx.Client
    ) -> None:
        self.guards = list(guards)
        self.runtime = runtime
        self.upstream = upstream
        self.client = client
        self.lock = threading.Lock()
        self.sessions: dict[str, Gateway] = {}
        self._sessions_lock = threading.Lock()

    def session(self, session_id: str) -> Gateway:
        with self._sessions_lock:
            gateway = self.sessions.get(session_id)
            if gateway is None:
                gateway = Gateway(self.guards, self.runtime, _discard, _discard, self.lock)
                self.sessions[session_id] = gateway
            return gateway

    def answer(self, gateway: Gateway, message: Any) -> Any | None:
        """The reply that answers a POST body without forwarding it, or None to forward it. In a batch,
        one blocked call holds back every request: an HTTP reply cannot merge a forwarded part."""
        batch = message if isinstance(message, list) else [message]
        calls = [m for m in batch if isinstance(m, dict) and m.get("method") == "tools/call"]
        answers = {id(m): gateway.check(m) for m in calls}
        if not any(answers.values()):
            return None
        if not isinstance(message, list):
            return answers[id(message)]
        replies = []
        for item in batch:
            if not isinstance(item, dict) or "id" not in item or "method" not in item:
                continue
            reply = answers.get(id(item))
            if reply is None:
                result = {"content": [{"type": "text", "text": NOT_SENT}], "isError": True}
                reply = {"jsonrpc": "2.0", "id": item["id"], "result": result}
            replies.append(reply)
        return replies


def _discard(data: bytes) -> None:
    return None


def _messages(data: bytes) -> list[Any]:
    try:
        message = json.loads(data)
    except ValueError:
        return []
    return message if isinstance(message, list) else [message]


class _Handler(BaseHTTPRequestHandler):
    server: _HttpServer
    protocol_version = "HTTP/1.0"  # one response per connection, ended by closing it

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        gateway = self.server.gateway.session(self.headers.get("Mcp-Session-Id", ""))
        messages = _messages(body)
        reply = self.server.gateway.answer(gateway, messages[0] if len(messages) == 1 else messages)
        if reply is not None:
            self._send_json(200, reply)
            return
        self._forward("POST", body, gateway)

    def do_GET(self) -> None:
        self._forward("GET", None, self.server.gateway.session(self.headers.get("Mcp-Session-Id", "")))

    def do_DELETE(self) -> None:
        self._forward("DELETE", None, self.server.gateway.session(self.headers.get("Mcp-Session-Id", "")))

    def _send_json(self, status: int, message: Any) -> None:
        data = json.dumps(message).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _forward(self, method: str, body: bytes | None, gateway: Gateway) -> None:
        upstream = self.server.gateway
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        try:
            with upstream.client.stream(method, upstream.upstream, headers=headers, content=body) as response:
                self.send_response(response.status_code)
                for key, value in response.headers.items():
                    if key.lower() not in HOP_BY_HOP:
                        self.send_header(key, value)
                self.end_headers()
                if response.headers.get("content-type", "").startswith("text/event-stream"):
                    self._relay_events(response, gateway)
                else:
                    data = response.read()
                    for item in _messages(data):
                        if isinstance(item, dict):
                            gateway.record_response(item)
                    self.wfile.write(data)
        except httpx.HTTPError as exc:
            self._send_json(
                502, {"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(exc)}}
            )

    def _relay_events(self, response: httpx.Response, gateway: Gateway) -> None:
        data: list[str] = []
        for line in response.iter_lines():
            self.wfile.write(line.encode("utf-8") + b"\n")
            self.wfile.flush()
            if line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
            elif not line and data:
                for item in _messages("\n".join(data).encode("utf-8")):
                    if isinstance(item, dict):
                        gateway.record_response(item)
                data = []


class _HttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], gateway: HttpGateway) -> None:
        super().__init__(address, _Handler)
        self.gateway = gateway


def http_server(gateway: HttpGateway, host: str, port: int) -> _HttpServer:
    """A server on (host, port) that applies `gateway`; call `serve_forever()` on it."""
    return _HttpServer((host, port), gateway)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ratchet.export.mcp_gateway", description=__doc__.split("\n")[0]
    )
    p.add_argument(
        "--harness", type=Path, required=True, help="harness JSON (a Harness as written by Ratchet)"
    )
    p.add_argument("--receipt", type=Path, required=True, help="the DSSE decision envelope that admitted it")
    p.add_argument(
        "--trusted-key", action="append", required=True, help="Ed25519 public key, hex; repeatable"
    )
    p.add_argument("--upstream", help="URL of an MCP server's Streamable HTTP endpoint, instead of a command")
    p.add_argument("--listen", default="127.0.0.1:8808", help="host:port the HTTP gateway listens on")
    p.add_argument("command", nargs="*", help="the stdio MCP server command, after --")
    return p


def main(
    argv: Sequence[str] | None = None, stdin: IO[bytes] | None = None, stdout: IO[bytes] | None = None
) -> int:
    args = _parser().parse_args(argv)
    try:
        if bool(args.upstream) == bool(args.command):
            raise ExportError("give either --upstream URL or a server command after --")
        harness = harness_from_json(json.loads(args.harness.read_text(encoding="utf-8")))
        envelope = args.receipt.read_bytes()
        verify_admission(harness, envelope, trusted_keys(args.trusted_key))
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"ratchet mcp gateway: harness {harness.version}, {len(harness.guards)} guard(s), "
        f"receipt sha256 {sha256_hex(envelope)}",
        file=sys.stderr,
    )
    with GuardRuntime(workers=1) as runtime:
        if args.upstream:
            host, _, port = args.listen.rpartition(":")
            with httpx.Client(timeout=None) as client:
                server = http_server(
                    HttpGateway(harness.guards, runtime, args.upstream, client), host, int(port)
                )
                print(f"listening on http://{args.listen}/ for {args.upstream}", file=sys.stderr, flush=True)
                with contextlib.suppress(KeyboardInterrupt):
                    server.serve_forever()
                server.server_close()
            return 0
        return serve(
            lambda to_client, to_server: Gateway(harness.guards, runtime, to_client, to_server),
            args.command,
            stdin or sys.stdin.buffer,
            stdout or sys.stdout.buffer,
        )


if __name__ == "__main__":
    sys.exit(main())
