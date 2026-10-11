"""Enforce an admitted harness's guards at an MCP gateway, before a `tools/call` reaches the server.

The gateway sits between an MCP client and a stdio MCP server and relays newline-delimited JSON-RPC both
ways. Every `tools/call` request is shown to every guard of the harness, in the sandboxed guard runtime,
as a `ToolCall(name, arguments, call_id=<request id>)` with the session's history so far. A block, or a
guard error (fail closed), never reaches the server: the client gets a tool result with `isError: true`
and the reasons, the same text the Ratchet worker gives its model. Everything else passes unchanged.

History uses the worker's event shapes: `tool_call` {"call", "view"}, `tool_result` {"result",
"decisions"} and `guard_block` {"call_id", "source", "message", "decisions"}. The view is {}: an MCP server
exposes no state to the gateway beyond what its results say, and those are in the history.

Only an admitted harness is enforced: the receipt must verify under a trusted key and admit exactly this
harness, as for the Claude Code hook export.

    python -m ratchet.export.mcp_gateway --harness h.json --receipt r.json --trusted-key <hex> \
        -- <server command> [args...]
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
from pathlib import Path
from typing import IO, Any

from ratchet.agent.guards import GuardDecision, GuardRuntime
from ratchet.export.claude_code import ExportError, harness_from_json, trusted_keys, verify_admission
from ratchet.types import Event, EventKind, Guard, ToolCall, sha256_hex

Write = Callable[[bytes], None]


def _line(message: Any) -> bytes:
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


class Gateway:
    """The relay's decisions, independent of how the two streams are connected."""

    def __init__(
        self, guards: Sequence[Guard], runtime: GuardRuntime, to_client: Write, to_server: Write
    ) -> None:
        self.guards = list(guards)
        self.runtime = runtime
        self.to_client = to_client
        self.to_server = to_server
        self.events: list[Event] = []
        self.pending: dict[str, str] = {}
        self.blocked = 0
        self.forwarded = 0
        self._lock = threading.Lock()

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
                self._result(item)
        self.to_client(raw)

    def _result(self, response: dict[str, Any]) -> None:
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
    p.add_argument("command", nargs="+", help="the MCP server command, after --")
    return p


def main(
    argv: Sequence[str] | None = None, stdin: IO[bytes] | None = None, stdout: IO[bytes] | None = None
) -> int:
    args = _parser().parse_args(argv)
    try:
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
        return serve(
            lambda to_client, to_server: Gateway(harness.guards, runtime, to_client, to_server),
            args.command,
            stdin or sys.stdin.buffer,
            stdout or sys.stdout.buffer,
        )


if __name__ == "__main__":
    sys.exit(main())
