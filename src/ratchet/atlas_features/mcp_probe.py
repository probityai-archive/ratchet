"""Probe 5: the MongoDB MCP server at runtime, read-only, as the improver would use it.

    uv run python -m ratchet.atlas_features.mcp_probe

Starts the pinned `mongodb-mcp-server` over stdio in read-only mode as the improver's restricted database
identity, lists its tools, performs one `find` over the training traces, and tries one held-out read that the
identity's role must refuse. It speaks MCP JSON-RPC directly (initialize, tools/list, tools/call), so
it needs no client library. The connection string reaches the server only through its environment.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import Any
from urllib.parse import quote_plus

from ratchet.atlas_features.common import ROOT, ProbeResult, cluster_srv_host, connect, secret, timed

SERVER_PACKAGE = "mongodb-mcp-server@3.0.5"
INSTALL_DIR = ROOT / "research" / "_raw" / "atlas-probes" / "mcp"
PROTOCOL = "2025-06-18"
TIMEOUT_S = 90.0
WRITE_TOOLS = ("insert-many", "update-many", "delete-many", "drop-collection", "drop-database")


class McpError(RuntimeError):
    """The server answered with a JSON-RPC error or did not answer in time."""


class StdioClient:
    """A line-delimited JSON-RPC client for one MCP server process."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self._lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._next = 0

    def _pump(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.put(line)
        self._lines.put("")

    def _send(self, message: dict[str, Any]) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(message) + "\n")
        self._proc.stdin.flush()

    def notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next += 1
        self._send({"jsonrpc": "2.0", "id": self._next, "method": method, "params": params or {}})
        while True:
            try:
                line = self._lines.get(timeout=TIMEOUT_S)
            except queue.Empty as exc:
                raise McpError(f"{method}: no answer within {TIMEOUT_S:.0f}s") from exc
            if not line:
                raise McpError(f"{method}: server exited")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == self._next:
                if "error" in msg:
                    raise McpError(f"{method}: {json.dumps(msg['error'])[:400]}")
                return dict(msg["result"])

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is not None:
                stream.close()


def server_env(uri: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "NVM_DIR", "NODE_PATH")}
    env.update(
        MDB_MCP_CONNECTION_STRING=uri,
        MDB_MCP_READ_ONLY="true",
        MDB_MCP_TELEMETRY="disabled",
        MDB_MCP_LOGGERS="stderr",
    )
    return env


def server_argv() -> list[str]:
    """The pinned server run from a private install (`npm install --prefix`), never the shared npx cache."""
    return ["node", str(INSTALL_DIR / "node_modules" / "mongodb-mcp-server" / "dist" / "esm" / "index.js")]


IMPROVER_USER = "ratchet-improver"
TRAIN_DB = "ratchet"
HELDOUT_DB = "ratchet_heldout"


def improver_uri() -> str:
    """The restricted identity's URI, built from the cluster host; the password never leaves the process."""
    password = secret("RATCHET_IMPROVER_PASSWORD", "mongodb", "hackathon-db-improver")
    return f"mongodb+srv://{IMPROVER_USER}:{quote_plus(password)}@{cluster_srv_host()}/?retryWrites=true&w=majority"


def find_call(database: str, collection: str) -> dict[str, Any]:
    return {
        "name": "find",
        "arguments": {
            "connectionId": "preconfigured",
            "database": database,
            "collection": collection,
            "filter": {},
            "limit": 1,
        },
    }


def tool_text(result: dict[str, Any]) -> str:
    return " ".join(str(c.get("text", "")) for c in result.get("content", []))


SENTINEL_ID = "featureprobe-mcp-sentinel"


def with_sentinel_trace(action: Callable[[], ProbeResult]) -> ProbeResult:
    """Run `action` with one clearly named trace in the training collection, removed afterwards.

    The read must return a real document to prove the identity can read traces; the store lane's collection
    may be empty at probe time, so the probe writes and removes its own sentinel with the app identity.
    """
    client = connect()
    coll = client[TRAIN_DB]["traces"]
    coll.delete_many({"_id": SENTINEL_ID})
    coll.insert_one({"_id": SENTINEL_ID, "run_id": SENTINEL_ID, "case_id": SENTINEL_ID, "probe": True})
    try:
        return action()
    finally:
        coll.delete_many({"_id": SENTINEL_ID})
        client.close()


def run() -> ProbeResult:
    return with_sentinel_trace(read_as_improver)


def read_as_improver() -> ProbeResult:
    """One training-traces read as the restricted identity, plus the held-out read the role must refuse."""
    argv = server_argv()
    train_call = find_call(TRAIN_DB, "traces")
    heldout_call = find_call(HELDOUT_DB, "cases")
    with timed() as elapsed:
        client = StdioClient(argv, server_env(improver_uri()))
        try:
            init = client.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL,
                    "capabilities": {},
                    "clientInfo": {"name": "ratchet-probe", "version": "0"},
                },
            )
            client.notify("notifications/initialized")
            tools = [t["name"] for t in client.request("tools/list")["tools"]]
            train = client.request("tools/call", train_call)
            heldout = client.request("tools/call", heldout_call)
        except McpError as exc:
            return ProbeResult(
                probe="mcp_server",
                works=False,
                summary="MCP server read failed",
                latency_s=elapsed(),
                pipeline={"argv": argv, "call": train_call},
                error=str(exc),
            )
        finally:
            client.close()
    train_text, heldout_text = tool_text(train), tool_text(heldout)
    ok = (
        not train.get("isError")
        and "resulted in" in train_text
        and not [t for t in tools if t in WRITE_TOOLS]
    )
    denied = bool(heldout.get("isError"))
    return ProbeResult(
        probe="mcp_server",
        works=ok and denied,
        summary=f"{IMPROVER_USER} read {TRAIN_DB}.traces via {SERVER_PACKAGE}; held-out read "
        + ("refused" if denied else "ALLOWED"),
        latency_s=elapsed(),
        pipeline={
            "argv": argv,
            "env": ["MDB_MCP_READ_ONLY=true", "MDB_MCP_TELEMETRY=disabled"],
            "training_read": train_call,
            "heldout_read": heldout_call,
        },
        data={
            "server": init.get("serverInfo"),
            "tool_count": len(tools),
            "tools": tools,
            "write_tools_exposed": [t for t in tools if t in WRITE_TOOLS],
            "training_read_isError": bool(train.get("isError")),
            "training_read_head": train_text[:300],
            "heldout_read_isError": denied,
            "heldout_read_head": heldout_text[:300],
        },
        error=None if ok and denied else (train_text if not ok else heldout_text)[:500],
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
