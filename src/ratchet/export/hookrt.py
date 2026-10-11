"""Ratchet hook runtime for Claude Code. Standard library only.

This module is used two ways, from one source. Ratchet imports it to parse Claude Code transcripts for
replay, and `ratchet.export.claude_code` pastes it verbatim into every generated PreToolUse hook, so the
installed hook builds the guard's inputs with exactly the code that proved the guard against the corpus.

Transcript mapping (one JSONL line per content block, as Claude Code writes them):
  assistant text or tool_use  ->  Event "model"      {"model", "content", "tool_calls"}
  each tool_use               ->  Event "tool_call"  {"call": {"name", "args", "call_id"}, "view"}
  user tool_result            ->  Event "tool_result" {"result": {"call_id", "ok", "data", "error"},
                                                       "decisions": []}
User prompts, thinking blocks, attachments and system lines carry no event: the contract's event kinds
have none for them, and the shapes above are the ones the Ratchet worker records. Tool results are kept
whole: cutting each to 4096 characters shrank the largest local session's history from 27.0 to 25.9 MB,
too little to justify hiding text from a guard.

The guard's `view` for a coding session is {"cwd", "git_branch"}: where the agent is and which branch is
checked out (detached HEAD is "HEAD", outside a repository ""). Replay reads both from the transcript line
that holds the call; the live hook reads `cwd` from the hook input and the branch from the repository.
`permission_mode` is not in the view: 174 of 231 local sessions record no permission-mode line, so replay
could not reconstruct it.

The history for a call is every event before the transcript line that holds the call. Claude Code writes
the transcript asynchronously, so a live hook may see a shorter history than replay does; it never sees a
longer one. A guard whose source provably never reads its history (see `reads_history`) gets an empty
history in replay and live alike, and its hook never reads the transcript.

Speed: parsing a 141 MB transcript at every tool call took 18 s. A generated hook therefore asks a session
server first: one per guard, started on demand from the hook file itself, listening on a socket in a
user-private directory. It tails each transcript incrementally and keeps one sandboxed guard child per
session, so a call costs one incremental read and one request. Any failure to reach it falls back to the
one-shot path, which gives the same verdict.
"""

from __future__ import annotations

import ast
import contextlib
import fcntl
import hashlib
import json
import os
import select
import socket
import struct
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any

GUARD_ERROR_REASON = "guard error"
RUNTIME_BEGIN = "# >>> ratchet hook runtime (verbatim ratchet/export/hookrt.py) >>>"
RUNTIME_END = "# <<< ratchet hook runtime <<<"
# Claude Code cancels a command hook after 600 s by default; the hook gives the server half of that.
SERVER_TIMEOUT_S = 300.0
SERVER_START_S = 5.0
SERVER_IDLE_S = 900.0
SERVER_SESSIONS = 8
# Wall time past the guard's CPU limit before the parent stops a guard that blocks without using CPU. The
# child enforces the limit itself in CPU time, counted from its loading and start markers.
GUARD_WALL_GRACE_S = 5.0
SOCKET_PATH_MAX = 107  # bytes in sockaddr_un.sun_path, less the terminating zero, on Linux


# --- transcript parsing -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallSite:
    """One tool call in a transcript, with what the guard saw at that point."""

    call: dict[str, Any]
    view: dict[str, Any]
    history_len: int
    line: int


@dataclass
class Transcript:
    events: list[dict[str, Any]] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    lines: int = 0
    malformed: int = 0

    def add(self, kind: str, payload: dict[str, Any], ts: float) -> None:
        self.events.append({"seq": len(self.events), "kind": kind, "payload": payload, "ts": ts})

    def feed(self, lines: Iterable[str | bytes]) -> Transcript:
        """Parse more JSONL lines. Lines that are not JSON objects are counted, not fatal."""
        for raw in lines:
            self.lines += 1
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except ValueError:
                self.malformed += 1
                continue
            if not isinstance(entry, dict):
                self.malformed += 1
                continue
            message = entry.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            ts = _timestamp(entry.get("timestamp"))
            if entry.get("type") == "assistant":
                _assistant(self, entry, message, self.lines, ts)
            elif entry.get("type") == "user":
                _user(self, message, ts)
        return self

    def site_of(self, call: Mapping[str, Any]) -> CallSite | None:
        """The call's own site: by tool use id, else the last call with the same name and arguments."""
        call_id = call.get("call_id")
        if call_id:
            for site in reversed(self.calls):
                if site.call["call_id"] == call_id:
                    return site
        for site in reversed(self.calls):
            if site.call["name"] == call.get("name") and site.call["args"] == call.get("args"):
                return site
        return None

    def history_before(self, call: Mapping[str, Any]) -> list[dict[str, Any]]:
        site = self.site_of(call)
        return self.events[: site.history_len] if site is not None else list(self.events)


def parse_transcript(lines: Iterable[str | bytes]) -> Transcript:
    return Transcript().feed(lines)


def coding_view(cwd: str, git_branch: str) -> dict[str, Any]:
    return {"cwd": cwd, "git_branch": git_branch}


def _timestamp(value: Any) -> float:
    if not isinstance(value, str):
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _blocks(content: list[Any], kind: str) -> list[dict[str, Any]]:
    return [b for b in content if isinstance(b, dict) and b.get("type") == kind]


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(b.get("text", "")) if b.get("type") == "text" else f"[{b.get('type')}]"
            for b in content
            if isinstance(b, dict)
        ]
        return "\n".join(parts)
    return ""


def _assistant(t: Transcript, entry: dict[str, Any], message: dict[str, Any], line: int, ts: float) -> None:
    content = message["content"]
    texts = [str(b.get("text", "")) for b in _blocks(content, "text")]
    calls = [
        {
            "name": str(b.get("name", "")),
            "args": b["input"] if isinstance(b.get("input"), dict) else {},
            "call_id": str(b.get("id", "")),
        }
        for b in _blocks(content, "tool_use")
    ]
    if not texts and not calls:
        return
    before = len(t.events)
    view = coding_view(str(entry.get("cwd") or ""), str(entry.get("gitBranch") or ""))
    model = {"model": str(message.get("model") or ""), "content": "\n".join(texts), "tool_calls": calls}
    t.add("model", model, ts)
    for call in calls:
        t.calls.append(CallSite(call, view, before, line))
        t.add("tool_call", {"call": call, "view": view}, ts)


def _user(t: Transcript, message: dict[str, Any], ts: float) -> None:
    for block in _blocks(message["content"], "tool_result"):
        text = _result_text(block.get("content"))
        failed = bool(block.get("is_error"))
        result = {
            "call_id": str(block.get("tool_use_id", "")),
            "ok": not failed,
            "data": None if failed else {"content": text},
            "error": text if failed else None,
        }
        t.add("tool_result", {"result": result, "decisions": []}, ts)


# --- does the guard read its history? ---------------------------------------------------------------------

# Names through which code can reach a frame, a code object or another function's arguments.
INTROSPECTION = frozenset(
    {
        "locals", "vars", "globals", "eval", "exec", "compile", "__import__", "getattr", "setattr",
        "delattr", "sys", "inspect", "gc", "builtins", "__builtins__", "importlib", "_getframe",
        "f_locals", "f_back", "f_globals", "tb_frame", "gi_frame", "cr_frame", "ag_frame", "__code__",
        "__closure__", "__globals__", "__dict__", "__self__", "__func__", "__traceback__",
        "__getattribute__", "attrgetter", "methodcaller",
    }
)  # fmt: skip


# Attribute names that lead from an object to a frame, a code object or a function's namespace.
FRAME_WORDS = tuple(
    sorted(w for w in INTROSPECTION if w.startswith(("f_", "tb_", "gi_", "cr_", "ag_", "_", "__")))
)
# Attributes that are a way in wherever they appear (`re.compile` is harmless; a bare `compile` is not).
ATTRIBUTE_WORDS = frozenset(
    {
        *FRAME_WORDS,
        "attrgetter",
        "methodcaller",
        "currentframe",
        "__subclasses__",
        "__mro__",
        "__bases__",
        "__base__",
    }
)


def _check_def(tree: ast.Module) -> ast.FunctionDef | None:
    """The module's one plain `def check(call, view, history)`, or None when it has no such single def."""
    defs = [
        n for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == "check"
    ]
    if len(defs) != 1 or not isinstance(defs[0], ast.FunctionDef) or defs[0].decorator_list:
        return None
    args = defs[0].args
    plain = [*args.posonlyargs, *args.args]
    if args.vararg or args.kwarg or args.kwonlyargs or len(plain) != 3:
        return None
    return defs[0]


def _could_reach(node: ast.AST, hidden: str, check: ast.FunctionDef) -> bool:
    """Whether this node names the hidden parameter, rebinds `check`, or names a way into frames."""
    if isinstance(node, ast.Name):
        rebinds = node.id == "check" and not isinstance(node.ctx, ast.Load)
        return rebinds or node.id == hidden or node.id in INTROSPECTION
    if isinstance(node, ast.Attribute):
        return node.attr in ATTRIBUTE_WORDS
    if isinstance(node, ast.Constant):  # a format field such as "{0.gi_frame}" reads an attribute by name
        return isinstance(node.value, str) and any(word in node.value for word in FRAME_WORDS)
    if isinstance(node, ast.Import | ast.ImportFrom):
        modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
        named = [m.split(".")[0] for m in modules] + [a.name for a in node.names]
        return any(n in INTROSPECTION for n in named)
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node is not check and node.name == "check"
    return False


def reads_history(source: str) -> bool:
    """False only when the guard provably never sees its history: the module defines `check` once, at top
    level, undecorated, with exactly three plain parameters; the third is never named anywhere; `check` is
    never rebound; and nothing names a way into frames or code objects, not even inside a string. Anything
    else counts as reading."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True
    check = _check_def(tree)
    if check is None:
        return True
    hidden = [*check.args.posonlyargs, *check.args.args][2].arg
    return any(_could_reach(node, hidden, check) for node in ast.walk(tree))


# --- live inputs ------------------------------------------------------------------------------------------


def _head_file(directory: Path) -> Path | None:
    dot = directory / ".git"
    if dot.is_dir():
        return dot / "HEAD"
    if dot.is_file():
        text = dot.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text[len("gitdir:") :].strip())
            return (gitdir if gitdir.is_absolute() else directory / gitdir) / "HEAD"
    return None


def git_branch(cwd: str) -> str:
    """The checked-out branch for `cwd`, as Claude Code records it: a name, "HEAD" when detached, else ""."""
    start = Path(cwd)
    for directory in (start, *start.parents):
        head = _head_file(directory)
        if head is None:
            continue
        try:
            value = head.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        if not value.startswith("ref:"):
            return "HEAD"
        ref = value[len("ref:") :].strip()
        return ref.removeprefix("refs/heads/")
    return ""


def transcript_path(event: Mapping[str, Any]) -> Path | None:
    """The transcript a call belongs to: the subagent's own file when the call came from a subagent. Claude
    Code names it `<session>/subagents/agent-<agent_id>.jsonl` (298 of 298 local subagent files match)."""
    path = event.get("transcript_path")
    if not isinstance(path, str) or not path:
        return None
    agent = event.get("agent_id")
    if isinstance(agent, str) and agent:
        own = Path(path).with_suffix("") / "subagents" / f"agent-{agent}.jsonl"
        if own.is_file():
            return own
    return Path(path)


def live_history(event: Mapping[str, Any], call: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = transcript_path(event)
    if path is None:
        return []
    try:
        with path.open(encoding="utf-8") as handle:
            transcript = parse_transcript(handle)
    except FileNotFoundError:
        return []
    return transcript.history_before(call)


def call_and_view(event: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | str:
    """The guard's call and view for a PreToolUse event, or why the event cannot be judged."""
    if event.get("hook_event_name") != "PreToolUse":
        return f"installed on {event.get('hook_event_name')!r}, expected PreToolUse"
    name, args = event.get("tool_name"), event.get("tool_input")
    if not isinstance(name, str) or not isinstance(args, dict):
        return "hook input has no tool_name and tool_input"
    call = {"name": name, "args": args, "call_id": str(event.get("tool_use_id") or "")}
    cwd = str(event.get("cwd") or "")
    return call, coding_view(cwd, git_branch(cwd) if cwd else "")


# --- the sandboxed guard child ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxSpec:
    guard_id: str
    guard_source: str
    child_source: str
    cpu_s: int
    memory_mb: int
    call_timeout_s: float
    startup_timeout_s: float
    require: tuple[str, ...]


@dataclass(frozen=True)
class Verdict:
    block: str | None
    error: str | None = None


class ChildError(Exception):
    """The child could not start, timed out, died or answered out of protocol."""


class GuardChild:
    """One sandboxed child running one guard, speaking the Ratchet guard child protocol. History is sent
    incrementally: the child keeps the events of the session it last saw."""

    def __init__(self, spec: SandboxSpec) -> None:
        self.spec = spec
        self._buffer = b""
        self._next = 0
        self.session: str | None = None
        self.sent = 0
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", spec.child_source],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={},
                cwd="/",
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise ChildError(f"sandbox did not start: {exc}") from exc
        setup = {
            "source": spec.guard_source,
            "cpu_s": spec.cpu_s,
            "memory_mb": spec.memory_mb,
            "timeout_s": spec.call_timeout_s,
            "require": list(spec.require),
        }
        deadline = time.monotonic() + spec.startup_timeout_s + spec.call_timeout_s
        try:
            self._send(setup, deadline)
            ready = self._recv(deadline)
            if ready.get("loading"):
                # The sandbox is up; loading the guard is untrusted work under the guard's own limit.
                ready = self._recv(time.monotonic() + spec.call_timeout_s + GUARD_WALL_GRACE_S)
        except ChildError:
            self.close()
            raise
        if not ready.get("ready"):
            self.close()
            raise ChildError(f"{ready.get('error')}; layers: {ready.get('isolation')}")

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        for stream in (self.proc.stdin, self.proc.stdout):
            if stream is not None:
                with contextlib.suppress(BrokenPipeError):
                    stream.close()

    def _send(self, obj: Any, deadline: float) -> None:
        assert self.proc.stdin is not None
        fd = self.proc.stdin.fileno()
        os.set_blocking(fd, False)
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        while data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise ChildError("timeout")
            try:
                data = data[os.write(fd, data) :]
            except BrokenPipeError as exc:
                raise ChildError(f"guard process exited with status {self.proc.wait()}") from exc

    def _recv(self, deadline: float) -> dict[str, Any]:
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise ChildError("timeout")
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                raise ChildError(f"guard process exited with status {self.proc.wait()}")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            reply = json.loads(line)
        except ValueError:
            reply = None
        if not isinstance(reply, dict):
            raise ChildError("guard process sent a malformed reply")
        return reply

    def ask(
        self,
        session: str | None,
        call: Mapping[str, Any],
        view: Mapping[str, Any],
        history: Sequence[dict[str, Any]],
    ) -> Verdict:
        """One evaluation. A child that fails here is closed; the verdict carries the error."""
        keep = min(self.sent, len(history)) if session is not None and session == self.session else 0
        index, self._next = self._next, self._next + 1
        request = {
            "i": index,
            "call": dict(call),
            "view": dict(view),
            "h_keep": keep,
            "h_add": list(history[keep:]),
        }
        spec = self.spec
        try:
            self._send(request, time.monotonic() + spec.startup_timeout_s + spec.call_timeout_s)
            # The child times the guard itself; this backstop also covers decoding a large history.
            reply = self._recv(time.monotonic() + spec.startup_timeout_s + spec.call_timeout_s + 0.5)
            if reply.get("s") == index:
                # The child decoded the request and started the guard, which it times in CPU time.
                reply = self._recv(time.monotonic() + spec.call_timeout_s + GUARD_WALL_GRACE_S)
            if reply.get("i") != index:
                raise ChildError("guard process sent a malformed reply")
        except ChildError as exc:
            self.close()
            return Verdict(None, str(exc))
        self.session, self.sent = session, len(history)
        if reply.get("error"):
            return Verdict(None, str(reply["error"]))
        block = reply.get("block")
        return Verdict(None if block is None else str(block))


def evaluate_once(
    spec: SandboxSpec, call: Mapping[str, Any], view: Mapping[str, Any], history: Sequence[dict[str, Any]]
) -> Verdict:
    """Run the guard once in a fresh sandboxed child."""
    try:
        child = GuardChild(spec)
    except ChildError as exc:
        return Verdict(None, str(exc))
    try:
        return child.ask(None, call, view, history)
    finally:
        child.close()


# --- the session server -----------------------------------------------------------------------------------


class Tail:
    """A transcript read incrementally: only complete lines are parsed, and a shrunk file starts over."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.pending = b""
        self.transcript = Transcript()
        self.child: GuardChild | None = None

    def update(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size < self.offset:
            self.offset, self.pending, self.transcript = 0, b"", Transcript()
        if size == self.offset:
            return
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            data = handle.read(size - self.offset)
        self.offset += len(data)
        *complete, self.pending = (self.pending + data).split(b"\n")
        self.transcript.feed(complete)

    def close(self) -> None:
        if self.child is not None:
            self.child.close()


class SessionServer:
    """Verdicts for many sessions of one guard: a tail and a sandboxed child per session, at most
    SERVER_SESSIONS of them, least recently used first out."""

    def __init__(self, spec: SandboxSpec, sessions: int = SERVER_SESSIONS) -> None:
        self.spec = spec
        self.limit = sessions
        self.history = reads_history(spec.guard_source)
        self.tails: OrderedDict[str, Tail] = OrderedDict()

    def _tail(self, key: str, path: Path | None) -> Tail:
        tail = self.tails.pop(key, None) or Tail(path or Path(os.devnull))
        self.tails[key] = tail
        while len(self.tails) > self.limit:
            self.tails.popitem(last=False)[1].close()
        return tail

    def verdict(self, event: Mapping[str, Any]) -> Verdict:
        prepared = call_and_view(event)
        if isinstance(prepared, str):
            return Verdict(None, prepared)
        call, view = prepared
        path = transcript_path(event) if self.history else None
        key = str(path) if path is not None else ""
        tail = self._tail(key, path)
        history: list[dict[str, Any]] = []
        if path is not None:
            tail.update()
            history = tail.transcript.history_before(call)
        if tail.child is None or not tail.child.alive:
            try:
                tail.child = GuardChild(self.spec)
            except ChildError as exc:
                return Verdict(None, str(exc))
        return tail.child.ask(key, call, view, history)

    def close(self) -> None:
        for tail in self.tails.values():
            tail.close()
        self.tails.clear()


def socket_dir() -> Path | None:
    """A directory only this user can enter, or None when that cannot be had."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    directory = (
        Path(base) / "ratchet-hook" if base else Path(tempfile.gettempdir()) / f"ratchet-hook-{os.getuid()}"
    )
    try:
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
    except OSError:
        return None
    if info.st_uid != os.getuid() or info.st_mode & 0o077 or not directory.is_dir() or directory.is_symlink():
        return None
    return directory


def server_key(config: HookConfig) -> str:
    spec = config.spec
    material = [config.guard_sha256, config.child_sha256, config.runtime_sha256, spec.guard_id, spec.cpu_s,
                spec.memory_mb, spec.call_timeout_s, spec.startup_timeout_s, list(spec.require)]  # fmt: skip
    return _sha256(json.dumps(material))[:24]


def _same_user(conn: socket.socket) -> bool:
    if not hasattr(socket, "SO_PEERCRED"):
        return True  # the socket directory is private to this user; no peer credential to read
    creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return int(struct.unpack("3i", creds)[1]) == os.getuid()


def _read_all(conn: socket.socket) -> bytes:
    chunks = []
    while chunk := conn.recv(1 << 16):
        chunks.append(chunk)
    return b"".join(chunks)


def serve_sessions(config: HookConfig, path: Path, idle_s: float = SERVER_IDLE_S) -> int:
    """Answer verdict requests on `path` until idle for `idle_s` or until the socket file is removed. One
    server per socket: a second one finds the lock held and exits at once."""
    lock = open(path.with_suffix(".lock"), "a")  # noqa: SIM115 - held for the server's lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        return 0
    server = SessionServer(config.spec)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
        path.chmod(0o600)
        listener.listen(16)
        listener.settimeout(min(idle_s, 1.0))
        last = time.monotonic()
        while True:
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                if time.monotonic() - last >= idle_s or not path.exists():
                    return 0
                continue
            last = time.monotonic()
            with conn:
                conn.settimeout(SERVER_TIMEOUT_S)
                if not _same_user(conn):
                    continue
                try:
                    event = json.loads(_read_all(conn))
                    verdict = (
                        server.verdict(event) if isinstance(event, dict) else Verdict(None, "bad request")
                    )
                except Exception as exc:  # fail closed: the client gets an error verdict
                    verdict = Verdict(None, f"{type(exc).__name__}: {exc}")
                with contextlib.suppress(OSError):
                    conn.sendall(json.dumps({"block": verdict.block, "error": verdict.error}).encode("utf-8"))
    finally:
        listener.close()
        server.close()
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        lock.close()


def _exchange(path: Path, event: Mapping[str, Any]) -> Verdict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(SERVER_TIMEOUT_S)
        conn.connect(str(path))
        conn.sendall(json.dumps(event).encode("utf-8"))
        conn.shutdown(socket.SHUT_WR)
        reply = json.loads(_read_all(conn))
    if not isinstance(reply, dict):
        raise ValueError("server sent a malformed reply")
    block, error = reply.get("block"), reply.get("error")
    return Verdict(None if block is None else str(block), None if error is None else str(error))


def ask_server(config: HookConfig, event: Mapping[str, Any]) -> Verdict | None:
    """The session server's verdict, starting the server when none is running; None when no server can be
    reached, so the caller evaluates the call itself."""
    directory = socket_dir()
    if directory is None or config.hook_file is None:
        return None
    path = directory / f"{server_key(config)}.sock"
    with contextlib.suppress(OSError, ValueError):
        return _exchange(path, event)
    starter = subprocess.Popen(
        [sys.executable, config.hook_file, "--serve", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + SERVER_START_S
        while time.monotonic() < deadline:
            with contextlib.suppress(OSError, ValueError):
                return _exchange(path, event)
            if starter.poll() not in (None, 0):  # it failed before detaching; 0 means it detached
                return None
            time.sleep(0.05)
        return None
    finally:
        with contextlib.suppress(subprocess.TimeoutExpired):
            starter.wait(timeout=SERVER_START_S)


# --- the hook ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HookConfig:
    """`hook_file` and `runtime_sha256` are set in a generated hook, which then re-hashes its own embedded
    runtime at every call and may start a session server from that file; they are None when this module is
    imported rather than embedded."""

    spec: SandboxSpec
    guard_sha256: str
    child_sha256: str
    runtime_sha256: str | None = None
    hook_file: str | None = None


def embedded_runtime(text: str) -> str | None:
    """The runtime source pasted between the markers of a generated hook, or None without markers."""
    begin, end = text.find(RUNTIME_BEGIN + "\n"), text.find("\n" + RUNTIME_END)
    if begin < 0 or end < begin:
        return None
    return text[begin + len(RUNTIME_BEGIN) + 1 : end].rstrip("\n") + "\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def integrity_error(config: HookConfig) -> str | None:
    spec = config.spec
    if _sha256(spec.guard_source) != config.guard_sha256:
        return "guard source does not match its recorded digest"
    if _sha256(spec.child_source) != config.child_sha256:
        return "sandbox child does not match its recorded digest"
    if config.hook_file is not None:
        runtime = embedded_runtime(Path(config.hook_file).read_text(encoding="utf-8"))
        if runtime is None or _sha256(runtime) != config.runtime_sha256:
            return "hook runtime does not match its recorded digest"
    return None


def decide(config: HookConfig, event: Mapping[str, Any], use_server: bool = True) -> Verdict:
    """The guard's verdict on one PreToolUse event. Every problem is an error verdict, never an allow."""
    problem = integrity_error(config)
    if problem is not None:
        return Verdict(None, problem)
    prepared = call_and_view(event)
    if isinstance(prepared, str):
        return Verdict(None, prepared)
    if use_server:
        answer = ask_server(config, event)
        if answer is not None:
            return answer
    call, view = prepared
    history = live_history(event, call) if reads_history(config.spec.guard_source) else []
    return evaluate_once(config.spec, call, view, history)


def block_message(guard_id: str, verdict: Verdict) -> str | None:
    if verdict.error is not None:
        return f"blocked by guard {guard_id}: {GUARD_ERROR_REASON}: {verdict.error}"
    if verdict.block is not None:
        return f"blocked by guard {guard_id}: {verdict.block}"
    return None


def run_hook(config: HookConfig, stdin: IO[str], stdout: IO[str], stderr: IO[str]) -> int:
    """Claude Code PreToolUse protocol: exit 0 with no output to leave the call to the normal permission
    flow, or deny with a JSON decision and exit 2, so the call is blocked even where JSON is ignored."""
    try:
        event = json.loads(stdin.read())
        if not isinstance(event, dict):
            raise ValueError("hook input is not a JSON object")
        verdict = decide(config, event)
    except Exception as exc:  # fail closed: a hook that cannot decide blocks
        verdict = Verdict(None, f"{type(exc).__name__}: {exc}")
    reason = block_message(config.spec.guard_id, verdict)
    if reason is None:
        return 0
    decision = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }
    stdout.write(json.dumps({"hookSpecificOutput": decision}))
    stderr.write(reason + "\n")
    return 2


def main(config: HookConfig, argv: Sequence[str]) -> int:
    """A generated hook's entry point: `--serve <socket>` detaches and runs the session server, anything
    else judges one call."""
    if len(argv) == 2 and argv[0] == "--serve":
        path = Path(argv[1])
        if integrity_error(config) is not None or len(os.fsencode(path)) > SOCKET_PATH_MAX:
            return 1
        if os.fork() > 0:
            return 0  # detach: the hook that started the server reaps this process at once
        return serve_sessions(config, path)
    return run_hook(config, sys.stdin, sys.stdout, sys.stderr)
