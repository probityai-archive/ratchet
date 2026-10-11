"""Guard runtime: evaluates untrusted guard source in sandboxed child processes, failing closed.

A guard's `source` defines `check(call, view, history) -> str | None`. Inside the sandbox the guard sees:
  call     a frozen `ToolCall` with `.name`, `.args` (dict) and `.call_id`
  view     the world's `StateView` as a plain JSON dict
  history  a list of frozen `Event`s (`.seq`, `.kind`, `.payload`, `.ts`); payloads are read-only mappings
Returning a string blocks the call and the string is the reason shown to the model; `None` allows it.
The module source is re-executed for every evaluation, so a guard cannot carry state from one call to
the next and a replay gives the same answer as the live run. Guards may import the modules listed in
`_guard_child.PRELOAD`; nothing else is reachable once the sandbox is closed.

Each guard gets its own child process (`python -I -S`, empty environment), so one guard can never forge
another guard's decision. The child applies rlimits, a new network namespace, a seccomp filter and a
Landlock ruleset before it reads the guard source (see `_guard_child`), and reports which layers took
effect; the runtime refuses to start a child that lacks a required layer.

Fail closed: a guard that does not compile, does not define `check`, raises, returns a non-string, runs
past its time limit or kills its process yields a BLOCK decision with reason `GUARD_ERROR_REASON` and the
underlying error in `GuardDecision.error`. Timed-out or dead children are killed and replaced.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import sys
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ratchet.types import Event, Guard, ToolCall

GUARD_ERROR_REASON = "guard error"
CHILD_PATH = Path(__file__).with_name("_guard_child.py")
ALL_LAYERS = ("rlimits", "netns", "seccomp", "landlock")


class GuardSourceError(ValueError):
    """Guard source was rejected: it does not compile or does not define a callable `check`."""


class SandboxUnavailableError(RuntimeError):
    """The child could not apply an isolation layer the runtime requires."""


@dataclass(frozen=True)
class GuardLimits:
    call_timeout_s: float = 2.0
    startup_timeout_s: float = 10.0
    cpu_s: int = 120
    memory_mb: int = 512
    # Layers that must take effect or the child is refused. Network isolation is required by the
    # contract; netns and seccomp each provide it, and both are required by default on Linux.
    require: tuple[str, ...] = ("rlimits", "netns", "seccomp", "landlock")
    # Use at most this Landlock ABI, to run the sandbox as it runs on an older kernel. None uses the
    # kernel's own ABI. Isolation never depends on it: seccomp alone refuses signals and sockets.
    landlock_max_abi: int | None = None


@dataclass(frozen=True)
class GuardDecision:
    guard_id: str
    blocked: bool
    reason: str | None
    error: str | None = None
    elapsed_s: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardItem:
    """One evaluation for batch mode."""

    guard: Guard
    call: ToolCall
    view: Mapping[str, Any]
    history: Sequence[Event] = field(default_factory=list)


def _error_decision(guard_id: str, error: str, elapsed: float = 0.0) -> GuardDecision:
    return GuardDecision(
        guard_id=guard_id, blocked=True, reason=GUARD_ERROR_REASON, error=error, elapsed_s=elapsed
    )


def compile_guard(source: str) -> None:
    """Parent-side syntax check. Compiling never executes the source."""
    try:
        compile(source, "<guard>", "exec")
    except (SyntaxError, ValueError) as exc:
        raise GuardSourceError(f"guard source does not compile: {exc}") from exc


class _ChildDiedError(Exception):
    pass


class GuardProcess:
    """One sandboxed child running one guard. Request/response, one line each, with hard deadlines."""

    def __init__(self, guard: Guard, limits: GuardLimits) -> None:
        self.guard = guard
        self.limits = limits
        self.isolation: dict[str, str] = {}
        self._buffer = b""
        self._sent_history: list[Event] = []
        self._next = 0
        self._proc = subprocess.Popen(
            [sys.executable, "-I", "-S", str(CHILD_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={},
            cwd="/",
            close_fds=True,
            start_new_session=True,
        )
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _start(self) -> None:
        setup = {
            "source": self.guard.source,
            "cpu_s": self.limits.cpu_s,
            "memory_mb": self.limits.memory_mb,
            "timeout_s": self.limits.call_timeout_s,
            "require": list(self.limits.require),
            "landlock_max_abi": self.limits.landlock_max_abi,
        }
        deadline = time.monotonic() + self.limits.startup_timeout_s + self.limits.call_timeout_s
        try:
            self._write(setup, deadline)
            reply = self._read(deadline)
        except (TimeoutError, _ChildDiedError) as exc:
            raise GuardSourceError(f"guard failed to start: {exc}") from exc
        self.isolation = dict(reply.get("isolation") or {})
        if not reply.get("ready"):
            error = str(reply.get("error"))
            if error.startswith("isolation unavailable"):
                raise SandboxUnavailableError(f"{error}; layers: {self.isolation}")
            raise GuardSourceError(error)

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is not None:
                stream.close()

    def _write(self, obj: Any, deadline: float) -> None:
        stdin = self._proc.stdin
        assert stdin is not None
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        fd = stdin.fileno()
        os.set_blocking(fd, False)
        while data:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timeout")
            _, writable, _ = select.select([], [fd], [], remaining)
            if not writable:
                raise TimeoutError("timeout")
            try:
                data = data[os.write(fd, data) :]
            except BrokenPipeError as exc:
                raise _ChildDiedError("guard process exited") from exc

    def _read(self, deadline: float) -> dict[str, Any]:
        stdout = self._proc.stdout
        assert stdout is not None
        fd = stdout.fileno()
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timeout")
            readable, _, _ = select.select([fd], [], [], remaining)
            if not readable:
                raise TimeoutError("timeout")
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                code = self._proc.wait()
                raise _ChildDiedError(f"guard process exited with status {code}")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        reply = json.loads(line)
        if not isinstance(reply, dict):
            raise _ChildDiedError("guard process sent a malformed reply")
        return reply

    def _encode(self, item: GuardItem, index: int) -> bytes:
        sent = self._sent_history
        history = item.history
        keep = 0
        limit = min(len(sent), len(history))
        while keep < limit and sent[keep] is history[keep]:
            keep += 1
        add = list(history[keep:])
        self._sent_history = sent[:keep] + add
        call = item.call
        request = {
            "i": index,
            "call": {"name": call.name, "args": call.args, "call_id": call.call_id},
            "view": item.view,
            "h_keep": keep,
            "h_add": [{"seq": e.seq, "kind": e.kind, "payload": e.payload, "ts": e.ts} for e in add],
        }
        return (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")

    def _decision(self, reply: dict[str, Any], elapsed: float) -> GuardDecision:
        guard_id = self.guard.id
        if reply.get("error"):
            return _error_decision(guard_id, str(reply["error"]), elapsed)
        block = reply.get("block")
        if block is None:
            return GuardDecision(guard_id=guard_id, blocked=False, reason=None, elapsed_s=elapsed)
        return GuardDecision(guard_id=guard_id, blocked=True, reason=str(block), elapsed_s=elapsed)

    def evaluate(self, call: ToolCall, view: Mapping[str, Any], history: Sequence[Event]) -> GuardDecision:
        """Evaluate once. Raises nothing: every failure becomes a fail-closed decision."""
        return self.evaluate_many([GuardItem(self.guard, call, view, history)])[0]

    def evaluate_many(self, items: Sequence[GuardItem], window: int = 64) -> list[GuardDecision]:
        """Pipelined evaluation: up to `window` requests in flight, replies read as they arrive.

        Returns decisions for a prefix of `items`. When the child times out or dies, the item it was
        working on gets a fail-closed decision, the child is killed, and the shorter list is returned so the
        caller can continue the rest on a fresh child. Each item's time limit runs from the moment the child
        could start it (its send time, or the previous reply, whichever is later)."""
        if not items:
            return []
        if not self.alive:
            return [_error_decision(self.guard.id, "guard process is not running")]
        stdin, stdout = self._proc.stdin, self._proc.stdout
        assert stdin is not None and stdout is not None
        in_fd, out_fd = stdin.fileno(), stdout.fileno()
        os.set_blocking(in_fd, False)
        limit = self.limits.call_timeout_s + 0.5
        out: list[GuardDecision] = []
        inflight: deque[tuple[int, float]] = deque()
        pending = b""
        sent = 0
        last_reply = time.monotonic()
        while len(out) < len(items):
            while not pending and sent < len(items) and len(inflight) < window:
                pending = self._encode(items[sent], self._next)
                inflight.append((self._next, time.monotonic()))
                self._next += 1
                sent += 1
            head_start = max(inflight[0][1], last_reply)
            remaining = head_start + limit - time.monotonic()
            if remaining <= 0:
                self.close()
                out.append(_error_decision(self.guard.id, "timeout", time.monotonic() - head_start))
                return out
            readable, writable, _ = select.select([out_fd], [in_fd] if pending else [], [], remaining)
            if writable:
                try:
                    pending = pending[os.write(in_fd, pending) :]
                except BrokenPipeError:
                    pending = b""
            if not readable:
                continue
            chunk = os.read(out_fd, 1 << 16)
            if not chunk:
                code = self._proc.wait()
                self.close()
                error = f"guard process exited with status {code}"
                out.append(_error_decision(self.guard.id, error, time.monotonic() - head_start))
                return out
            self._buffer += chunk
            now = time.monotonic()
            *lines, self._buffer = self._buffer.split(b"\n")
            for line in lines:
                index, started = inflight.popleft()
                try:
                    reply = json.loads(line)
                except ValueError:
                    reply = None
                if not isinstance(reply, dict) or reply.get("i") != index:
                    self.close()
                    out.append(_error_decision(self.guard.id, "guard process sent a malformed reply"))
                    return out
                out.append(self._decision(reply, now - max(started, last_reply)))
                last_reply = now
        return out


def _guard_key(guard: Guard) -> str:
    return guard.id + ":" + hashlib.sha256(guard.source.encode("utf-8")).hexdigest()


class GuardRuntime:
    """Owns the sandboxed guard processes for one run (live checks) or one replay (batch mode)."""

    def __init__(self, limits: GuardLimits | None = None, *, workers: int | None = None) -> None:
        self.limits = limits or GuardLimits()
        self.workers = max(1, workers or (os.cpu_count() or 1))
        self._live: dict[str, GuardProcess] = {}
        self._rejected: dict[str, str] = {}

    def __enter__(self) -> GuardRuntime:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for proc in self._live.values():
            proc.close()
        self._live.clear()

    def validate(self, guard: Guard) -> dict[str, str]:
        """Load the guard in a sandbox and return the isolation achieved; raise if the source is rejected."""
        compile_guard(guard.source)
        proc = GuardProcess(guard, self.limits)
        try:
            return dict(proc.isolation)
        finally:
            proc.close()

    def _process(self, guard: Guard) -> GuardProcess | str:
        key = _guard_key(guard)
        if key in self._rejected:
            return self._rejected[key]
        proc = self._live.get(key)
        if proc is not None and proc.alive:
            return proc
        try:
            compile_guard(guard.source)
            proc = GuardProcess(guard, self.limits)
        except GuardSourceError as exc:
            self._rejected[key] = str(exc)
            return str(exc)
        self._live[key] = proc
        return proc

    def check(
        self, guard: Guard, call: ToolCall, view: Mapping[str, Any], history: Sequence[Event]
    ) -> GuardDecision:
        proc = self._process(guard)
        if isinstance(proc, str):
            return _error_decision(guard.id, proc)
        return proc.evaluate(call, view, history)

    def check_all(
        self, guards: Iterable[Guard], call: ToolCall, view: Mapping[str, Any], history: Sequence[Event]
    ) -> list[GuardDecision]:
        """Every guard, in order. All decisions are returned so each one is recorded."""
        return [self.check(g, call, view, history) for g in guards]

    def batch(self, items: Sequence[GuardItem]) -> list[GuardDecision]:
        """Evaluate many items. Items for one guard are split into up to `workers` shards; each shard runs
        in its own sandboxed child, and shards run concurrently. Results come back in input order."""
        by_guard: dict[str, list[int]] = {}
        for index, item in enumerate(items):
            by_guard.setdefault(_guard_key(item.guard), []).append(index)
        shards: list[list[int]] = []
        total = len(items)
        for indices in by_guard.values():
            count = max(1, min(self.workers, round(self.workers * len(indices) / max(total, 1))))
            size = -(-len(indices) // count)
            shards.extend(indices[i : i + size] for i in range(0, len(indices), size))
        results: list[GuardDecision | None] = [None] * total

        def run(shard: list[int]) -> None:
            guard = items[shard[0]].guard
            pending = list(shard)
            while pending:
                try:
                    compile_guard(guard.source)
                    proc = GuardProcess(guard, self.limits)
                except GuardSourceError as exc:
                    for index in pending:
                        results[index] = _error_decision(guard.id, str(exc))
                    return
                try:
                    done = proc.evaluate_many([items[i] for i in pending])
                finally:
                    proc.close()
                for index, decision in zip(pending, done, strict=False):
                    results[index] = decision
                pending = pending[len(done) :]

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for future in [pool.submit(run, shard) for shard in shards]:
                future.result()
        done = [r for r in results if r is not None]
        assert len(done) == total
        return done
