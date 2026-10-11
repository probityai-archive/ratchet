"""Guard sandbox child. Runs ONE guard's untrusted source; stdlib only; started as `python -I -S <this file>`.

This file is executed by the guard runtime in a fresh interpreter with an empty environment. It reads one
JSON setup line, locks itself down, loads the guard, then answers one JSON request per line on stdin with
one JSON reply per line on a private copy of stdout.

Lockdown, applied before any untrusted byte is executed (each layer is reported back; none is skipped
silently):
  rlimits   CPU seconds, address space, no child processes, no file writes, no core dumps.
  netns     a new user + network namespace (`unshare(CLONE_NEWUSER | CLONE_NEWNET)`): no interface is up,
            so no TCP/UDP route exists, loopback included.
  seccomp   socket, socketpair, ptrace, process_vm_readv/writev and io_uring are refused with EPERM, which
            also closes filesystem unix sockets (the docker and session-bus sockets) that a network
            namespace does not cover. Every signal-sending call (kill, tkill, tgkill, rt_sigqueueinfo,
            rt_tgsigqueueinfo, pidfd_open, pidfd_send_signal) is refused too, so the guard cannot signal
            its parent on any kernel; Landlock's signal scoping needs ABI 6 (Linux 6.12) and is only a
            second layer where the kernel has it.
  landlock  every filesystem access right, TCP bind/connect, abstract unix sockets and signals to
            processes outside the sandbox are handled with no rule granting any of them back.

Coverage: the layer installers and `main` run only inside a locked-down child, where a coverage tracer
cannot write its data. They are exercised end to end by tests/agent/test_guards.py, which starts real
children, asserts every layer reports active, and asserts each escape attempt (network, unix sockets, file
reads and writes, fork, exec, signals, memory) is refused.

Startup : setup line in; {"loading": true} once locked down (untrusted code runs from here), then
          {"ready": bool, "isolation": {...}, "error"?: str}.
Request : {"i": n, "call": {...}, "view": {...}, "h_keep": k, "h_add": [event, ...]}
          The child keeps its last history; it keeps the first k events and appends `h_add`.
Replies : {"s": n} when the request is decoded and the guard starts, then
          {"i": n, "block": str | null, "error": str | null, "guard_s": float}.
          The guard's time limit is CPU time (ITIMER_PROF) measured from the start marker only.
"""

import builtins
import ctypes
import json
import os
import platform
import resource
import signal
import sys
import time
import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

# Modules a guard may import. Everything else fails to import once the filesystem is closed.
PRELOAD = (
    "re",
    "math",
    "decimal",
    "fractions",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "string",
    "operator",
    "statistics",
    "encodings.ascii",
    "encodings.latin_1",
    "encodings.utf_8",
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    call_id: str


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    payload: Any
    ts: float


class GuardTimeout(BaseException):
    pass


def freeze(value: Any) -> Any:
    """Deep read-only copy, so a guard cannot change the history the next evaluation sees."""
    if isinstance(value, dict):
        return types.MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value


# --- lockdown layers --------------------------------------------------------------------------------------


def apply_rlimits(
    cpu_s: int, memory_mb: int
) -> str:  # pragma: no cover - runs only inside the sandbox; see module docstring
    limit = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f"cpu={cpu_s}s as={memory_mb}MiB nproc=0 fsize=0 core=0"


def apply_netns() -> str:  # pragma: no cover - runs only inside the sandbox; see module docstring
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)
    return "new user and network namespace, no interface up"


_SYSCALLS = {
    # machine: (audit arch, x32 bit check, {name: nr})
    "x86_64": (
        0xC000003E,
        True,
        {
            "socket": 41,
            "socketpair": 53,
            "ptrace": 101,
            "process_vm_readv": 310,
            "process_vm_writev": 311,
            "io_uring_setup": 425,
            "io_uring_enter": 426,
            "io_uring_register": 427,
            "kill": 62,
            "tkill": 200,
            "tgkill": 234,
            "rt_sigqueueinfo": 129,
            "rt_tgsigqueueinfo": 297,
            "pidfd_open": 434,
            "pidfd_send_signal": 424,
        },
    ),
    "aarch64": (
        0xC00000B7,
        False,
        {
            "socket": 198,
            "socketpair": 199,
            "ptrace": 117,
            "process_vm_readv": 270,
            "process_vm_writev": 271,
            "io_uring_setup": 425,
            "io_uring_enter": 426,
            "io_uring_register": 427,
            "kill": 129,
            "tkill": 130,
            "tgkill": 131,
            "rt_sigqueueinfo": 138,
            "rt_tgsigqueueinfo": 240,
            "pidfd_open": 434,
            "pidfd_send_signal": 424,
        },
    ),
}


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(_SockFilter))]


def _libc() -> Any:
    return ctypes.CDLL(None, use_errno=True)


def _no_new_privs(libc: Any) -> None:  # pragma: no cover - runs only inside the sandbox; see module docstring
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")


def seccomp_program(machine: str) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """Classic BPF: kill on a foreign arch, EPERM for the denied syscalls (and x32 numbers), else allow."""
    if machine not in _SYSCALLS:
        raise OSError(f"no seccomp table for {machine}")
    arch, x32, table = _SYSCALLS[machine]
    ld, jeq, jge, ret = 0x20, 0x15, 0x35, 0x06
    allow, deny, kill = 0x7FFF0000, 0x00050000 | 1, 0x80000000  # RET_ALLOW, RET_ERRNO|EPERM, KILL_PROCESS
    checks: list[tuple[int, int, int, int]] = []
    if x32:
        checks.append((jge, 0, 0, 0x40000000))
    checks.extend((jeq, 0, 0, nr) for nr in sorted(table.values()))
    prog: list[tuple[int, int, int, int]] = [
        (ld, 0, 0, 4),
        (jeq, 1, 0, arch),
        (ret, 0, 0, kill),
        (ld, 0, 0, 0),
    ]
    for index, (code, _, _, k) in enumerate(checks):
        # On match jump to the deny instruction, which sits after the remaining checks and the allow.
        prog.append((code, len(checks) - index, 0, k))
    prog.append((ret, 0, 0, allow))
    prog.append((ret, 0, 0, deny))
    return prog, sorted(table)


def apply_seccomp() -> str:  # pragma: no cover - runs only inside the sandbox; see module docstring
    prog, names = seccomp_program(platform.machine())
    filters = (_SockFilter * len(prog))(*[_SockFilter(c, t, f, k) for c, t, f, k in prog])
    fprog = _SockFprog(len(prog), filters)
    libc = _libc()
    _no_new_privs(libc)
    if libc.prctl(22, 2, ctypes.byref(fprog), 0, 0) != 0:  # PR_SET_SECCOMP, SECCOMP_MODE_FILTER
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP) failed")
    return "denied: " + ",".join(names)


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


def landlock_ruleset(abi: int) -> tuple[_RulesetAttr, int, str]:
    """Handle every right this ABI knows, so that with no rules added each one is denied."""
    fs = (1 << 13) - 1
    if abi >= 2:
        fs |= 1 << 13  # REFER
    if abi >= 3:
        fs |= 1 << 14  # TRUNCATE
    if abi >= 5:
        fs |= 1 << 15  # IOCTL_DEV
    net = 0b11 if abi >= 4 else 0  # BIND_TCP, CONNECT_TCP
    scoped = 0b11 if abi >= 6 else 0  # ABSTRACT_UNIX_SOCKET, SIGNAL
    size = 8 if abi < 4 else (16 if abi < 6 else 24)
    parts = ["fs:all"] + (["tcp"] if net else []) + (["abstract-unix", "signal"] if scoped else [])
    return _RulesetAttr(fs, net, scoped), size, f"abi {abi}, denied " + ",".join(parts)


def apply_landlock(max_abi: int | None) -> str:  # pragma: no cover - runs only inside the sandbox
    libc = _libc()
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(ctypes.c_long(444), None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    if max_abi is not None:
        abi = min(abi, max_abi)  # emulate an older kernel; the rules of a lower ABI are a subset
    if abi < 1:
        raise OSError(ctypes.get_errno(), "landlock is not available")
    attr, size, description = landlock_ruleset(abi)
    fd = libc.syscall(ctypes.c_long(444), ctypes.byref(attr), ctypes.c_size_t(size), ctypes.c_uint32(0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    _no_new_privs(libc)
    if libc.syscall(ctypes.c_long(446), ctypes.c_int(fd), ctypes.c_uint32(0)) != 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    os.close(fd)
    return description


# --- protocol ---------------------------------------------------------------------------------------------


def _send(fd: int, obj: Any) -> None:
    data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    while data:
        data = data[os.write(fd, data) :]


def _on_alarm(signum: int, frame: Any) -> None:
    raise GuardTimeout()


def _evaluate(code: Any, call: ToolCall, view: Any, history: list[Event], timeout_s: float) -> dict[str, Any]:
    """Run the guard under a CPU-time alarm. Only the guard's own work counts against `timeout_s`: the
    request was decoded before the alarm starts, and CPU time does not advance while the machine's load
    keeps this process waiting. A guard that blocks without using CPU is stopped by the parent instead."""
    started = time.perf_counter()
    signal.setitimer(signal.ITIMER_PROF, timeout_s)
    try:
        namespace: dict[str, Any] = {"__name__": "guard", "__builtins__": builtins}
        exec(code, namespace)
        result = namespace["check"](call, view, history)
    except GuardTimeout:
        return {"block": None, "error": "timeout", "guard_s": time.perf_counter() - started}
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
        return {"block": None, "error": error, "guard_s": time.perf_counter() - started}
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
    elapsed = time.perf_counter() - started
    if result is None:
        return {"block": None, "error": None, "guard_s": elapsed}
    if isinstance(result, str):
        return {"block": result[:1000], "error": None, "guard_s": elapsed}
    error = f"check returned {type(result).__name__}, expected str or None"
    return {"block": None, "error": error, "guard_s": elapsed}


def _load(source: str) -> Any:
    code = compile(source, "<guard>", "exec")
    namespace: dict[str, Any] = {"__name__": "guard", "__builtins__": builtins}
    exec(code, namespace)
    if not callable(namespace.get("check")):
        raise ValueError("guard source does not define a callable check(call, view, history)")
    return code


def lockdown(
    setup: dict[str, Any],
) -> dict[str, str]:  # pragma: no cover - runs only inside the sandbox; see module docstring
    return run_layers(
        (
            ("rlimits", lambda: apply_rlimits(int(setup["cpu_s"]), int(setup["memory_mb"]))),
            ("netns", apply_netns),
            ("seccomp", apply_seccomp),
            ("landlock", lambda: apply_landlock(setup.get("landlock_max_abi"))),
        )
    )


def run_layers(layers: Iterable[tuple[str, Callable[[], str]]]) -> dict[str, str]:
    isolation: dict[str, str] = {}
    for name, layer in layers:
        try:
            isolation[name] = layer()
        except (OSError, ValueError, AttributeError) as exc:
            isolation[name] = f"UNAVAILABLE: {exc}"
    return isolation


def serve(
    setup: dict[str, Any],
    isolation: dict[str, str],
    lines: Iterable[bytes],
    send: Callable[[dict[str, Any]], None],
) -> None:
    """Report readiness, load the guard, then answer each request line until stdin closes."""
    missing = [n for n in setup["require"] if isolation.get(n, "UNAVAILABLE").startswith("UNAVAILABLE")]
    if missing:
        send({"ready": False, "isolation": isolation, "error": f"isolation unavailable: {missing}"})
        return
    timeout_s = float(setup["timeout_s"])
    signal.signal(signal.SIGPROF, _on_alarm)
    # Tells the parent that untrusted code starts now: time before this marker is sandbox startup.
    send({"loading": True})
    signal.setitimer(signal.ITIMER_PROF, timeout_s)
    try:
        code = _load(setup["source"])
    except GuardTimeout:
        send({"ready": False, "isolation": isolation, "error": "timeout while loading guard"})
        return
    except BaseException as exc:
        send({"ready": False, "isolation": isolation, "error": f"{type(exc).__name__}: {exc}"[:500]})
        return
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
    send({"ready": True, "isolation": isolation})
    history: list[Event] = []
    for line in lines:
        request = json.loads(line)
        del history[int(request["h_keep"]) :]
        history.extend(
            Event(seq=e["seq"], kind=e["kind"], payload=freeze(e["payload"]), ts=e["ts"])
            for e in request["h_add"]
        )
        raw = request["call"]
        call = ToolCall(name=raw["name"], args=raw["args"], call_id=raw["call_id"])
        snapshot = list(history)
        # Decoding is done; the guard's own time starts here, and the parent times it from this marker.
        send({"s": request["i"]})
        reply = _evaluate(code, call, request["view"], snapshot, timeout_s)
        reply["i"] = request["i"]
        send(reply)


def main() -> None:  # pragma: no cover - runs only inside the sandbox; see module docstring
    proto = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    setup = json.loads(sys.stdin.buffer.readline())
    for name in PRELOAD:
        __import__(name)
    isolation = lockdown(setup)
    serve(setup, isolation, sys.stdin.buffer, lambda obj: _send(proto, obj))


if __name__ == "__main__":
    main()
