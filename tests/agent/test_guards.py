from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator

import pytest
from ratchet_cases.types import Event, ToolCall

from ratchet.agent.guards import (
    GUARD_ERROR_REASON,
    INFRASTRUCTURE_ERROR_REASON,
    GuardInfrastructureError,
    GuardItem,
    GuardLimits,
    GuardProcess,
    GuardRuntime,
    GuardSourceError,
    SandboxUnavailableError,
)
from ratchet.types import Guard

CALL = ToolCall(name="refund", args={"amount": 50}, call_id="c1")
VIEW = {"limit": 10}


def guard(body: str, gid: str = "g") -> Guard:
    return Guard(id=gid, description="test guard", source=textwrap.dedent(body))


LIMIT_GUARD = guard(
    """
    def check(call, view, history):
        if call.name == "refund" and call.args.get("amount", 0) > view["limit"]:
            return f"refund {call.args['amount']} exceeds limit {view['limit']}"
        return None
    """,
    "limit",
)


@pytest.fixture(scope="module")
def runtime() -> Iterator[GuardRuntime]:
    with GuardRuntime(GuardLimits(call_timeout_s=1.0), workers=2) as rt:
        yield rt


def test_validate_reports_every_isolation_layer(runtime: GuardRuntime) -> None:
    isolation = runtime.validate(LIMIT_GUARD)
    assert set(isolation) == {"rlimits", "netns", "seccomp", "landlock"}
    assert not any(v.startswith("UNAVAILABLE") for v in isolation.values()), isolation


def test_block_and_allow(runtime: GuardRuntime) -> None:
    blocked = runtime.check(LIMIT_GUARD, CALL, VIEW, [])
    assert blocked.blocked and blocked.reason == "refund 50 exceeds limit 10" and blocked.error is None
    allowed = runtime.check(LIMIT_GUARD, ToolCall("refund", {"amount": 5}, "c2"), VIEW, [])
    assert not allowed.blocked and allowed.reason is None
    payload = allowed.to_payload()
    assert payload["guard_id"] == "limit" and payload["blocked"] is False


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("def check(call, view, history)\n    return None\n", "does not compile"),
        ("x = 1\n", "does not define a callable check"),
        ("check = 3\n", "does not define a callable check"),
        ("raise RuntimeError('boom at import')\n", "boom at import"),
        ("while True:\n    pass\n", "timeout"),
    ],
)
def test_rejected_sources(source: str, match: str) -> None:
    with GuardRuntime(GuardLimits(call_timeout_s=0.5)) as rt, pytest.raises(GuardSourceError, match=match):
        rt.validate(guard(source))


def test_rejected_source_fails_closed_on_check(runtime: GuardRuntime) -> None:
    bad = guard("x = 1\n", "bad")
    first = runtime.check(bad, CALL, VIEW, [])
    second = runtime.check(bad, CALL, VIEW, [])
    assert first.blocked and first.reason == GUARD_ERROR_REASON and "check" in (first.error or "")
    assert second.error == first.error


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("def check(call, view, history):\n    raise KeyError('x')\n", "KeyError"),
        ("def check(call, view, history):\n    return 42\n", "check returned int"),
        ("def check(call, view, history):\n    return view['missing']\n", "KeyError"),
    ],
)
def test_guard_errors_block(runtime: GuardRuntime, body: str, error: str) -> None:
    decision = runtime.check(guard(body, "err"), CALL, VIEW, [])
    assert decision.blocked and decision.reason == GUARD_ERROR_REASON and error in (decision.error or "")


def test_infinite_loop_times_out_and_child_is_replaced() -> None:
    looping = guard(
        """
        def check(call, view, history):
            if call.args.get("loop"):
                while True:
                    pass
            return None
        """
    )
    with GuardRuntime(GuardLimits(call_timeout_s=0.3)) as rt:
        started = time.monotonic()
        decision = rt.check(looping, ToolCall("x", {"loop": True}, "c"), {}, [])
        assert time.monotonic() - started < 2.0
        assert decision.blocked and decision.reason == GUARD_ERROR_REASON and decision.error == "timeout"
        assert not rt.check(looping, ToolCall("x", {}, "c"), {}, []).blocked


def test_guard_that_swallows_the_timer_is_killed_by_the_parent() -> None:
    stubborn = guard(
        """
        def check(call, view, history):
            while True:
                try:
                    while True:
                        pass
                except BaseException:
                    pass
        """
    )
    with GuardRuntime(GuardLimits(call_timeout_s=0.3, guard_wall_grace_s=0.3)) as rt:
        started = time.monotonic()
        decision = rt.check(stubborn, CALL, VIEW, [])
        assert decision.blocked and decision.error == "timeout" and time.monotonic() - started < 2.0


ESCAPES = {
    "tcp": "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=1)",
    "loopback": "import socket\nsocket.create_connection(('127.0.0.1', 8787), timeout=1)",
    "unix": "import socket\ns = socket.socket(socket.AF_UNIX)\ns.connect('/var/run/docker.sock')",
    "read_file": "open('/etc/hostname').read()",
    "list_home": "import os\nos.listdir('/home')",
    "write_file": "open('/tmp/ratchet-guard-escape', 'w').write('x')",
    "fork": "import os\nos.fork()",
    "exec": "import os\nos.execv('/bin/true', ['true'])",
    "import_new_module": "import sqlite3",
    "memory": "x = bytearray(2 * 1024 * 1024 * 1024)",
    "env": "import os\nassert not os.environ, dict(os.environ)\nraise RuntimeError('env empty')",
}


@pytest.mark.parametrize("name", sorted(ESCAPES))
def test_sandbox_refuses_escape(runtime: GuardRuntime, name: str) -> None:
    body = (
        "def check(call, view, history):\n" + textwrap.indent(ESCAPES[name], "    ") + "\n    return None\n"
    )
    decision = runtime.check(guard(body, f"escape-{name}"), CALL, VIEW, [])
    assert decision.blocked and decision.reason == GUARD_ERROR_REASON, (name, decision)
    assert not os.path.exists("/tmp/ratchet-guard-escape")


def test_history_is_frozen_and_state_does_not_carry_over(runtime: GuardRuntime) -> None:
    probe = guard(
        """
        seen = []
        def check(call, view, history):
            seen.append(1)
            if history:
                try:
                    history[0].payload["x"] = 1
                except TypeError:
                    pass
                else:
                    return "payload was writable"
            kinds = ",".join(e.kind for e in history)
            return f"{len(seen)}|{kinds}|{history[-1].payload['args'][0] if history else ''}"
        """,
        "probe",
    )
    e0 = Event(seq=0, kind="model", payload={"args": [7]}, ts=1.0)
    e1 = Event(seq=1, kind="tool_call", payload={"args": [8]}, ts=2.0)
    e2 = Event(seq=2, kind="tool_result", payload={"args": [9]}, ts=3.0)
    assert runtime.check(probe, CALL, VIEW, [e0]).reason == "1|model|7"
    assert runtime.check(probe, CALL, VIEW, [e0, e1]).reason == "1|model,tool_call|8"
    assert runtime.check(probe, CALL, VIEW, [e0, e2]).reason == "1|model,tool_result|9"
    assert runtime.check(probe, CALL, VIEW, []).reason == "1||"


def test_check_all_returns_one_decision_per_guard(runtime: GuardRuntime) -> None:
    allow = guard("def check(call, view, history):\n    return None\n", "allow")
    decisions = runtime.check_all([allow, LIMIT_GUARD], CALL, VIEW, [])
    assert [(d.guard_id, d.blocked) for d in decisions] == [("allow", False), ("limit", True)]


def test_batch_matches_live_checks_in_order() -> None:
    allow = guard("def check(call, view, history):\n    return None\n", "allow")
    bad = guard("nothing = 1\n", "bad")
    history = [Event(seq=i, kind="model", payload={"i": i}, ts=float(i)) for i in range(5)]
    items = []
    for amount in range(20):
        call = ToolCall("refund", {"amount": amount}, f"c{amount}")
        prefix = history[: amount % 5]
        items += [GuardItem(LIMIT_GUARD, call, VIEW, prefix), GuardItem(allow, call, VIEW, prefix)]
    items.append(GuardItem(bad, CALL, VIEW, []))
    items.append(GuardItem(bad, CALL, VIEW, []))
    with GuardRuntime(workers=3) as rt:
        results = rt.batch(items)
    assert len(results) == len(items)
    for item, result in zip(items, results, strict=True):
        assert result.guard_id == item.guard.id
        if item.guard.id == "limit":
            assert result.blocked == (item.call.args["amount"] > 10)
        elif item.guard.id == "allow":
            assert not result.blocked
        else:
            assert result.reason == GUARD_ERROR_REASON
    assert GuardRuntime(workers=1).batch([]) == []


def test_batch_recovers_after_a_timeout_mid_shard() -> None:
    sometimes = guard(
        """
        def check(call, view, history):
            if call.args["n"] == 2:
                while True:
                    pass
            return "n=%d" % call.args["n"]
        """
    )
    items = [GuardItem(sometimes, ToolCall("t", {"n": n}, str(n)), {}, []) for n in range(5)]
    with GuardRuntime(GuardLimits(call_timeout_s=0.3), workers=1) as rt:
        results = rt.batch(items)
    assert [r.reason for r in results] == ["n=0", "n=1", GUARD_ERROR_REASON, "n=3", "n=4"]
    assert results[2].error == "timeout"


def test_missing_required_layer_refuses_to_start() -> None:
    with pytest.raises(SandboxUnavailableError, match="no-such-layer"):
        GuardProcess(LIMIT_GUARD, GuardLimits(require=("rlimits", "no-such-layer")))


def test_dead_process_fails_closed() -> None:
    proc = GuardProcess(LIMIT_GUARD, GuardLimits())
    proc.close()
    decision = proc.evaluate(CALL, VIEW, [])
    assert decision.blocked and decision.error == "guard process is not running"


def test_child_exit_mid_call_fails_closed() -> None:
    quitter = guard("import os\ndef check(call, view, history):\n    os._exit(3)\n")
    proc = GuardProcess(quitter, GuardLimits())
    decision = proc.evaluate(CALL, VIEW, [])
    assert decision.blocked and "exited with status 3" in (decision.error or "")
    assert not proc.alive
    proc.close()


# Signals that are harmless if a regression lets them through, so the probe can never kill the test runner.
SIGNAL_PROBE = guard(
    """
    import os, signal

    def check(call, view, history):
        target = call.args["pid"]
        out = []
        for name, attempt in (
            ("kill0", lambda: os.kill(target, 0)),
            ("sigcont", lambda: os.kill(target, signal.SIGCONT)),
            ("pidfd", lambda: signal.pidfd_send_signal(os.pidfd_open(target), 0)),
            ("group", lambda: os.killpg(os.getpgid(target), 0)),
        ):
            try:
                attempt()
            except PermissionError:
                out.append(name + "=EPERM")
            except OSError as exc:
                out.append(name + "=" + type(exc).__name__)
            else:
                out.append(name + "=SENT")
        return " ".join(out)
    """,
    "signal-probe",
)


@pytest.mark.parametrize("max_abi", [None, 5, 1], ids=["kernel-abi", "abi5-no-signal-scope", "abi1"])
def test_guard_cannot_signal_its_parent_on_any_landlock_abi(max_abi: int | None) -> None:
    """Landlock scopes signals only from ABI 6 (Linux 6.12); the seccomp filter must refuse them anyway."""
    with GuardRuntime(GuardLimits(landlock_max_abi=max_abi)) as rt:
        isolation = rt.validate(SIGNAL_PROBE)
        if max_abi is not None:
            # The cap emulates an older kernel; a kernel older than the cap (6.8 has ABI 4) applies its own.
            applied = int(isolation["landlock"].removeprefix("abi ").split(",", 1)[0])
            assert 1 <= applied <= max_abi and "signal" not in isolation["landlock"], isolation["landlock"]
        denied = set(isolation["seccomp"].removeprefix("denied: ").split(","))
        assert {"kill", "tkill", "tgkill", "pidfd_open", "pidfd_send_signal"} <= denied, denied
        for target in (os.getpid(), os.getppid(), 1):
            decision = rt.check(SIGNAL_PROBE, ToolCall("x", {"pid": target}, "c"), {}, [])
            assert decision.blocked and decision.error is None, decision
            assert decision.reason is not None and "SENT" not in decision.reason, (target, decision.reason)
            assert decision.reason.count("=EPERM") >= 3, (target, decision.reason)


# --- guard time versus transport time -----------------------------------------------------------------


def _big_history(n: int, tag: int) -> list[Event]:
    payload = {"call": {"name": "lookup", "args": {"id": "INV-1", "amount": 50}}, "note": "x" * 120}
    return [Event(seq=i, kind="tool_call", payload={**payload, "tag": tag}, ts=float(i)) for i in range(n)]


@contextlib.contextmanager
def saturated_cpus() -> Iterator[int]:
    """Saturate every CPU this process may run on with busy loops while the block runs."""
    count = len(os.sched_getaffinity(0))
    hogs = [
        subprocess.Popen([sys.executable, "-c", "while True: pass"], stdin=subprocess.DEVNULL)
        for _ in range(count)
    ]
    try:
        yield count
    finally:
        for hog in hogs:
            hog.kill()
        for hog in hogs:
            hog.wait()


@pytest.fixture
def cpu_hog() -> Iterator[int]:
    with saturated_cpus() as count:
        yield count


INSTANT = guard(
    """
    def check(call, view, history):
        if call.args.get("loop"):
            while True:
                pass
        return "n=%d" % len(history)
    """,
    "instant",
)
HISTORY_EVENTS = 15_000  # where calibration starts; the size is only ever doubled from here
MAX_HISTORY_EVENTS = HISTORY_EVENTS * 16  # stays inside the child's memory limit
CALL_LIMIT_S = 0.2
BATCH_ITEMS = 3


def _load_items(events: int) -> list[GuardItem]:
    return [
        GuardItem(INSTANT, ToolCall("lookup", {}, f"c{k}"), VIEW, _big_history(events, k))
        for k in range(BATCH_ITEMS)
    ]


def _seconds_per_item(rt: GuardRuntime, events: int) -> float:
    items = _load_items(events)
    started = time.monotonic()
    rt.batch(items)
    return (time.monotonic() - started) / len(items)


@pytest.fixture(scope="module")
def history_events() -> int:
    """A history size at which moving one history into the child takes over twice a guard's time limit.

    The tests below only mean something when that holds, and how many events it takes depends on the
    machine. So double the size from HISTORY_EVENTS until the per-item time of the very batch the
    first test asserts on, measured on a saturated machine as that test runs, clears twice the limit.
    """
    with saturated_cpus(), GuardRuntime(GuardLimits(call_timeout_s=CALL_LIMIT_S), workers=1) as rt:
        events = HISTORY_EVENTS
        while _seconds_per_item(rt, events) <= 2 * CALL_LIMIT_S:
            assert events < MAX_HISTORY_EVENTS, (
                f"even {events} events take under {2 * CALL_LIMIT_S}s per item to move into the child "
                "here; raise MAX_HISTORY_EVENTS"
            )
            events *= 2
        return events


def test_decoding_a_large_history_under_load_is_never_a_guard_timeout(
    cpu_hog: int, history_events: int
) -> None:
    items = _load_items(history_events)
    with GuardRuntime(GuardLimits(call_timeout_s=CALL_LIMIT_S), workers=1) as rt:
        started = time.monotonic()
        results = rt.batch(items)
        wall = time.monotonic() - started
    # The scenario is real: getting each history into the child took longer than a guard may run ...
    assert wall / len(items) > CALL_LIMIT_S, wall
    # ... and none of that time was charged to the guard.
    assert [r.kind for r in results] == ["verdict"] * len(items), results
    assert [r.reason for r in results] == [f"n={history_events}"] * len(items)
    assert all(r.elapsed_s < CALL_LIMIT_S for r in results), [r.elapsed_s for r in results]


def test_a_looping_guard_still_times_out_under_the_same_load(cpu_hog: int, history_events: int) -> None:
    history = _big_history(history_events, 0)
    items = [
        GuardItem(INSTANT, ToolCall("lookup", {}, "a"), VIEW, history),
        GuardItem(INSTANT, ToolCall("lookup", {"loop": True}, "b"), VIEW, history),
        GuardItem(INSTANT, ToolCall("lookup", {}, "c"), VIEW, history),
    ]
    with GuardRuntime(GuardLimits(call_timeout_s=CALL_LIMIT_S), workers=1) as rt:
        results = rt.batch(items)
    assert [r.kind for r in results] == ["verdict", "guard_error", "verdict"], results
    assert results[1].reason == GUARD_ERROR_REASON and results[1].error == "timeout"


def test_a_guard_that_blocks_without_cpu_hits_the_wall_backstop() -> None:
    sleeper = guard("import time\ndef check(call, view, history):\n    time.sleep(30)\n", "sleeper")
    with GuardRuntime(GuardLimits(call_timeout_s=0.2, guard_wall_grace_s=0.3)) as rt:
        started = time.monotonic()
        decision = rt.check(sleeper, CALL, VIEW, [])
    assert decision.kind == "guard_error" and decision.error == "timeout" and time.monotonic() - started < 3.0


def test_transport_failure_is_an_infrastructure_error_not_a_verdict() -> None:
    limits = GuardLimits(transport_timeout_s=0.01)
    items = [
        GuardItem(INSTANT, ToolCall("lookup", {}, f"c{k}"), VIEW, _big_history(HISTORY_EVENTS, k))
        for k in range(2)
    ]
    with GuardRuntime(limits, workers=1) as rt:
        with pytest.raises(GuardInfrastructureError, match="failed in the sandbox, not in the guard") as info:
            rt.batch(items)
        assert {d.kind for d in info.value.decisions} == {"infrastructure_error"}
        assert {d.reason for d in info.value.decisions} == {INFRASTRUCTURE_ERROR_REASON}
        with pytest.raises(GuardInfrastructureError, match="transport timeout") as one:
            rt.check(INSTANT, CALL, VIEW, items[0].history)
        assert one.value.decisions[0].reason == INFRASTRUCTURE_ERROR_REASON


def test_sandbox_startup_failure_is_an_infrastructure_error() -> None:
    with GuardRuntime(GuardLimits(startup_timeout_s=0.001)) as rt:
        with pytest.raises(GuardInfrastructureError, match="failed to start"):
            rt.validate(LIMIT_GUARD)
        with pytest.raises(GuardInfrastructureError):
            rt.check(LIMIT_GUARD, CALL, VIEW, [])
        items = [GuardItem(LIMIT_GUARD, CALL, VIEW, [])]
        with pytest.raises(GuardInfrastructureError) as info:
            rt.batch(items)
        assert info.value.decisions[0].kind == "infrastructure_error"
