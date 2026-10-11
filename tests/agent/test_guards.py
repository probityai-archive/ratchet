from __future__ import annotations

import os
import textwrap
import time
from collections.abc import Iterator

import pytest

from ratchet.agent.guards import (
    GUARD_ERROR_REASON,
    GuardItem,
    GuardLimits,
    GuardProcess,
    GuardRuntime,
    GuardSourceError,
    SandboxUnavailableError,
)
from ratchet.types import Event, Guard, ToolCall

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
    with GuardRuntime(GuardLimits(call_timeout_s=0.3)) as rt:
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
