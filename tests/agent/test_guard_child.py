"""In-process tests of the sandbox child's pure parts. The lockdown itself is exercised end to end by
`test_guards.py`, which runs real children and checks that every escape attempt is refused."""

from __future__ import annotations

import json
import signal
import types
from typing import Any

import pytest

from ratchet.agent import _guard_child as child

ALLOW = "def check(call, view, history):\n    return None\n"


def _setup(source: str = ALLOW, require: tuple[str, ...] = (), timeout_s: float = 0.5) -> dict[str, Any]:
    return {"source": source, "cpu_s": 5, "memory_mb": 256, "timeout_s": timeout_s, "require": list(require)}


def _request(i: int, keep: int = 0, add: list[dict[str, Any]] | None = None, **args: Any) -> bytes:
    body = {
        "i": i,
        "call": {"name": "refund", "args": args, "call_id": f"c{i}"},
        "view": {"limit": 10},
        "h_keep": keep,
        "h_add": add or [],
    }
    return json.dumps(body).encode()


def _serve(setup: dict[str, Any], lines: list[bytes], isolation: dict[str, str] | None = None) -> list[Any]:
    sent: list[Any] = []
    child.serve(setup, isolation or {}, lines, sent.append)
    return sent


def test_freeze_is_deep_and_read_only() -> None:
    frozen = child.freeze({"a": [1, {"b": 2}], "c": "x"})
    assert isinstance(frozen, types.MappingProxyType) and frozen["a"][1]["b"] == 2
    assert isinstance(frozen["a"], tuple)
    with pytest.raises(TypeError):
        frozen["a"][1]["b"] = 3


def test_seccomp_program_shapes() -> None:
    prog, names = child.seccomp_program("x86_64")
    assert {"socket", "io_uring_setup", "kill", "tgkill", "tkill", "pidfd_send_signal"} <= set(names)
    deny_at = len(prog) - 1
    assert prog[-1] == (0x06, 0, 0, 0x00050001) and prog[-2] == (0x06, 0, 0, 0x7FFF0000)
    for index, (code, jt, _, _) in enumerate(prog[4:-2], start=4):
        assert code in (0x15, 0x35) and index + 1 + jt == deny_at
    arm, _ = child.seccomp_program("aarch64")
    assert arm[1][3] == 0xC00000B7 and arm[4][0] == 0x15
    with pytest.raises(OSError, match="no seccomp table"):
        child.seccomp_program("sparc")


@pytest.mark.parametrize(
    ("abi", "size", "net", "scoped", "text"),
    [(1, 8, 0, 0, "fs:all"), (3, 8, 0, 0, "fs:all"), (4, 16, 3, 0, "tcp"), (8, 24, 3, 3, "signal")],
)
def test_landlock_ruleset_per_abi(abi: int, size: int, net: int, scoped: int, text: str) -> None:
    attr, got_size, description = child.landlock_ruleset(abi)
    assert got_size == size and attr.handled_access_net == net and attr.scoped == scoped
    assert text in description and attr.handled_access_fs & 1 == 1
    if abi >= 5:
        assert attr.handled_access_fs == (1 << 16) - 1


def test_run_layers_records_failures() -> None:
    def broken() -> str:
        raise OSError("nope")

    isolation = child.run_layers((("a", lambda: "on"), ("b", broken)))
    assert isolation == {"a": "on", "b": "UNAVAILABLE: nope"}


def test_serve_refuses_when_a_required_layer_is_missing() -> None:
    sent = _serve(_setup(require=("netns",)), [], {"netns": "UNAVAILABLE: no"})
    assert sent == [
        {
            "ready": False,
            "isolation": {"netns": "UNAVAILABLE: no"},
            "error": "isolation unavailable: ['netns']",
        }
    ]


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("x = 1\n", "does not define a callable check"),
        ("def check(\n", "SyntaxError"),
        ("while True:\n    pass\n", "timeout while loading guard"),
    ],
)
def test_serve_reports_load_failures(source: str, error: str) -> None:
    sent = _serve(_setup(source, timeout_s=0.2), [])
    assert len(sent) == 1 and sent[0]["ready"] is False and error in sent[0]["error"]


def test_serve_answers_requests_and_tracks_history() -> None:
    source = (
        "def check(call, view, history):\n"
        "    if call.args.get('amount', 0) > view['limit']:\n"
        "        return 'over:' + ','.join(str(e.payload['n']) for e in history)\n"
        "    return None\n"
    )
    event = {"seq": 0, "kind": "model", "payload": {"n": 1}, "ts": 0.0}
    event2 = {"seq": 1, "kind": "tool_call", "payload": {"n": 2}, "ts": 0.0}
    lines = [
        _request(0, 0, [event], amount=50),
        _request(1, 1, [event2], amount=50),
        _request(2, 1, [], amount=50),
        _request(3, 0, [], amount=1),
    ]
    sent = _serve(_setup(source), lines)
    assert sent[0]["ready"] is True
    assert [r["block"] for r in sent[1:]] == ["over:1", "over:1,2", "over:1", None]
    assert [r["i"] for r in sent[1:]] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("    raise ValueError('bad')\n", "ValueError: bad"),
        ("    return 7\n", "check returned int"),
        ("    while True:\n        pass\n", "timeout"),
    ],
)
def test_evaluate_failures(body: str, error: str) -> None:
    code = child._load("def check(call, view, history):\n" + body)
    call = child.ToolCall(name="x", args={}, call_id="c")
    signal_was = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, child._on_alarm)
    try:
        reply = child._evaluate(code, call, {}, [], 0.2)
    finally:
        signal.signal(signal.SIGALRM, signal_was)
    assert reply["block"] is None and error in reply["error"]


def test_long_block_reason_is_truncated() -> None:
    code = child._load("def check(call, view, history):\n    return 'x' * 5000\n")
    reply = child._evaluate(code, child.ToolCall(name="x", args={}, call_id="c"), {}, [], 1.0)
    assert len(reply["block"]) == 1000


def test_send_writes_one_json_line(tmp_path: Any) -> None:
    import os

    path = tmp_path / "out"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT)
    child._send(fd, {"a": 1})
    os.close(fd)
    assert path.read_bytes() == b'{"a":1}\n'
