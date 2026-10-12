from __future__ import annotations

import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from ratchet.agent.guards import CHILD_PATH, GUARD_ERROR_REASON, GuardLimits
from ratchet.export import hookrt
from ratchet.export.hookrt import (
    GuardChild,
    HookConfig,
    SandboxSpec,
    Verdict,
    block_message,
    decide,
    evaluate_once,
    git_branch,
    live_history,
    parse_transcript,
    run_hook,
    transcript_path,
)

from .synth import Session, hook_event

CHILD = CHILD_PATH.read_text(encoding="utf-8")

ECHO_GUARD = """
import json
def check(call, view, history):
    return json.dumps({"call": [call.name, call.call_id, dict(call.args)], "view": dict(view),
                       "kinds": [e.kind for e in history], "seqs": [e.seq for e in history]})
"""
ALLOW_GUARD = "def check(call, view, history):\n    return None\n"
CRASH_GUARD = "def check(call, view, history):\n    raise RuntimeError('boom')\n"
LOOP_GUARD = "def check(call, view, history):\n    while True:\n        pass\n"


def spec(source: str, child: str = CHILD, **limits: Any) -> SandboxSpec:
    lim = GuardLimits(**limits)
    return SandboxSpec(
        "g", source, child, lim.cpu_s, lim.memory_mb, lim.call_timeout_s, lim.startup_timeout_s, lim.require
    )


def config(source: str, child: str = CHILD, **limits: Any) -> HookConfig:
    return HookConfig(spec(source, child, **limits), hookrt._sha256(source), hookrt._sha256(child))


def session() -> Session:
    s = Session(branch="feature/x").noise().prompt("please push").think().say("on it")
    s.bash("c1", "git status").result("c1", "clean")
    s.bash("c2", "git push").result(
        "c2", [{"type": "text", "text": "rejected"}, {"type": "image"}], is_error=True
    )
    return s


# --- parsing ----------------------------------------------------------------------------------------------


def test_parse_maps_blocks_to_worker_event_shapes() -> None:
    lines = [*session().text().splitlines(), "not json", "[1]", ""]
    t = parse_transcript(lines)
    assert [e["kind"] for e in t.events] == [
        "model", "model", "tool_call", "tool_result", "model", "tool_call", "tool_result",
    ]  # fmt: skip
    assert [e["seq"] for e in t.events] == list(range(7))
    assert t.malformed == 2 and t.lines == len(lines)
    assert t.events[0]["payload"] == {"model": "claude-test", "content": "on it", "tool_calls": []}
    call = {"name": "Bash", "args": {"command": "git status", "description": "run"}, "call_id": "c1"}
    assert t.events[1]["payload"]["tool_calls"] == [call]
    assert t.events[2]["payload"] == {"call": call, "view": {"cwd": "/work/repo", "git_branch": "feature/x"}}
    assert t.events[3]["payload"] == {
        "result": {"call_id": "c1", "ok": True, "data": {"content": "clean"}, "error": None},
        "decisions": [],
    }
    failed = t.events[6]["payload"]["result"]
    assert failed == {"call_id": "c2", "ok": False, "data": None, "error": "rejected\n[image]"}
    assert [(c.call["call_id"], c.history_len) for c in t.calls] == [("c1", 1), ("c2", 4)]
    assert t.events[0]["ts"] > 1.7e9


def test_parse_tolerates_odd_values() -> None:
    s = Session()
    s.lines.append(
        {"type": "assistant", "timestamp": "yesterday", "message": {"content": [{"type": "tool_use"}]}}
    )
    s.lines.append({"type": "user", "message": {"content": [{"type": "tool_result", "content": 7}]}})
    s.lines.append({"type": "assistant", "message": {"content": [{"type": "text", "text": "x" * 9}]}})
    s.result("c", "y" * 100_000)
    t = parse_transcript(s.text().encode("utf-8").splitlines())
    assert t.calls[0].call == {"name": "", "args": {}, "call_id": ""}
    assert t.calls[0].view == {"cwd": "", "git_branch": ""}
    assert t.events[0]["ts"] == 0.0
    assert t.events[2]["payload"]["result"]["data"] == {"content": ""}
    assert len(t.events[4]["payload"]["result"]["data"]["content"]) == 100_000


def test_history_before_finds_the_call_by_id_then_by_content() -> None:
    t = parse_transcript(session().text().splitlines())
    assert len(t.history_before({"call_id": "c2"})) == 4
    by_content = {"call_id": "zzz", "name": "Bash", "args": {"command": "git status", "description": "run"}}
    assert len(t.history_before(by_content)) == 1
    assert len(t.history_before({"call_id": "", "name": "Bash", "args": {}})) == 7


# --- live inputs ------------------------------------------------------------------------------------------


def test_git_branch_reads_head_like_claude_code_records_it(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/feature/x\n")
    (repo / "src" / "deep").mkdir(parents=True)
    assert git_branch(str(repo / "src" / "deep")) == "feature/x"
    (repo / ".git" / "HEAD").write_text("0123456789abcdef0123456789abcdef01234567\n")
    assert git_branch(str(repo)) == "HEAD"
    (repo / ".git" / "HEAD").write_text("ref: refs/remotes/origin/main\n")
    assert git_branch(str(repo)) == "refs/remotes/origin/main"

    worktree = tmp_path / "wt"
    worktree.mkdir()
    (tmp_path / "gitdirs" / "wt").mkdir(parents=True)
    (tmp_path / "gitdirs" / "wt" / "HEAD").write_text("ref: refs/heads/lane\n")
    (worktree / ".git").write_text("gitdir: ../gitdirs/wt\n")
    assert git_branch(str(worktree)) == "lane"
    (worktree / ".git").write_text(f"gitdir: {tmp_path / 'missing'}\n")
    assert git_branch(str(worktree)) == ""

    odd = tmp_path / "odd"
    odd.mkdir()
    (odd / ".git").write_text("not a gitdir file\n")
    assert git_branch(str(odd)) == ""


def test_transcript_path_prefers_the_subagent_file(tmp_path: Path) -> None:
    main = Session().write(tmp_path / "s1.jsonl")
    assert transcript_path({}) is None
    assert transcript_path({"transcript_path": str(main)}) == main
    assert transcript_path({"transcript_path": str(main), "agent_id": "a9"}) == main
    sub = Session().write(tmp_path / "s1" / "subagents" / "agent-a9.jsonl")
    assert transcript_path({"transcript_path": str(main), "agent_id": "a9"}) == sub


def test_live_history_is_empty_without_a_transcript(tmp_path: Path) -> None:
    call = {"name": "Bash", "args": {}, "call_id": "c"}
    assert live_history({}, call) == []
    assert live_history({"transcript_path": str(tmp_path / "missing.jsonl")}, call) == []
    path = session().write(tmp_path / "s.jsonl")
    assert len(live_history({"transcript_path": str(path)}, {**call, "call_id": "c2"})) == 4


# --- one sandboxed evaluation -----------------------------------------------------------------------------


def test_evaluate_once_runs_the_guard_in_the_real_sandbox() -> None:
    call = {"name": "Bash", "args": {"command": "ls"}, "call_id": "c9"}
    history = parse_transcript(session().text().splitlines()).events
    seen = json.loads(str(evaluate_once(spec(ECHO_GUARD), call, {"git_branch": "b"}, history).block))
    assert seen == {
        "call": ["Bash", "c9", {"command": "ls"}],
        "view": {"git_branch": "b"},
        "kinds": [e["kind"] for e in history],
        "seqs": list(range(7)),
    }
    assert evaluate_once(spec(ALLOW_GUARD), call, {}, []) == Verdict(None)
    assert evaluate_once(spec(CRASH_GUARD), call, {}, []) == Verdict(None, "RuntimeError: boom")


def child_script(*writes: bytes, reads: int = 2, sleep: float = 0.0) -> str:
    """A stand-in sandbox child: reads `reads` lines in turn and writes each of `writes` after one."""
    lines = ["import os, sys, time"]
    for index in range(max(reads, len(writes))):
        if index < reads:
            lines.append("sys.stdin.buffer.readline()")
        if index < len(writes):
            lines.append(f"os.write(1, {writes[index]!r})")
    lines.append(f"time.sleep({sleep})")
    return "\n".join(lines) + "\n"


READY = b'{"ready": true}\n'


def test_evaluate_once_fails_closed() -> None:
    call = {"name": "Bash", "args": {}, "call_id": "c"}
    assert evaluate_once(spec("x = 1\n"), call, {}, []).error is not None
    assert evaluate_once(spec(LOOP_GUARD, call_timeout_s=0.2), call, {}, []).error == "timeout"
    unavailable = evaluate_once(spec(ALLOW_GUARD, require=("no-such-layer",)), call, {}, [])
    assert str(unavailable.error).startswith("isolation unavailable")
    exits = evaluate_once(
        spec(ALLOW_GUARD, "import sys\nsys.stdin.buffer.readline()\nraise SystemExit(3)"), call, {}, []
    )
    assert exits.error == "guard process exited with status 3"
    malformed = "guard process sent a malformed reply"
    assert (
        evaluate_once(spec(ALLOW_GUARD, child_script(READY, b"not json\n")), call, {}, []).error == malformed
    )
    assert (
        evaluate_once(spec(ALLOW_GUARD, child_script(READY, b'{"i": 99}\n')), call, {}, []).error == malformed
    )
    assert evaluate_once(spec(ALLOW_GUARD, child_script(b"[1]\n")), call, {}, []).error == malformed
    slow = spec(ALLOW_GUARD, child_script(sleep=5), startup_timeout_s=0.1, call_timeout_s=0.1)
    assert evaluate_once(slow, call, {}, []) == Verdict(None, "timeout")
    gone = spec(ALLOW_GUARD, child_script(READY, reads=1))
    big = [{"seq": 0, "kind": "model", "payload": {"content": "x" * 2_000_000}, "ts": 0.0}]
    assert evaluate_once(gone, call, {}, big) == Verdict(None, "guard process exited with status 0")
    stuck = spec(
        ALLOW_GUARD, child_script(READY, reads=1, sleep=5), startup_timeout_s=0.2, call_timeout_s=0.1
    )
    history = [{"seq": 0, "kind": "model", "payload": {"content": "x" * 2_000_000}, "ts": 0.0}]
    assert evaluate_once(stuck, call, {}, history) == Verdict(None, "timeout")


def test_guard_child_sends_history_incrementally() -> None:
    events = parse_transcript(session().text().splitlines()).events
    call = {"name": "Bash", "args": {}, "call_id": "c"}
    child = GuardChild(spec(ECHO_GUARD))
    try:
        for session_key, n in (("s", 3), ("s", 7), ("s", 2), ("t", 1), (None, 4)):
            seen = json.loads(str(child.ask(session_key, call, {}, events[:n]).block))
            assert seen["seqs"] == list(range(n))
        assert (child.session, child.sent) == (None, 4)
    finally:
        child.close()
    assert not child.alive


def test_evaluate_once_reports_a_missing_interpreter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "executable", str(tmp_path / "no-python"))
    verdict = evaluate_once(spec(ALLOW_GUARD), {"name": "x", "args": {}, "call_id": ""}, {}, [])
    assert str(verdict.error).startswith("sandbox did not start")


# --- the hook ---------------------------------------------------------------------------------------------


def test_decide_builds_live_inputs_from_hook_input(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    path = session().write(tmp_path / "s.jsonl")
    seen = json.loads(
        str(decide(config(ECHO_GUARD), hook_event(path, "c2", "git push", cwd=str(repo))).block)
    )
    assert seen["view"] == {"cwd": str(repo), "git_branch": "main"}
    assert seen["kinds"] == ["model", "model", "tool_call", "tool_result"]
    assert seen["call"] == ["Bash", "c2", {"command": "git push", "description": "run"}]
    no_cwd = json.loads(str(decide(config(ECHO_GUARD), hook_event(None, "c", "ls", cwd="")).block))
    assert no_cwd["view"] == {"cwd": "", "git_branch": ""}


def test_decide_refuses_anything_it_cannot_vouch_for() -> None:
    good = config(ALLOW_GUARD)
    event = hook_event(None, "c", "ls")
    assert decide(good, event) == Verdict(None)
    tampered = HookConfig(good.spec, "0" * 64, good.child_sha256)
    assert decide(tampered, event).error == "guard source does not match its recorded digest"
    tampered = HookConfig(good.spec, good.guard_sha256, "0" * 64)
    assert decide(tampered, event).error == "sandbox child does not match its recorded digest"
    assert "expected PreToolUse" in str(decide(good, {**event, "hook_event_name": "PostToolUse"}).error)
    assert decide(good, {**event, "tool_input": "ls"}).error == "hook input has no tool_name and tool_input"


def run(cfg: HookConfig, stdin: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run_hook(cfg, io.StringIO(stdin), out, err)
    return code, out.getvalue(), err.getvalue()


def test_run_hook_speaks_the_pretooluse_protocol() -> None:
    assert run(config(ALLOW_GUARD), json.dumps(hook_event(None, "c", "ls"))) == (0, "", "")
    code, out, err = run(
        config("def check(c, v, h):\n    return 'no'\n"), json.dumps(hook_event(None, "c", "ls"))
    )
    assert code == 2 and err == "blocked by guard g: no\n"
    assert json.loads(out) == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "blocked by guard g: no",
        }
    }


@pytest.mark.parametrize("stdin", ["", "{not json", "[1, 2]"])
def test_run_hook_blocks_when_input_is_unreadable(stdin: str) -> None:
    code, out, err = run(config(ALLOW_GUARD), stdin)
    assert code == 2
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert err.startswith(f"blocked by guard g: {GUARD_ERROR_REASON}: ")


def test_block_message_matches_the_worker_format() -> None:
    assert hookrt.GUARD_ERROR_REASON == GUARD_ERROR_REASON
    assert block_message("g", Verdict(None)) is None
    assert block_message("g", Verdict("why")) == "blocked by guard g: why"
    assert block_message("g", Verdict(None, "boom")) == "blocked by guard g: guard error: boom"


def test_decide_rehashes_the_embedded_runtime(tmp_path: Path) -> None:
    runtime = Path(hookrt.__file__).read_text(encoding="utf-8")
    hook = tmp_path / "hook.py"
    hook.write_text(f"{hookrt.RUNTIME_BEGIN}\n{runtime}\n\n{hookrt.RUNTIME_END}\nX = 1\n", encoding="utf-8")
    good = config(ALLOW_GUARD)
    pinned = HookConfig(good.spec, good.guard_sha256, good.child_sha256, hookrt._sha256(runtime), str(hook))
    event = hook_event(None, "c", "ls")
    assert decide(pinned, event, use_server=False) == Verdict(None)
    hook.write_text(
        hook.read_text().replace('GUARD_ERROR_REASON = "guard error"', 'GUARD_ERROR_REASON = "x"')
    )
    assert decide(pinned, event, use_server=False).error == "hook runtime does not match its recorded digest"
    hook.write_text("no markers\n")
    assert hookrt.embedded_runtime(hook.read_text()) is None
    assert decide(pinned, event).error == "hook runtime does not match its recorded digest"


# --- history elision --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "reads"),
    [
        ("def check(call, view, history):\n    return None\n", False),
        ("import re\ndef check(c, v, h):\n    return 'x' if re.match('a', c.name) else None\n", False),
        ("def check(call, view, history):\n    return str(len(history))\n", True),
        ("def check(call, view, h):\n    return locals()['h']\n", True),
        ("import sys\ndef check(call, view, history):\n    return None\n", True),
        ("from operator import attrgetter\ndef check(call, view, history):\n    return None\n", True),
        ("def check(call, view, history):\n    return getattr(call, 'name')\n", True),
        ("def check(call, view, history):\n    try:\n        1/0\n    except Exception as e:\n"
         "        return str(e.__traceback__.tb_frame)\n", True),
        ("def check(call, view, history):\n    return 'history' if call.name == 'eval' else None\n", False),
        ("def check(call, view, history):\n    return '{0.gi_frame}'.format(call)\n", True),
        ("import re\nP = re.compile('x')\ndef check(c, v, h):\n    return P.pattern\n", False),
        ("def check(c, v, h):\n    return str(().__class__.__base__.__subclasses__())\n", True),
        ("def check(c, v, h):\n    return str(c.f_back)\n", True),
        ("def check(call, view, history):\n    return None\ncheck = print\n", True),
        ("def check(call, view, history):\n    def check(): pass\n    return None\n", True),
        ("def check(call, view, *rest):\n    return None\n", True),
        ("def check(call, view, history=None, *, k=1):\n    return None\n", True),
        ("def check(call, view):\n    return None\n", True),
        ("@staticmethod\ndef check(call, view, history):\n    return None\n", True),
        ("async def check(call, view, history):\n    return None\n", True),
        ("check = lambda call, view, history: None\n", True),
        ("def check(:\n", True),
    ],
)  # fmt: skip
def test_reads_history_is_false_only_when_provable(source: str, reads: bool) -> None:
    assert hookrt.reads_history(source) is reads


# --- incremental tailing and the session server -----------------------------------------------------------


def test_tail_parses_complete_lines_only_and_restarts_on_shrink(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    tail = hookrt.Tail(path)
    tail.update()
    assert tail.transcript.events == []
    text = session().text()
    cut = text.index("\n", len(text) // 2) + 10
    path.write_text(text[:cut])
    tail.update()
    partial = len(tail.transcript.events)
    path.write_text(text)
    tail.update()
    tail.update()
    whole = parse_transcript(text.splitlines())
    assert partial < len(whole.events)
    assert tail.transcript.events == whole.events and tail.transcript.lines == whole.lines
    path.write_text(Session().bash("z", "ls").text())
    tail.update()
    assert [c.call["call_id"] for c in tail.transcript.calls] == ["z"]
    tail.close()


def test_session_server_matches_the_one_shot_verdict(tmp_path: Path) -> None:
    path = session().write(tmp_path / "s.jsonl")
    other = Session().bash("d1", "ls").write(tmp_path / "t.jsonl")
    server = hookrt.SessionServer(spec(ECHO_GUARD), sessions=1)
    try:
        for event in (
            hook_event(path, "c2", "git push"),
            hook_event(other, "d1", "ls"),
            hook_event(path, "c1", "x"),
        ):
            assert server.verdict(event) == decide(config(ECHO_GUARD), event, use_server=False)
        assert list(server.tails) == [str(path)]
        dead = server.tails[str(path)].child
        assert dead is not None
        dead.close()
        assert server.verdict(hook_event(path, "c2", "git push")).block is not None
        assert server.verdict({"hook_event_name": "Stop"}).error is not None
    finally:
        server.close()
    plain = hookrt.SessionServer(spec(ALLOW_GUARD))
    try:
        assert plain.verdict(hook_event(path, "c2", "x")) == Verdict(None)
        assert plain.verdict(hook_event(other, "d1", "x")) == Verdict(None)
        assert list(plain.tails) == [""]
    finally:
        plain.close()
    broken = hookrt.SessionServer(spec(ALLOW_GUARD, require=("no-such-layer",)))
    assert str(broken.verdict(hook_event(None, "c", "ls")).error).startswith("isolation unavailable")


def test_socket_dir_is_private(monkeypatch: pytest.MonkeyPatch, private_runtime_dir: Path) -> None:
    directory = hookrt.socket_dir()
    assert directory == private_runtime_dir / "ratchet-hook"
    directory.chmod(0o755)
    assert hookrt.socket_dir() is None
    blocker = private_runtime_dir / "file"
    blocker.write_text("")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(blocker))
    assert hookrt.socket_dir() is None
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(private_runtime_dir))
    assert hookrt.socket_dir() == private_runtime_dir / f"ratchet-hook-{os.getuid()}"


def serving(cfg: HookConfig, path: Path, idle_s: float = 30.0) -> threading.Thread:
    thread = threading.Thread(target=hookrt.serve_sessions, args=(cfg, path, idle_s), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return thread


def test_serve_sessions_answers_until_its_socket_goes(tmp_path: Path, private_runtime_dir: Path) -> None:
    sock = private_runtime_dir / "s.sock"
    cfg = config(ECHO_GUARD)
    thread = serving(cfg, sock)
    path = session().write(tmp_path / "s.jsonl")
    event = hook_event(path, "c2", "git push")
    assert hookrt._exchange(sock, event) == decide(cfg, event, use_server=False)
    assert hookrt.serve_sessions(cfg, sock) == 0  # the lock is held: a second server leaves at once
    with socket.socket(socket.AF_UNIX) as conn:
        conn.connect(str(sock))
        conn.sendall(b"[1]")
        conn.shutdown(socket.SHUT_WR)
        assert json.loads(hookrt._read_all(conn))["error"] == "bad request"
    with socket.socket(socket.AF_UNIX) as conn:
        conn.connect(str(sock))
        conn.sendall(b"{not json")
        conn.shutdown(socket.SHUT_WR)
        assert json.loads(hookrt._read_all(conn))["error"].startswith("JSONDecodeError")
    with pytest.raises(ValueError, match="malformed"):
        reply = private_runtime_dir / "r.sock"
        fake = socket.socket(socket.AF_UNIX)
        fake.bind(str(reply))
        fake.listen(1)
        answer = threading.Thread(target=lambda: _answer(fake, b"[1]"), daemon=True)
        answer.start()
        hookrt._exchange(reply, event)
    sock.unlink()
    thread.join(timeout=5)
    assert not thread.is_alive()


def _answer(listener: socket.socket, data: bytes) -> None:
    conn, _ = listener.accept()
    with conn:
        hookrt._read_all(conn)
        conn.sendall(data)
    listener.close()


def test_serve_sessions_exits_when_idle(private_runtime_dir: Path) -> None:
    sock = private_runtime_dir / "idle.sock"
    thread = serving(config(ALLOW_GUARD), sock, idle_s=1.5)
    thread.join(timeout=5)
    assert not thread.is_alive() and not sock.exists()


def test_serve_sessions_turns_away_other_users(
    monkeypatch: pytest.MonkeyPatch, private_runtime_dir: Path
) -> None:
    sock = private_runtime_dir / "u.sock"
    thread = serving(config(ALLOW_GUARD), sock)
    monkeypatch.setattr(os, "getuid", lambda: -5)
    with pytest.raises((ValueError, OSError)):  # turned away unanswered: an empty reply or a reset
        hookrt._exchange(sock, hook_event(None, "c", "ls"))
    monkeypatch.undo()
    monkeypatch.setattr(socket, "SO_PEERCRED", None, raising=False)
    monkeypatch.delattr(socket, "SO_PEERCRED")
    assert hookrt._exchange(sock, hook_event(None, "c", "ls")) == Verdict(None)
    sock.unlink()
    thread.join(timeout=5)


def test_ask_server_needs_a_hook_file_and_a_private_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    event = hook_event(None, "c", "ls")
    assert hookrt.ask_server(config(ALLOW_GUARD), event) is None
    failing = tmp_path / "fails.py"
    failing.write_text("raise SystemExit(1)\n")
    cfg = config(ALLOW_GUARD)
    with_file = HookConfig(cfg.spec, cfg.guard_sha256, cfg.child_sha256, None, str(failing))
    started = time.monotonic()
    assert hookrt.ask_server(with_file, event) is None
    assert time.monotonic() - started < hookrt.SERVER_START_S
    quiet = tmp_path / "quiet.py"
    quiet.write_text("pass\n")
    monkeypatch.setattr(hookrt, "SERVER_START_S", 0.3)
    quiet_cfg = HookConfig(cfg.spec, cfg.guard_sha256, cfg.child_sha256, None, str(quiet))
    assert hookrt.ask_server(quiet_cfg, event) is None
    monkeypatch.setattr(hookrt, "socket_dir", lambda: None)
    assert hookrt.ask_server(with_file, event) is None


def test_main_serves_or_runs_one_call(monkeypatch: pytest.MonkeyPatch, private_runtime_dir: Path) -> None:
    good = config(ALLOW_GUARD)
    tampered = HookConfig(good.spec, "0" * 64, good.child_sha256)
    assert hookrt.main(tampered, ["--serve", str(private_runtime_dir / "x.sock")]) == 1
    assert hookrt.main(good, ["--serve", "/tmp/" + "x" * 120 + ".sock"]) == 1
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(hook_event(None, "c", "ls"))))
    assert hookrt.main(good, []) == 0
    monkeypatch.setattr(os, "fork", lambda: 4242)
    assert hookrt.main(good, ["--serve", str(private_runtime_dir / "p.sock")]) == 0
    monkeypatch.setattr(os, "fork", lambda: 0)
    sock = private_runtime_dir / "m.sock"
    thread = threading.Thread(target=hookrt.main, args=(good, ["--serve", str(sock)]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not sock.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    sock.unlink()
    thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.mark.parametrize("reply", [{"protocol": 99}, {"ready": False}])
def test_an_out_of_protocol_startup_reply_is_quoted_in_the_verdict(reply: dict[str, Any]) -> None:
    line = json.dumps(json.dumps(reply))
    child = f"import sys\nsys.stdin.buffer.readline()\nprint({line}, flush=True)\nsys.stdin.buffer.read()\n"
    verdict = evaluate_once(spec(ALLOW_GUARD, child), {"name": "Bash", "args": {}, "call_id": "c"}, {}, [])
    quoted = json.dumps(reply, sort_keys=True)
    assert verdict == Verdict(None, f"guard sandbox sent an out-of-protocol startup reply: {quoted}")
