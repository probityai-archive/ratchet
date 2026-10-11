from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ratchet.agent.guards import CHILD_PATH, GUARD_ERROR_REASON, GuardLimits
from ratchet.export import hookrt
from ratchet.export.hookrt import (
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


def test_evaluate_once_fails_closed() -> None:
    call = {"name": "Bash", "args": {}, "call_id": "c"}
    assert evaluate_once(spec("x = 1\n"), call, {}, []).error is not None
    assert evaluate_once(spec(LOOP_GUARD, call_timeout_s=0.2), call, {}, []).error == "timeout"
    unavailable = evaluate_once(spec(ALLOW_GUARD, require=("no-such-layer",)), call, {}, [])
    assert str(unavailable.error).startswith("isolation unavailable")
    assert evaluate_once(spec(ALLOW_GUARD, "raise SystemExit(3)"), call, {}, []).error == (
        "guard process exited with status 3"
    )
    ready_only = "import os\nos.write(1, b'{\"ready\": true}\\nnot json\\n')\n"
    assert evaluate_once(spec(ALLOW_GUARD, ready_only), call, {}, []).error == (
        "guard process exited with status 0"
    )
    listy = "import os\nos.write(1, b'[1]\\n')\n"
    assert evaluate_once(spec(ALLOW_GUARD, listy), call, {}, []).error == "guard process exited with status 0"
    slow = "import time\ntime.sleep(5)\n"
    assert evaluate_once(
        spec(ALLOW_GUARD, slow, startup_timeout_s=0.1, call_timeout_s=0.1), call, {}, []
    ) == (Verdict(None, "timeout"))


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
    assert decide(pinned, event) == Verdict(None)
    hook.write_text(
        hook.read_text().replace('GUARD_ERROR_REASON = "guard error"', 'GUARD_ERROR_REASON = "x"')
    )
    assert decide(pinned, event).error == "hook runtime does not match its recorded digest"
    hook.write_text("no markers\n")
    assert hookrt.embedded_runtime(hook.read_text()) is None
    assert decide(pinned, event).error == "hook runtime does not match its recorded digest"
