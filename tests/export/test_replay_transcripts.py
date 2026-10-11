from __future__ import annotations

import json
from pathlib import Path

import pytest

from ratchet.agent.guards import CHILD_PATH, GuardLimits, GuardRuntime
from ratchet.export import hookrt
from ratchet.export.replay_transcripts import (
    Prefix,
    TranscriptReplayReport,
    discover,
    main,
    replay_transcripts,
)
from ratchet.types import Event, Guard

from .synth import Session, hook_event

NO_RM = """
def check(call, view, history):
    if call.name == "Bash" and call.args.get("command", "").startswith("rm "):
        return "rm is not allowed"
    return None
"""
ECHO = """
import json
def check(call, view, history):
    return json.dumps({"view": dict(view), "kinds": [e.kind for e in history], "n": len(history)})
"""
CRASH_ON_RM = """
def check(call, view, history):
    if call.args.get("command", "").startswith("rm "):
        raise RuntimeError("boom")
    return None
"""


def corpus(root: Path) -> Path:
    one = Session(session_id="s-1", branch="main").prompt("clean").bash("a1", "ls").result("a1", "x")
    one.bash("a2", "rm -rf build").result("a2", "", is_error=True).say("done")
    one.write(root / "proj" / "s-1.jsonl")
    two = Session(session_id="s-2", branch="feature").bash("b1", "git status").result("b1", "ok")
    two.call("b2", "Read", {"file_path": "/work/repo/rm "}).result("b2", "text")
    two.write(root / "proj" / "s-2.jsonl")
    sub = Session(session_id="s-1", branch="main").bash("c1", "rm a").result("c1", "")
    sub.write(root / "proj" / "s-1" / "subagents" / "agent-7.jsonl")
    return root


def test_prefix_is_a_read_only_window() -> None:
    events = [Event(i, "model", {}, 0.0) for i in range(5)]
    p = Prefix(events, 3)
    assert len(p) == 3 and p[0] is events[0] and p[-1] is events[2]
    assert p[1:] == events[1:3] and p[5:] == []
    with pytest.raises(IndexError):
        p[3]
    with pytest.raises(IndexError):
        p[-4]


def test_replay_fires_on_exactly_the_expected_calls(tmp_path: Path) -> None:
    root = corpus(tmp_path)
    guard = Guard("no-rm", "", NO_RM)
    with GuardRuntime(workers=2) as rt:
        report = replay_transcripts(guard, discover([root]), rt, chunk=2, expect=["a2", "c1"])
    assert report.passed is True and report.missed == [] and report.unexpected == []
    assert (report.sessions, report.calls, report.unreadable, report.malformed_lines) == (3, 5, 0, 0)
    assert sorted((f.session, f.call_id, f.line) for f in report.fires) == [
        ("proj/s-1.jsonl", "a2", 4),
        ("proj/s-1/subagents/agent-7.jsonl", "c1", 1),
    ]
    assert report.calls_by_tool == {"Bash": 4, "Read": 1} and report.fires_by_tool == {"Bash": 2}
    assert report.check_s > 0 and report.checks_per_s > 0


def test_replay_reports_misses_unexpected_fires_and_errors(tmp_path: Path) -> None:
    root = corpus(tmp_path)
    with GuardRuntime(workers=1) as rt:
        wrong = replay_transcripts(Guard("no-rm", "", NO_RM), discover([root]), rt, expect=["b1"])
        broken = replay_transcripts(Guard("crash", "", CRASH_ON_RM), discover([root]), rt)
    assert wrong.passed is False and wrong.missed == ["b1"] and wrong.unexpected == ["a2", "c1"]
    assert broken.passed is None and not broken.fires
    assert sorted(e.call_id for e in broken.errors) == ["a2", "c1"]
    assert broken.errors[0].reason == "RuntimeError: boom"


def test_replay_shows_the_guard_what_the_live_hook_shows_it(tmp_path: Path) -> None:
    root = corpus(tmp_path)
    path = root / "proj" / "s-1.jsonl"
    with GuardRuntime(workers=1) as rt:
        report = replay_transcripts(Guard("echo", "", ECHO), [("s-1", path)], rt)
    replayed = {f.call_id: json.loads(f.reason) for f in report.fires}
    child, lim = CHILD_PATH.read_text(encoding="utf-8"), GuardLimits()
    spec = hookrt.SandboxSpec(
        "echo", ECHO, child, lim.cpu_s, lim.memory_mb, lim.call_timeout_s, lim.startup_timeout_s, lim.require
    )
    config = hookrt.HookConfig(spec, hookrt._sha256(ECHO), hookrt._sha256(child))
    for call_id, command in (("a1", "ls"), ("a2", "rm -rf build")):
        live = json.loads(
            str(hookrt.decide(config, hook_event(path, call_id, command, cwd="/work/repo")).block)
        )
        assert live["kinds"] == replayed[call_id]["kinds"]
        assert replayed[call_id]["view"] == {"cwd": "/work/repo", "git_branch": "main"}
    assert replayed["a2"]["kinds"] == ["model", "tool_call", "tool_result"]


def test_replay_counts_unreadable_and_malformed_input(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b"\xff\xfe not utf-8\n")
    odd = tmp_path / "odd.jsonl"
    odd.write_text("not json\n" + Session().bash("x", "rm a").text(), encoding="utf-8")
    sessions = [*discover([bad, odd]), ("gone", tmp_path / "missing.jsonl")]
    with GuardRuntime(workers=1) as rt:
        report = replay_transcripts(Guard("no-rm", "", NO_RM), sessions, rt)
    assert (report.unreadable, report.sessions, report.malformed_lines, len(report.fires)) == (2, 1, 1, 1)


def test_summary_carries_counts_and_never_content(tmp_path: Path) -> None:
    with GuardRuntime(workers=1) as rt:
        report = replay_transcripts(
            Guard("no-rm", "", NO_RM), discover([corpus(tmp_path)]), rt, expect=["a2"]
        )
    summary = json.dumps(report.summary())
    for secret in ("rm -rf", "rm is not allowed", "s-1.jsonl", '"a2"', "/work/repo"):
        assert secret not in summary
    assert report.summary()["fires"] == 2 and report.summary()["unexpected"] == 1
    full = report.to_payload()
    assert full["fire_sites"][0]["reason"] == "rm is not allowed" and full["unexpected_ids"] == ["c1"]
    assert TranscriptReplayReport("g", "x").checks_per_s == 0.0


def test_progress_is_reported_after_each_batch(tmp_path: Path) -> None:
    seen: list[int] = []
    with GuardRuntime(workers=1) as rt:
        replay_transcripts(
            Guard("no-rm", "", NO_RM),
            discover([corpus(tmp_path)]),
            rt,
            chunk=1,
            progress=lambda r: seen.append(r.calls),
        )
    assert seen == [1, 3, 5]  # the subagent file sorts first: one call, then two, then two


def test_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = corpus(tmp_path / "corpus")
    guard = tmp_path / "no_rm.py"
    guard.write_text(NO_RM, encoding="utf-8")
    expect = tmp_path / "expect.txt"
    expect.write_text("a2\n\nc1\n", encoding="utf-8")
    out = tmp_path / "out" / "report.json"
    assert (
        main(["--guard", str(guard), "--expect", str(expect), "--out", str(out), "--workers", "1", str(root)])
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["guard_id"] == "no_rm" and printed["passed"] is True
    assert json.loads(out.read_text())["expected"] == ["a2", "c1"]
    assert main(["--guard", str(guard), str(root)]) == 0
    crash = tmp_path / "crash.py"
    crash.write_text(CRASH_ON_RM, encoding="utf-8")
    assert main(["--guard", str(crash), "--guard-id", "c", str(root)]) == 1
    expect.write_text("b1\n", encoding="utf-8")
    assert main(["--guard", str(guard), "--expect", str(expect), str(root)]) == 1
