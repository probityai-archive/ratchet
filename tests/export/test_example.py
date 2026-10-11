"""The worked example in examples/claude-code: the guard's behaviour and the committed artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ratchet.agent.guards import GuardItem, GuardRuntime
from ratchet.export.claude_code import admitted_guard, harness_from_json, public_key, verify_hook
from ratchet.export.hookrt import reads_history
from ratchet.export.replay_transcripts import discover, replay_transcripts
from ratchet.improve.receipt import key_id, verify
from ratchet.types import Guard, ToolCall, sha256_hex

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "claude-code"
SOURCE = (EXAMPLE / "no_force_push_main.py").read_text(encoding="utf-8")
GUARD = Guard("no-force-push-main", "", SOURCE)

# (command, checked-out branch, blocked)
CASES = [
    ("git push --force", "main", True),
    ("git push --force", "feature", False),
    ("git push -f origin main", "x", True),
    ("git push origin main", "main", False),
    ("git push origin +HEAD:main", "f", True),
    ("git -C repo push -uf origin master", "f", True),
    ('git commit -m "git push --force origin main"', "main", False),
    ("echo 'git push -f origin main'", "main", False),
    ("cd r && git push --force-with-lease origin main", "f", True),
    ("git push --force-with-lease origin feat", "main", False),
    ("git push origin --delete main", "f", True),
    ("git push origin :main", "f", True),
    ("git push --force", "", True),
    ("GIT_SSH_COMMAND=x git push --force origin main", "", True),
    ("git push --mirror --force", "f", True),
    ("git push --all", "f", False),
    ("git push --force-if-includes origin main", "f", False),
    ("git push -o ci.skip origin feat", "main", False),
    ('bash -c "git push -f origin main"', "f", True),
    ("sudo sh -lc 'git push --force'", "main", True),
    ("bash -c 'echo git push -f origin main'", "main", False),
    ("eval git push -f origin main", "f", True),
    ("echo $(git push -f origin main)", "f", True),
    ("echo `git push -f origin main`", "f", True),
    ('echo "$(git push --force origin main)"', "f", True),
    ("echo '$(git push -f origin main)'", "f", False),
    ("x=$(git rev-parse HEAD) && git push origin $x:main", "f", False),
    ("git push origin\\ main", "f", False),
    ("echo " + "$(" * 10 + "ls" + ")" * 10, "f", True),
    ("git status; git push --force origin master", "main", True),
]


@pytest.fixture(scope="module")
def runtime() -> GuardRuntime:
    return GuardRuntime(workers=2)


def test_guard_blocks_exactly_the_destructive_pushes(runtime: GuardRuntime) -> None:
    items = [
        GuardItem(GUARD, ToolCall("Bash", {"command": c}, f"c{i}"), {"git_branch": b, "cwd": "/r"}, [])
        for i, (c, b, _) in enumerate(CASES)
    ]
    decisions = runtime.batch(items)
    assert [d.error for d in decisions] == [None] * len(CASES)
    assert [(c, d.blocked) for (c, _, _), d in zip(CASES, decisions, strict=True)] == [
        (c, want) for c, _, want in CASES
    ]
    other = runtime.check(GUARD, ToolCall("Edit", {"command": "git push -f"}, "e"), {}, [])
    odd = runtime.check(GUARD, ToolCall("Bash", {"command": 7}, "e"), {}, [])
    assert not other.blocked and not odd.blocked
    runtime.close()


def test_guard_never_reads_its_history() -> None:
    assert reads_history(SOURCE) is False


def test_committed_artifacts_agree() -> None:
    hook = (EXAMPLE / "hooks" / "ratchet_no_force_push_main.py").read_text(encoding="utf-8")
    header = verify_hook(hook)
    envelope = (EXAMPLE / "receipt.json").read_bytes()
    key = public_key((EXAMPLE / "public_key.txt").read_text(encoding="utf-8"))
    decision = verify(envelope, {key_id(key): key})
    harness = harness_from_json(json.loads((EXAMPLE / "harness.json").read_text(encoding="utf-8")))
    guard, provenance = admitted_guard(harness, "no-force-push-main", envelope, {key_id(key): key})
    assert guard.source == SOURCE
    assert header["decision_receipt_sha256"] == sha256_hex(envelope) == provenance.decision_receipt_sha256
    assert header["harness_version"] == harness.version == decision["candidate"]
    assert header["decision_signers"] == key_id(key)
    replay = json.loads((EXAMPLE / "replay.json").read_text(encoding="utf-8"))
    assert replay == decision["checks"][0]["evidence"]
    assert replay["passed"] is True and replay["errors"] == 0
    settings = json.loads((EXAMPLE / "settings.json").read_text(encoding="utf-8"))
    handler = settings["hooks"]["PreToolUse"][0]["hooks"][0]
    assert handler["args"] == ["${CLAUDE_PROJECT_DIR}/.claude/hooks/ratchet_no_force_push_main.py"]
    assert handler["onFailure"] == "block"


def test_replay_over_the_example_transcripts_meets_its_expectation() -> None:
    expected = (EXAMPLE / "expected_fires.txt").read_text(encoding="utf-8").split()
    with GuardRuntime(workers=1) as rt:
        report = replay_transcripts(GUARD, discover([EXAMPLE / "transcripts"]), rt, expect=expected)
    assert report.passed is True and len(report.fires) == 3
