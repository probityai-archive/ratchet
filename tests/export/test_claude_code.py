from __future__ import annotations

import ast
import json
import stat
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from ratchet_cases.types import Event, ToolCall, sha256_hex

from ratchet.agent.guards import CHILD_PATH, GuardLimits, GuardRuntime
from ratchet.export import hookrt
from ratchet.export.claude_code import (
    HEADER_KEYS,
    RUNTIME_PATH,
    ExportError,
    admitted_guard,
    export_hook,
    harness_from_json,
    main,
    public_key,
    py_literal,
    read_header,
    render_hook,
    settings_entry,
    text_sha256,
    tuple_literal,
    verify_hook,
)
from ratchet.export.hookrt import parse_transcript
from ratchet.improve.admit import Check, Decision
from ratchet.improve.receipt import Signer, receipt
from ratchet.types import Guard, Harness
from ratchet.verify.dsse import raw_public

from .synth import Session, hook_event

NO_RM = """
def check(call, view, history):
    if call.name == "Bash" and call.args.get("command", "").startswith("rm "):
        return "rm is not allowed (" + str(len(history)) + " earlier events)"
    return None
"""


def harness(source: str = NO_RM) -> Harness:
    return Harness.build(
        parent=None,
        system_prompt="",
        procedures=[],
        guards=[Guard("no-rm", "never rm", source)],
        tool_allowlist=["Bash"],
        created_by="human",
    )


def decision(h: Harness, admitted: bool = True) -> Decision:
    return Decision(
        parent="root",
        candidate=h.version,
        candidate_digest=h.digest,
        admitted=admitted,
        checks=[Check("replay", admitted, {"fires": 1})],
        decided_at="2026-10-11T12:00:00+00:00",
    )


@pytest.fixture(scope="module")
def signer() -> Signer:
    return Signer.generate()


def keys(signer: Signer) -> dict[str, Any]:
    return {signer.keyid: signer.public_key}


def test_admitted_guard_carries_provenance(signer: Signer) -> None:
    h = harness()
    envelope = receipt(decision(h), signer)
    guard, prov = admitted_guard(h, "no-rm", envelope, keys(signer))
    assert guard.source == NO_RM
    assert prov.guard_id == "no-rm" and prov.guard_sha256 == text_sha256(NO_RM)
    assert (prov.harness_version, prov.harness_digest) == (h.version, h.digest)
    assert prov.decision_receipt_sha256 == sha256_hex(envelope)
    assert prov.decision_signers == signer.keyid
    assert prov.decided_at == "2026-10-11T12:00:00+00:00"
    other = Signer.generate()
    both = {**keys(signer), other.keyid: other.public_key}
    assert admitted_guard(h, "no-rm", envelope, both)[1].decision_signers == signer.keyid


def test_admitted_guard_refuses_what_the_receipt_does_not_admit(signer: Signer) -> None:
    h = harness()
    good = receipt(decision(h), signer)
    with pytest.raises(ExportError, match="does not verify"):
        admitted_guard(h, "no-rm", good, keys(Signer.generate()))
    with pytest.raises(ExportError, match="does not match its digest"):
        admitted_guard(replace(h, system_prompt="changed"), "no-rm", good, keys(signer))
    with pytest.raises(ExportError, match="did not admit"):
        admitted_guard(h, "no-rm", receipt(decision(h, admitted=False), signer), keys(signer))
    with pytest.raises(ExportError, match="not harness"):
        admitted_guard(harness(NO_RM + "\n"), "no-rm", good, keys(signer))
    with pytest.raises(ExportError, match="has no guard"):
        admitted_guard(h, "other", good, keys(signer))


@pytest.mark.parametrize(
    "text",
    [
        "plain",
        "a\nb",
        'two\n"""quotes"""',
        "both\n\"\"\"\n'''",
        "ends in a backslash\n\\",
        'ends in a quote\n"',
        'tab\there "q"',
    ],
)
def test_py_literal_is_exact(text: str) -> None:
    assert ast.literal_eval(py_literal(text)) == text
    assert py_literal("a\nb") == 'r"""a\nb"""'
    assert py_literal('x\n"""') == "r'''x\n\"\"\"'''"
    assert py_literal("one line") == '"one line"'
    assert tuple_literal([]) == "()" and tuple_literal(["a"]) == '("a",)'
    assert tuple_literal(["a", "b"]) == '("a", "b")'


def render(signer: Signer, source: str = NO_RM, limits: GuardLimits | None = None) -> str:
    h = harness(source)
    guard, prov = admitted_guard(h, "no-rm", receipt(decision(h), signer), keys(signer))
    return render_hook(guard, prov, limits)


def test_rendered_hook_carries_and_matches_its_header(signer: Signer) -> None:
    text = render(signer)
    header = verify_hook(text)
    assert set(header) == set(HEADER_KEYS)
    assert header["sandbox_child_sha256"] == text_sha256(CHILD_PATH.read_text(encoding="utf-8"))
    assert header["hook_runtime_sha256"] == text_sha256(RUNTIME_PATH.read_text(encoding="utf-8"))
    assert header["sandbox_require"] == "rlimits,netns,seccomp,landlock"
    assert text.startswith("#!/usr/bin/env python3\n")
    assert read_header(render(signer, limits=GuardLimits(require=())))["sandbox_require"] == "none"


def test_verify_hook_catches_every_kind_of_edit(signer: Signer) -> None:
    text = render(signer)
    with pytest.raises(ExportError, match="guard_sha256"):
        verify_hook(text.replace('startswith("rm ")', 'startswith("rmdir ")'))
    with pytest.raises(ExportError, match="hook_runtime_sha256"):
        verify_hook(text.replace('GUARD_ERROR_REASON = "guard error"', 'GUARD_ERROR_REASON = "ok"'))
    with pytest.raises(ExportError, match="sandbox_child_sha256"):
        verify_hook(text.replace('"statistics",', '"statistics", "os",', 1))
    head = read_header(text)
    with pytest.raises(ExportError, match="GUARD_SHA256"):
        verify_hook(text.replace(f'GUARD_SHA256 = "{head["guard_sha256"]}"', f'GUARD_SHA256 = "{"0" * 64}"'))
    with pytest.raises(ExportError, match="RUNTIME_SHA256"):
        verify_hook(
            text.replace(
                f'RUNTIME_SHA256 = "{head["hook_runtime_sha256"]}"', f'RUNTIME_SHA256 = "{"0" * 64}"'
            )
        )
    with pytest.raises(ExportError, match="missing"):
        verify_hook(text.replace("# decided_at:", "# decided:"))
    with pytest.raises(ExportError, match="markers"):
        verify_hook(text.replace("# <<< ratchet hook runtime <<<", ""))


def run_hook_file(path: Path, event: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path)], input=json.dumps(event), capture_output=True, text=True, timeout=60
    )


def test_installed_hook_decides_like_the_guard_runtime(signer: Signer, tmp_path: Path) -> None:
    h = harness()
    out = tmp_path / "hooks" / "ratchet_no_rm.py"
    export_hook(h, "no-rm", receipt(decision(h), signer), keys(signer), out)
    assert out.stat().st_mode & stat.S_IXUSR
    transcript = Session().prompt("tidy up").bash("c1", "ls").result("c1", "a b").bash("c2", "rm -rf a")
    path = transcript.write(tmp_path / "s.jsonl")

    blocked = run_hook_file(out, hook_event(path, "c2", "rm -rf a"))
    assert blocked.returncode == 2
    reason = json.loads(blocked.stdout)["hookSpecificOutput"]["permissionDecisionReason"]
    assert reason == "blocked by guard no-rm: rm is not allowed (3 earlier events)"
    assert blocked.stderr == reason + "\n"
    history = [
        Event(e["seq"], e["kind"], e["payload"], e["ts"])
        for e in parse_transcript(transcript.text().splitlines()).events[:3]
    ]
    with GuardRuntime(workers=1) as rt:
        direct = rt.check(h.guards[0], ToolCall("Bash", {"command": "rm -rf a"}, "c2"), {}, history)
    assert reason == f"blocked by guard no-rm: {direct.reason}"

    allowed = run_hook_file(out, hook_event(path, "c1", "ls"))
    assert (allowed.returncode, allowed.stdout, allowed.stderr) == (0, "", "")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('startswith("rm ")', 'startswith("never ")', "guard source does not match its recorded digest"),
        (
            'GUARD_ERROR_REASON = "guard error"',
            'GUARD_ERROR_REASON = "ok"',
            "hook runtime does not match its recorded digest",
        ),
        ("# <<< ratchet hook runtime <<<", "", "hook runtime does not match its recorded digest"),
        ('"statistics",', '"statistics", "os",', "sandbox child does not match its recorded digest"),
    ],
)
def test_tampered_hook_denies_every_call(
    signer: Signer, tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = tmp_path / "hook.py"
    path.write_text(render(signer).replace(old, new, 1), encoding="utf-8")
    result = run_hook_file(path, hook_event(None, "c", "ls"))
    assert result.returncode == 2
    assert message in result.stderr


def test_generated_hook_is_lint_and_format_clean(signer: Signer, tmp_path: Path) -> None:
    path = tmp_path / "hook.py"
    path.write_text(render(signer), encoding="utf-8")
    for args in (["check", "--no-cache", str(path)], ["format", "--no-cache", "--check", str(path)]):
        done = subprocess.run(
            [sys.executable, "-m", "ruff", *args], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stdout + done.stderr


def test_settings_entry_blocks_when_the_hook_cannot_start() -> None:
    entry = settings_entry("python3", "${CLAUDE_PROJECT_DIR}/.claude/hooks/h.py", "Bash")
    handler = entry["hooks"]["PreToolUse"][0]["hooks"][0]
    assert entry["hooks"]["PreToolUse"][0]["matcher"] == "Bash"
    assert handler == {
        "type": "command",
        "command": "python3",
        "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/h.py"],
        "onFailure": "block",
    }


def write_inputs(tmp_path: Path, signer: Signer, admitted: bool = True) -> list[str]:
    h = harness()
    (tmp_path / "harness.json").write_text(json.dumps(asdict(h)), encoding="utf-8")
    (tmp_path / "receipt.json").write_bytes(receipt(decision(h, admitted), signer))
    key = raw_public(signer.public_key).hex()
    return [
        "--harness",
        "harness.json",
        "--guard-id",
        "no-rm",
        "--receipt",
        "receipt.json",
        "--trusted-key",
        key,
    ]


def test_command_line_exports_and_verifies(
    signer: Signer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    args = write_inputs(tmp_path, signer)
    assert main([*args, "--out", ".claude/hooks/no_rm.py", "--require", ""]) == 0
    printed = capsys.readouterr().out
    assert '"${CLAUDE_PROJECT_DIR}/.claude/hooks/no_rm.py"' in printed
    assert read_header((tmp_path / ".claude/hooks/no_rm.py").read_text())["sandbox_require"] == "none"
    assert main(["--verify", ".claude/hooks/no_rm.py"]) == 0
    assert json.loads(capsys.readouterr().out)["guard_id"] == "no-rm"
    absolute = tmp_path / "abs.py"
    assert main([*args, "--out", str(absolute)]) == 0
    assert f'"{absolute}"' in capsys.readouterr().out


def test_command_line_refuses(signer: Signer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = write_inputs(tmp_path, signer, admitted=False)
    assert main(args) == 1
    assert main([*args, "--out", "h.py"]) == 1
    assert main([*args[:-1], "abcd", "--out", "h.py"]) == 1
    assert not (tmp_path / "h.py").exists()
    with pytest.raises(ExportError):
        public_key("zz")


def test_harness_round_trips_through_json() -> None:
    h = harness()
    assert harness_from_json(json.loads(json.dumps(asdict(h)))) == h


def test_decide_starts_and_reuses_the_session_server(signer: Signer, tmp_path: Path) -> None:
    h = harness()
    out = tmp_path / "hook.py"
    prov = export_hook(h, "no-rm", receipt(decision(h), signer), keys(signer), out)
    lim = GuardLimits()
    child = CHILD_PATH.read_text(encoding="utf-8")
    spec = hookrt.SandboxSpec(
        "no-rm",
        NO_RM,
        child,
        lim.cpu_s,
        lim.memory_mb,
        lim.call_timeout_s,
        lim.startup_timeout_s,
        lim.require,
    )
    runtime = RUNTIME_PATH.read_text(encoding="utf-8")
    cfg = hookrt.HookConfig(spec, prov.guard_sha256, text_sha256(child), text_sha256(runtime), str(out))
    path = Session().bash("c1", "ls").result("c1", "a").bash("c2", "rm -rf a").write(tmp_path / "s.jsonl")
    event = hook_event(path, "c2", "rm -rf a")
    first = hookrt.decide(cfg, event)
    assert first == hookrt.decide(cfg, event, use_server=False)
    assert first.block == "rm is not allowed (3 earlier events)"
    directory = hookrt.socket_dir()
    assert directory is not None
    assert (directory / f"{hookrt.server_key(cfg)}.sock").exists()
    assert hookrt.decide(cfg, hook_event(path, "c1", "ls")) == hookrt.Verdict(None)
