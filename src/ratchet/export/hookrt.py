"""Ratchet hook runtime for Claude Code. Standard library only.

This module is used two ways, from one source. Ratchet imports it to parse Claude Code transcripts for
replay, and `ratchet.export.claude_code` pastes it verbatim into every generated PreToolUse hook, so the
installed hook builds the guard's inputs with exactly the code that proved the guard against the corpus.

Transcript mapping (one JSONL line per content block, as Claude Code writes them):
  assistant text or tool_use  ->  Event "model"      {"model", "content", "tool_calls"}
  each tool_use               ->  Event "tool_call"  {"call": {"name", "args", "call_id"}, "view"}
  user tool_result            ->  Event "tool_result" {"result": {"call_id", "ok", "data", "error"},
                                                       "decisions": []}
User prompts, thinking blocks, attachments and system lines carry no event: the contract's event kinds
have none for them, and the shapes above are the ones the Ratchet worker records. Tool results are kept
whole: cutting each to 4096 characters shrank the largest local session's history from 27.0 to 25.9 MB,
too little to justify hiding text from a guard.

The guard's `view` for a coding session is {"cwd", "git_branch"}: where the agent is and which branch is
checked out (detached HEAD is "HEAD", outside a repository ""). Replay reads both from the transcript line
that holds the call; the live hook reads `cwd` from the hook input and the branch from the repository.

The history for a call is every event before the transcript line that holds the call. Claude Code writes
the transcript asynchronously, so a live hook may see a shorter history than replay does; it never sees a
longer one.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any

GUARD_ERROR_REASON = "guard error"
RUNTIME_BEGIN = "# >>> ratchet hook runtime (verbatim ratchet/export/hookrt.py) >>>"
RUNTIME_END = "# <<< ratchet hook runtime <<<"


# --- transcript parsing -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallSite:
    """One tool call in a transcript, with what the guard saw at that point."""

    call: dict[str, Any]
    view: dict[str, Any]
    history_len: int
    line: int


@dataclass
class Transcript:
    events: list[dict[str, Any]] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    lines: int = 0
    malformed: int = 0

    def add(self, kind: str, payload: dict[str, Any], ts: float) -> None:
        self.events.append({"seq": len(self.events), "kind": kind, "payload": payload, "ts": ts})

    def site_of(self, call: Mapping[str, Any]) -> CallSite | None:
        """The call's own site: by tool use id, else the last call with the same name and arguments."""
        call_id = call.get("call_id")
        if call_id:
            for site in reversed(self.calls):
                if site.call["call_id"] == call_id:
                    return site
        for site in reversed(self.calls):
            if site.call["name"] == call.get("name") and site.call["args"] == call.get("args"):
                return site
        return None

    def history_before(self, call: Mapping[str, Any]) -> list[dict[str, Any]]:
        site = self.site_of(call)
        return self.events[: site.history_len] if site is not None else list(self.events)


def coding_view(cwd: str, git_branch: str) -> dict[str, Any]:
    return {"cwd": cwd, "git_branch": git_branch}


def _timestamp(value: Any) -> float:
    if not isinstance(value, str):
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _blocks(content: list[Any], kind: str) -> list[dict[str, Any]]:
    return [b for b in content if isinstance(b, dict) and b.get("type") == kind]


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = [
            str(b.get("text", "")) if b.get("type") == "text" else f"[{b.get('type')}]"
            for b in content
            if isinstance(b, dict)
        ]
        text = "\n".join(parts)
    else:
        text = ""
    return text


def _assistant(t: Transcript, entry: dict[str, Any], message: dict[str, Any], line: int, ts: float) -> None:
    content = message["content"]
    texts = [str(b.get("text", "")) for b in _blocks(content, "text")]
    calls = [
        {
            "name": str(b.get("name", "")),
            "args": b["input"] if isinstance(b.get("input"), dict) else {},
            "call_id": str(b.get("id", "")),
        }
        for b in _blocks(content, "tool_use")
    ]
    if not texts and not calls:
        return
    before = len(t.events)
    view = coding_view(str(entry.get("cwd") or ""), str(entry.get("gitBranch") or ""))
    model = {"model": str(message.get("model") or ""), "content": "\n".join(texts), "tool_calls": calls}
    t.add("model", model, ts)
    for call in calls:
        t.calls.append(CallSite(call, view, before, line))
        t.add("tool_call", {"call": call, "view": view}, ts)


def _user(t: Transcript, message: dict[str, Any], ts: float) -> None:
    for block in _blocks(message["content"], "tool_result"):
        text = _result_text(block.get("content"))
        failed = bool(block.get("is_error"))
        result = {
            "call_id": str(block.get("tool_use_id", "")),
            "ok": not failed,
            "data": None if failed else {"content": text},
            "error": text if failed else None,
        }
        t.add("tool_result", {"result": result, "decisions": []}, ts)


def parse_transcript(lines: Iterable[str | bytes]) -> Transcript:
    """Parse Claude Code transcript JSONL. Lines that are not JSON objects are counted, not fatal."""
    t = Transcript()
    for number, raw in enumerate(lines, 1):
        t.lines = number
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except ValueError:
            t.malformed += 1
            continue
        if not isinstance(entry, dict):
            t.malformed += 1
            continue
        message = entry.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        ts = _timestamp(entry.get("timestamp"))
        if entry.get("type") == "assistant":
            _assistant(t, entry, message, number, ts)
        elif entry.get("type") == "user":
            _user(t, message, ts)
    return t


# --- live inputs ------------------------------------------------------------------------------------------


def _head_file(directory: Path) -> Path | None:
    dot = directory / ".git"
    if dot.is_dir():
        return dot / "HEAD"
    if dot.is_file():
        text = dot.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text[len("gitdir:") :].strip())
            return (gitdir if gitdir.is_absolute() else directory / gitdir) / "HEAD"
    return None


def git_branch(cwd: str) -> str:
    """The checked-out branch for `cwd`, as Claude Code records it: a name, "HEAD" when detached, else ""."""
    start = Path(cwd)
    for directory in (start, *start.parents):
        head = _head_file(directory)
        if head is None:
            continue
        try:
            value = head.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        if not value.startswith("ref:"):
            return "HEAD"
        ref = value[len("ref:") :].strip()
        return ref.removeprefix("refs/heads/")
    return ""


def transcript_path(event: Mapping[str, Any]) -> Path | None:
    """The transcript a call belongs to: the subagent's own file when the call came from a subagent."""
    path = event.get("transcript_path")
    if not isinstance(path, str) or not path:
        return None
    agent = event.get("agent_id")
    if isinstance(agent, str) and agent:
        own = Path(path).with_suffix("") / "subagents" / f"agent-{agent}.jsonl"
        if own.is_file():
            return own
    return Path(path)


def live_history(event: Mapping[str, Any], call: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = transcript_path(event)
    if path is None:
        return []
    try:
        with path.open(encoding="utf-8") as handle:
            transcript = parse_transcript(handle)
    except FileNotFoundError:
        return []
    return transcript.history_before(call)


# --- one sandboxed evaluation -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxSpec:
    guard_id: str
    guard_source: str
    child_source: str
    cpu_s: int
    memory_mb: int
    call_timeout_s: float
    startup_timeout_s: float
    require: tuple[str, ...]


@dataclass(frozen=True)
class Verdict:
    block: str | None
    error: str | None = None


def _replies(stdout: bytes) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            reply = json.loads(line)
        except ValueError:
            return out
        if not isinstance(reply, dict):
            return out
        out.append(reply)
    return out


def evaluate_once(
    spec: SandboxSpec, call: Mapping[str, Any], view: Mapping[str, Any], history: list[dict[str, Any]]
) -> Verdict:
    """Run the guard once in a fresh sandboxed child, speaking the Ratchet guard child protocol."""
    setup = {
        "source": spec.guard_source,
        "cpu_s": spec.cpu_s,
        "memory_mb": spec.memory_mb,
        "timeout_s": spec.call_timeout_s,
        "require": list(spec.require),
    }
    request = {"i": 0, "call": dict(call), "view": dict(view), "h_keep": 0, "h_add": history}
    data = "".join(json.dumps(obj, separators=(",", ":")) + "\n" for obj in (setup, request))
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", spec.child_source],
            input=data.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={},
            cwd="/",
            close_fds=True,
            start_new_session=True,
            timeout=spec.startup_timeout_s + spec.call_timeout_s + 0.5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Verdict(None, "timeout")
    except OSError as exc:
        return Verdict(None, f"sandbox did not start: {exc}")
    replies = _replies(proc.stdout)
    if not replies:
        return Verdict(None, f"guard process exited with status {proc.returncode}")
    ready = replies[0]
    if not ready.get("ready"):
        return Verdict(None, f"{ready.get('error')}; layers: {ready.get('isolation')}")
    if len(replies) < 2 or replies[1].get("i") != 0:
        return Verdict(None, f"guard process exited with status {proc.returncode}")
    reply = replies[1]
    if reply.get("error"):
        return Verdict(None, str(reply["error"]))
    block = reply.get("block")
    return Verdict(None if block is None else str(block))


# --- the hook ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HookConfig:
    """`hook_file` and `runtime_sha256` are set in a generated hook, which then re-hashes its own embedded
    runtime at every call; they are None when this module is imported rather than embedded."""

    spec: SandboxSpec
    guard_sha256: str
    child_sha256: str
    runtime_sha256: str | None = None
    hook_file: str | None = None


def embedded_runtime(text: str) -> str | None:
    """The runtime source pasted between the markers of a generated hook, or None without markers."""
    begin, end = text.find(RUNTIME_BEGIN + "\n"), text.find("\n" + RUNTIME_END)
    if begin < 0 or end < begin:
        return None
    return text[begin + len(RUNTIME_BEGIN) + 1 : end].rstrip("\n") + "\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def decide(config: HookConfig, event: Mapping[str, Any]) -> Verdict:
    """The guard's verdict on one PreToolUse event. Every problem is an error verdict, never an allow."""
    spec = config.spec
    if _sha256(spec.guard_source) != config.guard_sha256:
        return Verdict(None, "guard source does not match its recorded digest")
    if _sha256(spec.child_source) != config.child_sha256:
        return Verdict(None, "sandbox child does not match its recorded digest")
    if config.hook_file is not None:
        runtime = embedded_runtime(Path(config.hook_file).read_text(encoding="utf-8"))
        if runtime is None or _sha256(runtime) != config.runtime_sha256:
            return Verdict(None, "hook runtime does not match its recorded digest")
    if event.get("hook_event_name") != "PreToolUse":
        return Verdict(None, f"installed on {event.get('hook_event_name')!r}, expected PreToolUse")
    name, args = event.get("tool_name"), event.get("tool_input")
    if not isinstance(name, str) or not isinstance(args, dict):
        return Verdict(None, "hook input has no tool_name and tool_input")
    call = {"name": name, "args": args, "call_id": str(event.get("tool_use_id") or "")}
    cwd = str(event.get("cwd") or "")
    view = coding_view(cwd, git_branch(cwd) if cwd else "")
    return evaluate_once(spec, call, view, live_history(event, call))


def block_message(guard_id: str, verdict: Verdict) -> str | None:
    if verdict.error is not None:
        return f"blocked by guard {guard_id}: {GUARD_ERROR_REASON}: {verdict.error}"
    if verdict.block is not None:
        return f"blocked by guard {guard_id}: {verdict.block}"
    return None


def run_hook(config: HookConfig, stdin: IO[str], stdout: IO[str], stderr: IO[str]) -> int:
    """Claude Code PreToolUse protocol: exit 0 with no output to leave the call to the normal permission
    flow, or deny with a JSON decision and exit 2, so the call is blocked even where JSON is ignored."""
    try:
        event = json.loads(stdin.read())
        if not isinstance(event, dict):
            raise ValueError("hook input is not a JSON object")
        verdict = decide(config, event)
    except Exception as exc:  # fail closed: a hook that cannot decide blocks
        verdict = Verdict(None, f"{type(exc).__name__}: {exc}")
    reason = block_message(config.spec.guard_id, verdict)
    if reason is None:
        return 0
    decision = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }
    stdout.write(json.dumps({"hookSpecificOutput": decision}))
    stderr.write(reason + "\n")
    return 2
