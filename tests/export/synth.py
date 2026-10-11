"""Synthetic Claude Code transcripts, written to the documented JSONL shape: one content block per line,
assistant `tool_use` blocks and user `tool_result` blocks, plus the bookkeeping lines a real session has.
Every value here is invented for tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Session:
    session_id: str = "00000000-0000-4000-8000-000000000001"
    cwd: str = "/work/repo"
    branch: str = "main"
    lines: list[dict[str, Any]] = field(default_factory=list)
    _n: int = 0

    def _base(self, kind: str) -> dict[str, Any]:
        self._n += 1
        return {
            "type": kind,
            "uuid": f"u-{self._n}",
            "parentUuid": f"u-{self._n - 1}" if self._n > 1 else None,
            "sessionId": self.session_id,
            "timestamp": f"2026-10-11T12:00:{self._n % 60:02d}.000Z",
            "cwd": self.cwd,
            "gitBranch": self.branch,
            "isSidechain": False,
            "version": "2.1.300",
        }

    def prompt(self, text: str) -> Session:
        self.lines.append({**self._base("user"), "message": {"role": "user", "content": text}})
        return self

    def say(self, text: str) -> Session:
        block = {"type": "text", "text": text}
        message = {"id": f"m-{self._n}", "model": "claude-test", "role": "assistant", "content": [block]}
        self.lines.append({**self._base("assistant"), "message": message})
        return self

    def think(self) -> Session:
        block = {"type": "thinking", "thinking": "", "signature": "x"}
        message = {"id": f"m-{self._n}", "model": "claude-test", "role": "assistant", "content": [block]}
        self.lines.append({**self._base("assistant"), "message": message})
        return self

    def call(self, call_id: str, name: str, args: dict[str, Any]) -> Session:
        block = {"type": "tool_use", "id": call_id, "name": name, "input": args, "caller": {"type": "direct"}}
        message = {"id": f"m-{self._n}", "model": "claude-test", "role": "assistant", "content": [block]}
        self.lines.append({**self._base("assistant"), "message": message})
        return self

    def bash(self, call_id: str, command: str) -> Session:
        return self.call(call_id, "Bash", {"command": command, "description": "run"})

    def result(self, call_id: str, content: Any, is_error: bool = False) -> Session:
        block = {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": is_error}
        self.lines.append({**self._base("user"), "message": {"role": "user", "content": [block]}})
        return self

    def noise(self) -> Session:
        self.lines.append(
            {"type": "permission-mode", "permissionMode": "default", "sessionId": self.session_id}
        )
        self.lines.append({**self._base("attachment"), "attachment": {"type": "todo"}})
        self.lines.append({"type": "ai-title", "aiTitle": "t", "sessionId": self.session_id})
        return self

    def text(self) -> str:
        return "".join(json.dumps(line) + "\n" for line in self.lines)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text(), encoding="utf-8")
        return path


def hook_event(
    transcript: Path | None, call_id: str, command: str, cwd: str = "/work/repo", **extra: Any
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "session_id": "00000000-0000-4000-8000-000000000001",
        "cwd": cwd,
        "permission_mode": "default",
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": command, "description": "run"},
        "tool_use_id": call_id,
    }
    if transcript is not None:
        event["transcript_path"] = str(transcript)
    event.update(extra)
    return event
