"""Shared helpers for the terminal adapter tests: a tiny local task and a fake container.

`FakeContainer` answers the world's and the grader's scripts the way a real container would, from a small
model: staged commands are looked up in `commands` (command text -> output, exit code, seconds, new cwd),
files written through `put_file` are kept in `files`, and the verifier reports `reward` and the CTRF or pytest
text it is given. Every script and file it received is recorded for assertions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ratchet.bench.terminal.container import ExecResult
from ratchet.bench.terminal.grade import COLLECT, RUN_TESTS
from ratchet.bench.terminal.world import PROBE, SETUP_SCRIPT, STATE_DIR

TASK_TOML = """schema_version = "1.1"

[task]
name = "local/{name}"

[metadata]
difficulty = "{difficulty}"
category = "software-engineering"

[verifier]
timeout_sec = 60.0

[verifier.env]
MODE = "strict"

[agent]
timeout_sec = {timeout}

[environment]
docker_image = "local/{name}:1"
cpus = 1
memory_mb = 1024
"""


def make_task(
    root: Path,
    name: str = "tiny-task",
    *,
    difficulty: str = "hard",
    timeout: float = 900.0,
    image: bool = True,
) -> Path:
    d = root / name
    (d / "tests").mkdir(parents=True)
    (d / "solution").mkdir()
    toml = TASK_TOML.format(name=name, difficulty=difficulty, timeout=timeout)
    if not image:
        toml = toml.replace(f'docker_image = "local/{name}:1"\n', "")
    (d / "task.toml").write_text(toml, encoding="utf-8")
    (d / "instruction.md").write_text("Write hello to /app/out.txt.\n", encoding="utf-8")
    (d / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n", encoding="utf-8"
    )
    (d / "solution" / "solve.sh").write_text("echo hello > /app/out.txt\n", encoding="utf-8")
    return d


@dataclass
class Command:
    output: str = ""
    exit_code: int = 0
    seconds: int = 0
    cwd: str | None = None


@dataclass
class FakeContainer:
    name: str = "fake-1"
    commands: dict[str, Command] = field(default_factory=dict)
    cwd: str = "/app"
    files: dict[str, bytes] = field(default_factory=dict)
    changed: list[str] = field(default_factory=list)
    bulk: dict[str, int] = field(default_factory=dict)
    processes: list[tuple[int, str]] = field(default_factory=list)
    reward_txt: str = "1"
    reward_json: str = ""
    ctrf: str = ""
    stdout: str = ""
    setup_ok: bool = True
    drop_marker: bool = False
    put_fails: set[str] = field(default_factory=set)
    tests_timeout: bool = False
    scripts: list[str] = field(default_factory=list)
    dirs: list[tuple[Path, str]] = field(default_factory=list)
    envs: list[Mapping[str, str] | None] = field(default_factory=list)
    closed: bool = False

    def probe_text(self) -> str:
        lines = [f"cwd {self.cwd}", f"total {len(self.changed)}"]
        lines += [f"B\t{n}\t{top}" for top, n in sorted(self.bulk.items())]
        for path in sorted(self.changed):
            data = self.files.get(path)
            lines.append(
                f"F\t{len(data) if data is not None else 2000000}\t{'ab' * 32 if data else '-'}\t{path}"
            )
        lines += [f"P\t{pid}\t{cmd} " for pid, cmd in self.processes]
        return "\n".join(lines) + "\n"

    def exec(
        self, script: str, timeout_s: float, *, env: Mapping[str, str] | None = None, user: str | None = None
    ) -> ExecResult:
        self.scripts.append(script)
        self.envs.append(env)
        if script == SETUP_SCRIPT:
            return ExecResult(0 if self.setup_ok else 1, "ready\n" if self.setup_ok else "no /proc\n")
        if script == PROBE:
            return ExecResult(0, self.probe_text())
        if script == RUN_TESTS:
            return ExecResult(-1 if self.tests_timeout else 0, "test.sh exited 0\n", self.tests_timeout)
        if script == COLLECT:
            return ExecResult(
                0,
                f"@@reward_json\n{self.reward_json}\n@@reward_txt\n{self.reward_txt}\n"
                f"@@ctrf\n{self.ctrf}\n@@stdout\n{self.stdout}",
            )
        found = re.search(r"@@ratchet-[0-9a-f]+@@", script)
        assert found is not None, "unknown script"
        cmd = self.files.get(f"{STATE_DIR}/cmd", b"").decode("utf-8")
        step = self.commands.get(cmd, Command(f"ran {cmd}"))
        if step.cwd:
            self.cwd = step.cwd
        if self.drop_marker:
            return ExecResult(-1, step.output, True)
        text = (
            f"{step.output}\n{found.group(0)}\nrc {step.exit_code}\nelapsed {step.seconds}\n"
            + self.probe_text()
        )
        return ExecResult(0, text)

    def put_file(self, path: str, data: bytes) -> ExecResult:
        if path in self.put_fails:
            return ExecResult(1, "Permission denied")
        self.files[path] = data
        if not path.startswith(STATE_DIR):
            self.changed.append(path)
        return ExecResult(0, "")

    def put_dir(self, source: Path, dest: str) -> ExecResult:
        self.dirs.append((source, dest))
        return ExecResult(0, "")

    def close(self) -> None:
        self.closed = True


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now
