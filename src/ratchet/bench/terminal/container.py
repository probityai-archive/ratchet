"""One task container, driven through the docker command line.

`Container` is the surface the world and the grader use: run a shell script inside, write a file, copy a
directory in, and remove the container. `DockerContainer` implements it with `docker run`, `docker exec`,
`docker cp` and `docker rm`; every docker invocation goes through one injectable `run` function, so the tests
drive it without a docker daemon. The container is started from the task's prebuilt image with the task's CPU
and memory limits and kept alive with `sleep infinity`, the way the task suite expects to be run.
"""

from __future__ import annotations

import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    output: str
    timed_out: bool = False


class Container(Protocol):
    @property
    def name(self) -> str: ...

    def exec(
        self, script: str, timeout_s: float, *, env: Mapping[str, str] | None = None, user: str | None = None
    ) -> ExecResult: ...

    def put_file(self, path: str, data: bytes) -> ExecResult: ...

    def put_dir(self, source: Path, dest: str) -> ExecResult: ...

    def close(self) -> None: ...


class ContainerError(RuntimeError):
    """A docker command failed where the run cannot continue (start, copy)."""


@dataclass(frozen=True)
class Completed:
    returncode: int
    output: bytes
    timed_out: bool = False


Run = Callable[[Sequence[str], bytes | None, float | None], Completed]


def subprocess_run(argv: Sequence[str], stdin: bytes | None, timeout_s: float | None) -> Completed:
    """Run one command with stdout and stderr merged. A timeout kills the client process."""
    try:
        done = subprocess.run(
            list(argv), input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.output if isinstance(exc.output, bytes) else b""
        return Completed(-1, out, timed_out=True)
    return Completed(done.returncode, done.stdout)


def _text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


class DockerContainer:
    """A running container. Build one with `DockerContainer.start`."""

    def __init__(self, name: str, run: Run = subprocess_run, docker: str = "docker") -> None:
        self._name = name
        self._run = run
        self._docker = docker
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def start(
        cls,
        image: str,
        *,
        cpus: float = 1.0,
        memory_mb: int = 2048,
        name: str | None = None,
        run: Run = subprocess_run,
        docker: str = "docker",
        pull_timeout_s: float = 1800.0,
    ) -> DockerContainer:
        cname = name or f"ratchet-term-{uuid.uuid4().hex[:12]}"
        argv = [
            docker, "run", "-d", "--pull", "missing", "--name", cname,
            "--cpus", f"{cpus:g}", "--memory", f"{memory_mb}m",
            image, "sh", "-c", "sleep infinity",
        ]  # fmt: skip
        done = run(argv, None, pull_timeout_s)
        if done.returncode != 0 or done.timed_out:
            run([docker, "rm", "-f", cname], None, 120.0)
            state = "timed out" if done.timed_out else f"exited {done.returncode}"
            raise ContainerError(f"docker run {image} {state}: {_text(done.output)[-500:]}")
        return cls(cname, run, docker)

    def exec(
        self, script: str, timeout_s: float, *, env: Mapping[str, str] | None = None, user: str | None = None
    ) -> ExecResult:
        argv = [self._docker, "exec", "-i"]
        if user:
            argv += ["-u", user]
        for key, value in sorted((env or {}).items()):
            argv += ["-e", f"{key}={value}"]
        argv += [self._name, "sh", "-s"]
        done = self._run(argv, script.encode("utf-8"), timeout_s)
        return ExecResult(done.returncode, _text(done.output), done.timed_out)

    def put_file(self, path: str, data: bytes) -> ExecResult:
        argv = [
            self._docker, "exec", "-i", self._name,
            "sh", "-c", 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"', "sh", path,
        ]  # fmt: skip
        done = self._run(argv, data, 120.0)
        return ExecResult(done.returncode, _text(done.output), done.timed_out)

    def put_dir(self, source: Path, dest: str) -> ExecResult:
        made = self._run([self._docker, "exec", self._name, "mkdir", "-p", dest], None, 120.0)
        if made.returncode != 0:
            raise ContainerError(f"mkdir {dest} in {self._name} failed: {_text(made.output)[-300:]}")
        done = self._run([self._docker, "cp", f"{source}/.", f"{self._name}:{dest}"], None, 600.0)
        if done.returncode != 0 or done.timed_out:
            raise ContainerError(f"docker cp {source} -> {dest} failed: {_text(done.output)[-300:]}")
        return ExecResult(done.returncode, _text(done.output))

    def close(self) -> None:
        if not self.closed:
            self._run([self._docker, "rm", "-f", self._name], None, 120.0)
            self.closed = True
