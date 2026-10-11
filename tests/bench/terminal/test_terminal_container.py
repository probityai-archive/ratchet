from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from ratchet.bench.terminal.container import Completed, ContainerError, DockerContainer, subprocess_run


class FakeDocker:
    def __init__(self, replies: Sequence[Completed] = ()) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[str], bytes | None, float | None]] = []

    def __call__(self, argv: Sequence[str], stdin: bytes | None, timeout_s: float | None) -> Completed:
        self.calls.append((list(argv), stdin, timeout_s))
        return self.replies.pop(0) if self.replies else Completed(0, b"ok")


def test_start_runs_the_image_with_the_task_limits() -> None:
    docker = FakeDocker()
    c = DockerContainer.start("img:1", cpus=2, memory_mb=4096, name="box", run=docker)
    argv = docker.calls[0][0]
    assert argv[:2] == ["docker", "run"]
    assert argv[argv.index("--cpus") : argv.index("--cpus") + 2] == ["--cpus", "2"]
    assert "4096m" in argv and "img:1" in argv and argv[-3:] == ["sh", "-c", "sleep infinity"]
    assert c.name == "box"


def test_start_failure_removes_the_container_and_raises() -> None:
    docker = FakeDocker([Completed(125, b"pull access denied")])
    with pytest.raises(ContainerError, match="exited 125: pull access denied"):
        DockerContainer.start("img:1", name="box", run=docker)
    assert docker.calls[1][0] == ["docker", "rm", "-f", "box"]
    docker = FakeDocker([Completed(-1, b"", timed_out=True)])
    with pytest.raises(ContainerError, match="timed out"):
        DockerContainer.start("img:1", run=docker)
    assert docker.calls[0][0][docker.calls[0][0].index("--name") + 1].startswith("ratchet-term-")


def test_exec_sends_the_script_on_stdin_with_user_and_env() -> None:
    docker = FakeDocker([Completed(3, b"out\xff")])
    c = DockerContainer("box", docker)
    r = c.exec("echo hi", 5.0, env={"B": "2", "A": "1"}, user="root")
    argv, stdin, timeout = docker.calls[0]
    assert argv == ["docker", "exec", "-i", "-u", "root", "-e", "A=1", "-e", "B=2", "box", "sh", "-s"]
    assert (stdin, timeout) == (b"echo hi", 5.0)
    assert (r.exit_code, r.output, r.timed_out) == (3, "out�", False)


def test_put_file_and_put_dir(tmp_path: Path) -> None:
    docker = FakeDocker()
    c = DockerContainer("box", docker)
    assert c.put_file("/a/b.txt", b"data").exit_code == 0
    argv, stdin, _ = docker.calls[0]
    assert argv[-2:] == ["sh", "/a/b.txt"] and stdin == b"data"
    c.put_dir(tmp_path, "/tests")
    assert docker.calls[1][0] == ["docker", "exec", "box", "mkdir", "-p", "/tests"]
    assert docker.calls[2][0] == ["docker", "cp", f"{tmp_path}/.", "box:/tests"]


def test_put_dir_failures_raise(tmp_path: Path) -> None:
    c = DockerContainer("box", FakeDocker([Completed(1, b"read-only")]))
    with pytest.raises(ContainerError, match="mkdir /tests"):
        c.put_dir(tmp_path, "/tests")
    c = DockerContainer("box", FakeDocker([Completed(0, b""), Completed(1, b"no space")]))
    with pytest.raises(ContainerError, match="docker cp"):
        c.put_dir(tmp_path, "/tests")


def test_close_removes_once() -> None:
    docker = FakeDocker()
    c = DockerContainer("box", docker)
    c.close()
    c.close()
    assert [a for a, _, _ in docker.calls] == [["docker", "rm", "-f", "box"]]


def test_subprocess_run_merges_output_and_reports_timeouts() -> None:
    done = subprocess_run(["sh", "-c", "echo out; echo err >&2; exit 3"], None, 10.0)
    assert done.returncode == 3 and b"out" in done.output and b"err" in done.output
    assert subprocess_run(["sh", "-s"], b"echo $((2+2))", 10.0).output == b"4\n"
    late = subprocess_run(["sh", "-c", "sleep 5"], None, 0.05)
    assert late.timed_out and late.returncode == -1
