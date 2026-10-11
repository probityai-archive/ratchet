from __future__ import annotations

import pytest
from terminal_testkit import Clock, Command, FakeContainer

from ratchet.bench.terminal import world as W
from ratchet.bench.terminal.container import ExecResult
from ratchet.types import ToolCall, ToolResult


def make(
    fc: FakeContainer | None = None, clock: Clock | None = None, timeout: float = 900.0
) -> W.TerminalWorld:
    return W.TerminalWorld(fc or FakeContainer(), "tiny-task", timeout, clock=clock or Clock())


def run(world: W.TerminalWorld, cmd: str, **extra: object) -> ToolResult:
    return world.call(ToolCall(W.RUN_COMMAND, {"cmd": cmd, **extra}, "c1"))


def test_setup_runs_once_and_failure_raises() -> None:
    fc = FakeContainer()
    make(fc)
    assert fc.scripts[:2] == [W.SETUP_SCRIPT, W.PROBE]
    with pytest.raises(RuntimeError, match="world setup failed"):
        make(FakeContainer(setup_ok=False))


def test_tools_and_names() -> None:
    world = make()
    assert world.tool_names() == ["run_command", "write_file"]
    names = [t["function"]["name"] for t in world.tools()]
    assert names == ["run_command", "write_file"]
    world.tools()[0]["function"]["name"] = "changed"
    assert world.tools()[0]["function"]["name"] == "run_command"


def test_run_command_returns_output_and_updates_the_view() -> None:
    fc = FakeContainer(commands={"cd /src && make": W_cmd("built", cwd="/src")})
    fc.processes = [(42, "python -m http.server")]
    world = make(fc)
    result = world.call(ToolCall("run_command", {"cmd": "cd /src && make", "timeout_s": 30}, "c1"))
    assert result.ok and result.error is None
    assert result.data == {"exit_code": 0, "timed_out": False, "output": "built", "cwd": "/src"}
    assert fc.files[f"{W.STATE_DIR}/cmd"] == b"cd /src && make"
    script = fc.scripts[-1]
    assert "timeout -k 5 30 " in script
    view = world.view().to_json()
    assert view["cwd"] == "/src" and view["last_exit_code"] == 0 and view["calls"] == 1
    assert view["processes"] == [{"pid": 42, "cmd": "python -m http.server"}]
    assert view["processes_total"] == 1


def W_cmd(output: str = "", exit_code: int = 0, seconds: int = 0, cwd: str | None = None) -> Command:
    return Command(output, exit_code, seconds, cwd)


def test_a_failing_command_is_a_failed_result_with_its_output() -> None:
    world = make(FakeContainer(commands={"false": W_cmd("nope", 1)}))
    result = run(world, "false")
    assert not result.ok and result.error == "exit code 1"
    assert result.data is not None and result.data["output"] == "nope"


def test_a_killed_command_is_reported_as_timed_out() -> None:
    world = make(FakeContainer(commands={"sleep 100": W_cmd("", 124, 10)}))
    result = run(world, "sleep 100", timeout_s=10)
    assert result.error == "killed after 10 s"
    assert world.view().to_json()["last_timed_out"] is True
    # A command that exits 124 by itself, well before its limit, did not time out.
    world = make(FakeContainer(commands={"exit 124": W_cmd("", 124, 0)}))
    assert run(world, "exit 124", timeout_s=10).error == "exit code 124"


def test_a_command_whose_script_never_finished() -> None:
    world = make(FakeContainer(commands={"yes": W_cmd("y\ny\n")}, drop_marker=True))
    result = run(world, "yes")
    assert result.error == "the command did not finish"
    assert result.data is not None and result.data["timed_out"] is True
    assert result.data["output"] == "y\ny\n"


def test_timeouts_are_clamped_to_the_time_left() -> None:
    clock = Clock()
    fc = FakeContainer()
    world = make(fc, clock, timeout=100.0)
    clock.now += 70
    run(world, "x", timeout_s=500)
    assert "timeout -k 5 30 " in fc.scripts[-1]
    run(world, "x", timeout_s="soon")
    assert "timeout -k 5 30 " in fc.scripts[-1]
    run(world, "x", timeout_s=-3)
    assert "timeout -k 5 30 " in fc.scripts[-1]
    clock.now += 40
    assert world.expired and world.view().to_json()["seconds_left"] == 0
    result = run(world, "x")
    assert result.error == "the agent's time limit for this task has passed"


def test_bad_arguments_and_unknown_tools() -> None:
    world = make()
    assert run(world, "  ").error == "run_command needs a non-empty string 'cmd'"
    bad = world.call(ToolCall("write_file", {"path": "a"}, "c2"))
    assert bad.error == "write_file needs string 'path' and 'content'"
    assert world.call(ToolCall("rm_rf", {}, "c3")).error == "unknown tool 'rm_rf'"


def test_staging_failure_is_reported() -> None:
    world = make(FakeContainer(put_fails={f"{W.STATE_DIR}/cmd"}))
    assert str(run(world, "ls").error).startswith("could not stage the command")


def test_write_file_resolves_relative_paths_and_hashes_the_content() -> None:
    fc = FakeContainer()
    world = make(fc)
    result = world.call(ToolCall("write_file", {"path": "out.txt", "content": "hello\n"}, "c1"))
    assert result.ok and result.data is not None
    assert result.data["path"] == "/app/out.txt" and result.data["bytes"] == 6
    assert fc.files["/app/out.txt"] == b"hello\n"
    view = world.view().to_json()
    assert view["files_changed"] == [{"path": "/app/out.txt", "size": 6, "sha256": "ab" * 32}]
    assert view["files_changed_total"] == 1
    absolute = world.call(ToolCall("write_file", {"path": "/etc/x", "content": ""}, "c2"))
    assert absolute.data is not None and absolute.data["path"] == "/etc/x"
    fc.put_fails.add("/app/no")
    failed = world.call(ToolCall("write_file", {"path": "no", "content": "x"}, "c3"))
    assert not failed.ok and failed.error is not None and "could not write /app/no" in failed.error


def test_digest_follows_the_container_state() -> None:
    fc = FakeContainer()
    world = make(fc)
    before = world.digest()
    world.call(ToolCall("write_file", {"path": "f", "content": "1"}, "c1"))
    assert world.digest() != before


def test_a_failed_probe_keeps_the_last_view() -> None:
    fc = FakeContainer()
    world = make(fc)
    original = fc.exec

    def broken(script: str, timeout_s: float, **kw: object) -> ExecResult:
        if script == W.PROBE:
            return ExecResult(1, "")
        return original(script, timeout_s)

    fc.exec = broken  # type: ignore[method-assign]
    world.call(ToolCall("write_file", {"path": "f", "content": "1"}, "c1"))
    assert world.probe_errors == 1 and world.view().to_json()["files_changed"] == []


def test_parse_probe_reads_every_kind_of_line() -> None:
    lines = ["cwd /w", "rc 2", "elapsed 7", "total 3", "B\t120\t/usr", "F\t5\t-\t/w/big", "nofind"]
    lines += [f"P\t{i}\tworker {i}" for i in range(W.MAX_PROCESSES + 3)]
    p = W.parse_probe("\n".join(lines))
    assert (p.cwd, p.exit_code, p.elapsed_s, p.files_total) == ("/w", 2, 7, 3)
    assert p.bulk == {"/usr": 120}
    assert p.files == [{"path": "/w/big", "size": 5, "sha256": None}]
    assert len(p.processes) == W.MAX_PROCESSES and p.processes_total == W.MAX_PROCESSES + 3
    assert W.parse_probe("cwd \n").cwd == "/"


def test_long_text_is_clipped() -> None:
    p = W.parse_probe("P\t1\t" + "x" * 1000)
    assert len(p.processes[0]["cmd"]) == W.MAX_TEXT and p.processes[0]["cmd"].endswith("...")


def test_the_command_script_cuts_long_output_and_marks_its_end() -> None:
    script = W.command_script(0.2, "@@m@@")
    assert "timeout -k 5 1 " in script and "@@m@@" in script
    assert f"head -c {W.OUTPUT_HEAD}" in script and f"tail -c {W.OUTPUT_TAIL}" in script
