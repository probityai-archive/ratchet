from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from terminal_testkit import Clock, Command, FakeContainer, make_task

from ratchet.agent.model import ModelTurn, ScriptedModel, scripted_turn
from ratchet.bench.terminal import container as C
from ratchet.bench.terminal import runner as R
from ratchet.bench.terminal.container import Container
from ratchet.bench.terminal.tasks import TaskSpec, TerminalCase, load_suite, task_cases
from ratchet.bench.terminal.world import TerminalWorld
from ratchet.types import Event


def one_case(tmp_path: Path) -> TerminalCase:
    make_task(tmp_path, "tiny-task")
    return task_cases(load_suite(tmp_path), ["tiny-task"], "train")[0]


def factory(fc: FakeContainer) -> tuple[list[TaskSpec], Callable[[TaskSpec], Container]]:
    seen: list[TaskSpec] = []

    def make(task: TaskSpec) -> Container:
        seen.append(task)
        return fc

    return seen, make


def test_seed_harness() -> None:
    h = R.seed_harness()
    assert h.verify_digest() and h.parent is None and h.created_by == "seed"
    assert h.tool_allowlist == ["run_command", "write_file"]
    assert h.guards == [] and h.procedures == []
    assert R.seed_harness().version == h.version


def test_package_version_names_the_pinned_package() -> None:
    assert R.package_version() == "0.2.18"


def test_a_run_uses_the_tools_and_is_graded_by_the_tests(tmp_path: Path) -> None:
    case = one_case(tmp_path)
    fc = FakeContainer(commands={"cat /app/out.txt": Command("hello")}, reward_txt="1")
    seen, make = factory(fc)
    model = ScriptedModel(
        [
            scripted_turn(None, [("write_file", {"path": "out.txt", "content": "hello\n"})], id_prefix="a"),
            scripted_turn(None, [("run_command", {"cmd": "cat /app/out.txt"})], id_prefix="b"),
            scripted_turn("Done."),
        ]
    )
    runner = R.TerminalRunner(container_factory=make, clock=Clock())
    trace = runner(R.seed_harness(), case, model, None, run_id="run-1")
    assert trace.outcome.passed and trace.outcome.reasons == []
    assert trace.outcome.cost.model_calls == 3 and trace.outcome.cost.tool_calls == 2
    assert fc.closed and [t.name for t in seen] == ["tiny-task"]
    first = model.requests[0][0]
    assert first[0]["content"] == R.SEED_PROMPT
    assert first[1]["content"].endswith("Write hello to /app/out.txt.\n")
    views = [e.payload["view"] for e in trace.events if e.kind == "tool_call"]
    assert views[1]["files_changed"][0]["path"] == "/app/out.txt"


def test_a_failed_run_carries_the_failing_tests(tmp_path: Path) -> None:
    case = one_case(tmp_path)
    fc = FakeContainer(
        reward_txt="0", ctrf='{"results": {"tests": [{"name": "test_out", "status": "failed"}]}}'
    )
    _, make = factory(fc)
    runner = R.TerminalRunner(container_factory=make)
    trace = runner(R.seed_harness(), case, ScriptedModel([scripted_turn("Nothing to do.")]), None, run_id="r")
    assert not trace.outcome.passed and trace.outcome.reasons == ["test test_out failed"]


def test_the_container_is_removed_when_the_run_fails(tmp_path: Path) -> None:
    case = one_case(tmp_path)
    fc = FakeContainer(setup_ok=False)
    _, make = factory(fc)
    with pytest.raises(RuntimeError, match="world setup failed"):
        R.TerminalRunner(container_factory=make)(R.seed_harness(), case, ScriptedModel([]), None, run_id="r")
    assert fc.closed


class Ticking:
    """A model that spends `step` seconds of the world's clock per call."""

    def __init__(self, clock: Clock, step: float) -> None:
        self.clock, self.step, self.calls = clock, step, 0

    @property
    def name(self) -> str:
        return "ticking"

    def complete(self, messages: object, tools: object) -> ModelTurn:
        self.calls += 1
        self.clock.now += self.step
        return scripted_turn(None, [("run_command", {"cmd": "make"})], id_prefix=f"t{self.calls}_")


def test_the_time_limit_ends_the_run_and_the_tests_still_decide(tmp_path: Path) -> None:
    case = one_case(tmp_path)
    clock = Clock()
    fc = FakeContainer(reward_txt="1")
    _, make = factory(fc)
    model = Ticking(clock, 400.0)
    runner = R.TerminalRunner(container_factory=make, clock=clock)
    trace = runner(R.seed_harness(), case, model, None, run_id="r")
    final = next(e for e in trace.events if e.kind == "final").payload
    assert final["complete"] is True and model.calls == 3
    assert trace.outcome.passed
    deadline = [e for e in trace.events if e.kind == "model"][-1].payload
    assert deadline["finish_reason"] == R.AGENT_TIMEOUT
    assert trace.outcome.cost.model_calls == 3


def test_deadline_model_passes_through_until_the_limit() -> None:
    clock = Clock()
    world = TerminalWorld(FakeContainer(), "t", 5.0, clock=clock)
    inner = ScriptedModel([scripted_turn("hi")])
    m = R.DeadlineModel(inner, world)
    assert m.name == "scripted" and m.complete([], []).content == "hi"
    clock.now += 6
    turn = m.complete([], [])
    assert turn.finish_reason == R.AGENT_TIMEOUT and turn.tool_calls == [] and turn.input_tokens == 0


def test_agent_cost_skips_deadline_replies() -> None:
    events = [
        Event(0, "model", {"usage": {"input_tokens": 5, "output_tokens": 2}, "finish_reason": "stop"}, 1.0),
        Event(1, "tool_call", {}, 2.0),
        Event(2, "model", {"usage": {}, "finish_reason": R.AGENT_TIMEOUT}, 4.5),
    ]
    cost = R.agent_cost(events)
    assert (cost.model_calls, cost.input_tokens, cost.output_tokens, cost.tool_calls, cost.wall_s) == (
        1, 5, 2, 1, 3.5,
    )  # fmt: skip
    assert R.agent_cost([]).wall_s == 0.0


def test_ratchet_case_and_start_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = one_case(tmp_path)
    rc = R.ratchet_case(case)
    assert (rc.id, rc.family, rc.variant, rc.split, rc.ticket_id) == (
        "term-tiny-task", "terminal", "software-engineering", "train", R.TICKET,
    )  # fmt: skip
    started: dict[str, object] = {}

    def fake_start(image: str, **kw: object) -> str:
        started.update(kw, image=image)
        return "container"

    monkeypatch.setattr(C.DockerContainer, "start", staticmethod(fake_start))
    assert R.start_container(case.task) == "container"  # type: ignore[comparison-overlap]
    assert started == {"image": "local/tiny-task:1", "cpus": 1.0, "memory_mb": 1024}
