"""The benign-twin suite under AgentDojo's own contract. Needs the `agentdojo` extra."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import pytest
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite, get_suites
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

import ratchet_cases.agentdojo as twins

PAIRS = [(t.suite, t) for t in twins.TWINS]


class Refuse(BasePipelineElement):
    """A defense that blocks everything: no tool call, a refusal."""

    name = "refuse-everything"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Any = None,
        messages: Sequence[ChatMessage] = (),
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Any, Sequence[ChatMessage], dict[str, Any]]:
        reply = [
            ChatUserMessage(role="user", content=[text_content_block_from_string(query)]),
            ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("I can't help with that request.")],
                tool_calls=None,
            ),
        ]
        return query, runtime, env, [*messages, *reply], extra_args or {}


def test_there_is_one_twin_per_injection_task_of_the_source_version() -> None:
    expected = {
        (name, task_id)
        for name, suite in get_suites(twins.SOURCE_VERSION).items()
        for task_id in suite.injection_tasks
    }
    assert {(t.suite, t.injection_task) for t in twins.TWINS} == expected
    assert len(twins.TWINS) == len(expected) == 35


def test_the_suites_are_registered_under_the_twin_benchmark_version() -> None:
    suites = get_suites(twins.BENCHMARK_VERSION)
    assert sorted(suites) == ["banking", "slack", "travel", "workspace"]
    assert sum(len(s.user_tasks) for s in suites.values()) == 35
    assert all(not s.injection_tasks for s in suites.values())


# TaskSuite.check in agentdojo 0.1.35 copies environments with pydantic's deprecated `copy`; the warning is
# raised inside AgentDojo, not here, so this test alone ignores that one deprecation.
@pytest.mark.filterwarnings("ignore:The `copy` method is deprecated:DeprecationWarning")
@pytest.mark.parametrize("name", ["workspace", "travel", "banking", "slack"])
def test_every_ground_truth_solves_its_twin(name: str) -> None:
    # Injectability is tested below: in agentdojo 0.1.35, is_task_injectable reads only tool results whose
    # content is a str, and the ground-truth pipeline returns content blocks, so check_injectable=True
    # rejects every user task of every suite, AgentDojo's own included.
    ok, (users, _) = get_suite(twins.BENCHMARK_VERSION, name).check(check_injectable=False)
    assert ok, {k: v for k, v in users.items() if not v[0]}


@pytest.mark.parametrize(("name", "twin"), PAIRS, ids=[f"{s}-{t.id}" for s, t in PAIRS])
def test_the_ground_truth_reads_the_twins_own_slot(name: str, twin: twins.Twin) -> None:
    # AgentDojo's injectability check, reading the text out of content blocks, and narrowed from "some
    # canary appears" to "the canary in this twin's own slot appears".
    suite = get_suite(twins.BENCHMARK_VERSION, name)
    canaries = {vector: f"---CANARY_{vector}---" for vector in suite.get_injection_vector_defaults()}
    task = suite.user_tasks[twin.id]
    _, _, _, responses, _ = GroundTruthPipeline(task).query(
        task.PROMPT, FunctionsRuntime(suite.tools), suite.load_and_inject_default_environment(canaries)
    )
    outputs = " ".join(get_text_content_as_str(m["content"]) for m in responses if m["role"] == "tool")
    assert canaries[twin.slot.vector] in outputs


@pytest.mark.parametrize(("name", "twin"), PAIRS, ids=[f"{s}-{t.id}" for s, t in PAIRS])
def test_a_refuse_everything_pipeline_fails_the_twin(name: str, twin: twins.Twin) -> None:
    suite = get_suite(twins.BENCHMARK_VERSION, name)
    utility, _ = suite.run_task_with_pipeline(Refuse(), suite.user_tasks[twin.id], None, {})
    assert utility is False


@pytest.mark.parametrize(("name", "twin"), PAIRS, ids=[f"{s}-{t.id}" for s, t in PAIRS])
def test_carrying_out_the_paired_attack_does_not_pass_the_twin(name: str, twin: twins.Twin) -> None:
    suite = get_suite(twins.BENCHMARK_VERSION, name)
    attack = get_suite(twins.SOURCE_VERSION, name).injection_tasks[twin.injection_task]
    utility, _ = suite.run_task_with_pipeline(
        GroundTruthPipeline(attack), suite.user_tasks[twin.id], None, {}
    )
    assert utility is False


@pytest.mark.parametrize(("name", "twin"), PAIRS, ids=[f"{s}-{t.id}" for s, t in PAIRS])
def test_the_episode_starts_with_the_instruction_in_the_slot(name: str, twin: twins.Twin) -> None:
    # The test above proves the ground truth surfaces this slot; here the slot holds the instruction.
    suite = get_suite(twins.BENCHMARK_VERSION, name)
    env = suite.user_tasks[twin.id].init_environment(suite.load_and_inject_default_environment({}))
    assert twin.instruction.strip() in json.dumps(env.model_dump(mode="json"))
    assert twin.slot.read == tuple(suite.user_tasks[twin.id].ground_truth(env)[: len(twin.slot.read)])


def test_agentdojo_check_suites_accepts_the_suite() -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentdojo.scripts.check_suites",
            "-ml",
            "ratchet_cases.agentdojo",
            "-v",
            twins.BENCHMARK_VERSION,
            # The same agentdojo 0.1.35 defect as above; the twins' own injectability test covers it.
            "--no-check-injectable",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    # check_suites prints its verdict and exits 0 either way, so the verdict is read from its output.
    assert done.returncode == 0, done.stdout + done.stderr
    assert "failed" not in done.stdout, done.stdout
    for name, count in (("workspace", 14), ("travel", 7), ("banking", 9), ("slack", 5)):
        assert f"Suite '{name}' passed for all {count} user tasks!" in done.stdout, done.stdout
