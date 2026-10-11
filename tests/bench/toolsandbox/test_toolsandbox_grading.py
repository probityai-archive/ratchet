from __future__ import annotations

import polars as pl
import pytest
from tool_sandbox.common.evaluation import (
    Milestone,
    SnapshotConstraint,
    snapshot_similarity,
)
from tool_sandbox.common.execution_context import DatabaseNamespace
from toolsandbox_testkit import make_world

from ratchet.bench.toolsandbox.cases import scenarios
from ratchet.bench.toolsandbox.gate import GATE
from ratchet.bench.toolsandbox.grading import (
    EXACT,
    PASS_SIMILARITY,
    Graded,
    Score,
    describe,
    message_label,
    reasons_for,
)
from ratchet.types import ToolCall


def _graded(
    similarity: float, milestones: tuple[Score, ...] = (), minefields: tuple[Score, ...] = ()
) -> Graded:
    return Graded(similarity, similarity, 0.0, 4, milestones, minefields)


def test_passed_is_at_the_declared_threshold_and_exact_is_one() -> None:
    assert PASS_SIMILARITY == 0.9
    assert _graded(0.9).passed and not _graded(0.8999).passed
    assert _graded(1.0).exact and _graded(1.0 - 1e-12).exact and not _graded(0.999).exact
    assert EXACT < 1.0


def test_reasons_name_each_unmet_milestone_and_reached_minefield_only() -> None:
    graded = _graded(
        0.0,
        milestones=(Score(0, 1.0, 5, "SETTING has {}", "m"), Score(1, 0.4, 7, "SANDBOX has {}", "m")),
        minefields=(Score(0, 0.0, None, "never", "m"), Score(1, 1.0, 9, "CONTACT unchanged", "m")),
    )
    assert reasons_for(graded) == [
        "milestone 2 not met: SANDBOX has {}",
        "minefield 2 reached: CONTACT unchanged",
    ]


def test_a_reason_does_not_change_with_the_match_so_diagnose_can_follow_it() -> None:
    a = _graded(0.0, milestones=(Score(0, 0.0, None, "SETTING has x", "no message"),))
    b = _graded(0.0, milestones=(Score(0, 0.5, 9, "SETTING has x", "message 9 (...)"),))
    assert reasons_for(a) == reasons_for(b)


def test_graded_serialises_every_score() -> None:
    graded = _graded(1.0, milestones=(Score(0, 1.0, 5, "d", "m"),))
    out = graded.to_json()
    assert out["similarity"] == 1.0 and out["turn_count"] == 4
    assert out["milestones"] == [
        {"index": 0, "similarity": 1.0, "snapshot_index": 5, "description": "d", "matched": "m"}
    ]
    assert out["minefields"] == []


def test_describe_shows_the_states_a_milestone_expects() -> None:
    evaluation = scenarios()["cellular_off"].evaluation
    first, second = evaluation.milestone_matcher.milestones
    assert describe(first).startswith('SETTING has {"cellular": false}')
    assert (
        'SANDBOX has {"sender": "AGENT", "recipient": "USER", "content": "Cellular service is turned off"}'
        in (describe(second))
    )
    assert "unchanged" in describe(second)


def test_describe_cuts_a_long_milestone() -> None:

    big = Milestone(
        [
            SnapshotConstraint(
                DatabaseNamespace.REMINDER,
                snapshot_similarity,
                target_dataframe=pl.DataFrame({"content": ["x" * 400]}),
            )
        ]
    )
    text = describe(big)
    assert len(text) <= 220 and text.endswith("...")


def test_describe_a_constraint_without_rows_points_at_its_reference_milestone() -> None:
    m = Milestone(
        [
            SnapshotConstraint(
                DatabaseNamespace.CONTACT, snapshot_similarity, reference_milestone_node_index=2
            )
        ],
        guardrail_database_list=[],
    )
    assert describe(m) == "CONTACT follows milestone 2"


def test_message_label_names_the_message_or_why_there_is_none() -> None:
    world = make_world("cellular_off")
    ctx = world._context
    with GATE.held(ctx):
        assert message_label(ctx, None) == "no message"
        assert message_label(ctx, 15).startswith("message 15 (user to agent: Turn off cellular")
        assert message_label(ctx, 9999) == "message 9999"


@pytest.mark.parametrize("calls", [0, 1])
def test_grade_scores_the_conversation_with_the_benchmarks_own_evaluation(calls: int) -> None:
    world = make_world("cellular_off")
    if calls:
        world.call(ToolCall("set_cellular_service_status", {"on": False}, "c1"))
    graded = world.grade()
    assert [round(s.similarity, 2) for s in graded.milestones] == ([1.0, 0.0] if calls else [0.0, 0.0])
    assert graded.similarity == graded.milestone_similarity
