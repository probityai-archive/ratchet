"""The benchmark's own grade of a conversation, and the reasons an improver can read beside it.

The grade is `tool_sandbox`'s: `Evaluation.evaluate` over the execution context, which maps the scenario's
milestones (states the conversation must pass through) and minefields (states it must never reach) onto the
conversation's database snapshots and returns a similarity in [0, 1]. Nothing here changes a score. This
module adds three things around it:

- `Graded`, the same numbers as plain data with each milestone's and minefield's own similarity, so a record
  keeps what the benchmark computed and a report can total it any way the benchmark does;
- reasons in words, one per milestone not fully met and per minefield reached, naming the state expected (or
  forbidden), so `diagnose` can group failures and locate the step that caused them;
- `PASS_SIMILARITY`, Ratchet's binary reading of the continuous score (see the constant).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tool_sandbox.common.evaluation import (
    Evaluation,
    EvaluationResult,
    guardrail_similarity,
)
from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
)

# Ratchet's worker, replay and admission gates need pass or fail; the benchmark scores a scenario between 0
# and 1. A scenario counts as passed at 0.9 or more, which tolerates the wording gap between a reply and the
# reference text (a milestone on free text is scored by ROUGE-L, so a correct reply phrased differently scores
# below 1) while a missed state change, which scores 0 or 0.5, fails. The value was fixed before any run. It
# never replaces the benchmark's score: every report leads with the similarity and also counts scenarios at
# exactly 1.
PASS_SIMILARITY = 0.9
EXACT = 1.0 - 1e-9
_MAX_TEXT = 220


@dataclass(frozen=True)
class Score:
    """One milestone or minefield: its index in the scenario, the snapshot it matched, its similarity."""

    index: int
    similarity: float
    snapshot_index: int | None
    description: str
    matched: str

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "similarity": self.similarity,
            "snapshot_index": self.snapshot_index,
            "description": self.description,
            "matched": self.matched,
        }


@dataclass(frozen=True)
class Graded:
    """`EvaluationResult` as plain data, with a score per milestone and per minefield."""

    similarity: float
    milestone_similarity: float
    minefield_similarity: float
    turn_count: int
    milestones: tuple[Score, ...]
    minefields: tuple[Score, ...]

    @property
    def passed(self) -> bool:
        return self.similarity >= PASS_SIMILARITY

    @property
    def exact(self) -> bool:
        return self.similarity >= EXACT

    def to_json(self) -> dict[str, Any]:
        return {
            "similarity": self.similarity,
            "milestone_similarity": self.milestone_similarity,
            "minefield_similarity": self.minefield_similarity,
            "turn_count": self.turn_count,
            "milestones": [s.to_json() for s in self.milestones],
            "minefields": [s.to_json() for s in self.minefields],
        }


def _short(text: str) -> str:
    return text if len(text) <= _MAX_TEXT else text[: _MAX_TEXT - 3] + "..."


def _cell(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def describe(milestone: Any) -> str:
    """What a milestone or minefield asks for: per database, the rows it expects (or that nothing changes)."""
    parts: list[str] = []
    for constraint in milestone.snapshot_constraints:
        name = str(constraint.database_namespace)
        if constraint.snapshot_constraint is guardrail_similarity:
            parts.append(f"{name} unchanged")
        elif constraint.target_dataframe is not None:
            rows = ", ".join(_cell(r) for r in constraint.target_dataframe.to_dicts())
            parts.append(f"{name} has {rows}")
        else:
            parts.append(f"{name} follows milestone {constraint.reference_milestone_node_index}")
    return _short("; ".join(parts))


def _scores(matcher: Any, mapping: Any, context: ExecutionContext) -> tuple[Score, ...]:
    out: list[Score] = []
    for i, milestone in enumerate(matcher.milestones):
        snapshot, similarity = mapping.get(i, (None, 0.0))
        out.append(
            Score(i, float(similarity), snapshot, describe(milestone), message_label(context, snapshot))
        )
    return tuple(out)


def grade(evaluation: Evaluation, context: ExecutionContext, max_messages: int) -> Graded:
    """The benchmark's evaluation of the conversation in `context` (the global context must be `context`)."""
    result: EvaluationResult = evaluation.evaluate(execution_context=context, max_turn_count=max_messages)
    return Graded(
        similarity=float(result.similarity),
        milestone_similarity=float(result.milestone_similarity),
        minefield_similarity=float(result.minefield_similarity),
        turn_count=int(result.turn_count),
        milestones=_scores(evaluation.milestone_matcher, result.milestone_mapping, context),
        minefields=_scores(evaluation.minefield_matcher, result.minefield_mapping, context),
    )


def message_label(context: ExecutionContext, snapshot_index: int | None) -> str:
    """`sender to recipient: text` for the conversation message at `snapshot_index`, or why there is none."""
    if snapshot_index is None:
        return "no message"
    sandbox = context.get_database(
        DatabaseNamespace.SANDBOX, get_all_history_snapshots=True, drop_sandbox_message_index=False
    )
    rows = sandbox.filter(sandbox["sandbox_message_index"] == snapshot_index).to_dicts()
    if not rows:
        return f"message {snapshot_index}"
    row = rows[0]
    text = _short(str(row["content"]).replace("\n", " "))
    return (
        f"message {snapshot_index} ({str(row['sender']).lower()} to {str(row['recipient']).lower()}: {text})"
    )


def reasons_for(graded: Graded) -> list[str]:
    """One sentence per milestone not fully met and per minefield reached, from the first to the last.

    The sentence names only the state expected or forbidden, never the similarity or the message it matched,
    so it reads the same at every step of a conversation that has not changed it; `diagnose` finds the step
    that caused a failure by watching where a reason starts to hold. The match detail is in `Graded`.
    """
    reasons = [
        f"milestone {s.index + 1} not met: {s.description}" for s in graded.milestones if s.similarity < EXACT
    ]
    reasons += [
        f"minefield {s.index + 1} reached: {s.description}" for s in graded.minefields if s.similarity > 0.0
    ]
    return reasons
