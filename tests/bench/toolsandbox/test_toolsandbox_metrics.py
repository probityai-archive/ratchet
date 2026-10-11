from __future__ import annotations

import pytest

from ratchet.bench.toolsandbox.metrics import (
    ALL,
    LOST,
    NOT_RUN,
    ORDER,
    RAN,
    Row,
    counted_categories,
    group,
    summarize,
    summarize_category,
)


def _row(name: str, cats: tuple[str, ...], sim: float | None = None, status: str = RAN, **kw: float) -> Row:
    scores = (
        {}
        if sim is None
        else {
            "similarity": sim,
            "milestone_similarity": kw.get("m", sim),
            "minefield_similarity": kw.get("f", 0.0),
            "turn_count": 4,
        }
    )
    return Row(name, name, cats, "heldout", status, **scores)  # type: ignore[arg-type]


def test_a_row_reads_its_scores() -> None:
    assert _row("a", (), 0.95).passed and not _row("a", (), 0.95).exact
    assert _row("a", (), 1.0).exact
    assert not _row("a", (), 0.5).passed
    lost = Row("x", "x", (), "heldout", LOST)
    assert not lost.ran and not lost.passed and not lost.exact


def test_a_scenario_counts_in_every_category_it_carries_and_in_all() -> None:
    row = _row("a", ("STATE_DEPENDENCY", "SINGLE_TOOL_CALL"), 1.0)
    assert counted_categories(row) == ["STATE_DEPENDENCY", "SINGLE_TOOL_CALL", ALL]


def test_the_three_distractor_scrambled_forms_do_not_count_under_three_distractors() -> None:
    plain = _row("a", ("THREE_DISTRACTION_TOOLS", "CANONICALIZATION"), 1.0)
    scrambled = _row("b", ("THREE_DISTRACTION_TOOLS", "TOOL_NAME_SCRAMBLED"), 1.0)
    assert "THREE_DISTRACTION_TOOLS" in counted_categories(plain)
    assert counted_categories(scrambled) == ["TOOL_NAME_SCRAMBLED", ALL]
    grouped = group([plain, scrambled])
    assert [r.case_id for r in grouped["THREE_DISTRACTION_TOOLS"]] == ["a"]
    assert [r.case_id for r in grouped[ALL]] == ["a", "b"]


def test_means_are_over_those_that_ran_and_over_all_with_the_rest_as_zero() -> None:
    rows = [
        _row("a", ("C",), 1.0, m=1.0),
        _row("b", ("C",), 0.5, m=0.6, f=0.0),
        Row("c", "c", ("C",), "heldout", NOT_RUN, detail="PermissionError: no key"),
        Row("d", "d", ("C",), "heldout", LOST, detail="endpoint"),
    ]
    s = summarize_category(rows)
    assert (s["n"], s["ran"], s["not_run"], s["lost"]) == (4, 2, 1, 1)
    assert s["similarity_ran"] == pytest.approx(0.75)
    assert s["similarity_all"] == pytest.approx(0.375)
    assert s["milestone_ran"] == pytest.approx(0.8) and s["milestone_all"] == pytest.approx(0.4)
    assert s["minefield_ran"] == 0.0 and s["turns_ran"] == 4.0
    assert (s["passed"], s["exact"]) == (1, 1)


def test_a_category_with_nothing_that_ran_has_no_mean() -> None:
    s = summarize_category([Row("c", "c", ("C",), "heldout", NOT_RUN)])
    assert s["similarity_ran"] is None and s["similarity_all"] == 0.0 and s["n"] == 1
    empty = summarize_category([])
    assert empty["n"] == 0 and empty["similarity_all"] is None


def test_summarize_lists_the_report_order_then_any_other_category() -> None:
    rows = [_row("a", ("ZZ_OTHER", "CANONICALIZATION", "STATE_DEPENDENCY"), 1.0)]
    keys = list(summarize(rows))
    assert keys == [ALL, "STATE_DEPENDENCY", "CANONICALIZATION", "ZZ_OTHER"]
    assert ORDER[0] == ALL
