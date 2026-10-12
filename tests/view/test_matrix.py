from __future__ import annotations

import io

import pytest
from rich.console import Console

from ratchet.improve.replay import GuardInput
from ratchet.types import Guard
from ratchet.view.data import decision_info
from ratchet.view.matrix import (
    CaseInfo,
    Matrix,
    MatrixColumn,
    MatrixView,
    classify,
    compute_matrix,
    layout,
    letters,
    pick_rows,
    show_matrix,
)

from .fakes import (
    REJECTED_RUNS,
    RULES,
    FakeRunner,
    decision_doc,
    fresh_evidence,
    info,
    make_trace,
    sample_guards,
    sample_traces,
    walks,
)

KEYS = ["g-retry", "g-all-refunds", "g-outside-mail", "g-any-mail", "g-broken"]


def build(decisions: dict[str, bool] | None = None) -> Matrix:
    guards = {g.id: g for g in sample_guards()}
    cols = []
    for i, key in enumerate(KEYS):
        d = None
        if decisions and key in decisions:
            d = decision_info(
                decision_doc(
                    "p",
                    f"cand{i}",
                    decisions[key],
                    checks=[
                        {"name": "replay", "passed": decisions[key], "evidence": {}},
                        {"name": "fresh", "passed": False, "evidence": fresh_evidence(REJECTED_RUNS)},
                    ],
                )
            )
        cols.append(MatrixColumn(guards[key], f"cand{i}", letters(5)[i], d))
    return compute_matrix("parent000001", cols, sample_traces(), info, walks(), FakeRunner(RULES))


def text(renderable: object, width: int = 120, height: int = 35) -> str:
    c = Console(width=width, height=height, record=True, file=io.StringIO(), force_terminal=False)
    c.print(renderable)
    return c.export_text()


def test_rows_are_grouped_and_ordered() -> None:
    m = build()
    kinds = [r.kind for r in m.rows]
    assert kinds == sorted(kinds, key=("trap_failure", "past_success", "benign_twin", "other_failure").index)
    assert m.rows_of("trap_failure") == [0, 1, 2, 3]
    assert [m.rows[i].case_id for i in m.rows_of("benign_twin")] == ["mail.w2", "refund.w1"]
    assert m.rows[m.rows_of("other_failure")[0]].case_id == "lone.x"


def test_classify() -> None:
    passed, failed = sample_traces()[4], sample_traces()[0]
    assert classify(failed, CaseInfo("trap", "f")) == "trap_failure"
    assert classify(failed, CaseInfo("must_succeed", "f")) == "other_failure"
    assert classify(failed, None) == "other_failure"
    assert classify(passed, CaseInfo("trap", "f")) == "past_success"
    assert classify(failed, CaseInfo("benign_twin", "f")) == "benign_twin"


def test_pick_rows_keeps_newest_trace_per_case() -> None:
    old = make_trace("old", "refund.t1", False, [("a", {})])
    new = make_trace("new", "refund.t1", False, [("a", {})])
    picked, rows = pick_rows([old, new], info)
    assert [t.run_id for t in picked] == ["new"]
    assert rows[0].family == "refund"


def test_cells_fire_where_the_guard_blocks() -> None:
    m = build()
    retry = 0
    fired = {m.rows[r].case_id for r in range(len(m.rows)) if m.fired(r, retry)}
    assert fired == {"refund.t1", "refund.t2"}
    assert m.hits(0) == (2, 2)
    assert m.hits(1) == (1, 1)
    assert m.count("past_success", 1) == 2
    assert m.count("benign_twin", 1) == 1
    assert m.model_calls == 0
    assert m.checks > 0
    cell = m.cells[(m.rows_of("trap_failure")[2], 0)]
    assert cell.fired
    assert cell.seq == 5


def test_targets_fall_back_to_every_trap_when_origin_is_unknown() -> None:
    m = build()
    assert len(m.targets[0]) == 2
    assert len(m.targets[4]) == 4


def test_errors_are_cells_not_crashes() -> None:
    m = build()
    assert m.errors(4) == len(m.rows)
    assert "guard error" in " ".join(m.replay_reasons(4))
    assert m.cells[(0, 4)].error is not None


def test_error_never_hides_an_earlier_fire() -> None:
    def flaky(i: GuardInput) -> bool:
        if i.call.name == "refund" and i.call.args.get("retry"):
            return True
        if i.call.name == "get_charge":
            return False
        raise ValueError("x")

    guard = Guard("g-flaky", "d", "s", [])
    col = MatrixColumn(guard, "c", "A")
    m = compute_matrix("p", [col], sample_traces(), info, walks(), FakeRunner({"g-flaky": flaky}))
    t2 = next(i for i, r in enumerate(m.rows) if r.case_id == "refund.t2")
    assert m.cells[(t2, 0)].fired


def test_verdicts_without_a_decision() -> None:
    m = build()
    assert m.verdict(0).label == "CLEARED"
    assert m.verdict(0).ok
    v = m.verdict(1)
    assert v.label == "REJECTED"
    assert v.reasons == ["blocks 2 past successes", "fires on 1 benign twin"]
    assert m.verdict(3).reasons == ["blocks 1 past success", "fires on 1 benign twin"]


def test_verdict_with_a_decision_reports_the_decision() -> None:
    m = build({"g-retry": True, "g-all-refunds": False})
    assert m.verdict(0).label == "ADMITTED"
    assert m.verdict(1).label == "REJECTED"
    assert any(r.startswith("protected case broke") for r in m.verdict(1).reasons)
    assert m.verdict(1).replay == "11/11 checked, REJECTED"
    assert "blocks 2 past successes" in m.verdict(1).reasons
    # a decision that rejects a guard replay found clean says why, from the decision
    clean_but_rejected = build({"g-retry": False})
    assert clean_but_rejected.verdict(0).label == "REJECTED"
    assert any(r.startswith("protected case broke") for r in clean_but_rejected.verdict(0).reasons)
    assert clean_but_rejected.verdict(0).replay == "11/11 checked, CLEARED"


def test_no_trap_failures_means_nothing_proved() -> None:
    traces = [t for t in sample_traces() if t.outcome.passed]
    col = MatrixColumn(sample_guards()[0], "c", "A")
    m = compute_matrix("p", [col], traces, info, walks(), FakeRunner(RULES))
    assert m.replay_reasons(0) == ["no trap failure recorded to prove it on"]


def test_runner_returning_wrong_count_is_an_error() -> None:
    class Short:
        def check_batch(self, guard: Guard, inputs: object) -> list[object]:
            return []

    col = MatrixColumn(sample_guards()[0], "c", "A")
    with pytest.raises(RuntimeError, match="verdicts"):
        compute_matrix("p", [col], sample_traces(), info, walks(), Short())  # type: ignore[arg-type]


def test_letters() -> None:
    assert letters(3) == ["A", "B", "C"]
    assert letters(28)[26] == "27"


def test_full_render_contains_the_story() -> None:
    out = text(MatrixView(build({"g-retry": True, "g-all-refunds": False}), full=True))
    assert "replay matrix" in out
    for group in ("TRAP FAILURES (4)", "PAST SUCCESSES (4)", "BENIGN TWINS (2)", "OTHER FAILURES (1)"):
        assert group in out
    assert "refund.t1" in out
    assert "● BLOCK @3" in out
    assert "✗ BLOCK @3" in out
    assert "○ missed" not in out
    assert "· pass" in out
    assert "! error" in out
    assert "ADMITTED" in out
    assert "REJECTED" in out
    assert "11 traces," in out
    assert "0 model calls" in out
    assert "g-all-refunds" in out
    assert "blocks 2 past successes; fires on 1 benign twin" in out


def test_missed_target_is_shown() -> None:
    guard = Guard("g-retry", "d", "s", ["r-t1@4"])
    cols = [MatrixColumn(guard, "c", "A")]
    miss = FakeRunner({"g-retry": lambda i: False})
    m = compute_matrix("p", cols, sample_traces(), info, walks(), miss)
    out = text(MatrixView(m, full=True))
    assert "○ missed" in out
    assert "misses 1 of 1 target failures" in out


def test_focus_layout_fits_a_small_terminal_and_keeps_offenders() -> None:
    m = build()
    full = layout(m, budget=100)
    assert sum(1 for ln in full if ln.kind == "row") == len(m.rows)
    small = layout(m, budget=14)
    shown = {m.rows[ln.row].case_id for ln in small if ln.kind == "row"}
    assert {"ordinary.o1", "refund.t5"} <= shown  # past successes that guard B blocks outrank clean rows
    assert any(ln.kind == "more" for ln in small)
    assert len(small) <= 14
    assert len(layout(m, budget=9, full=True)) == len(full)


def test_render_in_a_35_row_terminal_fits() -> None:
    out = text(MatrixView(build(), height=35, width=120))
    assert len(out.rstrip().splitlines()) <= 35
    narrow = text(MatrixView(build(), height=24, width=120))
    assert "more rows (--full shows all)" in narrow


def test_show_matrix_prints_once_without_a_terminal() -> None:
    buf = io.StringIO()
    c = Console(file=buf, width=120, height=35, force_terminal=False)
    show_matrix(c, build())
    assert buf.getvalue().count("replay matrix") == 1


def test_show_matrix_animates_on_a_terminal() -> None:
    buf = io.StringIO()
    c = Console(file=buf, width=120, height=35, force_terminal=True, color_system="truecolor")
    show_matrix(c, build(), delay=0.001)
    out = buf.getvalue()
    assert "0 model calls" in out
    assert "\x1b[" in out


def test_partial_reveal_shows_no_footer() -> None:
    view = MatrixView(build(), full=True)
    view.reveal = 3
    out = text(view)
    assert "0 model calls" not in out
    assert "VERDICT" not in out
    assert "TRAP FAILURES" in out


def test_empty_matrix_prints() -> None:
    m = compute_matrix(
        "p", [MatrixColumn(sample_guards()[0], "c", "A")], [], info, walks(), FakeRunner(RULES)
    )
    buf = io.StringIO()
    show_matrix(Console(file=buf, force_terminal=True, width=100, height=30), m, delay=0.01)
    assert "0 traces" in buf.getvalue()


def test_a_guard_that_fires_after_the_failing_step_is_late_not_a_hit() -> None:
    guard = Guard(
        "g-retry", "d", "s", ["r-t1@3"]
    )  # t1 failed at seq 3 (the first refund); the guard waits for 5
    m = compute_matrix(
        "p", [MatrixColumn(guard, "c", "A")], sample_traces(), info, walks(), FakeRunner(RULES)
    )
    t1 = next(i for i, r in enumerate(m.rows) if r.case_id == "refund.t1")
    assert m.cells[(t1, 0)].late
    assert not m.fired(t1, 0)
    assert m.hits(0) == (0, 1)
    assert m.replay_reasons(0)[0] == "misses 1 of 1 target failures"
    assert "only after the failing step" in m.replay_reasons(0)[1]
    assert "◐ late @5" in text(MatrixView(m, full=True))


def test_audit_cohort_traces_are_left_out_of_the_matrix() -> None:
    def with_audit(case_id: str) -> CaseInfo | None:
        got = info(case_id)
        return CaseInfo(got.kind, got.family, "audit") if got and case_id == "refund.t2" else got

    picked, rows = pick_rows(sample_traces(), with_audit)
    assert "refund.t2" not in {r.case_id for r in rows}
    assert len(picked) == len(rows) == 10


def test_replay_coverage_is_n_of_n_when_complete_and_incomplete_when_a_guard_errors() -> None:
    m = build()
    assert m.coverage(0) == (11, 11)
    assert m.verdict(0).replay == "11/11 checked, CLEARED"
    assert m.coverage(4) == (0, 11)
    assert m.verdict(4).label == "INCOMPLETE"
    assert m.verdict(4).replay == "0/11 checked, INCOMPLETE"
    out = text(MatrixView(m, full=True))
    assert "INCOMPLETE" in out
    assert "replay 11/11 checked, CLEARED" in out
    assert "replay 0/11 checked, INCOMPLETE" in out


def test_a_trace_whose_views_were_not_recorded_cannot_be_checked() -> None:
    from ratchet_cases.types import Event, Trace

    t = sample_traces()[0]
    bare = [Event(e.seq, e.kind, {k: v for k, v in e.payload.items() if k != "view"}, e.ts) for e in t.events]
    traces = [Trace(t.run_id, t.case_id, t.harness_version, t.model, bare, t.outcome), *sample_traces()[1:]]
    col = MatrixColumn(sample_guards()[0], "c", "A")
    m = compute_matrix("p", [col], traces, info, walks(), FakeRunner(RULES))
    assert m.coverage(0) == (10, 11)
    assert m.verdict(0).label == "INCOMPLETE"
    assert "rebuilt world differs" in " ".join(m.replay_reasons(0))


def test_matrix_header_says_the_traces_are_recorded() -> None:
    out = text(MatrixView(build(), full=True))
    assert "RECORDED 1970-01-01T" in out
