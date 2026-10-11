from __future__ import annotations

from ratchet.view.admission import (
    fresh_view,
    parse_coverage,
    procedures_not_applicable,
    replay_rows,
)

from .fakes import STABLE, fresh_evidence, replay_report


def test_parse_coverage() -> None:
    assert parse_coverage("9/12 traces checked") == (9, 12)
    assert parse_coverage("nothing") == (None, None)
    assert parse_coverage(None) == (None, None)


def test_replay_rows_cover_guards_and_the_composed_runtime() -> None:
    ev = {
        "reports": [
            replay_report("g1"),
            replay_report("g2", "INCOMPLETE", "3/9 traces checked", errors=[1, 2]),
        ],
        "composed": replay_report("c", "REJECTED", "9/9 traces checked", false_fires=["a", "b"]),
        "procedures": {"p-2": "NOT APPLICABLE", "p-1": "NOT APPLICABLE"},
    }
    rows = replay_rows(ev)
    assert [r.guard_id for r in rows] == ["g1", "g2", "(all guards composed)"]
    assert rows[0].summary == "11/11 checked, CLEARED"
    assert rows[1].summary == "3/9 checked, INCOMPLETE"
    assert rows[1].errors == 2
    assert rows[2].false_fires == 2
    assert procedures_not_applicable(ev) == ["p-1", "p-2"]


def test_a_row_without_coverage_or_verdict_says_so() -> None:
    (row,) = replay_rows({"reports": [{"guard_id": "g", "passed": True}]})
    assert row.coverage == "no coverage recorded"
    assert row.verdict == "CLEARED"
    (bad,) = replay_rows({"reports": [{"guard_id": "g", "passed": False, "false_positives": 3}]})
    assert bad.verdict == "REJECTED"
    assert bad.false_fires == 3


def test_fresh_view_orders_broken_first_and_labels_every_role() -> None:
    ev = fresh_evidence(
        {
            "stable": ["fail", "fail", "pass", "pass"],
            "kept": ["pass", "pass", "pass", "pass"],
            "broke": ["pass", "fail", "pass", "fail"],
            "still": ["fail", "fail", "fail", "fail"],
            "nodata": ["fail", "fail", "pass", "pass"],
        }
    )
    ev["invalid_evidence"] = ["nodata"]
    v = fresh_view(ev)
    assert v is not None
    assert [(r.case_id, r.role) for r in v.rows] == [
        ("broke", "broken"),
        ("nodata", "invalid"),
        ("stable", "stable"),
        ("kept", "protected"),
        ("still", "unrecovered"),
    ]
    assert v.broken == ["broke"]
    assert v.protected == ["broke", "kept"]
    assert v.discordance == 1
    assert v.invalid == ["nodata"]
    assert v.stable_c1 == ["nodata", "stable"]


def test_fresh_view_pending_confirmation_and_missing_gate() -> None:
    v = fresh_view(fresh_evidence(STABLE, pending=True))
    assert v is not None
    assert v.confirm_pending
    assert all(r.outcomes[3] == "pending" for r in v.rows)
    assert fresh_view({}) is None
    assert fresh_view({"outcomes": "nope"}) is None
