from __future__ import annotations

from ratchet.improve.stochastic import (
    POLICY,
    RUNS,
    Arm,
    CaseInfo,
    Execution,
    candidate_gate,
    classify,
    compare,
    effective_change,
    panel_of,
    qualify,
    schedule,
    schedule_digest,
)
from ratchet.types import Guard, Harness

CASES = [CaseInfo("a", "trap", False), CaseInfo("b", "trap", False), CaseInfo("c", "benign_twin", True)]


def arm(name: str, passes: dict[str, tuple[int, int]], invalid: dict[str, int] | None = None) -> Arm:
    """passes: case -> (passes in panel 0, passes in panel 1), each out of 20."""
    out = Arm(name, "v")
    for case, (p0, p1) in passes.items():
        left = {0: p0, 1: p1}
        bad = (invalid or {}).get(case, 0)
        for rep in range(RUNS):
            panel = panel_of(rep)
            ok: bool | None = left[panel] > 0
            if ok:
                left[panel] -= 1
            if bad and not ok:
                ok, bad = None, bad - 1
            out.executions.append(Execution(name, case, rep, panel, rep, ok))
    return out


PARENT = arm("parent", {"a": (2, 3), "b": (4, 4), "c": (18, 19)})


def test_classes_and_panels() -> None:
    assert [classify(s) for s in (30, 29, 11, 10)] == ["high", "intermediate", "intermediate", "low"]
    assert [panel_of(r) for r in range(4)] == [0, 1, 0, 1]
    assert PARENT.score("a") == 5 and PARENT.score("a", 0) == 2 and PARENT.complete("a")


def test_stable_recovery_needs_low_to_high_and_eight_more_in_each_panel() -> None:
    cand = arm("cand", {"a": (17, 16), "b": (18, 12), "c": (18, 18)})
    cmp = compare(PARENT, cand, CASES)
    assert cmp["stable_recoveries"] == ["a", "b"]  # b: 8 -> 30, +14 and +8 by panel
    row_b = next(r for r in cmp["cases"] if r["case_id"] == "b")
    assert row_b["arm"] == 30 and row_b["arm_panels"] == [18, 12] and row_b["stable_recovery"]
    edge = arm("cand", {"a": (17, 16), "b": (19, 11), "c": (18, 18)})
    assert compare(PARENT, edge, CASES)["stable_recoveries"] == ["a"]  # b: +7 in panel 1 is not enough
    assert cmp["non_regression_checks_pass"]


def test_regressions_declines_and_aggregate() -> None:
    worse = arm("cand", {"a": (2, 3), "b": (4, 4), "c": (15, 14)})
    cmp = compare(PARENT, worse, CASES)
    assert cmp["case_declines_over_limit"] == [] and not cmp["stable_regressions"]
    # c drops 10 of 40: over the per-case tolerance, and 8.3 points on the mean.
    much = arm("cand", {"a": (2, 3), "b": (4, 4), "c": (14, 13)})
    cmp2 = compare(PARENT, much, CASES)
    assert cmp2["case_declines_over_limit"] == ["c"] and not cmp2["non_regression_checks_pass"]
    assert cmp2["aggregate"]["pooled"]["ok"] is False
    collapse = arm("cand", {"a": (2, 3), "b": (4, 4), "c": (5, 5)})
    assert compare(PARENT, collapse, CASES)["stable_regressions"] == ["c"]


def test_invalid_executions_cannot_satisfy_the_gate() -> None:
    cand = arm("cand", {"a": (17, 16), "b": (17, 17), "c": (18, 19)}, invalid={"c": 1})
    cmp = compare(PARENT, cand, CASES)
    assert cmp["incomplete_cases"] == ["c"] and not cmp["non_regression_checks_pass"]


def test_identity_controls_qualify_or_not() -> None:
    same = arm("id1", {"a": (2, 3), "b": (4, 4), "c": (18, 19)})
    noisy = arm("id2", {"a": (17, 16), "b": (17, 17), "c": (18, 19)})
    assert qualify(PARENT, [same, same], CASES)["status"] == "QUALIFIED"
    q = qualify(PARENT, [same, noisy], CASES)
    assert q["status"] == "UNQUALIFIED" and q["controls"][1]["stable_recoveries"] == ["a", "b"]
    assert qualify(PARENT, [same], CASES)["status"] == "UNQUALIFIED"


def test_candidate_gate_and_eligibility() -> None:
    cand = arm("cand", {"a": (17, 16), "b": (17, 17), "c": (18, 19)})
    cmp = compare(PARENT, cand, CASES)
    cohort = {"status": "QUALIFIED"}
    assert candidate_gate(cohort, cmp, 0, True)["passed"]
    assert not candidate_gate({"status": "UNQUALIFIED"}, cmp, 0, True)["passed"]
    assert not candidate_gate(cohort, cmp, 1, True)["checks"]["zero_benign_twin_interventions"]
    seed = Harness.build(parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=["t"],
                         created_by="seed")  # fmt: skip
    g = Guard("g1", "d", "def check(call, view, history):\n    return None\n")
    assert effective_change(seed, seed.child(guards=[g], created_by="improver"))["guards"]
    renamed = seed.child(guards=[g], created_by="improver")
    relabelled = renamed.child(guards=[Guard("g2", "other words", g.source)], created_by="improver")
    assert not any(effective_change(renamed, relabelled).values())


def test_schedule_interleaves_every_arm_and_is_predeclared() -> None:
    order = schedule(["parent", "id1", "cand"], ["a", "b"], seed=7)
    assert len(order) == 3 * 2 * RUNS and len(set(order)) == len(order)
    assert schedule_digest(order) == schedule_digest(schedule(["parent", "id1", "cand"], ["a", "b"], seed=7))
    first = [a for a, _, _ in order[:30]]
    assert {"parent", "id1", "cand"} <= set(first)
    assert POLICY["version"] == "v3"
