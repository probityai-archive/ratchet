from __future__ import annotations

import pytest

from ratchet.bench.transfer.analysis import discordant_pair_p, paired_rows, summarize
from ratchet.improve.evaluate import CaseResult, EvalReport
from ratchet.types import Cost

CASES = [
    ("h1", "heldout", "trap", True),
    ("h2", "heldout", "trap", True),
    ("h3", "heldout", "benign_twin", True),
    ("h4", "heldout", "must_succeed", True),
    ("a1", "audit", "trap", False),
]


def report(
    version: str,
    passed: dict[str, bool],
    interventions: dict[str, int] | None = None,
    errors: set[str] | frozenset[str] = frozenset(),
    mode: str = "FRESH",
) -> EvalReport:
    iv = interventions or {}
    results = [
        CaseResult(cid, split, ms, kind, passed[cid], [], f"run-{cid}", Cost(), iv.get(cid, 0),  # type: ignore[arg-type]
                   error="boom" if cid in errors else None, mode=mode)  # type: ignore[arg-type]
        for cid, split, kind, ms in CASES
    ]  # fmt: skip
    return EvalReport(version, results, 1.0)


def outcomes(*ids: str) -> dict[str, bool]:
    return {cid: cid in ids for cid, *_ in CASES}


def test_discordant_pair_p_known_values() -> None:
    assert discordant_pair_p(0, 0) == 1.0
    assert discordant_pair_p(5, 0) == pytest.approx(2 / 32)
    assert discordant_pair_p(0, 5) == pytest.approx(2 / 32)
    assert discordant_pair_p(3, 3) == 1.0
    assert discordant_pair_p(10, 2) == pytest.approx(0.0386, abs=1e-4)
    with pytest.raises(ValueError):
        discordant_pair_p(-1, 0)


def test_summary_counts_every_case_and_repeat() -> None:
    base = [report("v0", outcomes("h4")), report("v0", outcomes("h3", "h4"))]
    cand = [
        report("v1", outcomes("h1", "h2", "h4"), {"h3": 1}),
        report("v1", outcomes("h1", "h2", "h3", "h4")),
    ]
    s = summarize(base, cand, "heldout")
    assert s.cases == 4 and s.repeats == 2
    assert s.base.passes == (1, 2)
    assert s.cand.passes == (3, 4)
    assert s.base.strict_passes == 1 and s.cand.strict_passes == 3
    assert s.base.noise_d == 1 and s.cand.noise_d == 1
    # repeat 1: h1,h2 b; repeat 2: h1,h2 b, h3 neither
    assert (s.b, s.c) == (4, 0)
    assert (s.strict_b, s.strict_c) == (2, 0)
    assert s.p_value == pytest.approx(discordant_pair_p(4, 0))
    assert s.cand.benign_twin_interventions == 1
    assert s.base.must_succeed_failures == 5 and s.cand.must_succeed_failures == 1
    assert s.by_kind["trap"]["cases"] == 2 and s.by_kind["trap"]["b"] == 4
    payload = s.to_payload()
    assert payload["base"]["passes"] == [1, 2] and payload["split"] == "heldout"


def test_regressions_errors_and_non_fresh_are_counted() -> None:
    base = [report("v0", outcomes("h1", "h4", "a1"))]
    cand = [report("v1", outcomes("h1"), errors={"h4"}, mode="CACHED")]
    s = summarize(base, cand, "heldout")
    assert (s.b, s.c) == (0, 1)
    assert s.cand.must_succeed_failures == 3
    assert s.cand.errors == 1
    assert s.cand.non_fresh == 4 and s.base.non_fresh == 0
    a = summarize(base, cand, "audit")
    assert a.cases == 1 and (a.b, a.c) == (0, 1)


def test_mismatched_inputs_are_refused() -> None:
    r = report("v0", outcomes())
    short = EvalReport("v1", r.results[:2], 1.0)
    with pytest.raises(ValueError):
        paired_rows([r], [short])
    with pytest.raises(ValueError):
        paired_rows([r, short], [r, r])
    with pytest.raises(ValueError):
        paired_rows([r], [r, r])
    with pytest.raises(ValueError):
        paired_rows([], [r])
    reordered = EvalReport("v1", list(reversed(r.results)), 1.0)
    with pytest.raises(ValueError):
        paired_rows([r], [reordered])
    with pytest.raises(ValueError):
        summarize([r], [r], "train")
