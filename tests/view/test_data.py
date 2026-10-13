from __future__ import annotations

from typing import Any

import pytest

from ratchet.view.admission import check_reasons, replay_rows
from ratchet.view.data import (
    DecisionInfo,
    build_lineage,
    decision_info,
    find_decision_doc,
    receipt_id,
    resolve_candidates,
    stats_from_eval,
)

from .fakes import FakeStore, decision_doc, eval_doc, scenario

MISSING = {"guard_id": "g", "verdict": "MISSING"}
MISSING_EV = {"reports": [MISSING]}
HITS = [{"literal": "ch_1"}, {"literal": "3200"}]


def rep(**kw: Any) -> dict[str, Any]:
    return {"guard_id": "g", "verdict": "REJECTED", "coverage": "12/12 traces checked", "targets_hit": 1,
            "targets_total": 1, "false_fires": [], "benign_twin_fires": 0, "errors": 0, **kw}  # fmt: skip


def fresh(**kw: Any) -> dict[str, Any]:
    base = {"same_manifest": True, "invalid_evidence": [], "stable_recoveries": ["a", "b"]}
    base |= {"stable_recoveries_c1": ["a", "b"], "protected_broken": [], "confirm_pending": False}
    return {**base, **kw}


@pytest.mark.parametrize(
    ("check", "expected"),
    [
        ({"name": "replay", "passed": True}, []),
        (
            {"name": "replay", "passed": False, "evidence": {"reports": [
                rep(false_fires=["a", "b", "c"], benign_twin_fires=1, targets_hit=1, targets_total=3)]}},
            [
                "fires 2 times on past successes",
                "fires 1 time on benign twins",
                "misses 2 of 3 target failures",
            ],
        ),
        (
            {
                "name": "replay",
                "passed": False,
                "evidence": {"reports": [rep(false_fires=["a", "b"], benign_twin_fires=1)]},
            },
            ["fires 1 time on past successes", "fires 1 time on benign twins"],
        ),
        (
            {"name": "replay", "passed": False, "evidence": {"reports": [
                rep(verdict="INCOMPLETE", coverage="9/12 traces checked", errors=2)]}},
            ["replay INCOMPLETE (9/12 checked, 2 errors)"],
        ),
        (
            {"name": "replay", "passed": False, "evidence": {"reports": [rep()], "composed": {
                "verdict": "REJECTED", "passed": False, "coverage": "12/12 traces checked"}}},
            ["composed guards: 12/12 checked, REJECTED"],
        ),
        ({"name": "replay", "passed": False, "evidence": MISSING_EV}, ["no replay recorded for g"]),
        ({"name": "replay", "passed": False, "evidence": {}}, ["replay failed"]),
        ({"name": "fresh", "passed": False, "evidence": fresh(same_manifest=False)},
         ["the four runs did not use the same case manifest"]),
        ({"name": "fresh", "passed": False, "evidence": fresh(invalid_evidence=["a", "b"])},
         ["2 cases lack valid FRESH evidence"]),
        ({"name": "fresh", "passed": False, "evidence": fresh(invalid_evidence=["a"])},
         ["1 case lacks valid FRESH evidence"]),
        ({"name": "fresh", "passed": False, "evidence": fresh(protected_broken=["p1", "p2"])},
         ["protected case broke: p1, p2"]),
        ({"name": "fresh", "passed": False, "evidence": fresh(confirm_pending=True)},
         ["awaiting the confirming run C2"]),
        ({"name": "fresh", "passed": False, "evidence": fresh(
            confirm_pending=True, stable_recoveries_c1=["a"], stable_recoveries=[])},
         ["stable recoveries so far 1, needs 2; awaiting the confirming run C2"]),
        ({"name": "fresh", "passed": False, "evidence": fresh(stable_recoveries=["a"])},
         ["1 stable recoveries over C1 and C2, needs 2"]),
        ({"name": "fresh", "passed": False, "evidence": fresh()}, ["paired recovery not shown"]),
        ({"name": "no_regression", "passed": False, "evidence": {"regressed": ["a"]}},
         ["regresses 1 must-succeed case"]),
        ({"name": "no_regression", "passed": False, "evidence": {"regressed": ["a", "b"]}},
         ["regresses 2 must-succeed cases"]),
        ({"name": "benign_twins", "passed": False, "evidence": {"intervened": ["a", "b"]}},
         ["intervenes on 2 benign twins"]),
        ({"name": "heldout", "passed": False, "evidence": fresh(
            confirm_pending=True, initial_ok=False, protected_broken=["p"], stable_recoveries_c1=[],
            stable_recoveries=[])},
         ["protected case broke: p", "0 stable recoveries on C1, needs 2", "refused after C1; C2 not run"]),
        ({"name": "overfit_lint", "passed": False, "evidence": {"hits": HITS}},
         ["copies 2 literal(s) from training traces (ch_1, 3200)"]),
        ({"name": "structure", "passed": False, "evidence": {
            "violations": ["tools_granted", "prompt_prefix_lost", "duplicate_ids", "weird"],
            "tools_granted": ["x"], "duplicate_ids": ["g1"]}},
         ["grants tools: x", "rewrites the existing prompt", "duplicate ids: g1", "weird"]),
        ({"name": "structure", "passed": False, "evidence": {}}, ["structure rule broken"]),
        ({"name": "novel", "passed": False}, ["novel failed"]),
    ],
)  # fmt: skip
def test_check_reasons(check: dict[str, Any], expected: list[str]) -> None:
    assert check_reasons(check) == expected


def test_stats_count_false_interventions_outside_traps() -> None:
    doc = eval_doc(
        "v",
        (1, 2),
        (3, 4),
        [
            {"interventions": 2, "kind": "trap"},
            {"interventions": 1, "kind": "benign_twin"},
            {"interventions": 3, "kind": "must_succeed"},
        ],
    )
    s = stats_from_eval(doc)
    assert (s.train_passed, s.train_total, s.heldout_passed, s.heldout_total) == (1, 2, 3, 4)
    assert (s.interventions, s.false_interventions, s.model_calls) == (6, 4, 12)


def test_receipt_id() -> None:
    assert receipt_id(None) is None
    assert receipt_id("") is None
    assert receipt_id("not base64!!") is None
    rid = receipt_id("c2lnbmVk")
    assert rid is not None
    assert rid.startswith("rcpt:")
    assert len(rid) == 15


def test_decision_info_properties() -> None:
    d = decision_info(
        decision_doc("p", "c", False, kind="guard", change={"description": "block x"}, generation=3)
    )
    assert (d.kind, d.generation, d.what, d.failed) == ("guard", 3, "block x", ["replay"])
    assert decision_info(decision_doc("p", "c", True, kind="procedure", change={"name": "n"})).what == "n"
    assert decision_info(decision_doc("p", "c", True, kind="prompt_edit")).what == "prompt edit"
    comp = decision_info(
        decision_doc("p", "c", True, kind="", extra_context={"composite_of": ["a", "b"], "kind": None})
    )
    assert comp.kind == "composite"
    assert comp.what == "composite of 2 changes"
    assert decision_info(decision_doc("p", "c", True, kind="", extra_context={"kind": None})).what == "change"
    assert d.reasons(skip=("replay",)) == []


def test_lineage_merges_stored_versions_with_decisions() -> None:
    store, seed, good = scenario()
    (root,) = build_lineage(store)
    assert root.version == seed.version
    assert root.status == "seed"
    assert root.stats is not None
    assert root.stats.heldout_passed == 3
    by = {n.version: n for n in root.walk()}
    assert by[good.version].status == "promoted"
    assert by["rej00000001"].status == "rejected"
    assert by["rej00000001"].stored is False
    assert [c.version for c in root.children] == [good.version, "rej00000001", "rej00000002"]
    decision = by[good.version].decision
    assert decision is not None
    assert decision.receipt is not None


def test_lineage_admitted_but_unstored_and_duplicate_decisions() -> None:
    store, seed, _ = scenario()
    store.decision_docs.append(decision_doc(seed.version, "adm00000001", True, generation=2))
    store.decision_docs.append(decision_doc(seed.version, "adm00000001", False, generation=2))
    store.decision_docs.append({**decision_doc("ghost", "orphan0001", False), "type": "decision"})
    store.decision_docs.append({"_id": "x", "type": "human_approval"})
    roots = build_lineage(store)
    nodes = {n.version: n for r in roots for n in r.walk()}
    assert nodes["adm00000001"].status == "admitted"
    assert [r.version for r in roots] == [seed.version, "orphan0001"]


def test_lineage_empty() -> None:
    assert build_lineage(FakeStore()) == []


def test_resolve_candidates_default_is_latest_generation() -> None:
    store, _, good = scenario()
    refs = resolve_candidates(store, [])
    assert [r.candidate for r in refs] == [good.version, "rej00000001", "rej00000002"]
    assert [g.id for g in refs[0].guards] == ["g-retry"]
    assert refs[2].guards == []
    assert "no guard to replay" in refs[2].note


def test_resolve_candidates_by_version_id_prefix_and_bare_harness() -> None:
    store, seed, good = scenario()
    assert resolve_candidates(store, ["rej00000001"])[0].guards[0].id == "g-all-refunds"
    assert resolve_candidates(store, ["c-g-all-refunds"])[0].candidate == "rej00000001"
    assert resolve_candidates(store, [good.version[:6]])[0].candidate == good.version
    store.decision_docs.clear()
    bare = resolve_candidates(store, [good.version])[0]
    assert bare.decision is None
    assert [g.id for g in bare.guards] == ["g-retry"]
    assert bare.parent == seed.version
    unknown = resolve_candidates(store, ["nope"])[0]
    assert unknown.guards == []
    assert "unknown candidate" in unknown.note
    assert resolve_candidates(store, [seed.version])[0].note.startswith("unknown candidate")


def test_resolve_composite_collects_each_pick_guard() -> None:
    store, seed, _ = scenario()
    store.decision_docs.append(
        decision_doc(
            seed.version,
            "combo0000001",
            True,
            kind="",
            generation=2,
            extra_context={"kind": None, "composite_of": ["c-g-all-refunds", "c-g-all-refunds", "c-p-1"]},
        )
    )
    ref = resolve_candidates(store, ["combo0000001"])[0]
    assert [g.id for g in ref.guards] == ["g-all-refunds"]


def test_resolve_candidates_without_decisions_is_empty() -> None:
    assert resolve_candidates(FakeStore(), []) == []


def test_find_decision_doc() -> None:
    store, seed, good = scenario()
    assert find_decision_doc(store, f"{seed.version}->{good.version}") is not None
    assert find_decision_doc(store, good.version) is not None
    assert find_decision_doc(store, good.version[:5]) is not None
    assert find_decision_doc(store, "zzz") is None
    assert find_decision_doc(store, "") is None
    assert isinstance(decision_info(store.decision_docs[0]), DecisionInfo)


def _signed_replay(hit: int, total: int, ids: list[str]) -> dict[str, Any]:
    return {
        "name": "replay",
        "passed": False,
        "evidence": {
            "reports": [
                {
                    "guard_id": "g",
                    "passed": False,
                    "verdict": "REJECTED",
                    "coverage": "6/6 traces checked",
                    "targets_hit": hit,
                    "targets_total": total,
                    "target_failures": ids,
                    "false_fires": [],
                    "errors": 0,
                }
            ]
        },
    }


def test_a_step_named_by_several_reasons_is_one_target_in_show_decision() -> None:
    ids = ["r1@13", "r1@13", "r1@13", "r2@10", "r2@10", "r3@4", "r3@4", "r3@4"]
    check = _signed_replay(0, 8, ids)
    assert check_reasons(check) == ["misses 3 of 3 target failures"]
    rows = replay_rows(check["evidence"])
    assert (rows[0].targets_hit, rows[0].targets_total) == (0, 3)
    assert replay_rows(_signed_replay(8, 8, ids)["evidence"])[0].targets_total == 3
    assert replay_rows(_signed_replay(8, 8, ids)["evidence"])[0].targets_hit == 3


def test_distinct_ids_and_mixed_rows_keep_the_signed_counts() -> None:
    plain = replay_rows(_signed_replay(1, 3, ["a@1", "b@2", "c@3"])["evidence"])[0]
    assert (plain.targets_hit, plain.targets_total) == (1, 3)
    repeated = ["a@1", "a@1", "a@1", "b@2", "b@2", "c@3", "c@3", "c@3"]
    mixed = replay_rows(_signed_replay(5, 8, repeated)["evidence"])[0]
    assert (mixed.targets_hit, mixed.targets_total) == (5, 8)
