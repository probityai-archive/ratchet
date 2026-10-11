from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict

import pytest

from ratchet.cases import CASE_KINDS, all_cases, by_family, by_kind, by_split, families, get, suite_digest
from ratchet.cases.registry import frozen_digest


def test_eight_families_with_two_train_and_two_heldout_at_least() -> None:
    fams = families()
    assert len(fams) == 8 and len({f.id for f in fams}) == 8
    for fam in fams:
        cases = by_family(fam.id)
        splits = Counter(c.split for c in cases)
        assert splits["train"] >= 2 and splits["heldout"] >= 2, fam.id
        assert fam.rationale and fam.sources
        assert any(c.must_succeed for c in cases), fam.id


def test_case_ids_are_stable_and_unique() -> None:
    cases = all_cases()
    assert len(cases) == 37
    assert len({c.id for c in cases}) == len(cases)
    for case in cases:
        assert case.id == f"{case.family}.{case.variant}"
        assert get(case.id) is case
    assert len(by_split("train")) + len(by_split("heldout")) == len(cases)


def test_every_trap_family_has_a_benign_twin_in_each_split() -> None:
    for fam in families():
        cases = by_family(fam.id)
        if fam.id == "ordinary":
            assert {c.kind for c in cases} == {"must_succeed"}
            continue
        for split in ("train", "heldout"):
            kinds = {c.kind for c in cases if c.split == split}
            assert {"trap", "benign_twin"} <= kinds, (fam.id, split)
    assert sum(len(by_kind(k)) for k in CASE_KINDS) == len(all_cases())
    assert all(c.trap for c in by_kind("trap")) and not any(c.trap for c in by_kind("benign_twin"))


def test_heldout_differs_from_train_in_surface_details() -> None:
    for fam in families():
        train = by_family(fam.id)
        names = Counter(c.initial_state["tickets"]["tickets"][c.ticket_id]["subject"] for c in train)
        assert max(names.values()) == 1, fam.id
        parties = [
            tuple(
                sorted([*c.initial_state["customers"]["customers"], *c.initial_state["billing"]["accounts"]])
            )
            for c in train
        ]
        assert len(set(parties)) == len(parties), fam.id


def test_must_succeed_means_the_ticket_must_end_solved() -> None:
    for case in all_cases():
        assert case.must_succeed == all(s == "solved" for s in case.grader.ticket_status.values())


def test_cases_round_trip_as_json_and_match_the_frozen_digest() -> None:
    for case in all_cases():
        blob = json.dumps(asdict(case), sort_keys=True)
        assert json.loads(blob)["id"] == case.id
    assert suite_digest() == frozen_digest()


def test_unknown_lookups_raise() -> None:
    with pytest.raises(KeyError):
        get("nope.v1")
    with pytest.raises(KeyError):
        by_family("nope")
