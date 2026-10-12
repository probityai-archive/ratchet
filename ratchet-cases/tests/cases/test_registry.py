from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict

import pytest

from ratchet_cases import SUITE_DIGEST, __version__, released_digest
from ratchet_cases.cases import (
    CASE_KINDS,
    SPLITS,
    all_cases,
    by_family,
    by_kind,
    by_split,
    families,
    get,
    suite_digest,
)


def test_eight_families_with_train_heldout_and_audit_cases() -> None:
    fams = families()
    assert len(fams) == 8 and len({f.id for f in fams}) == 8
    for fam in fams:
        cases = by_family(fam.id)
        splits = Counter(c.split for c in cases)
        assert splits["train"] >= 2 and splits["heldout"] >= 2 and splits["audit"] >= 1, fam.id
        assert fam.rationale
        assert any(c.must_succeed for c in cases), fam.id


def test_case_ids_are_stable_and_unique() -> None:
    cases = all_cases()
    assert len(cases) == 52
    assert len({c.id for c in cases}) == len(cases)
    for case in cases:
        assert case.id == f"{case.family}.{case.variant}"
        assert get(case.id) is case
    assert sum(len(by_split(s)) for s in SPLITS) == len(cases)


def test_every_trap_family_has_a_benign_twin_in_each_split() -> None:
    for fam in families():
        cases = by_family(fam.id)
        if fam.id == "ordinary":
            assert {c.kind for c in cases} == {"must_succeed"}
            continue
        for split in SPLITS:
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


def test_no_held_out_or_audit_case_reuses_a_training_id_name_or_amount() -> None:
    """The overfitting screen rejects literals from training traces, so later splits must not share them."""

    def literals(split: str) -> set[str]:
        found: set[str] = set()
        for case in by_split(split):  # type: ignore[arg-type]
            state = case.initial_state
            found |= set(state["customers"]["customers"]) | set(state["payments"]["charges"])
            found |= set(state["orders"]["orders"]) | set(state["billing"]["accounts"]) | {case.ticket_id}
            found |= {c["name"] for c in state["customers"]["customers"].values()}
            found |= {str(c["amount_cents"]) for c in state["payments"]["charges"].values()}
        return found

    train = literals("train")
    for split in ("heldout", "audit"):
        assert not train & literals(split), split


def test_must_succeed_means_the_ticket_must_end_solved() -> None:
    for case in all_cases():
        assert case.must_succeed == all(s == "solved" for s in case.grader.ticket_status.values())


def test_cases_round_trip_as_json_and_match_the_frozen_digest() -> None:
    for case in all_cases():
        blob = json.dumps(asdict(case), sort_keys=True)
        assert json.loads(blob)["id"] == case.id
    assert suite_digest() == SUITE_DIGEST == released_digest(__version__)


def test_unknown_lookups_raise() -> None:
    with pytest.raises(KeyError):
        get("nope.v1")
    with pytest.raises(KeyError):
        by_family("nope")
