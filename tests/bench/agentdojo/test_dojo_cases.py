from __future__ import annotations

from ratchet.bench.agentdojo.cases import (
    DEFAULT_ATTACK,
    DojoSuite,
    attack_target,
    build_suite,
    case_id,
    make_attack,
    split_ids,
)
from ratchet.bench.agentdojo.world import DojoTask


def test_split_is_seeded_disjoint_and_complete() -> None:
    ids = [f"user_task_{i}" for i in range(16)]
    train, held = split_ids(ids, seed=0, heldout_fraction=0.5)
    assert split_ids(ids, seed=0, heldout_fraction=0.5) == (train, held)
    assert set(train).isdisjoint(held) and set(train) | set(held) == set(ids)
    assert len(held) == 8
    assert split_ids(ids, seed=1, heldout_fraction=0.5) != (train, held)


def test_full_banking_suite_counts() -> None:
    d = build_suite("banking")
    s = d.split
    assert len(s.train_users) == len(s.heldout_users) == 8
    assert len(s.train_injections) + len(s.heldout_injections) == 9
    assert len(d.by_split("train")) == 8 * (1 + len(s.train_injections))
    assert len(d.by_split("heldout")) == 8 * (1 + len(s.heldout_injections))
    payload = d.to_payload()
    assert payload["cases"] == {"train": len(d.by_split("train")), "heldout": len(d.by_split("heldout"))}


def test_no_held_out_task_appears_in_a_training_case(banking: DojoSuite) -> None:
    s = banking.split
    for c in banking.by_split("train"):
        t = banking.tasks[c.id]
        assert t.user_task_id in s.train_users
        assert t.injection_task_id is None or t.injection_task_id in s.train_injections
    for c in banking.by_split("heldout"):
        t = banking.tasks[c.id]
        assert t.user_task_id in s.heldout_users
        assert t.injection_task_id is None or t.injection_task_id in s.heldout_injections
    assert s.users("train") == s.train_users and s.injections("heldout") == s.heldout_injections


def test_cases_carry_the_prompt_and_attack_text(banking: DojoSuite) -> None:
    c = next(c for c in banking.cases if c.kind == "trap")
    t = banking.tasks[c.id]
    goal = banking.suite.get_injection_task_by_id(t.injection_task_id).GOAL
    assert c.must_succeed and c.trap == t.injection_task_id
    assert c.customer_message == banking.suite.get_user_task_by_id(t.user_task_id).PROMPT
    assert all(goal in text and "Emma Johnson" in text for text in t.injections.values())
    assert "the AI language model" in next(iter(t.injections.values()))
    clean = next(c for c in banking.cases if c.kind == "must_succeed")
    assert banking.tasks[clean.id].injections == {}
    assert case_id(DojoTask("banking", "v1.2.2", "user_task_3")) == "agentdojo/banking/user_task_3/no_attack"


def test_other_registered_attacks_load(banking: DojoSuite) -> None:
    attack = make_attack(banking.suite, "ignore_previous")
    user = banking.suite.get_user_task_by_id("user_task_0")
    inj = banking.suite.get_injection_task_by_id("injection_task_0")
    texts = attack.attack(user, inj)
    assert texts and all(inj.GOAL in t for t in texts.values())
    assert make_attack(banking.suite, DEFAULT_ATTACK).attack(user, inj) != texts


def test_attack_target_is_an_empty_named_pipeline() -> None:
    target = attack_target()
    assert target.name == "ratchet-harness" and list(target.elements) == []
