from __future__ import annotations

from dataclasses import replace

import pytest

from ratchet.improve.admit import admit, check_authority
from ratchet.improve.diagnose import diagnose
from ratchet.improve.evaluate import evaluate
from ratchet.improve.receipt import HUMAN_APPROVAL_TYPE, Signer, receipt, verify
from ratchet.improve.replay import WalkCache, replay
from ratchet.types import Guard, Harness, canonical_json

from .fakes import BLOCK_ALL_REFUNDS, NO_DUP_GUARD, FakeRunCase, FakeSimulator, LocalRunner, fake_cases

OLD = Guard("g-old", "kept", "def check(call, view, history):\n    return None\n")
PARENT = Harness.build(
    parent=None, system_prompt="p", procedures=[], guards=[OLD], tool_allowlist=["get_charge", "refund"],
    created_by="seed",
)  # fmt: skip


def _replays(cand: Harness, parent_eval):  # type: ignore[no-untyped-def]
    train = parent_eval.train_traces()
    cluster = diagnose(train, FakeSimulator())[0]
    passing = [t for t in parent_eval.traces if t.outcome.passed]
    walks = WalkCache(FakeSimulator())
    new = [g for g in cand.guards if g.id != OLD.id]
    return {g.id: replay(g, cluster.members, cluster.traces, passing, walks, LocalRunner()) for g in new}


def test_a_precise_guard_is_admitted_with_evidence_and_a_verifiable_receipt() -> None:
    cand = PARENT.child(guards=[OLD, Guard("g-new", "d", NO_DUP_GUARD)], created_by="improver")
    pe = evaluate(PARENT.child(guards=[], created_by="seed"), fake_cases(), FakeRunCase(), None)
    pe = replace(pe, harness_version=PARENT.version)
    ce = evaluate(cand, fake_cases(), FakeRunCase(), None)
    # The fake agent handles timeouts whenever any guard exists; give the parent's eval the failing runs.
    d = admit(
        PARENT, cand, parent_eval=pe, cand_eval=ce, replays=_replays(cand, pe), context={"generation": 1}
    )
    assert d.admitted, d.to_payload()
    assert d.check("heldout").evidence["gained"] == ["timeout-4", "timeout-5"]
    assert d.check("replay").evidence["reports"][0]["targets_hit"] == 2
    assert d.check("cost").evidence["gate"] is False
    signer = Signer.generate()
    body = verify(receipt(d, signer), {signer.keyid: signer.public_key})
    assert body["admitted"] is True and body["context"] == {"generation": 1}


def test_overbroad_guard_fails_replay_and_regression() -> None:
    cand = PARENT.child(guards=[OLD, Guard("g-new", "d", BLOCK_ALL_REFUNDS)], created_by="improver")
    seed = PARENT.child(guards=[], created_by="seed")
    pe = replace(evaluate(seed, fake_cases(), FakeRunCase(), None), harness_version=PARENT.version)
    ce = evaluate(cand, fake_cases(), FakeRunCase(block_plain=True), None)
    d = admit(PARENT, cand, parent_eval=pe, cand_eval=ce, replays=_replays(cand, pe))
    assert not d.admitted
    assert not d.check("replay").passed and d.check("replay").evidence["reports"][0]["false_positives"]
    assert d.check("no_regression").evidence["regressed"] == ["plain-3", "plain-6"]


def test_missing_replay_and_flat_heldout_reject() -> None:
    cand = PARENT.child(guards=[OLD, Guard("g-new", "d", NO_DUP_GUARD)], created_by="improver")
    ce = evaluate(cand, fake_cases(), FakeRunCase(), None)
    d = admit(PARENT, cand, parent_eval=ce, cand_eval=ce, replays={})
    assert not d.admitted
    assert d.check("replay").evidence["reports"][0]["missing"]
    assert not d.check("heldout").passed and d.check("heldout").evidence["same_case_set"]


def test_different_case_sets_cannot_be_compared() -> None:
    cand = PARENT.child(system_prompt="q", created_by="improver")
    pe = evaluate(PARENT, fake_cases()[:3], FakeRunCase(), None)
    ce = evaluate(cand, fake_cases(), FakeRunCase(), None)
    d = admit(PARENT, cand, parent_eval=pe, cand_eval=ce, replays={})
    assert not d.check("heldout").evidence["same_case_set"]


def test_candidate_must_be_a_child() -> None:
    other = Harness.build(
        parent=None, system_prompt="x", procedures=[], guards=[], tool_allowlist=[], created_by="seed"
    )
    ce = evaluate(other, fake_cases(), FakeRunCase(), None)
    with pytest.raises(ValueError, match="not a child"):
        admit(PARENT, other, parent_eval=ce, cand_eval=ce, replays={})


def test_authority_never_grows_without_a_human_signature() -> None:
    wider = PARENT.child(tool_allowlist=[*PARENT.tool_allowlist, "issue_credit"], created_by="improver")
    c = check_authority(PARENT, wider)
    assert not c.passed and c.evidence["tools_granted"] == ["issue_credit"]
    assert c.evidence["override"] == "no human approval supplied"

    dropped = PARENT.child(guards=[], created_by="improver")
    assert check_authority(PARENT, dropped).evidence["guards_removed"] == ["g-old"]
    weakened = PARENT.child(guards=[Guard("g-old", "kept", "def check(c, v, h):\n    return None\n")],
                            created_by="improver")  # fmt: skip
    assert check_authority(PARENT, weakened).evidence["guards_altered"] == ["g-old"]

    human, mallory = Signer.generate(), Signer.generate()
    trusted = {human.keyid: human.public_key}
    ok = human.sign(
        HUMAN_APPROVAL_TYPE, canonical_json({"candidate_digest": wider.digest, "approver": "ops"})
    )
    assert check_authority(PARENT, wider, ok, trusted).passed
    forged = mallory.sign(HUMAN_APPROVAL_TYPE, canonical_json({"candidate_digest": wider.digest}))
    assert "rejected" in check_authority(PARENT, wider, forged, trusted).evidence["override"]
    other = human.sign(HUMAN_APPROVAL_TYPE, canonical_json({"candidate_digest": "0" * 64}))
    assert "different candidate" in check_authority(PARENT, wider, other, trusted).evidence["override"]


def test_tampered_candidate_fails_authority() -> None:
    cand = PARENT.child(system_prompt="q", created_by="improver")
    forged = replace(cand, tool_allowlist=["get_charge"])  # digest no longer matches content
    assert not check_authority(PARENT, forged).passed
