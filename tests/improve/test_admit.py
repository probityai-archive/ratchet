from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from ratchet.improve.admit import admit, check_authority, check_heldout
from ratchet.improve.diagnose import diagnose
from ratchet.improve.evaluate import EvalReport, evaluate
from ratchet.improve.overfit import training_literals
from ratchet.improve.receipt import HUMAN_APPROVAL_TYPE, Signer, receipt, verify
from ratchet.improve.replay import ReplayReport, WalkCache, replay
from ratchet.types import Guard, Harness, canonical_json

from .fakes import BLOCK_ALL_REFUNDS, NO_DUP_GUARD, FakeRunCase, FakeSimulator, LocalRunner, fake_cases

OLD = Guard("g-old", "kept", "def check(call, view, history):\n    return None\n")
PARENT = Harness.build(
    parent=None, system_prompt="p", procedures=[], guards=[OLD], tool_allowlist=["get_charge", "refund"],
    created_by="seed",
)  # fmt: skip
HELDOUT = [c for c in fake_cases() if c.split == "heldout"]


def _replays(cand: Harness, pe: EvalReport) -> dict[str, ReplayReport]:
    cluster = diagnose(pe.train_traces(), FakeSimulator())[0]
    passing = [t for t in pe.traces if t.outcome.passed]
    benign = [t for t in pe.traces if t.case_id.startswith("plain")]
    walks = WalkCache(FakeSimulator())
    new = [g for g in cand.guards if g.id != OLD.id]
    return {
        g.id: replay(g, cluster.members, cluster.traces, passing, walks, LocalRunner(), benign) for g in new
    }


def _admit(cand: Harness, pe: EvalReport | None = None, ce: EvalReport | None = None, **kw: Any):  # type: ignore[no-untyped-def]
    pe = pe or evaluate(PARENT, fake_cases(), FakeRunCase(), None)
    rerun = kw.pop("rerun", None) or evaluate(PARENT, HELDOUT, FakeRunCase(), None)
    ce = ce or evaluate(cand, fake_cases(), FakeRunCase(), None)
    replays = kw.pop("replays", None)
    return admit(
        PARENT, cand, parent_eval=pe, parent_rerun=rerun, cand_eval=ce,
        replays=_replays(cand, pe) if replays is None else replays,
        literals=training_literals(pe.train_traces()), visible_case_ids=["timeout-1", "timeout-2", "plain-3"],
        suite_digest="abc", **kw,
    )  # fmt: skip


def test_a_precise_guard_is_admitted_with_evidence_and_a_verifiable_receipt() -> None:
    cand = PARENT.child(guards=[OLD, Guard("g-new", "d", NO_DUP_GUARD)], created_by="improver")
    d = _admit(cand, context={"generation": 1})
    assert d.admitted, d.to_payload()
    h = d.check("heldout").evidence
    assert (
        h["gained"] == ["timeout-4", "timeout-5"]
        and h["parent_runs_passed"] == [1, 1]
        and h["noise_floor"] == 1
    )
    assert len(h["parent_runs"]) == 2 and h["candidate_cached_calls"] == 0
    assert d.check("replay").evidence["reports"][0]["targets_hit"] == 2
    assert d.check("benign_twins").passed and d.check("overfit_lint").passed
    assert d.check("cost").evidence["gate"] is False
    assert (
        d.rates["candidate"]["recovery_rate"] == 1.0
        and d.rates["candidate"]["false_intervention_rate"] == 0.0
    )
    assert d.rates["parent"]["recovery_rate"] == 0.0
    signer = Signer.generate()
    body = verify(receipt(d, signer), {signer.keyid: signer.public_key})
    assert body["admitted"] is True and body["context"] == {"generation": 1}
    assert body["visible_case_ids"] == ["plain-3", "timeout-1", "timeout-2"] and body["suite_digest"] == "abc"


def test_overbroad_guard_fails_replay_regression_and_benign_twins() -> None:
    cand = PARENT.child(guards=[OLD, Guard("g-new", "d", BLOCK_ALL_REFUNDS)], created_by="improver")
    d = _admit(cand)
    assert not d.admitted
    rep = d.check("replay").evidence["reports"][0]
    assert rep["false_positives"] and rep["benign_twin_fires"] >= 1
    assert d.check("no_regression").evidence["regressed"] == ["plain-3", "plain-6"]
    assert d.check("benign_twins").evidence["intervened"] == ["plain-6"]
    assert d.rates["candidate"]["false_intervention_rate"] == 1.0


def test_copied_literals_fail_the_overfit_screen() -> None:
    src = NO_DUP_GUARD.replace(
        "return None\n", "return 'ch_1' if call.args.get('charge_id') == 'ch_1' else None\n", 1
    )
    cand = PARENT.child(guards=[OLD, Guard("g-new", "d", src)], created_by="improver")
    d = _admit(cand)
    assert not d.check("overfit_lint").passed
    assert {"where": "guard g-new", "category": "id", "literal": "ch_1"} in d.check("overfit_lint").evidence[
        "hits"
    ]


def test_gain_must_clear_the_parents_better_run_and_be_fresh() -> None:
    cand = PARENT.child(guards=[OLD, Guard("g-new", "d", NO_DUP_GUARD)], created_by="improver")
    pe = evaluate(PARENT, fake_cases(), FakeRunCase(), None)
    ce = evaluate(cand, fake_cases(), FakeRunCase(), None)
    lucky = replace(
        ce, harness_version=PARENT.version, results=[r for r in ce.results if r.split == "heldout"]
    )
    c = check_heldout(pe, lucky, ce)  # the parent's second run happened to pass everything
    assert not c.passed and c.evidence["noise_floor"] == 3
    cached = replace(
        ce, results=[replace(r, cached_calls=1) if r.case_id == "timeout-4" else r for r in ce.results]
    )
    c2 = check_heldout(pe, evaluate(PARENT, HELDOUT, FakeRunCase(), None), cached)
    assert (
        c2.evidence["excluded_cached_passes"] == ["timeout-4"] and c2.evidence["candidate_passed_fresh"] == 2
    )


def test_missing_replay_and_flat_heldout_reject() -> None:
    cand = PARENT.child(guards=[OLD, Guard("g-new", "d", NO_DUP_GUARD)], created_by="improver")
    ce = evaluate(cand, fake_cases(), FakeRunCase(), None)
    d = _admit(cand, pe=ce, ce=ce, rerun=evaluate(cand, HELDOUT, FakeRunCase(), None), replays={})
    assert not d.admitted
    assert d.check("replay").evidence["reports"][0]["missing"]
    assert not d.check("heldout").passed and d.check("heldout").evidence["same_case_set"]


def test_different_case_sets_cannot_be_compared() -> None:
    cand = PARENT.child(system_prompt="q", created_by="improver")
    pe = evaluate(PARENT, fake_cases()[:3], FakeRunCase(), None)
    d = _admit(cand, pe=pe, replays={})
    assert not d.check("heldout").evidence["same_case_set"]


def test_candidate_must_be_a_child() -> None:
    other = Harness.build(
        parent=None, system_prompt="x", procedures=[], guards=[], tool_allowlist=[], created_by="seed"
    )
    with pytest.raises(ValueError, match="not a child"):
        _admit(other, replays={})


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
