from __future__ import annotations

from ratchet.improve.overfit import lint, new_text, training_literals
from ratchet.types import Event, Guard, Harness, Procedure

from .fakes import bad_timeout, make_trace

PARENT = Harness.build(parent=None, system_prompt="Base.", procedures=[], guards=[], tool_allowlist=[],
                       created_by="seed")  # fmt: skip


def _traces():  # type: ignore[no-untyped-def]
    t = make_trace(
        "timeout-1", [("payments_refund", {"charge_id": "ch_3MoK8", "amount_cents": 3200})], "run-a"
    )
    extra = Event(99, "tool_result", {"result": {"data": {
        "customer": {"name": "Maya Okafor", "email": "maya@fastmail.example"},
        "policy": {"max_refund_cents": 50000, "doc_id": "pol_refund-v2"},
        "flag": True,
    }}}, 0.0)  # fmt: skip
    return [t.__class__(t.run_id, t.case_id, t.harness_version, t.model, [*t.events, extra], t.outcome)]


def test_literals_come_from_records_not_policy() -> None:
    lit = training_literals(_traces())
    assert {"ch_3MoK8", "timeout-1", "run-a"} <= lit.ids
    assert 3200 in lit.amounts and 50000 not in lit.amounts
    assert "pol_refund-v2" not in lit.ids
    assert {"Maya Okafor", "maya@fastmail.example"} <= lit.names


def test_lint_flags_ids_amounts_and_names_in_new_text_only() -> None:
    lit = training_literals(_traces())
    general = Guard("g1", "no duplicate refunds", "def check(call, view, history):\n    return None\n")
    copied = Guard(
        "g2",
        "for Maya Okafor",
        "def check(call, view, history):\n    "
        "return 'x' if call.args.get('charge_id') == 'ch_3MoK8' else None\n",
    )
    proc = Procedure("p1", "refund 32.00", "when asked", ["refund 3200 cents"])
    prompt = "Base.\n\nCall run-a again."
    cand = PARENT.child(
        guards=[general, copied], procedures=[proc], system_prompt=prompt, created_by="improver"
    )
    hits = lint(PARENT, cand, lit)
    got = {(h["where"], h["category"], h["literal"]) for h in hits}
    assert ("guard g2", "id", "ch_3MoK8") in got and ("guard g2", "name", "Maya Okafor") in got
    assert ("procedure p1", "amount", "3200") in got
    assert ("system prompt", "id", "run-a") in got
    assert not any(h["where"] == "guard g1" for h in hits)
    assert lint(cand, cand, lit) == [] and [w for w, _ in new_text(cand, cand)] == []


def test_ids_inside_longer_tokens_do_not_match() -> None:
    lit = training_literals([bad_timeout(1)])
    g = Guard("g", "d", "def check(call, view, history):\n    return 'ch_10' if False else None\n")
    assert lint(PARENT, PARENT.child(guards=[g], created_by="improver"), lit) == []
