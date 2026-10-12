"""The worker against the real simulated company, its real grader and the real guard sandbox."""

from __future__ import annotations

from ratchet_cases.cases.build import Company
from ratchet_cases.cases.model import Case, Expect

from ratchet.agent.model import ModelTurn, ScriptedModel, scripted_turn
from ratchet.agent.seed import seed_harness
from ratchet.agent.worker import run_case
from ratchet.types import Guard


def _refund_case() -> Case:
    company = Company()
    company.customer("cus_1", "Ada Park", "ada@example.com", "pm_1")
    company.charge("ch_1", "cus_1", 4200, days_ago=3, description="Blue mug")
    company.ticket(
        "T-1",
        "cus_1",
        "Refund please",
        "My mug arrived broken, please refund charge ch_1.",
        [{"type": "refund", "charge_id": "ch_1"}],
    )
    return Case(
        id="it-refund",
        family="integration",
        variant="v1",
        split="train",
        must_succeed=True,
        customer_message="My mug arrived broken, please refund charge ch_1.",
        initial_state=company.build(),
        faults=[],
        grader=Expect(
            ticket_status={"T-1": "solved"},
            refunds={"ch_1": {"total_cents": 4200, "count": 1, "status": "succeeded"}},
        ),
        ticket_id="T-1",
        kind="must_succeed",
    )


REFUND = {"charge_id": "ch_1", "amount_cents": 4200, "idempotency_key": "rf-ch_1", "reason": "broken"}


def _script() -> list[ModelTurn]:
    return [
        scripted_turn(tool_calls=[("tickets_get", {"ticket_id": "T-1"})], id_prefix="a"),
        scripted_turn(tool_calls=[("payments_refund", REFUND)], id_prefix="b"),
        scripted_turn(
            tool_calls=[
                ("tickets_add_note", {"ticket_id": "T-1", "body": "Refunded ch_1 in full."}),
                ("tickets_set_status", {"ticket_id": "T-1", "status": "solved"}),
            ],
            id_prefix="c",
        ),
        scripted_turn("Refunded 42.00 and solved the ticket."),
    ]


def test_seed_harness_solves_a_real_case() -> None:
    trace = run_case(seed_harness(), _refund_case(), ScriptedModel(_script()))
    assert trace.outcome.passed, trace.outcome.reasons
    assert trace.outcome.cost.model_calls == 4 and trace.outcome.cost.tool_calls == 4
    first_call = next(e for e in trace.events if e.kind == "tool_call")
    assert first_call.payload["view"]["payments"]["charges"]["ch_1"]["amount_cents"] == 4200


def test_real_guard_blocks_a_real_refund() -> None:
    guard = Guard(
        id="no-refund-before-note",
        description="refunds need a note first",
        source=(
            "def check(call, view, history):\n"
            "    if call.name == 'payments_refund' and not view['tickets']['tickets']['T-1']['notes']:\n"
            "        return 'add a ticket note explaining the refund first'\n"
            "    return None\n"
        ),
    )
    harness = seed_harness().child(guards=[guard], created_by="improver")
    trace = run_case(harness, _refund_case(), ScriptedModel(_script()))
    assert not trace.outcome.passed
    assert any("refunded 0 cents" in r or "0 refunds" in r for r in trace.outcome.reasons), (
        trace.outcome.reasons
    )
    block = next(e for e in trace.events if e.kind == "guard_block")
    assert block.payload["decisions"][0]["reason"] == "add a ticket note explaining the refund first"
