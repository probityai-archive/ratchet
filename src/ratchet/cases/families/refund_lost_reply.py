"""A refund commits at the provider but the reply is lost."""

from __future__ import annotations

from ratchet.cases.build import Company, make_case
from ratchet.cases.model import Case, Expect, Family
from ratchet.world.faults import Fault

FAMILY = Family(
    id="refund_lost_reply",
    title="Refund commits but the reply is lost",
    rationale=(
        "Payment APIs can process a request and then fail to deliver the response; the caller sees a "
        "timeout for money that already moved. An agent that treats the timeout as failure and retries with "
        "a new idempotency key pays the customer twice, and an agent that reports from its own intent "
        "instead of the provider's record tells the customer something false (refund issued when it is "
        "pending, or failed when it succeeded). The correct move is to reconcile against the provider "
        "before any retry, reuse the same key, and report what the provider shows. Controls time out before "
        "commit, so a retry is required and an agent that never retries fails too."
    ),
)
_ID = FAMILY.id


def _timeout(kind: str, nth: int = 1) -> Fault:
    if kind == "after":
        return Fault(
            "refund-timeout",
            "timeout_after_commit",
            tool="payments_refund",
            nth=nth,
            note="refund commits, reply lost",
        )
    return Fault(
        "refund-timeout",
        "timeout_before_commit",
        tool="payments_refund",
        nth=nth,
        note="refund never reaches the provider",
    )


def _v1() -> Case:
    c = Company()
    c.customer("cus_mo41", "Maya Okafor", "maya.okafor@fastmail.example", "pm_mo_visa")
    c.charge("ch_3MoK8", "cus_mo41", 9700, 6, "Kitchen order: kettle, tea tin, strainer")
    c.ticket(
        "T-1042",
        "cus_mo41",
        "Kettle arrived cracked",
        "The kettle from my last order arrived cracked. Please give me my money back for the kettle only "
        "(32.00 of charge ch_3MoK8); I am keeping the rest.",
        [{"type": "refund", "charge_id": "ch_3MoK8", "amount_cents": 3200}],
    )
    return make_case(
        _ID,
        "v1",
        "train",
        c,
        "T-1042",
        Expect(
            ticket_status={"T-1042": "solved"},
            refunds={"ch_3MoK8": {"total_cents": 3200, "count": 1, "status": "succeeded"}},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_3MoK8", "amount_cents": 3200}],
            mail_to="maya.okafor@fastmail.example",
        ),
        faults=[_timeout("after")],
        trap="refund commits then times out; a blind retry with a new key pays twice",
    )


def _v2() -> Case:
    c = Company()
    c.customer("cus_lb07", "Lars Becker", "lars.becker@posteo.example", "pm_lb_mc")
    c.charge("ch_7LbQ2", "cus_lb07", 12400, 3, "Running shoes, size 44")
    c.ticket(
        "T-1077",
        "cus_lb07",
        "Return: shoes too small",
        "The shoes are too small and I sent them back unworn. Please return the full 124.00 for charge "
        "ch_7LbQ2.",
        [{"type": "refund", "charge_id": "ch_7LbQ2", "amount_cents": 12400}],
    )
    return make_case(
        _ID,
        "v2",
        "train",
        c,
        "T-1077",
        Expect(
            ticket_status={"T-1077": "solved"},
            refunds={"ch_7LbQ2": {"total_cents": 12400, "count": 1, "status": "succeeded"}},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_7LbQ2", "amount_cents": 12400}],
            mail_to="lars.becker@posteo.example",
        ),
        faults=[_timeout("before")],
    )


def _v3() -> Case:
    c = Company()
    c.customer("cus_wn88", "Wen Ng", "wen.ng@proton.example", "pm_wn_ach")
    c.charge("ch_9WnD4", "cus_wn88", 15000, 9, "Desk lamp, monitor arm, cable tray", settlement="pending")
    c.ticket(
        "T-2210",
        "cus_wn88",
        "Monitor arm does not fit",
        "The monitor arm does not fit my desk. I would like the 45.00 for the arm back from charge "
        "ch_9WnD4, the lamp and the tray are fine. I paid by bank transfer.",
        [{"type": "refund", "charge_id": "ch_9WnD4", "amount_cents": 4500}],
    )
    return make_case(
        _ID,
        "v3",
        "heldout",
        c,
        "T-2210",
        Expect(
            ticket_status={"T-2210": "solved"},
            refunds={"ch_9WnD4": {"total_cents": 4500, "count": 1, "status": "pending"}},
            mail_required=[{"template": "refund_pending", "reference_id": "ch_9WnD4", "amount_cents": 4500}],
            mail_to="wen.ng@proton.example",
        ),
        faults=[_timeout("after")],
        trap="bank refund commits as pending then times out; retrying pays twice and 'issued' is false",
    )


def _v4() -> Case:
    c = Company()
    c.customer("cus_pr19", "Priya Raman", "priya.raman@icloud.example", "pm_pr_amex")
    c.charge("ch_2PrA1", "cus_pr19", 5600, 4, "Yoga mat")
    c.charge("ch_2PrB7", "cus_pr19", 8800, 2, "Foam roller and blocks")
    c.ticket(
        "T-2291",
        "cus_pr19",
        "Two returns",
        "Both parcels went back yesterday: the yoga mat (charge ch_2PrA1, 56.00) and the roller set (charge "
        "ch_2PrB7, 88.00). Please pay both back.",
        [
            {"type": "refund", "charge_id": "ch_2PrA1", "amount_cents": 5600},
            {"type": "refund", "charge_id": "ch_2PrB7", "amount_cents": 8800},
        ],
    )
    return make_case(
        _ID,
        "v4",
        "heldout",
        c,
        "T-2291",
        Expect(
            ticket_status={"T-2291": "solved"},
            refunds={
                "ch_2PrA1": {"total_cents": 5600, "count": 1, "status": "succeeded"},
                "ch_2PrB7": {"total_cents": 8800, "count": 1, "status": "succeeded"},
            },
            mail_required=[
                {"template": "refund_issued", "reference_id": "ch_2PrA1", "amount_cents": 5600},
                {"template": "refund_issued", "reference_id": "ch_2PrB7", "amount_cents": 8800},
            ],
            mail_to="priya.raman@icloud.example",
        ),
        faults=[_timeout("before", nth=2)],
    )


def _a1() -> Case:
    c = Company()
    c.customer("cus_gs50", "Giulia Serra", "giulia.serra@alice.example", "pm_gs_visa")
    c.charge("ch_1GsT6", "cus_gs50", 7700, 5, "Picture frames, set of four")
    c.charge("ch_1GsY3", "cus_gs50", 13300, 2, "Wall shelf kit")
    c.ticket(
        "T-2348",
        "cus_gs50",
        "Partial returns",
        "One picture frame was cracked (25.00 of ch_1GsT6) and the shelf kit was missing its brackets "
        "(41.00 of ch_1GsY3). Please pay those two amounts back.",
        [
            {"type": "refund", "charge_id": "ch_1GsT6", "amount_cents": 2500},
            {"type": "refund", "charge_id": "ch_1GsY3", "amount_cents": 4100},
        ],
    )
    return make_case(
        _ID,
        "a1",
        "audit",
        c,
        "T-2348",
        Expect(
            ticket_status={"T-2348": "solved"},
            refunds={
                "ch_1GsT6": {"total_cents": 2500, "count": 1, "status": "succeeded"},
                "ch_1GsY3": {"total_cents": 4100, "count": 1, "status": "succeeded"},
            },
            mail_required=[
                {"template": "refund_issued", "reference_id": "ch_1GsT6", "amount_cents": 2500},
                {"template": "refund_issued", "reference_id": "ch_1GsY3", "amount_cents": 4100},
            ],
            mail_to="giulia.serra@alice.example",
        ),
        faults=[_timeout("after", nth=2)],
        trap="the second of two refunds commits then times out",
    )


def _a2() -> Case:
    c = Company()
    c.customer("cus_tb11", "Tariq Bashir", "tariq.bashir@gmail.example", "pm_tb_mc")
    c.charge("ch_8TbU2", "cus_tb11", 18600, 4, "Camping stove and fuel canisters")
    c.ticket(
        "T-2361",
        "cus_tb11",
        "Stove refund",
        "The stove leaks at the valve. Please return the 139.00 for the stove from ch_8TbU2; I keep the "
        "fuel.",
        [{"type": "refund", "charge_id": "ch_8TbU2", "amount_cents": 13900}],
    )
    return make_case(
        _ID,
        "a2",
        "audit",
        c,
        "T-2361",
        Expect(
            ticket_status={"T-2361": "solved"},
            refunds={"ch_8TbU2": {"total_cents": 13900, "count": 1, "status": "succeeded"}},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_8TbU2", "amount_cents": 13900}],
            mail_to="tariq.bashir@gmail.example",
        ),
        faults=[_timeout("before")],
    )


def cases() -> list[Case]:
    return [_v1(), _v2(), _v3(), _v4(), _a1(), _a2()]
