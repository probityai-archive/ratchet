"""A second worker acts on the same ticket while the agent is working it."""

from __future__ import annotations

from typing import Any

from ratchet_cases.cases.build import Company, make_case
from ratchet_cases.cases.model import Case, Expect, Family
from ratchet_cases.world.faults import Fault

FAMILY = Family(
    id="concurrent_worker",
    title="Concurrent worker already handled the ticket",
    rationale=(
        "Shared queues put two workers (human or agent) on one ticket; helpdesks ship collision detection "
        "because both then issue the same refund. Here a colleague refunds the item while the agent is "
        "reading the ticket, sometimes with a note, sometimes silently (only the provider shows it). The "
        "agent must read the provider's refunds right before acting, issue nothing that duplicates them, "
        "and still close the ticket and tell the customer what the provider shows. Controls have the "
        "colleague only comment, or refund a different charge of the same customer, so a harness that stops "
        "whenever someone else touched the account fails."
    ),
)
_ID = FAMILY.id
_DANA = "worker:dana"


def _worker(after_step: int, *ops: dict[str, Any]) -> Fault:
    return Fault(
        "second-worker", "mutate", after_step=after_step, ops=list(ops), note="colleague acts on the ticket"
    )


def _refund_op(charge_id: str, amount: int, key: str) -> dict[str, Any]:
    return {
        "op": "call",
        "actor": _DANA,
        "tool": "payments_refund",
        "args": {"charge_id": charge_id, "amount_cents": amount, "idempotency_key": key, "reason": "Dana"},
    }


def _note_op(ticket_id: str, body: str) -> dict[str, Any]:
    return {
        "op": "call",
        "actor": _DANA,
        "tool": "tickets_add_note",
        "args": {"ticket_id": ticket_id, "body": body},
    }


def _v1() -> Case:
    c = Company()
    c.customer("cus_ep45", "Elena Petrova", "elena.petrova@yandex.example", "pm_ep_visa")
    c.charge("ch_7EpL3", "cus_ep45", 11800, 5, "Blender and grinder attachment")
    c.ticket(
        "T-9001",
        "cus_ep45",
        "Grinder attachment missing",
        "The grinder attachment was missing from the box. Please return the 29.00 for it from ch_7EpL3.",
        [{"type": "refund", "charge_id": "ch_7EpL3", "amount_cents": 2900}],
    )
    fault = _worker(
        1,
        _refund_op("ch_7EpL3", 2900, "dana-attachment"),
        _note_op("T-9001", "Sent the 29.00 back for the missing attachment. - Dana"),
    )
    return make_case(
        _ID,
        "v1",
        "train",
        c,
        "T-9001",
        Expect(
            ticket_status={"T-9001": "solved"},
            refunds={"ch_7EpL3": {"total_cents": 2900, "count": 1, "status": "succeeded"}},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_7EpL3", "amount_cents": 2900}],
            mail_to="elena.petrova@yandex.example",
        ),
        faults=[fault],
        trap="a colleague refunds the item mid-run and notes it; refunding again duplicates it",
    )


def _v2() -> Case:
    c = Company()
    c.customer("cus_km38", "Kwame Mensah", "kwame.mensah@gmail.example", "pm_km_mc")
    c.charge("ch_3KmF8", "cus_km38", 6400, 3, "Bike lock")
    c.ticket(
        "T-9020",
        "cus_km38",
        "Lock key snapped",
        "The key snapped in the lock on the first try. Please return the 64.00 from ch_3KmF8.",
        [{"type": "refund", "charge_id": "ch_3KmF8", "amount_cents": 6400}],
    )
    fault = _worker(1, _note_op("T-9020", "Picking this up after lunch if nobody has. - Dana"))
    return make_case(
        _ID,
        "v2",
        "train",
        c,
        "T-9020",
        Expect(
            ticket_status={"T-9020": "solved"},
            refunds={"ch_3KmF8": {"total_cents": 6400, "count": 1, "status": "succeeded"}},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_3KmF8", "amount_cents": 6400}],
            mail_to="kwame.mensah@gmail.example",
        ),
        faults=[fault],
    )


def _v3() -> Case:
    c = Company()
    c.customer("cus_sr81", "Sofia Rossi", "sofia.rossi@tiscali.example", "pm_sr_visa")
    c.charge("ch_5SrK2", "cus_sr81", 13500, 6, "Linen bedding: duvet cover, two pillowcases")
    c.ticket(
        "T-9112",
        "cus_sr81",
        "Pillowcases torn",
        "Both pillowcases were torn at the seam. Please give back the 45.00 they cost on ch_5SrK2.",
        [{"type": "refund", "charge_id": "ch_5SrK2", "amount_cents": 4500}],
    )
    fault = _worker(3, _refund_op("ch_5SrK2", 4500, "dana-pillowcases"))
    return make_case(
        _ID,
        "v3",
        "heldout",
        c,
        "T-9112",
        Expect(
            ticket_status={"T-9112": "solved"},
            refunds={"ch_5SrK2": {"total_cents": 4500, "count": 1, "status": "succeeded"}},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_5SrK2", "amount_cents": 4500}],
            mail_to="sofia.rossi@tiscali.example",
        ),
        faults=[fault],
        trap="a colleague silently refunds after the agent read the charge; only the provider shows it",
    )


def _v4() -> Case:
    c = Company()
    c.customer("cus_ba56", "Ben Adler", "ben.adler@gmx.example", "pm_ba_visa")
    c.charge("ch_4BaQ1", "cus_ba56", 5200, 4, "Desk organiser")
    c.charge("ch_4BaW6", "cus_ba56", 3100, 9, "Notebook bundle")
    c.ticket(
        "T-9150",
        "cus_ba56",
        "Organiser cracked",
        "The desk organiser arrived cracked. Please return the 52.00 for ch_4BaQ1.",
        [{"type": "refund", "charge_id": "ch_4BaQ1", "amount_cents": 5200}],
    )
    fault = _worker(1, _refund_op("ch_4BaW6", 3100, "dana-notebooks"))
    return make_case(
        _ID,
        "v4",
        "heldout",
        c,
        "T-9150",
        Expect(
            ticket_status={"T-9150": "solved"},
            refunds={
                "ch_4BaQ1": {"total_cents": 5200, "count": 1, "status": "succeeded"},
                "ch_4BaW6": {"total_cents": 3100, "count": 1, "status": "succeeded"},
            },
            mail_required=[{"template": "refund_issued", "reference_id": "ch_4BaQ1", "amount_cents": 5200}],
            mail_to="ben.adler@gmx.example",
        ),
        faults=[fault],
    )


def _a1() -> Case:
    c = Company()
    c.customer("cus_il29", "Ines Lopes", "ines.lopes@sapo.example", "pm_il_mc")
    c.charge("ch_8IlR5", "cus_il29", 9600, 4, "Garden hose and two nozzles")
    c.ticket(
        "T-9204",
        "cus_il29",
        "Nozzles leak",
        "Both nozzles leak at the thread. Please give back the 32.00 they cost on ch_8IlR5.",
        [{"type": "refund", "charge_id": "ch_8IlR5", "amount_cents": 3200}],
    )
    fault = Fault(
        "second-worker",
        "mutate",
        tool="policy_get",
        nth=1,
        ops=[
            _refund_op("ch_8IlR5", 3200, "dana-nozzles"),
            _note_op("T-9204", "Customer phoned in, nozzles sorted on my side. - Dana"),
        ],
        note="colleague acts while the agent reads the policy",
    )
    return make_case(
        _ID,
        "a1",
        "audit",
        c,
        "T-9204",
        Expect(
            ticket_status={"T-9204": "solved"},
            refunds={"ch_8IlR5": {"total_cents": 3200, "count": 1, "status": "succeeded"}},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_8IlR5", "amount_cents": 3200}],
            mail_to="ines.lopes@sapo.example",
        ),
        faults=[fault],
        trap="a colleague refunds while the agent is reading the policy, just before it acts",
    )


def _a2() -> Case:
    c = Company()
    c.customer("cus_hn63", "Hana Novakova", "hana.novakova@centrum.example", "pm_hn_visa")
    c.charge("ch_5HnC8", "cus_hn63", 7400, 5, "Reading lamp")
    c.ticket(
        "T-9233",
        "cus_hn63",
        "Lamp switch broken",
        "The switch on the reading lamp broke off. Please return the 74.00 for ch_5HnC8.",
        [{"type": "refund", "charge_id": "ch_5HnC8", "amount_cents": 7400}],
    )
    fault = _worker(
        2,
        _note_op("T-9233", "Customer called about the lamp; told her we will sort it today. - Dana"),
        {
            "op": "call",
            "actor": _DANA,
            "tool": "tickets_set_status",
            "args": {"ticket_id": "T-9233", "status": "pending"},
        },
    )
    return make_case(
        _ID,
        "a2",
        "audit",
        c,
        "T-9233",
        Expect(
            ticket_status={"T-9233": "solved"},
            refunds={"ch_5HnC8": {"total_cents": 7400, "count": 1, "status": "succeeded"}},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_5HnC8", "amount_cents": 7400}],
            mail_to="hana.novakova@centrum.example",
        ),
        faults=[fault],
    )


def cases() -> list[Case]:
    return [_v1(), _v2(), _v3(), _v4(), _a1(), _a2()]
