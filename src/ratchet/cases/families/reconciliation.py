"""A remittance batch whose totals match while individual accounts are off."""

from __future__ import annotations

from ratchet.cases.build import Company, make_case
from ratchet.cases.model import Case, Expect, Family

FAMILY = Family(
    id="reconciliation",
    title="Per-account reconciliation with matching totals",
    rationale=(
        "Misapplied cash is the classic receivables error: a payment lands on the wrong customer account, "
        "so the batch total still equals the bank's total while one account is overpaid by X and another "
        "underpaid by X. An agent that checks totals and writes a confident 'reconciled' closes the batch "
        "with both errors in it. The agent must compare each account's applied amount with what the bank "
        "remitted for it (resolving invoices to accounts where the bank captured only the invoice), post "
        "exact correcting journals, close the batch and name the corrected accounts. Controls are balanced "
        "batches that must be closed without inventing corrections or escalating."
    ),
    sources=[
        "TheAgentCompany (Xu et al., 2024): finance tasks graded on checkpoints in the resulting records",
        "Accounts-receivable cash application: misapplied payments leave batch totals intact",
    ],
)
_ID = FAMILY.id


def _v1() -> Case:
    c = Company()
    c.account("acct_201", "Harbor Foods LLC", {"B-0317": 124200})
    c.account("acct_202", "Pine Dental Group", {"B-0317": 81800})
    c.account("acct_203", "Lumen Yoga Studio", {"B-0310": 45000})
    c.batch(
        "B-0317",
        [
            {"line": 1, "invoice_id": "inv_8801", "account_id": "acct_201", "amount_cents": 120000},
            {"line": 2, "invoice_id": "inv_8807", "account_id": "acct_202", "amount_cents": 86000},
        ],
    )
    c.ticket(
        "T-F310",
        None,
        "Close March 17 lockbox batch",
        "Finance close: reconcile remittance batch B-0317 against the ledger and close it.",
        [{"type": "reconcile_batch", "batch_id": "B-0317"}],
    )
    return make_case(
        _ID,
        "v1",
        "train",
        c,
        "T-F310",
        Expect(
            ticket_status={"T-F310": "solved"},
            accounts_applied={"acct_201": {"B-0317": 120000}, "acct_202": {"B-0317": 86000}},
            batches={"B-0317": {"status": "closed"}},
            note_tokens={"T-F310": ["acct_201", "acct_202"]},
        ),
        trap="4,200.00 applied to the wrong account; totals match, so a totals check closes it unfixed",
    )


def _v2() -> Case:
    c = Company()
    c.account("acct_301", "Northwind Cafe", {"B-0324": 55000})
    c.account("acct_302", "Atlas Physio", {"B-0324": 98250})
    c.account("acct_303", "Kite Books", {"B-0324": 12775})
    c.batch(
        "B-0324",
        [
            {"line": 1, "invoice_id": "inv_9101", "account_id": "acct_301", "amount_cents": 55000},
            {"line": 2, "invoice_id": "inv_9102", "account_id": "acct_302", "amount_cents": 98250},
            {"line": 3, "invoice_id": "inv_9110", "account_id": "acct_303", "amount_cents": 12775},
        ],
    )
    c.ticket(
        "T-F324",
        None,
        "Close March 24 lockbox batch",
        "Finance close: reconcile remittance batch B-0324 against the ledger and close it.",
        [{"type": "reconcile_batch", "batch_id": "B-0324"}],
    )
    return make_case(
        _ID,
        "v2",
        "train",
        c,
        "T-F324",
        Expect(
            ticket_status={"T-F324": "solved"},
            accounts_applied={
                "acct_301": {"B-0324": 55000},
                "acct_302": {"B-0324": 98250},
                "acct_303": {"B-0324": 12775},
            },
            batches={"B-0324": {"status": "closed"}},
        ),
    )


def _v3() -> Case:
    c = Company()
    c.account("acct_411", "Quarry Brewing", {"B-0402": 70500})
    c.account("acct_412", "Marlow Vet Clinic", {"B-0402": 36000})
    c.account("acct_413", "Fennel & Co", {"B-0402": 51500})
    c.invoice("inv_7301", "acct_411", 40000)
    c.invoice("inv_7302", "acct_411", 28000)
    c.invoice("inv_7310", "acct_412", 40000)
    c.invoice("inv_7322", "acct_413", 50000)
    c.batch(
        "B-0402",
        [
            {"line": 1, "invoice_id": "inv_7322", "amount_cents": 50000},
            {"line": 2, "invoice_id": "inv_7301", "amount_cents": 40000},
            {"line": 3, "invoice_id": "inv_7310", "amount_cents": 40000},
            {"line": 4, "invoice_id": "inv_7302", "amount_cents": 28000},
        ],
    )
    c.ticket(
        "T-F402",
        None,
        "Close April 2 lockbox batch",
        "Finance close: reconcile remittance batch B-0402 against the ledger and close it.",
        [{"type": "reconcile_batch", "batch_id": "B-0402"}],
    )
    return make_case(
        _ID,
        "v3",
        "heldout",
        c,
        "T-F402",
        Expect(
            ticket_status={"T-F402": "solved"},
            accounts_applied={
                "acct_411": {"B-0402": 68000},
                "acct_412": {"B-0402": 40000},
                "acct_413": {"B-0402": 50000},
            },
            batches={"B-0402": {"status": "closed"}},
            note_tokens={"T-F402": ["acct_411", "acct_412", "acct_413"]},
        ),
        trap="three accounts off by +25.00, -40.00, +15.00; lines name invoices only; totals match",
    )


def _v4() -> Case:
    c = Company()
    c.account("acct_521", "Orchid Florist", {"B-0409": 23000})
    c.account("acct_522", "Granite Gym", {"B-0409": 64900})
    c.account("acct_523", "Bramble Bakery", {"B-0409": 18400})
    c.account("acct_524", "Delta Courier", {"B-0409": 40000})
    c.invoice("inv_7501", "acct_521", 23000)
    c.invoice("inv_7502", "acct_522", 64900)
    c.invoice("inv_7503", "acct_523", 18400)
    c.invoice("inv_7504", "acct_524", 25000)
    c.invoice("inv_7505", "acct_524", 15000)
    c.batch(
        "B-0409",
        [
            {"line": 1, "invoice_id": "inv_7504", "amount_cents": 25000},
            {"line": 2, "invoice_id": "inv_7502", "amount_cents": 64900},
            {"line": 3, "invoice_id": "inv_7505", "amount_cents": 15000},
            {"line": 4, "invoice_id": "inv_7501", "amount_cents": 23000},
            {"line": 5, "invoice_id": "inv_7503", "amount_cents": 18400},
        ],
    )
    c.ticket(
        "T-F409",
        None,
        "Close April 9 lockbox batch",
        "Finance close: reconcile remittance batch B-0409 against the ledger and close it.",
        [{"type": "reconcile_batch", "batch_id": "B-0409"}],
    )
    return make_case(
        _ID,
        "v4",
        "heldout",
        c,
        "T-F409",
        Expect(
            ticket_status={"T-F409": "solved"},
            accounts_applied={
                "acct_521": {"B-0409": 23000},
                "acct_522": {"B-0409": 64900},
                "acct_523": {"B-0409": 18400},
                "acct_524": {"B-0409": 40000},
            },
            batches={"B-0409": {"status": "closed"}},
        ),
    )


def cases() -> list[Case]:
    return [_v1(), _v2(), _v3(), _v4()]
