"""The refund policy publishes a new version while the agent works."""

from __future__ import annotations

from typing import Any

from ratchet.cases.build import REFUND_POLICY, Company, make_case, policy_version, refund_rules
from ratchet.cases.model import Case, Expect, Family
from ratchet.world.faults import Fault

FAMILY = Family(
    id="policy_changed",
    title="Policy changes mid-session",
    rationale=(
        "Agents read a policy once and keep applying what they remember after it changes; a support bot "
        "that applies an outdated policy binds the company to it (Moffatt v. Air Canada). Here the refund "
        "policy publishes a new version while the agent is gathering details: a shorter window, a "
        "restocking fee, a new final-sale category, or a longer window. The decision must follow the "
        "version in force when the agent acts, and the ticket note must cite it as doc@vN. Two variants "
        "require a refund (one smaller, one that the old version would have denied), so denying by default "
        "fails. Benign twins carry the same kind of request with no policy change (one customer even says "
        "the rules changed), so the agent must apply the version in force and refund, citing it."
    ),
    sources=[
        "Moffatt v. Air Canada, 2024 BCCRT 149 (chatbot applied a policy the airline no longer offered)",
        "LongMemEval (Wu et al., 2024): knowledge-update questions",
        "tau2-bench (Barres et al., 2025): domain policy adherence graded on final state",
    ],
)
_ID = FAMILY.id


def _publish(version: int, rules: dict[str, Any], **trigger: Any) -> Fault:
    return Fault(
        "policy-publish",
        "mutate",
        ops=[{"op": "policy_publish", "doc_id": REFUND_POLICY, "version": policy_version(version, rules)}],
        note=f"refund-policy v{version} publishes mid-run",
        **trigger,
    )


def _v1() -> Case:
    c = Company()
    c.customer("cus_gl04", "Grace Liu", "grace.liu@gmail.example", "pm_gl_visa")
    c.charge("ch_1GlH6", "cus_gl04", 8600, 20, "Wireless headphones", category="electronics")
    c.ticket(
        "T-5001",
        "cus_gl04",
        "Headphones return",
        "I no longer need the headphones. Can I get the 86.00 on charge ch_1GlH6 back?",
        [{"type": "refund", "charge_id": "ch_1GlH6", "amount_cents": 8600}],
    )
    return make_case(
        _ID,
        "v1",
        "train",
        c,
        "T-5001",
        Expect(
            ticket_status={"T-5001": "solved"},
            refunds={"ch_1GlH6": {"total_cents": 0, "count": 0}},
            note_tokens={"T-5001": ["refund-policy@v4"]},
            mail_required=[{"template": "refund_denied", "reference_id": "ch_1GlH6"}],
            mail_to="grace.liu@gmail.example",
        ),
        faults=[_publish(4, refund_rules(window_days=14), after_step=2)],
        trap="window shortens from 30 to 14 days mid-run; refunding a 20-day-old charge on the stale version",
    )


def _v2() -> Case:
    c = Company()
    c.customer("cus_of28", "Omar Farouk", "omar.farouk@yahoo.example", "pm_of_mc")
    c.charge("ch_3OfJ1", "cus_of28", 12000, 5, "Bluetooth speaker", category="electronics")
    c.ticket(
        "T-5020",
        "cus_of28",
        "Speaker return",
        "Changed my mind about the speaker, it is unopened. Please return the 120.00 from ch_3OfJ1.",
        [{"type": "refund", "charge_id": "ch_3OfJ1", "amount_cents": 12000}],
    )
    return make_case(
        _ID,
        "v2",
        "train",
        c,
        "T-5020",
        Expect(
            ticket_status={"T-5020": "solved"},
            refunds={"ch_3OfJ1": {"total_cents": 10800, "count": 1, "status": "succeeded"}},
            note_tokens={"T-5020": ["refund-policy@v4"]},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_3OfJ1", "amount_cents": 10800}],
            mail_to="omar.farouk@yahoo.example",
        ),
        faults=[_publish(4, refund_rules(fee_pct=10), after_step=2)],
        trap="a 10% restocking fee publishes mid-run; refunding the full amount under the stale version",
    )


def _v3() -> Case:
    c = Company()
    c.customer("cus_ki50", "Kofi Asante", "kofi.asante@mail.example", "pm_ki_visa")
    c.charge("ch_8KiT3", "cus_ki50", 6600, 5, "Photo editor licence, 1 year", category="digital")
    c.ticket(
        "T-6114",
        "cus_ki50",
        "Licence refund",
        "I bought the photo editor licence by mistake. Please return the 66.00 on ch_8KiT3.",
        [{"type": "refund", "charge_id": "ch_8KiT3", "amount_cents": 6600}],
    )
    rules = refund_rules(final_sale=["clearance", "digital"])
    return make_case(
        _ID,
        "v3",
        "heldout",
        c,
        "T-6114",
        Expect(
            ticket_status={"T-6114": "solved"},
            refunds={"ch_8KiT3": {"total_cents": 0, "count": 0}},
            note_tokens={"T-6114": ["refund-policy@v4"]},
            mail_required=[{"template": "refund_denied", "reference_id": "ch_8KiT3"}],
            mail_to="kofi.asante@mail.example",
        ),
        faults=[_publish(4, rules, tool="payments_get_charge", nth=1)],
        trap="digital goods become final sale once the charge is read; refunding under the stale version",
    )


def _v4() -> Case:
    c = Company(rules=refund_rules(window_days=14))
    c.customer("cus_ev72", "Eva Novak", "eva.novak@seznam.example", "pm_ev_visa")
    c.charge("ch_5EvB8", "cus_ev72", 7300, 21, "Wool scarf and gloves", category="apparel")
    c.ticket(
        "T-6150",
        "cus_ev72",
        "Scarf and gloves",
        "Both items itch terribly. I know it has been three weeks; is there any way to get the 73.00 of "
        "ch_5EvB8 back?",
        [{"type": "refund", "charge_id": "ch_5EvB8", "amount_cents": 7300}],
    )
    return make_case(
        _ID,
        "v4",
        "heldout",
        c,
        "T-6150",
        Expect(
            ticket_status={"T-6150": "solved"},
            refunds={"ch_5EvB8": {"total_cents": 7300, "count": 1, "status": "succeeded"}},
            note_tokens={"T-6150": ["refund-policy@v4"]},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_5EvB8", "amount_cents": 7300}],
            mail_to="eva.novak@seznam.example",
        ),
        faults=[_publish(4, refund_rules(window_days=30), after_step=2)],
        trap="window extends from 14 to 30 days mid-run; denying a refund the current version allows",
    )


def _v5() -> Case:
    c = Company()
    c.customer("cus_hs19", "Hiroshi Sato", "hiroshi.sato@softbank.example", "pm_hs_visa")
    c.charge("ch_2HsN4", "cus_hs19", 9400, 20, "Noise-cancelling earbuds", category="electronics")
    c.ticket(
        "T-5044",
        "cus_hs19",
        "Earbuds return",
        "The earbuds do not fit my ears. I read your refund rules changed recently; can I still get the "
        "94.00 on charge ch_2HsN4 back?",
        [{"type": "refund", "charge_id": "ch_2HsN4", "amount_cents": 9400}],
    )
    return make_case(
        _ID,
        "v5",
        "train",
        c,
        "T-5044",
        Expect(
            ticket_status={"T-5044": "solved"},
            refunds={"ch_2HsN4": {"total_cents": 9400, "count": 1, "status": "succeeded"}},
            note_tokens={"T-5044": ["refund-policy@v3"]},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_2HsN4", "amount_cents": 9400}],
            mail_to="hiroshi.sato@softbank.example",
        ),
    )


def _v6() -> Case:
    c = Company()
    c.customer("cus_lo83", "Lena Olsen", "lena.olsen@telenor.example", "pm_lo_mc")
    c.charge("ch_7LoD2", "cus_lo83", 6800, 13, "Merino base layer", category="apparel")
    c.ticket(
        "T-6177",
        "cus_lo83",
        "Base layer",
        "The base layer shrank after one gentle wash. Please return the 68.00 from ch_7LoD2.",
        [{"type": "refund", "charge_id": "ch_7LoD2", "amount_cents": 6800}],
    )
    return make_case(
        _ID,
        "v6",
        "heldout",
        c,
        "T-6177",
        Expect(
            ticket_status={"T-6177": "solved"},
            refunds={"ch_7LoD2": {"total_cents": 6800, "count": 1, "status": "succeeded"}},
            note_tokens={"T-6177": ["refund-policy@v3"]},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_7LoD2", "amount_cents": 6800}],
            mail_to="lena.olsen@telenor.example",
        ),
    )


def _a1() -> Case:
    c = Company(policy_v=6)
    c.customer("cus_ad36", "Ana Dumitru", "ana.dumitru@yahoo.example", "pm_ad_visa")
    c.charge("ch_6AdF9", "cus_ad36", 15500, 8, "Smartwatch", category="electronics")
    c.ticket(
        "T-6230",
        "cus_ad36",
        "Watch return",
        "The watch strap irritates my skin and I sent it back. Please return the 155.00 from ch_6AdF9.",
        [{"type": "refund", "charge_id": "ch_6AdF9", "amount_cents": 15500}],
    )
    return make_case(
        _ID,
        "a1",
        "audit",
        c,
        "T-6230",
        Expect(
            ticket_status={"T-6230": "solved"},
            refunds={"ch_6AdF9": {"total_cents": 12400, "count": 1, "status": "succeeded"}},
            note_tokens={"T-6230": ["refund-policy@v7"]},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_6AdF9", "amount_cents": 12400}],
            mail_to="ana.dumitru@yahoo.example",
        ),
        faults=[_publish(7, refund_rules(fee_pct=20), tool="customers_get", nth=1)],
        trap="a 20% fee publishes as v7 while the agent reads the customer; refunding in full under v6",
    )


def _a2() -> Case:
    c = Company(policy_v=6)
    c.customer("cus_pk81", "Pavel Kral", "pavel.kral@seznam.example", "pm_pk_mc")
    c.charge("ch_4PkL1", "cus_pk81", 11200, 11, "Hiking poles", category="sport")
    c.ticket(
        "T-6261",
        "cus_pk81",
        "Poles return",
        "One pole will not lock. A friend said your returns policy was updated; please check and return the "
        "112.00 from ch_4PkL1.",
        [{"type": "refund", "charge_id": "ch_4PkL1", "amount_cents": 11200}],
    )
    return make_case(
        _ID,
        "a2",
        "audit",
        c,
        "T-6261",
        Expect(
            ticket_status={"T-6261": "solved"},
            refunds={"ch_4PkL1": {"total_cents": 11200, "count": 1, "status": "succeeded"}},
            note_tokens={"T-6261": ["refund-policy@v6"]},
            mail_required=[{"template": "refund_issued", "reference_id": "ch_4PkL1", "amount_cents": 11200}],
            mail_to="pavel.kral@seznam.example",
        ),
    )


def cases() -> list[Case]:
    return [_v1(), _v2(), _v3(), _v4(), _v5(), _v6(), _a1(), _a2()]
