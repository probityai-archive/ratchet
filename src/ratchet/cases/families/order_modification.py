"""An order change where the customer adds one more change before the irreversible step."""

from __future__ import annotations

from typing import Any

from ratchet.cases.build import Company, address, make_case
from ratchet.cases.model import Case, Expect, Family
from ratchet.world.faults import Fault

FAMILY = Family(
    id="order_modification",
    title="Order modification with a late addition",
    rationale=(
        "Item changes to a pending order can be made once; the agent must collect "
        "every change before calling the tool, and the price difference must be settled exactly. Customers "
        "add changes mid-conversation, and an agent that acts on its first read locks the order with half "
        "the request and cannot apply the rest. Here a follow-up message lands on the ticket while the "
        "agent looks up the order or the catalog. The final items must reflect every change, and charges "
        "minus refunds must equal the new order total. Controls have a single change (one with a refund "
        "delta) or several changes stated up front, which must still go in one call. A benign twin mentions "
        "a second change and then withdraws it, with nothing pending."
    ),
)
_ID = FAMILY.id


def _shop(c: Company) -> None:
    c.product(
        "prod_tee",
        "Organic tee",
        {
            "sku-tee-blue-m": ({"color": "blue", "size": "M"}, 2400),
            "sku-tee-blue-l": ({"color": "blue", "size": "L"}, 2600),
            "sku-tee-red-m": ({"color": "red", "size": "M"}, 2400),
            "sku-tee-red-l": ({"color": "red", "size": "L"}, 2600),
        },
    )
    c.product(
        "prod_mug",
        "Stoneware mug",
        {
            "sku-mug-red": ({"glaze": "red"}, 1800),
            "sku-mug-blue": ({"glaze": "blue"}, 1900),
        },
    )


def _modify(order_id: str, *changes: tuple[str, str]) -> dict[str, Any]:
    return {
        "type": "modify_order",
        "order_id": order_id,
        "changes": [{"line_id": line, "new_sku": sku} for line, sku in changes],
    }


def _followup(ticket_id: str, body: str, requests: list[dict[str, Any]], tool: str) -> Fault:
    return Fault(
        "customer-followup",
        "mutate",
        tool=tool,
        nth=1,
        ops=[{"op": "customer_message", "ticket_id": ticket_id, "body": body, "requests": requests}],
        note="customer adds a change mid-run",
    )


def _v1() -> Case:
    c = Company()
    _shop(c)
    c.customer("cus_hw22", "Hannah Weber", "hannah.weber@web.example", "pm_hw_visa")
    c.order(
        "ord_hw_510",
        "cus_hw22",
        "ch_1HwS4",
        [("prod_tee", "sku-tee-blue-m"), ("prod_mug", "sku-mug-red")],
        address("Lindenstrasse 4", "Berlin", "10969", "DE"),
    )
    c.ticket(
        "T-1201",
        "cus_hw22",
        "Change tee size",
        "Please change the tee on order ord_hw_510 (line L1) from M to L, same blue.",
        [_modify("ord_hw_510", ("L1", "sku-tee-blue-l"))],
    )
    fault = _followup(
        "T-1201",
        "Sorry, one more thing before it goes out: please also make the mug (L2) blue.",
        [_modify("ord_hw_510", ("L2", "sku-mug-blue"))],
        "orders_get",
    )
    return make_case(
        _ID,
        "v1",
        "train",
        c,
        "T-1201",
        Expect(
            ticket_status={"T-1201": "solved"},
            orders={
                "ord_hw_510": {
                    "skus": ["sku-tee-blue-l", "sku-mug-blue"],
                    "status": "pending_modified",
                    "total_cents": 4500,
                }
            },
            new_charges={"ord_hw_510": 300},
            paid_equals_total=["ord_hw_510"],
            mail_required=[{"template": "order_updated", "reference_id": "ord_hw_510"}],
            mail_to="hannah.weber@web.example",
        ),
        faults=[fault],
        trap="customer adds a second change after the order is read; one-shot modify locks it out",
    )


def _v2() -> Case:
    c = Company()
    c.product(
        "prod_lamp",
        "Arc floor lamp",
        {
            "sku-lamp-brass": ({"finish": "brass"}, 8900),
            "sku-lamp-black": ({"finish": "black"}, 7900),
        },
    )
    c.customer("cus_ys40", "Yusuf Sahin", "yusuf.sahin@gmail.example", "pm_ys_mc")
    c.order(
        "ord_ys_330",
        "cus_ys40",
        "ch_8YsG2",
        [("prod_lamp", "sku-lamp-brass")],
        address("Bagdat Cd 210", "Istanbul", "34728", "TR"),
    )
    c.ticket(
        "T-1240",
        "cus_ys40",
        "Lamp finish",
        "On order ord_ys_330 I would prefer the black lamp instead of brass (line L1).",
        [_modify("ord_ys_330", ("L1", "sku-lamp-black"))],
    )
    return make_case(
        _ID,
        "v2",
        "train",
        c,
        "T-1240",
        Expect(
            ticket_status={"T-1240": "solved"},
            orders={
                "ord_ys_330": {"skus": ["sku-lamp-black"], "status": "pending_modified", "total_cents": 7900}
            },
            refunds={"ch_8YsG2": {"total_cents": 1000, "count": 1, "status": "succeeded"}},
            paid_equals_total=["ord_ys_330"],
            mail_required=[{"template": "order_updated", "reference_id": "ord_ys_330"}],
            mail_to="yusuf.sahin@gmail.example",
        ),
    )


def _v3() -> Case:
    c = Company()
    c.product(
        "prod_pack",
        "Daypack",
        {
            "sku-pack-grey-20l": ({"color": "grey", "volume": "20l"}, 6500),
            "sku-pack-grey-30l": ({"color": "grey", "volume": "30l"}, 7500),
        },
    )
    c.product(
        "prod_bottle",
        "Insulated bottle",
        {
            "sku-bottle-steel": ({"finish": "steel"}, 2200),
            "sku-bottle-black": ({"finish": "black"}, 2400),
        },
    )
    c.customer("cus_lm09", "Luis Moreno", "luis.moreno@telefonica.example", "pm_lm_visa")
    c.order(
        "ord_lm_918",
        "cus_lm09",
        "ch_6LmP0",
        [("prod_bottle", "sku-bottle-steel"), ("prod_pack", "sku-pack-grey-20l")],
        address("Calle Mayor 19", "Madrid", "28013", "ES"),
    )
    c.ticket(
        "T-1388",
        "cus_lm09",
        "Bigger daypack",
        "For order ord_lm_918, could you swap the 20 l daypack (line L2) for the 30 l one?",
        [_modify("ord_lm_918", ("L2", "sku-pack-grey-30l"))],
    )
    fault = _followup(
        "T-1388",
        "And the bottle (L1) in black please, if it is not too late.",
        [_modify("ord_lm_918", ("L1", "sku-bottle-black"))],
        "catalog_get_product",
    )
    return make_case(
        _ID,
        "v3",
        "heldout",
        c,
        "T-1388",
        Expect(
            ticket_status={"T-1388": "solved"},
            orders={
                "ord_lm_918": {
                    "skus": ["sku-bottle-black", "sku-pack-grey-30l"],
                    "status": "pending_modified",
                    "total_cents": 9900,
                }
            },
            new_charges={"ord_lm_918": 1200},
            paid_equals_total=["ord_lm_918"],
            mail_required=[{"template": "order_updated", "reference_id": "ord_lm_918"}],
            mail_to="luis.moreno@telefonica.example",
        ),
        faults=[fault],
        trap="the extra change arrives while the agent checks the catalog, on a different line",
    )


def _outfitters(c: Company) -> None:
    """A different shop for held-out and audit cases, so nothing about the training catalog carries over."""
    c.product(
        "prod_cap",
        "Field cap",
        {
            "sku-cap-navy-s": ({"color": "navy", "size": "S"}, 2100),
            "sku-cap-navy-l": ({"color": "navy", "size": "L"}, 2300),
            "sku-cap-olive-s": ({"color": "olive", "size": "S"}, 2100),
            "sku-cap-olive-l": ({"color": "olive", "size": "L"}, 2300),
        },
    )
    c.product(
        "prod_tumbler",
        "Travel tumbler",
        {
            "sku-tumbler-white": ({"finish": "white"}, 1650),
            "sku-tumbler-sage": ({"finish": "sage"}, 1750),
        },
    )
    c.product(
        "prod_chair",
        "Folding chair",
        {
            "sku-chair-walnut": ({"wood": "walnut"}, 12900),
            "sku-chair-ash": ({"wood": "ash"}, 11400),
        },
    )


def _v4() -> Case:
    c = Company()
    _outfitters(c)
    c.customer("cus_ao77", "Amara Obi", "amara.obi@yahoo.example", "pm_ao_visa")
    c.order(
        "ord_ao_603",
        "cus_ao77",
        "ch_2AoX5",
        [("prod_tumbler", "sku-tumbler-sage"), ("prod_cap", "sku-cap-olive-s")],
        address("14 Allen Ave", "Lagos", "100271", "NG"),
    )
    c.ticket(
        "T-1405",
        "cus_ao77",
        "Two changes",
        "On order ord_ao_603: the tumbler (L1) in white instead of sage, and the cap (L2) in L instead of S.",
        [_modify("ord_ao_603", ("L1", "sku-tumbler-white"), ("L2", "sku-cap-olive-l"))],
    )
    return make_case(
        _ID,
        "v4",
        "heldout",
        c,
        "T-1405",
        Expect(
            ticket_status={"T-1405": "solved"},
            orders={
                "ord_ao_603": {
                    "skus": ["sku-tumbler-white", "sku-cap-olive-l"],
                    "status": "pending_modified",
                    "total_cents": 3950,
                }
            },
            new_charges={"ord_ao_603": 100},
            paid_equals_total=["ord_ao_603"],
            mail_required=[{"template": "order_updated", "reference_id": "ord_ao_603"}],
            mail_to="amara.obi@yahoo.example",
        ),
        trap="two changes stated up front; applying them one call at a time locks the order after the first",
    )


def _v5() -> Case:
    c = Company()
    _outfitters(c)
    c.customer("cus_mk64", "Marta Kowalska", "marta.kowalska@wp.example", "pm_mk_mc")
    c.order(
        "ord_mk_745",
        "cus_mk64",
        "ch_4MkE7",
        [("prod_tumbler", "sku-tumbler-white"), ("prod_cap", "sku-cap-navy-s")],
        address("ul. Prosta 7", "Warsaw", "00-838", "PL"),
    )
    c.ticket(
        "T-1466",
        "cus_mk64",
        "Cap size",
        "Please switch the cap on order ord_mk_745 (line L2) to size L. I wondered about the tumbler colour "
        "too, but I will keep it white. That is everything.",
        [_modify("ord_mk_745", ("L2", "sku-cap-navy-l"))],
    )
    return make_case(
        _ID,
        "v5",
        "heldout",
        c,
        "T-1466",
        Expect(
            ticket_status={"T-1466": "solved"},
            orders={
                "ord_mk_745": {
                    "skus": ["sku-tumbler-white", "sku-cap-navy-l"],
                    "status": "pending_modified",
                    "total_cents": 3950,
                }
            },
            new_charges={"ord_mk_745": 200},
            paid_equals_total=["ord_mk_745"],
            mail_required=[{"template": "order_updated", "reference_id": "ord_mk_745"}],
            mail_to="marta.kowalska@wp.example",
        ),
    )


def _a1() -> Case:
    c = Company()
    _outfitters(c)
    c.customer("cus_ej05", "Emma Johansson", "emma.johansson@telia.example", "pm_ej_visa")
    c.order(
        "ord_ej_877",
        "cus_ej05",
        "ch_7EjN3",
        [("prod_cap", "sku-cap-olive-s"), ("prod_chair", "sku-chair-walnut")],
        address("Drottninggatan 50", "Stockholm", "11121", "SE"),
    )
    c.ticket(
        "T-1520",
        "cus_ej05",
        "Chair wood",
        "Please make the folding chair on order ord_ej_877 (line L2) ash instead of walnut.",
        [_modify("ord_ej_877", ("L2", "sku-chair-ash"))],
    )
    fault = _followup(
        "T-1520",
        "Also the cap (L1) in L please, I just checked the size chart.",
        [_modify("ord_ej_877", ("L1", "sku-cap-olive-l"))],
        "customers_get",
    )
    return make_case(
        _ID,
        "a1",
        "audit",
        c,
        "T-1520",
        Expect(
            ticket_status={"T-1520": "solved"},
            orders={
                "ord_ej_877": {
                    "skus": ["sku-cap-olive-l", "sku-chair-ash"],
                    "status": "pending_modified",
                    "total_cents": 13700,
                }
            },
            refunds={"ch_7EjN3": {"total_cents": 1300, "count": 1, "status": "succeeded"}},
            paid_equals_total=["ord_ej_877"],
            mail_required=[{"template": "order_updated", "reference_id": "ord_ej_877"}],
            mail_to="emma.johansson@telia.example",
        ),
        faults=[fault],
        trap="the extra change lands early, while the agent reads the customer; net delta is a refund",
    )


def _a2() -> Case:
    c = Company()
    _outfitters(c)
    c.customer("cus_ns18", "Nikos Stavrou", "nikos.stavrou@otenet.example", "pm_ns_mc")
    c.order(
        "ord_ns_391",
        "cus_ns18",
        "ch_9NsK6",
        [("prod_cap", "sku-cap-navy-l"), ("prod_tumbler", "sku-tumbler-sage")],
        address("Ermou 22", "Athens", "10563", "GR"),
    )
    c.ticket(
        "T-1544",
        "cus_ns18",
        "Cap colour",
        "On order ord_ns_391, please switch the cap (L1) from navy to olive, same size L.",
        [_modify("ord_ns_391", ("L1", "sku-cap-olive-l"))],
    )
    fault = _followup("T-1544", "Thanks! Nothing else to change, the tumbler is perfect.", [], "orders_get")
    return make_case(
        _ID,
        "a2",
        "audit",
        c,
        "T-1544",
        Expect(
            ticket_status={"T-1544": "solved"},
            orders={
                "ord_ns_391": {
                    "skus": ["sku-cap-olive-l", "sku-tumbler-sage"],
                    "status": "pending_modified",
                    "total_cents": 4050,
                }
            },
            paid_equals_total=["ord_ns_391"],
            mail_required=[{"template": "order_updated", "reference_id": "ord_ns_391"}],
            mail_to="nikos.stavrou@otenet.example",
        ),
        faults=[fault],
    )


def cases() -> list[Case]:
    return [_v1(), _v2(), _v3(), _v4(), _v5(), _a1(), _a2()]
