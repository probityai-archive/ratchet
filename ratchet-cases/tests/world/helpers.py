"""A small company for world tests."""

from __future__ import annotations

from typing import Any

from ratchet_cases.cases.build import Company, address
from ratchet_cases.types import ToolCall, ToolResult
from ratchet_cases.world import Fault, World


def company() -> Company:
    c = Company()
    c.customer("cus_a", "Ann", "ann@example.test", "pm_a")
    c.customer("cus_b", "Bob", "bob@example.test", "pm_b")
    c.charge("ch_a", "cus_a", 10000, 3, "Thing")
    c.subscription("sub_a", "cus_a", "Plan", 1000, 10)
    c.product("prod_x", "X", {"sku-x-1": ({"v": "1"}, 1000), "sku-x-2": ({"v": "2"}, 1500)})
    c.product("prod_y", "Y", {"sku-y-1": ({"v": "1"}, 500)})
    c.state["catalog"]["products"]["prod_y"]["variants"]["sku-y-1"]["available"] = False
    c.order("ord_a", "cus_a", "ch_o", [("prod_x", "sku-x-1")], address("1 A St", "Town", "11111"))
    c.order(
        "ord_s",
        "cus_a",
        "ch_s",
        [("prod_x", "sku-x-1")],
        address("1 A St", "Town", "11111"),
        status="shipped",
    )
    c.ticket("T-1", "cus_a", "Subject", "Body", [])
    c.approval("apr_1", "ch_a", 10000, expires_in_minutes=2)
    c.inbound("msg_1", "ann@example.test", "Hi", "Hello")
    c.account("acct_1", "Acme", {"B-1": 100})
    c.invoice("inv_1", "acct_1", 100)
    c.batch("B-1", [{"line": 1, "invoice_id": "inv_1", "amount_cents": 100}])
    return c


def world(faults: list[Fault] | None = None) -> World:
    return World(company().build(), faults)


class Caller:
    def __init__(self, w: World) -> None:
        self.w = w
        self.n = 0

    def __call__(self, name: str, actor: str = "agent", **args: Any) -> ToolResult:
        self.n += 1
        return self.w.call(ToolCall(name, args, f"t{self.n}"), actor)

    def ok(self, name: str, **args: Any) -> dict[str, Any]:
        result = self(name, **args)
        assert result.ok, result.error
        assert result.data is not None
        return result.data
