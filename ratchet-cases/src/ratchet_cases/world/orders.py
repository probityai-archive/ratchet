"""Orders: line items, shipping address and status. Item changes are allowed once, while pending.

`orders_modify_items` may be called only once per order: after it the order is
`pending_modified` and its items are locked. The tool never moves money; the price difference must be
settled at the payment provider.
"""

from __future__ import annotations

import copy

from ratchet_cases.world.catalog import Catalog
from ratchet_cases.world.system import JSON, Context, System, ToolDef, ToolError, array, obj, params, string

_ADDRESS = obj(
    {
        "line1": string("Street and number."),
        "city": string("City."),
        "postal_code": string("Postal code."),
        "country": string("Two-letter country code."),
    },
    ["line1", "city", "postal_code", "country"],
    "A shipping address.",
)
_CHANGE = obj(
    {"line_id": string("Order line to change."), "new_sku": string("Variant sku to switch the line to.")},
    ["line_id", "new_sku"],
)
_OPEN = ("pending", "pending_modified")


class Orders(System):
    name = "orders"

    def __init__(self, state: JSON, ctx: Context, catalog: Catalog) -> None:
        super().__init__(state, ctx)
        self.catalog = catalog

    @staticmethod
    def default_state() -> JSON:
        return {"orders": {}}

    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                "orders_get",
                "Read one order: status, lines (sku, product, price), total, shipping address, the charge "
                "that paid for it, and the gift message if any.",
                params({"order_id": string("Order id, for example ord_5521.")}),
                self._get,
            ),
            ToolDef(
                "orders_list",
                "List a customer's orders.",
                params({"customer_id": string("Customer id.")}),
                self._list,
            ),
            ToolDef(
                "orders_modify_items",
                "Switch one or more lines of a pending order to other variants of the same product. This "
                "can be done ONCE per order: afterwards the items are locked, so include every change in "
                "one call. Returns the new total and price_difference_cents (positive: customer owes more; "
                "negative: customer is owed). Does not move money.",
                params(
                    {
                        "order_id": string("Order id."),
                        "changes": array("Every line change, all in this one call.", _CHANGE),
                    }
                ),
                self._modify_items,
            ),
            ToolDef(
                "orders_update_address",
                "Change the shipping address of an order that has not shipped.",
                params({"order_id": string("Order id."), "address": _ADDRESS}),
                self._update_address,
            ),
            ToolDef(
                "orders_cancel",
                "Cancel an order that has not shipped. Cannot be undone. Does not move money.",
                params({"order_id": string("Order id."), "reason": string("Why the order is cancelled.")}),
                self._cancel,
            ),
        ]

    def _order(self, order_id: str) -> JSON:
        return self.row("orders", order_id, "order")

    def _get(self, args: JSON, actor: str) -> JSON:
        return {"order": copy.deepcopy(self._order(args["order_id"]))}

    def _list(self, args: JSON, actor: str) -> JSON:
        rows = [
            copy.deepcopy(o) for o in self.state["orders"].values() if o["customer_id"] == args["customer_id"]
        ]
        return {"orders": sorted(rows, key=lambda o: o["id"])}

    def _modify_items(self, args: JSON, actor: str) -> JSON:
        order = self._order(args["order_id"])
        if order["status"] == "pending_modified":
            raise ToolError("order items were already modified once and are now locked")
        if order["status"] != "pending":
            raise ToolError(f"order is {order['status']}; only pending orders can be modified")
        if not args["changes"]:
            raise ToolError("changes must not be empty")
        lines = {line["line_id"]: line for line in order["lines"]}
        updates: list[tuple[JSON, str, int]] = []
        for change in args["changes"]:
            line = lines.get(change["line_id"])
            if line is None:
                raise ToolError(f"order line not found: {change['line_id']}")
            found = self.catalog.variant(change["new_sku"])
            if found is None:
                raise ToolError(f"sku not found: {change['new_sku']}")
            product_id, variant = found
            if product_id != line["product_id"]:
                raise ToolError(f"{change['new_sku']} is not a variant of {line['product_id']}")
            if not variant["available"]:
                raise ToolError(f"{change['new_sku']} is not available")
            updates.append((line, change["new_sku"], variant["price_cents"]))
        old_total = order["total_cents"]
        for line, sku, price in updates:
            line["sku"] = sku
            line["price_cents"] = price
        order["total_cents"] = sum(line["price_cents"] for line in order["lines"])
        order["status"] = "pending_modified"
        difference = order["total_cents"] - old_total
        self.record(actor, "modify_items", args, {"price_difference_cents": difference})
        return {"order": copy.deepcopy(order), "price_difference_cents": difference}

    def _update_address(self, args: JSON, actor: str) -> JSON:
        order = self._order(args["order_id"])
        if order["status"] not in _OPEN:
            raise ToolError(f"order is {order['status']}; the address can no longer change")
        order["address"] = dict(args["address"])
        self.record(actor, "update_address", args, {})
        return {"order": copy.deepcopy(order)}

    def _cancel(self, args: JSON, actor: str) -> JSON:
        order = self._order(args["order_id"])
        if order["status"] not in _OPEN:
            raise ToolError(f"order is {order['status']}; it can no longer be cancelled")
        order["status"] = "cancelled"
        order["cancel_reason"] = args["reason"]
        self.record(actor, "cancel", args, {})
        return {"order": copy.deepcopy(order)}
