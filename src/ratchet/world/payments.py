"""The payment provider: charges and refunds, with idempotency keys.

The provider knows nothing about tickets, approvals or policy. A refund it accepts is final. A fault can make
a call commit and still return a timeout, which is exactly how real provider integrations double-refund.
"""

from __future__ import annotations

import copy

from ratchet.world.system import JSON, System, ToolDef, ToolError, integer, params, string


def refunded_cents(state: JSON, charge_id: str) -> int:
    """Cents refunded on a charge, pending refunds included (they cannot be refunded twice either)."""
    return sum(r["amount_cents"] for r in state["refunds"].values() if r["charge_id"] == charge_id)


class Payments(System):
    name = "payments"

    @staticmethod
    def default_state() -> JSON:
        return {"charges": {}, "refunds": {}, "payment_methods": {}, "idempotency": {}}

    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                "payments_get_charge",
                "Read one charge at the payment provider: amount, currency, creation time, description, the "
                "order it paid for, and how much has been refunded so far.",
                params({"charge_id": string("Charge id, for example ch_3Pz.")}),
                self._get_charge,
            ),
            ToolDef(
                "payments_list_charges",
                "List every charge for a customer at the payment provider, oldest first.",
                params({"customer_id": string("Customer id.")}),
                self._list_charges,
            ),
            ToolDef(
                "payments_list_refunds",
                "List every refund on a charge as the provider records it, with status (succeeded or "
                "pending), amount, idempotency key and who issued it. This is the source of truth for what "
                "was refunded.",
                params({"charge_id": string("Charge id.")}),
                self._list_refunds,
            ),
            ToolDef(
                "payments_refund",
                "Refund part or all of a charge. Money leaves immediately and cannot be recalled. Pass an "
                "idempotency_key unique to this refund: repeating a call with the same key and the same "
                "arguments returns the original refund instead of paying again. A timeout does not mean the "
                "refund failed; check payments_list_refunds before retrying.",
                params(
                    {
                        "charge_id": string("Charge to refund."),
                        "amount_cents": integer("Amount to refund, in cents."),
                        "idempotency_key": string("Caller-chosen key that identifies this refund."),
                        "reason": string("Why the refund is issued."),
                    }
                ),
                self._refund,
            ),
            ToolDef(
                "payments_charge",
                "Charge a customer's stored payment method, for example to collect a price difference on an "
                "order. Pass an idempotency_key unique to this charge.",
                params(
                    {
                        "customer_id": string("Customer to charge."),
                        "payment_method_id": string("Stored payment method id, for example pm_visa_4242."),
                        "amount_cents": integer("Amount in cents."),
                        "idempotency_key": string("Caller-chosen key that identifies this charge."),
                        "description": string("What the charge is for."),
                    },
                    {"order_id": string("Order the charge belongs to, if any.")},
                ),
                self._charge,
            ),
        ]

    def _charge_view(self, charge: JSON) -> JSON:
        view = copy.deepcopy(charge)
        view["refunded_cents"] = refunded_cents(self.state, charge["id"])
        return view

    def _get_charge(self, args: JSON, actor: str) -> JSON:
        return {"charge": self._charge_view(self.row("charges", args["charge_id"], "charge"))}

    def _list_charges(self, args: JSON, actor: str) -> JSON:
        rows = [c for c in self.state["charges"].values() if c["customer_id"] == args["customer_id"]]
        rows.sort(key=lambda c: (c["created_at"], c["id"]))
        return {"charges": [self._charge_view(c) for c in rows]}

    def _list_refunds(self, args: JSON, actor: str) -> JSON:
        self.row("charges", args["charge_id"], "charge")
        rows = [
            copy.deepcopy(r) for r in self.state["refunds"].values() if r["charge_id"] == args["charge_id"]
        ]
        rows.sort(key=lambda r: (r["created_at"], r["id"]))
        return {
            "charge_id": args["charge_id"],
            "refunds": rows,
            "refunded_cents": refunded_cents(self.state, args["charge_id"]),
        }

    def _replay(self, key: str, fingerprint: JSON, table: str, label: str) -> JSON | None:
        prior = self.state["idempotency"].get(key)
        if prior is None:
            return None
        if prior["fingerprint"] != fingerprint:
            raise ToolError("idempotency_key_reused: this key was already used for a different request")
        return {label: copy.deepcopy(self.state[table][prior["id"]]), "idempotent_replay": True}

    def _refund(self, args: JSON, actor: str) -> JSON:
        fingerprint = {"op": "refund", "charge_id": args["charge_id"], "amount_cents": args["amount_cents"]}
        replay = self._replay(args["idempotency_key"], fingerprint, "refunds", "refund")
        if replay is not None:
            return replay
        charge = self.row("charges", args["charge_id"], "charge")
        amount = args["amount_cents"]
        if amount <= 0:
            raise ToolError("amount_cents must be positive")
        refundable = charge["amount_cents"] - refunded_cents(self.state, charge["id"])
        if amount > refundable:
            raise ToolError(f"amount_exceeds_refundable: {refundable} cents remain refundable")
        refund_id = self.ctx.next_id("re")
        refund = {
            "id": refund_id,
            "charge_id": charge["id"],
            "amount_cents": amount,
            "status": "pending" if charge.get("settlement") == "pending" else "succeeded",
            "idempotency_key": args["idempotency_key"],
            "reason": args["reason"],
            "created_at": self.ctx.now(),
            "issued_by": actor,
        }
        self.state["refunds"][refund_id] = refund
        self.state["idempotency"][args["idempotency_key"]] = {"fingerprint": fingerprint, "id": refund_id}
        self.record(actor, "refund", args, {"refund_id": refund_id})
        return {"refund": copy.deepcopy(refund)}

    def _charge(self, args: JSON, actor: str) -> JSON:
        fingerprint = {
            "op": "charge",
            "customer_id": args["customer_id"],
            "payment_method_id": args["payment_method_id"],
            "amount_cents": args["amount_cents"],
        }
        replay = self._replay(args["idempotency_key"], fingerprint, "charges", "charge")
        if replay is not None:
            return replay
        method = self.row("payment_methods", args["payment_method_id"], "payment method")
        if method["customer_id"] != args["customer_id"]:
            raise ToolError("payment method does not belong to this customer")
        if args["amount_cents"] <= 0:
            raise ToolError("amount_cents must be positive")
        charge_id = self.ctx.next_id("ch")
        charge = {
            "id": charge_id,
            "customer_id": args["customer_id"],
            "payment_method_id": args["payment_method_id"],
            "amount_cents": args["amount_cents"],
            "currency": "usd",
            "created_at": self.ctx.now(),
            "description": args["description"],
            "order_id": args.get("order_id"),
            "category": "adjustment",
            "settlement": "instant",
            "issued_by": actor,
        }
        self.state["charges"][charge_id] = charge
        self.state["idempotency"][args["idempotency_key"]] = {"fingerprint": fingerprint, "id": charge_id}
        self.record(actor, "charge", args, {"charge_id": charge_id})
        return {"charge": self._charge_view(charge)}
