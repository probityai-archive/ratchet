from __future__ import annotations

from ratchet.cases.build import address

from .helpers import Caller, world


def _refund(call: Caller, amount: int, key: str) -> object:
    return call("payments_refund", charge_id="ch_a", amount_cents=amount, idempotency_key=key, reason="r")


def test_payments_refund_rules_and_idempotency() -> None:
    call = Caller(world())
    first = call.ok("payments_refund", charge_id="ch_a", amount_cents=4000, idempotency_key="k", reason="r")
    again = call.ok("payments_refund", charge_id="ch_a", amount_cents=4000, idempotency_key="k", reason="r")
    assert again["idempotent_replay"] and again["refund"]["id"] == first["refund"]["id"]
    assert call(
        "payments_refund", charge_id="ch_a", amount_cents=10, idempotency_key="k", reason="r"
    ).error == ("idempotency_key_reused: this key was already used for a different request")
    assert call(
        "payments_refund", charge_id="ch_a", amount_cents=0, idempotency_key="z", reason="r"
    ).error == ("amount_cents must be positive")
    assert call(
        "payments_refund", charge_id="ch_a", amount_cents=6001, idempotency_key="y", reason="r"
    ).error == ("amount_exceeds_refundable: 6000 cents remain refundable")
    assert call("payments_get_charge", charge_id="ch_nope").error == "charge not found: ch_nope"
    assert call.ok("payments_get_charge", charge_id="ch_a")["charge"]["refunded_cents"] == 4000
    assert [c["id"] for c in call.ok("payments_list_charges", customer_id="cus_a")["charges"]] == [
        "ch_a",
        "ch_o",
        "ch_s",
    ]


def test_payments_charge() -> None:
    call = Caller(world())
    args = {
        "customer_id": "cus_a",
        "payment_method_id": "pm_a",
        "amount_cents": 500,
        "idempotency_key": "c",
        "description": "diff",
        "order_id": "ord_a",
    }
    made = call.ok("payments_charge", **args)["charge"]
    assert made["order_id"] == "ord_a" and made["refunded_cents"] == 0
    assert call.ok("payments_charge", **args)["idempotent_replay"]
    assert call("payments_charge", **{**args, "idempotency_key": "d", "payment_method_id": "pm_b"}).error == (
        "payment method does not belong to this customer"
    )
    assert call("payments_charge", **{**args, "idempotency_key": "e", "amount_cents": -1}).error == (
        "amount_cents must be positive"
    )


def test_subscriptions() -> None:
    call = Caller(world())
    assert call.ok("subscriptions_list", customer_id="cus_a")["subscriptions"][0]["id"] == "sub_a"
    sub = call.ok("subscriptions_cancel", subscription_id="sub_a", at_period_end=False)["subscription"]
    assert sub["status"] == "cancelled"
    assert call("subscriptions_cancel", subscription_id="sub_a", at_period_end=True).error == (
        "subscription is already cancelled"
    )
    assert call.ok("subscriptions_get", subscription_id="sub_a")["subscription"]["cancelled_at"]


def test_orders_modify_once_and_validation() -> None:
    call = Caller(world())

    def modify(*changes: tuple[str, str], order: str = "ord_a") -> object:
        return call(
            "orders_modify_items",
            order_id=order,
            changes=[{"line_id": line, "new_sku": sku} for line, sku in changes],
        ).error

    assert modify() == "changes must not be empty"
    assert modify(("L9", "sku-x-2")) == "order line not found: L9"
    assert modify(("L1", "sku-nope")) == "sku not found: sku-nope"
    assert modify(("L1", "sku-y-1")) == "sku-y-1 is not a variant of prod_x"
    assert modify(("L1", "sku-x-2"), order="ord_s") == "order is shipped; only pending orders can be modified"
    done = call.ok("orders_modify_items", order_id="ord_a", changes=[{"line_id": "L1", "new_sku": "sku-x-2"}])
    assert done["price_difference_cents"] == 500 and done["order"]["status"] == "pending_modified"
    assert modify(("L1", "sku-x-1")) == "order items were already modified once and are now locked"
    assert [o["id"] for o in call.ok("orders_list", customer_id="cus_a")["orders"]] == ["ord_a", "ord_s"]


def test_orders_unavailable_variant() -> None:
    w = world()
    w._state["catalog"]["products"]["prod_x"]["variants"]["sku-x-2"]["available"] = False
    call = Caller(w)
    assert call(
        "orders_modify_items", order_id="ord_a", changes=[{"line_id": "L1", "new_sku": "sku-x-2"}]
    ).error == ("sku-x-2 is not available")


def test_orders_address_and_cancel() -> None:
    call = Caller(world())
    new = address("2 B St", "City", "22222")
    assert call.ok("orders_update_address", order_id="ord_a", address=new)["order"]["address"] == new
    assert call("orders_update_address", order_id="ord_s", address=new).error == (
        "order is shipped; the address can no longer change"
    )
    assert (
        call("orders_cancel", order_id="ord_s", reason="r").error
        == "order is shipped; it can no longer be cancelled"
    )
    assert call.ok("orders_cancel", order_id="ord_a", reason="r")["order"]["status"] == "cancelled"
    assert call.ok("catalog_get_product", product_id="prod_x")["product"]["name"] == "X"


def test_tickets() -> None:
    call = Caller(world())
    assert call("tickets_add_note", ticket_id="T-1", body="  ").error == "note body must not be empty"
    assert call.ok("tickets_add_note", ticket_id="T-1", body="n")["notes"] == 1
    assert call.ok("tickets_set_status", ticket_id="T-1", status="pending")["status"] == "pending"
    assert call.ok("tickets_escalate", ticket_id="T-1", reason="why")["status"] == "escalated"
    ticket = call.ok("tickets_get", ticket_id="T-1")["ticket"]
    assert ticket["escalation"]["reason"] == "why"
    assert call("tickets_get", ticket_id="T-9").error == "ticket not found: T-9"


def test_approvals_expire_with_the_clock() -> None:
    call = Caller(world())
    assert call.ok("approvals_list", subject_id="ch_a")["approvals"][0]["status"] == "granted"
    call.ok("tickets_get", ticket_id="T-1")
    assert call.ok("approvals_get", approval_id="apr_1")["approval"]["status"] == "expired"


def test_policy_without_a_version_in_force() -> None:
    w = world()
    w._state["policy"]["docs"]["future"] = {
        "versions": [{"version": 1, "published_at": "2099-01-01T00:00:00Z"}]
    }
    assert Caller(w)("policy_get", doc_id="future").error == "no version of future is in force"


def test_mail_and_customers() -> None:
    call = Caller(world())
    assert call.ok("mail_get", message_id="msg_1")["message"]["body"] == "Hello"
    sent = call.ok("mail_send", to="ann@example.test", template="info", reference_id="T-1", body="b")
    assert sent["message_id"] == "out_sim0001"
    assert call("customers_update_email", customer_id="cus_a", email="nope").error == "invalid email address"
    assert (
        call.ok("customers_update_email", customer_id="cus_a", email="a@b.test")["customer"]["email"]
        == "a@b.test"
    )
    assert call.ok("customers_get", customer_id="cus_a")["customer"]["email"] == "a@b.test"


def test_billing() -> None:
    call = Caller(world())
    batch = call.ok("billing_get_batch", batch_id="B-1")["batch"]
    assert batch["received_total_cents"] == 100 == batch["applied_total_cents"] and batch["account_ids"] == [
        "acct_1"
    ]
    assert call.ok("billing_get_remittance", batch_id="B-1")["lines"][0]["invoice_id"] == "inv_1"
    assert call.ok("billing_get_invoice", invoice_id="inv_1")["invoice"]["account_id"] == "acct_1"
    args = {"account_id": "acct_1", "batch_id": "B-1", "amount_cents": 5, "memo": "m", "idempotency_key": "j"}
    assert call("billing_post_journal", **{**args, "amount_cents": 0, "idempotency_key": "z"}).error == (
        "amount_cents must not be zero"
    )
    call.ok("billing_post_journal", **args)
    assert call.ok("billing_post_journal", **args)["idempotent_replay"]
    assert call("billing_post_journal", **{**args, "amount_cents": 6}).error == (
        "idempotency_key_reused: this key was already used for a different entry"
    )
    assert call.ok("billing_get_account", account_id="acct_1")["account"]["applied"]["B-1"] == 105
    call.ok("billing_close_batch", batch_id="B-1")
    assert call("billing_close_batch", batch_id="B-1").error == "batch is already closed"
    assert call("billing_post_journal", **{**args, "idempotency_key": "q"}).error == "batch is closed"
