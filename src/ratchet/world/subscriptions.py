"""Subscriptions: plans that renew each period until cancelled."""

from __future__ import annotations

import copy

from ratchet.world.system import JSON, System, ToolDef, ToolError, boolean, params, string


class Subscriptions(System):
    name = "subscriptions"

    @staticmethod
    def default_state() -> JSON:
        return {"subscriptions": {}}

    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                "subscriptions_get",
                "Read one subscription: plan, price, status, current period end and whether it is set to "
                "cancel at period end.",
                params({"subscription_id": string("Subscription id, for example sub_Q1.")}),
                self._get,
            ),
            ToolDef(
                "subscriptions_list",
                "List a customer's subscriptions.",
                params({"customer_id": string("Customer id.")}),
                self._list,
            ),
            ToolDef(
                "subscriptions_cancel",
                "Cancel a subscription. With at_period_end=true it stays active until the current period "
                "ends and then stops renewing; with false it ends now. Cancellation cannot be undone.",
                params(
                    {
                        "subscription_id": string("Subscription id."),
                        "at_period_end": boolean(
                            "True to stop at the end of the paid period, false to end now."
                        ),
                    }
                ),
                self._cancel,
            ),
        ]

    def _get(self, args: JSON, actor: str) -> JSON:
        sub = self.row("subscriptions", args["subscription_id"], "subscription")
        return {"subscription": copy.deepcopy(sub)}

    def _list(self, args: JSON, actor: str) -> JSON:
        rows = [
            copy.deepcopy(s)
            for s in self.state["subscriptions"].values()
            if s["customer_id"] == args["customer_id"]
        ]
        return {"subscriptions": sorted(rows, key=lambda s: s["id"])}

    def _cancel(self, args: JSON, actor: str) -> JSON:
        sub = self.row("subscriptions", args["subscription_id"], "subscription")
        if sub["status"] == "cancelled" or sub["cancel_at_period_end"]:
            raise ToolError("subscription is already cancelled")
        if args["at_period_end"]:
            sub["cancel_at_period_end"] = True
        else:
            sub["status"] = "cancelled"
            sub["cancelled_at"] = self.ctx.now()
        self.record(actor, "cancel", args, {"status": sub["status"]})
        return {"subscription": copy.deepcopy(sub)}
