"""Accounts receivable: remittance batches from the bank and how each payment was applied to accounts.

A batch is reconciled when, for every account, the amount applied from the batch equals the amount the bank
remitted for that account. Matching batch totals prove nothing about individual accounts.
"""

from __future__ import annotations

import copy

from ratchet.world.system import JSON, System, ToolDef, ToolError, integer, params, string


def applied_total(state: JSON, batch_id: str) -> int:
    return sum(acct["applied"].get(batch_id, 0) for acct in state["accounts"].values())


class Billing(System):
    name = "billing"

    @staticmethod
    def default_state() -> JSON:
        return {"batches": {}, "accounts": {}, "invoices": {}, "journals": [], "idempotency": {}}

    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                "billing_get_batch",
                "Read a remittance batch: status, total cents the bank received, total cents applied to "
                "accounts, and the accounts it touched.",
                params({"batch_id": string("Batch id, for example B-0317.")}),
                self._get_batch,
            ),
            ToolDef(
                "billing_get_remittance",
                "Read the bank's remittance lines for a batch: what was received, per line, with the "
                "invoice and (when the bank captured it) the account it was for.",
                params({"batch_id": string("Batch id.")}),
                self._get_remittance,
            ),
            ToolDef(
                "billing_get_account",
                "Read a receivable account: name and the cents applied to it from each batch.",
                params({"account_id": string("Account id, for example acct_204.")}),
                self._get_account,
            ),
            ToolDef(
                "billing_get_invoice",
                "Read an invoice: the account it belongs to and its amount.",
                params({"invoice_id": string("Invoice id, for example inv_9001.")}),
                self._get_invoice,
            ),
            ToolDef(
                "billing_post_journal",
                "Post a correcting journal entry that changes the cents applied to an account from a batch. "
                "Positive amount_cents applies more to the account, negative applies less. Pass an "
                "idempotency_key unique to this entry.",
                params(
                    {
                        "account_id": string("Account id."),
                        "batch_id": string("Batch id."),
                        "amount_cents": integer("Signed correction in cents."),
                        "memo": string("Why the correction is made."),
                        "idempotency_key": string("Caller-chosen key that identifies this entry."),
                    }
                ),
                self._post_journal,
            ),
            ToolDef(
                "billing_close_batch",
                "Mark a batch reconciled and closed.",
                params({"batch_id": string("Batch id.")}),
                self._close_batch,
            ),
        ]

    def _batch(self, batch_id: str) -> JSON:
        return self.row("batches", batch_id, "batch")

    def _get_batch(self, args: JSON, actor: str) -> JSON:
        batch = self._batch(args["batch_id"])
        accounts = sorted(a["id"] for a in self.state["accounts"].values() if batch["id"] in a["applied"])
        return {
            "batch": {
                "id": batch["id"],
                "status": batch["status"],
                "received_total_cents": sum(line["amount_cents"] for line in batch["remittance"]),
                "applied_total_cents": applied_total(self.state, batch["id"]),
                "account_ids": accounts,
            }
        }

    def _get_remittance(self, args: JSON, actor: str) -> JSON:
        batch = self._batch(args["batch_id"])
        return {"batch_id": batch["id"], "lines": copy.deepcopy(batch["remittance"])}

    def _get_account(self, args: JSON, actor: str) -> JSON:
        return {"account": copy.deepcopy(self.row("accounts", args["account_id"], "account"))}

    def _get_invoice(self, args: JSON, actor: str) -> JSON:
        return {"invoice": copy.deepcopy(self.row("invoices", args["invoice_id"], "invoice"))}

    def _post_journal(self, args: JSON, actor: str) -> JSON:
        prior = self.state["idempotency"].get(args["idempotency_key"])
        if prior is not None:
            entry = self.state["journals"][prior]
            same = (entry["account_id"], entry["batch_id"], entry["amount_cents"]) == (
                args["account_id"],
                args["batch_id"],
                args["amount_cents"],
            )
            if not same:
                raise ToolError("idempotency_key_reused: this key was already used for a different entry")
            return {"journal": copy.deepcopy(entry), "idempotent_replay": True}
        account = self.row("accounts", args["account_id"], "account")
        batch = self._batch(args["batch_id"])
        if batch["status"] != "open":
            raise ToolError("batch is closed")
        if args["amount_cents"] == 0:
            raise ToolError("amount_cents must not be zero")
        account["applied"][batch["id"]] = account["applied"].get(batch["id"], 0) + args["amount_cents"]
        entry = {
            "id": self.ctx.next_id("jr"),
            "account_id": account["id"],
            "batch_id": batch["id"],
            "amount_cents": args["amount_cents"],
            "memo": args["memo"],
            "posted_at": self.ctx.now(),
            "posted_by": actor,
        }
        self.state["idempotency"][args["idempotency_key"]] = len(self.state["journals"])
        self.state["journals"].append(entry)
        self.record(actor, "post_journal", args, {"journal_id": entry["id"]})
        return {"journal": copy.deepcopy(entry)}

    def _close_batch(self, args: JSON, actor: str) -> JSON:
        batch = self._batch(args["batch_id"])
        if batch["status"] != "open":
            raise ToolError("batch is already closed")
        batch["status"] = "closed"
        self.record(actor, "close_batch", args, {})
        return {"batch_id": batch["id"], "status": "closed"}
