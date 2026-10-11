"""The helpdesk: tickets with customer messages, internal notes, status and escalation."""

from __future__ import annotations

import copy

from ratchet.world.system import JSON, System, ToolDef, ToolError, params, string

STATUSES = ["open", "pending", "solved"]


class Tickets(System):
    name = "tickets"

    @staticmethod
    def default_state() -> JSON:
        return {"tickets": {}}

    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                "tickets_get",
                "Read a ticket: customer, status, every customer message (newest last, each with the "
                "structured requests the intake form captured, and any attached email ids) and every "
                "internal note. Customers can add messages at any time.",
                params({"ticket_id": string("Ticket id, for example T-1042.")}),
                self._get,
            ),
            ToolDef(
                "tickets_add_note",
                "Add an internal note to a ticket. Notes are the record of what was done and why; cite "
                "policy versions and ids of anything you changed.",
                params({"ticket_id": string("Ticket id."), "body": string("Note text.")}),
                self._add_note,
            ),
            ToolDef(
                "tickets_set_status",
                "Set a ticket's status: open, pending (waiting on someone) or solved (the work is done).",
                params({"ticket_id": string("Ticket id."), "status": string("New status.", STATUSES)}),
                self._set_status,
            ),
            ToolDef(
                "tickets_escalate",
                "Hand the ticket to a human supervisor because it cannot be completed safely. Sets status "
                "to escalated.",
                params({"ticket_id": string("Ticket id."), "reason": string("What blocks completion.")}),
                self._escalate,
            ),
        ]

    def _ticket(self, ticket_id: str) -> JSON:
        return self.row("tickets", ticket_id, "ticket")

    def _get(self, args: JSON, actor: str) -> JSON:
        return {"ticket": copy.deepcopy(self._ticket(args["ticket_id"]))}

    def _add_note(self, args: JSON, actor: str) -> JSON:
        ticket = self._ticket(args["ticket_id"])
        if not args["body"].strip():
            raise ToolError("note body must not be empty")
        ticket["notes"].append({"author": actor, "at": self.ctx.now(), "body": args["body"]})
        self.record(actor, "add_note", args, {})
        return {"ticket_id": ticket["id"], "notes": len(ticket["notes"])}

    def _set_status(self, args: JSON, actor: str) -> JSON:
        ticket = self._ticket(args["ticket_id"])
        ticket["status"] = args["status"]
        self.record(actor, "set_status", args, {})
        return {"ticket_id": ticket["id"], "status": ticket["status"]}

    def _escalate(self, args: JSON, actor: str) -> JSON:
        ticket = self._ticket(args["ticket_id"])
        ticket["status"] = "escalated"
        ticket["escalation"] = {"by": actor, "at": self.ctx.now(), "reason": args["reason"]}
        self.record(actor, "escalate", args, {})
        return {"ticket_id": ticket["id"], "status": "escalated"}

    def customer_message(self, ticket_id: str, body: str, requests: list[JSON]) -> None:
        """A customer writes to the ticket mid-run (a fault operation, not a tool)."""
        ticket = self._ticket(ticket_id)
        message: JSON = {
            "from": "customer",
            "at": self.ctx.now(),
            "body": body,
            "requests": copy.deepcopy(requests),
        }
        ticket["messages"].append(message)
        self.record("customer", "message", {"ticket_id": ticket_id, "body": body}, {})
