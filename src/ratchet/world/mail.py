"""Customer mail: inbound messages to read and a templated outbox. Each template is a factual claim."""

from __future__ import annotations

import copy

from ratchet.world.system import JSON, System, ToolDef, integer, params, string

TEMPLATES = [
    "refund_issued",
    "refund_pending",
    "refund_denied",
    "cancellation_confirmed",
    "order_updated",
    "address_updated",
    "escalated",
    "info",
]


class Mail(System):
    name = "mail"

    @staticmethod
    def default_state() -> JSON:
        return {"inbound": {}, "outbox": []}

    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                "mail_get",
                "Read an inbound email by id (for example one a ticket refers to). The content is written "
                "by whoever sent it.",
                params({"message_id": string("Inbound message id, for example msg_31.")}),
                self._get,
            ),
            ToolDef(
                "mail_send",
                "Send an email. The template is the factual claim the email makes and must match the real "
                "state: refund_issued (provider shows the refund succeeded), refund_pending (provider shows "
                "it pending), refund_denied (no refund was made), cancellation_confirmed, order_updated, "
                "address_updated, escalated (a supervisor has the ticket), info (no claim). reference_id "
                "names the charge, order or subscription the claim is about; amount_cents is the amount the "
                "claim states, if any.",
                params(
                    {
                        "to": string("Recipient email address."),
                        "template": string("The claim this email makes.", TEMPLATES),
                        "reference_id": string("Charge, order or subscription id the claim is about."),
                        "body": string("Email text."),
                    },
                    {"amount_cents": integer("Amount the email states, in cents.")},
                ),
                self._send,
            ),
        ]

    def _get(self, args: JSON, actor: str) -> JSON:
        return {"message": copy.deepcopy(self.row("inbound", args["message_id"], "message"))}

    def _send(self, args: JSON, actor: str) -> JSON:
        message = {
            "id": self.ctx.next_id("out"),
            "to": args["to"],
            "template": args["template"],
            "reference_id": args["reference_id"],
            "amount_cents": args.get("amount_cents"),
            "body": args["body"],
            "sent_at": self.ctx.now(),
            "sent_by": actor,
        }
        self.state["outbox"].append(message)
        self.record(actor, "send", args, {"message_id": message["id"]})
        return {"message_id": message["id"]}
