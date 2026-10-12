"""A deterministic scripted agent for the demo scenario. It calls no model.

Each turn is computed from the transcript alone: the plan below is replayed from the start against the tool
results already in the conversation, and the next call it yields is the turn. The same transcript always
gives the same turn, so two runs of one harness on one case are identical by construction.

The plan handles refund tickets the careless way a hurried agent would: it reads the ticket, the customer
and the charge, refunds, and on a timeout retries with a new idempotency key. When a guard blocks a refund
it reads the provider's refunds instead of retrying, and reports the status the provider shows.
"""

from __future__ import annotations

import json
import re
from collections.abc import Generator, Sequence
from typing import Any

from ratchet.agent.model import Message, ModelTurn, ToolSpec, scripted_turn

NAME = "scripted-demo-worker"
LABEL = "demo scenario, scripted agent (no model call)"

Call = tuple[str, dict[str, Any]]
Plan = Generator[Call, dict[str, Any], str]

_TICKET = re.compile(r"\bT-[0-9A-Z]+\b")


def _find(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for v in value.values():
            found = _find(v, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for v in value:
            found = _find(v, key)
            if found is not None:
                return found
    return None


def _plan(ticket_id: str) -> Plan:
    ticket = yield ("tickets_get", {"ticket_id": ticket_id})
    customer_id = _find(ticket, "customer_id")
    requests = [r for r in (_find(ticket, "requests") or []) if r.get("type") == "refund"]
    customer = yield ("customers_get", {"customer_id": customer_id})
    email = _find(customer, "email")
    status: dict[str, str] = {}
    for r in requests:
        charge, amount = r["charge_id"], r["amount_cents"]
        yield ("payments_get_charge", {"charge_id": charge})
        args = {"charge_id": charge, "amount_cents": amount, "reason": f"ticket {ticket_id}"}
        res = yield ("payments_refund", {**args, "idempotency_key": f"{ticket_id}-{charge}-1"})
        if not res.get("ok") and "blocked by guard" not in str(res.get("error")):
            res = yield ("payments_refund", {**args, "idempotency_key": f"{ticket_id}-{charge}-2"})
        status[charge] = "issued"
        if not res.get("ok") and "blocked by guard" in str(res.get("error")):
            listed = yield ("payments_list_refunds", {"charge_id": charge})
            rows = _find(listed, "refunds") or []
            if any(isinstance(x, dict) and x.get("status") == "pending" for x in rows):
                status[charge] = "pending"
    for r in requests:
        charge, amount = r["charge_id"], r["amount_cents"]
        yield ("mail_send", {
            "to": email, "template": f"refund_{status[charge]}", "reference_id": charge,
            "amount_cents": amount,
            "body": f"Your refund of {amount / 100:.2f} on {charge} is {status[charge]}.",
        })  # fmt: skip
    yield ("tickets_set_status", {"ticket_id": ticket_id, "status": "solved"})
    return f"Ticket {ticket_id} handled."


def _results(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "tool":
            try:
                out.append(json.loads(str(m.get("content"))))
            except json.JSONDecodeError:
                out.append({"ok": False, "error": str(m.get("content"))})
    return out


class ScriptedDemoWorker:
    """The scripted agent behind `ratchet demo double-refund`. It never calls a model."""

    @property
    def name(self) -> str:
        return NAME

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        text = " ".join(str(m.get("content") or "") for m in messages if m.get("role") in ("user", "system"))
        found = _TICKET.findall(text)
        if not found:
            return scripted_turn("No ticket id in the request.")
        results = _results(messages)
        plan = _plan(found[-1])
        try:
            call = next(plan)
            for res in results:
                call = plan.send(res)
        except StopIteration as stop:
            return scripted_turn(str(stop.value), id_prefix=f"t{len(results)}-")
        return scripted_turn(tool_calls=[call], id_prefix=f"t{len(results)}-")
