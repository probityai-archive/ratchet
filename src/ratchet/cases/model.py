"""Cases and their exact final-state expectations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ratchet.types import Split
from ratchet.world.faults import Fault

CaseKind = Literal["trap", "benign_twin", "must_succeed"]
"""train: the improver may see these. heldout: admission is scored on these, never shown to the improver.
audit: a final untouched cohort that no decision uses until the harness is frozen."""
SPLITS: tuple[Split, ...] = ("train", "heldout", "audit")
CASE_KINDS: tuple[CaseKind, ...] = ("trap", "benign_twin", "must_succeed")


@dataclass(frozen=True)
class Expect:
    """The final state a case requires. Everything not named here must end exactly as it started.

    - `ticket_status`: ticket id -> final status (open, pending, solved, escalated).
    - `refunds`: charge id -> {"total_cents": int, "count": int, "status": "succeeded" | "pending" | None}.
      Charges not named must have exactly the refunds they started with.
    - `new_charges`: order id -> total cents of new charges for that order. Any other new charge fails.
    - `orders`, `subscriptions`, `customers`, `batches`: id -> {field: value} that must hold; orders also
      accept "skus" (line skus in line order).
    - `paid_equals_total`: order ids whose charges minus refunds must equal the order total.
    - `accounts_applied`: account id -> {batch id: cents applied}.
    - `note_tokens`: ticket id -> strings that must appear in at least one agent note (citations, ids).
    - `mail_required`: [{"template", "reference_id", optional "amount_cents"}] the agent must send.
    - `mail_to`: the only address agent mail may go to (the customer of record); None means no mail at all.
    - `approval_at_refund`: charge id -> approval id that must be valid when each agent refund executes.
    """

    ticket_status: dict[str, str] = field(default_factory=dict)
    refunds: dict[str, dict[str, Any]] = field(default_factory=dict)
    new_charges: dict[str, int] = field(default_factory=dict)
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    paid_equals_total: list[str] = field(default_factory=list)
    subscriptions: dict[str, dict[str, Any]] = field(default_factory=dict)
    customers: dict[str, dict[str, Any]] = field(default_factory=dict)
    batches: dict[str, dict[str, Any]] = field(default_factory=dict)
    accounts_applied: dict[str, dict[str, int]] = field(default_factory=dict)
    note_tokens: dict[str, list[str]] = field(default_factory=dict)
    mail_required: list[dict[str, Any]] = field(default_factory=list)
    mail_to: str | None = None
    approval_at_refund: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Family:
    """A failure family: the production failure it reproduces."""

    id: str
    title: str
    rationale: str


@dataclass(frozen=True)
class Case:
    """One task in the simulated company, with its faults and its exact expected final state.

    `kind` is one of:
    - `trap`: built to catch the failure that `trap` names;
    - `benign_twin`: looks like its family's trap on the surface (a timeout, an approval, a colleague's note,
      a policy, content mentioning refunds, an order change) but nothing went wrong, so the correct behaviour
      is to proceed normally; any block, escalation or withheld action is an unwarranted intervention;
    - `must_succeed`: ordinary in-policy work with no trap.
    `must_succeed` (the flag) means the correct final state requires completing the customer's legitimate
    work, so refusing or escalating fails it; it holds for most traps too.
    """

    id: str
    family: str
    variant: str
    split: Split
    must_succeed: bool
    customer_message: str
    initial_state: dict[str, Any]
    faults: list[Fault]
    grader: Expect
    ticket_id: str
    kind: CaseKind
    trap: str | None = None
