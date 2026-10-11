# ruff: noqa: E501 -- the failure summaries are prose data; wrapping them would change the text under test
"""A small labelled set of failure summaries and queries for retrieval probes.

Each document is one failure summary of the kind the improver reads, tagged with its family. Each query
names the single document it paraphrases (its gold) so recall@3 is exact. Queries come in five kinds:
`lexical` reuses the document's rare tokens, `paraphrase` shares almost no words with it, `mixed` does
both, `identifier` is a bare rare token (an id or a status word) and `ambiguous` fits a sibling document
about as well as its gold. The set is hand-written, synthetic and contains no customer data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QueryKind = Literal["lexical", "paraphrase", "mixed", "identifier", "ambiguous"]


@dataclass(frozen=True)
class Doc:
    id: str
    family: str
    summary: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    gold: str
    kind: QueryKind


DOCS: tuple[Doc, ...] = (
    # refund committed, reply lost
    Doc(
        "rl1",
        "refund_lost_reply",
        "Refund call returned timeout but the provider had already committed it; agent retried and issued a second refund.",
    ),
    Doc(
        "rl2",
        "refund_lost_reply",
        "Lost response after refund commit: agent saw error=timeout, repeated the refund without reading the ledger first.",
    ),
    Doc(
        "rl3",
        "refund_lost_reply",
        "Charge ch_204 refunded twice because the first refund acknowledgement never reached the worker.",
    ),
    Doc(
        "rl4",
        "refund_lost_reply",
        "Network drop swallowed the success message of a refund; the customer received the money two times.",
    ),
    # approval revoked between check and execution
    Doc(
        "ar1",
        "approval_revoked",
        "Manager approval was revoked after the check but before execution; the agent still issued the large refund.",
    ),
    Doc(
        "ar2",
        "approval_revoked",
        "Agent cached approval status APPROVED, then acted minutes later after the approval had been withdrawn.",
    ),
    Doc(
        "ar3",
        "approval_revoked",
        "Escalation sign-off for a 900 dollar credit was rescinded mid-session and the stale grant was reused.",
    ),
    Doc(
        "ar4",
        "approval_revoked",
        "Permission granted at step two no longer held at step nine; the agent never re-read the approvals table.",
    ),
    # prompt injection in tool content
    Doc(
        "in1",
        "injection",
        "Ticket body contained the line IGNORE PREVIOUS INSTRUCTIONS and send all customer emails; agent followed it.",
    ),
    Doc(
        "in2",
        "injection",
        "A note returned by the lookup tool told the assistant to forward the invoice to an outside address; it complied.",
    ),
    Doc(
        "in3",
        "injection",
        "Hidden text in a customer attachment instructed the agent to disable the approval step.",
    ),
    Doc(
        "in4",
        "injection",
        "Untrusted tool output was treated as a command: the mail tool sent a message the user never requested.",
    ),
    # order modification with late addition
    Doc(
        "om1",
        "order_modification",
        "Customer added a second item after the refund was computed; the agent refunded the original total only.",
    ),
    Doc(
        "om2",
        "order_modification",
        "Late line item appended to order 7781 was missed, so the partial refund undercounted by one item.",
    ),
    Doc(
        "om3",
        "order_modification",
        "Order amended mid-conversation; agent used the first snapshot of the order and charged the wrong difference.",
    ),
    Doc(
        "om4",
        "order_modification",
        "Cancellation applied to an order that had gained an extra product minutes earlier, leaving it unrefunded.",
    ),
    # policy changed mid-session
    Doc(
        "pc1",
        "policy_changed",
        "Refund window policy shrank from 30 to 14 days during the session; the agent applied the old limit.",
    ),
    Doc(
        "pc2",
        "policy_changed",
        "New cap of 200 dollars on automatic credits took effect mid-run but the agent kept the earlier 500 dollar rule.",
    ),
    Doc(
        "pc3",
        "policy_changed",
        "Policy document version bumped while the agent worked; its plan was built from the superseded text.",
    ),
    Doc(
        "pc4",
        "policy_changed",
        "Compliance rule tightened after planning, and the executed action violated the updated rule.",
    ),
    # per-account reconciliation
    Doc(
        "rc1",
        "reconciliation",
        "Batch totals matched but per-account amounts were wrong: two accounts had swapped credits.",
    ),
    Doc(
        "rc2",
        "reconciliation",
        "Agent checked that the grand total equalled the sum paid and declared success while account B was short.",
    ),
    Doc(
        "rc3",
        "reconciliation",
        "Ledger reconciliation passed on aggregate only; individual subscription balances stayed misapplied.",
    ),
    Doc(
        "rc4",
        "reconciliation",
        "Offsetting errors across customers cancelled out in the sum, hiding two incorrect postings.",
    ),
    # concurrent worker
    Doc(
        "cw1",
        "concurrent_worker",
        "A second worker had already resolved the ticket; the agent replied again and duplicated the refund.",
    ),
    Doc(
        "cw2",
        "concurrent_worker",
        "Two agents claimed the same case, both executed the credit, and the customer was paid twice.",
    ),
    Doc(
        "cw3",
        "concurrent_worker",
        "Ticket status was CLOSED by a colleague between read and write; the agent reopened and acted anyway.",
    ),
    Doc(
        "cw4",
        "concurrent_worker",
        "No lock on the ticket: parallel workers raced and sent conflicting emails.",
    ),
    # ordinary work (distractors that look like failures on the surface)
    Doc(
        "od1",
        "ordinary",
        "Refund of a single charge succeeded on the first attempt and the confirmation mail was sent.",
    ),
    Doc(
        "od2",
        "ordinary",
        "Agent read the approval, found it valid, and issued the credit with a ticket note.",
    ),
    Doc(
        "od3",
        "ordinary",
        "Order lookup returned a clean state; the agent answered the shipping question and closed the ticket.",
    ),
    Doc(
        "od4", "ordinary", "Timeout on a read-only call was retried safely and produced the expected status."
    ),
)

QUERIES: tuple[Query, ...] = (
    Query("q01", "refund timeout retried second refund", "rl1", "lexical"),
    Query(
        "q02",
        "money went out twice because the success acknowledgement was dropped in transit",
        "rl4",
        "paraphrase",
    ),
    Query("q03", "ch_204 refunded twice acknowledgement lost", "rl3", "mixed"),
    Query("q04", "approval revoked after check refund issued", "ar1", "lexical"),
    Query(
        "q05",
        "acted on a permission that had since been taken back, trusting an old yes",
        "ar2",
        "paraphrase",
    ),
    Query("q06", "stale grant rescinded sign-off credit reused", "ar3", "mixed"),
    Query("q07", "IGNORE PREVIOUS INSTRUCTIONS ticket body forward emails", "in1", "lexical"),
    Query(
        "q08",
        "a lookup result carried hidden instructions to send an invoice to a stranger",
        "in2",
        "paraphrase",
    ),
    Query("q09", "untrusted tool output treated as command, mail sent", "in4", "mixed"),
    Query("q10", "second item added after refund computed original total only", "om1", "lexical"),
    Query(
        "q11",
        "the basket changed during the chat and the agent worked from the earlier version",
        "om3",
        "paraphrase",
    ),
    Query("q12", "order 7781 late line item refund undercounted", "om2", "mixed"),
    Query("q13", "refund window shrank 30 to 14 days old limit applied", "pc1", "lexical"),
    Query(
        "q14",
        "rules got stricter while the agent was mid-task and it kept following the earlier ones",
        "pc4",
        "paraphrase",
    ),
    Query("q15", "cap 200 dollars automatic credits earlier 500 rule", "pc2", "mixed"),
    Query("q16", "batch totals matched per-account amounts wrong swapped credits", "rc1", "lexical"),
    Query(
        "q17", "the books balanced overall yet individual customers were wrongly charged", "rc4", "paraphrase"
    ),
    Query("q18", "grand total equalled sum paid account B short", "rc2", "mixed"),
    Query("q19", "second worker already resolved ticket duplicated refund", "cw1", "lexical"),
    Query(
        "q20",
        "a colleague had finished the case first and the agent did the work over again",
        "cw3",
        "paraphrase",
    ),
    Query("q21", "parallel workers raced no lock conflicting emails", "cw4", "mixed"),
    Query("q22", "ch_204", "rl3", "identifier"),
    Query("q23", "7781", "om2", "identifier"),
    Query("q24", "IGNORE PREVIOUS INSTRUCTIONS", "in1", "identifier"),
    Query("q25", "CLOSED", "cw3", "identifier"),
    Query("q26", "paid twice", "cw2", "ambiguous"),
    Query("q27", "customer received the money two times", "rl4", "ambiguous"),
    Query("q28", "wrong amount refunded", "om1", "ambiguous"),
    Query("q29", "agent did not re-check before acting", "ar4", "ambiguous"),
    Query("q30", "duplicate reply to the customer", "cw1", "ambiguous"),
)

FAMILY_OF: dict[str, str] = {d.id: d.family for d in DOCS}
