"""Activation: make an admitted candidate the active worker harness, separately from admitting it.

Order is fixed: the decision and its signed receipt are stored first; activation then runs in one MongoDB
transaction (`ratchet.store.activation.MongoActivator`, in the `atlas` extra) that (1) returns the earlier
result when the idempotency key was already used, (2) checks that the stored decision admits exactly this
candidate against exactly the expected active parent and that the stored envelope has the expected digest,
and (3) moves the active pointer only if it still names the expected version and revision (compare and
swap). A missing receipt, a rejected decision, or a stale active
reference leaves the active harness untouched.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ratchet_cases.types import sha256_hex

ACTIVE_ID = "worker"


class ActivationError(RuntimeError):
    """The activation was refused; the active harness did not change."""


@dataclass(frozen=True)
class Activation:
    version: str
    revision: int
    previous: str | None
    decision_id: str
    receipt_digest: str
    idempotency_key: str
    replayed: bool = False


class Activator(Protocol):
    def active(self) -> tuple[str | None, int]: ...

    def bootstrap(self, version: str) -> tuple[str | None, int]: ...

    def activate(
        self,
        *,
        candidate: str,
        expected: str | None,
        expected_revision: int,
        decision_id: str,
        receipt_digest: str,
        idempotency_key: str,
    ) -> Activation: ...


def envelope_digest(envelope_b64: str) -> str:
    return sha256_hex(base64.b64decode(envelope_b64))


def check_binding(decision: dict[str, Any] | None, candidate: str, expected: str | None, digest: str) -> None:
    """The stored decision must exist, admit this candidate against the expected parent, and carry the
    receipt whose digest the caller names."""
    if decision is None:
        raise ActivationError("no stored decision: the receipt must be persisted before activation")
    if not decision.get("admitted"):
        raise ActivationError("the stored decision did not admit this candidate")
    if decision.get("candidate") != candidate or decision.get("parent") != expected:
        raise ActivationError("the stored decision names a different candidate or parent")
    env = decision.get("envelope")
    if not isinstance(env, str) or envelope_digest(env) != digest:
        raise ActivationError("the stored receipt is missing or its digest differs")


class MemoryActivator:
    """The same rules over an in-memory decision map, for tests and dry runs."""

    def __init__(self, lookup: Callable[[str], dict[str, Any] | None]) -> None:
        self.lookup = lookup
        self.pointer: tuple[str | None, int] = (None, 0)
        self.log: dict[str, Activation] = {}

    def active(self) -> tuple[str | None, int]:
        return self.pointer

    def bootstrap(self, version: str) -> tuple[str | None, int]:
        if self.pointer[0] is None:
            self.pointer = (version, 1)
        return self.pointer

    def activate(
        self,
        *,
        candidate: str,
        expected: str | None,
        expected_revision: int,
        decision_id: str,
        receipt_digest: str,
        idempotency_key: str,
    ) -> Activation:
        done = self.log.get(idempotency_key)
        if done is not None:
            if done.version != candidate:
                raise ActivationError("idempotency key already used for a different candidate")
            return Activation(
                done.version,
                done.revision,
                done.previous,
                done.decision_id,
                done.receipt_digest,
                idempotency_key,
                replayed=True,
            )
        check_binding(self.lookup(decision_id), candidate, expected, receipt_digest)
        if self.pointer != (expected, expected_revision):
            raise ActivationError("stale active reference: the active version or revision moved")
        act = Activation(
            candidate, expected_revision + 1, expected, decision_id, receipt_digest, idempotency_key
        )
        self.pointer = (candidate, expected_revision + 1)
        self.log[idempotency_key] = act
        return act
