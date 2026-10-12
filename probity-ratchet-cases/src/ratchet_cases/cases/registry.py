"""Every case, by stable id, split and family."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from types import ModuleType

from ratchet_cases.cases.families import (
    approval_revoked,
    concurrent_worker,
    injection,
    order_modification,
    ordinary,
    policy_changed,
    reconciliation,
    refund_lost_reply,
)
from ratchet_cases.cases.model import Case, CaseKind, Family
from ratchet_cases.types import Split, canonical_json, sha256_hex, to_jsonable

_MODULES: tuple[ModuleType, ...] = (
    refund_lost_reply,
    approval_revoked,
    policy_changed,
    injection,
    concurrent_worker,
    reconciliation,
    order_modification,
    ordinary,
)


def families() -> list[Family]:
    found: list[Family] = [m.FAMILY for m in _MODULES]
    return found


@cache
def _all() -> tuple[Case, ...]:
    built: list[Case] = [case for module in _MODULES for case in module.cases()]
    ids = [c.id for c in built]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    return tuple(built)


def all_cases() -> list[Case]:
    """Every case in a stable order (family order, then variant)."""
    return list(_all())


def by_split(split: Split) -> list[Case]:
    return [c for c in _all() if c.split == split]


def by_kind(kind: CaseKind) -> list[Case]:
    return [c for c in _all() if c.kind == kind]


def suite_digest() -> str:
    """SHA-256 over the canonical JSON of every case definition, in id order. `DIGESTS.json` pins it."""
    return sha256_hex(canonical_json([to_jsonable(c) for c in sorted(_all(), key=lambda c: c.id)]))


DIGESTS_PATH = Path(__file__).with_name("DIGESTS.json")


def released_digests() -> dict[str, str]:
    """Every released version of the package mapped to the suite digest it shipped."""
    loaded = json.loads(DIGESTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in loaded.items()
    ):
        raise ValueError(f"{DIGESTS_PATH.name} must map version strings to digest strings")
    return dict(loaded)


def by_family(family: str) -> list[Case]:
    found = [c for c in _all() if c.family == family]
    if not found:
        raise KeyError(f"unknown family: {family}")
    return found


def get(case_id: str) -> Case:
    for case in _all():
        if case.id == case_id:
            return case
    raise KeyError(f"unknown case: {case_id}")
