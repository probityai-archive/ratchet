"""Countersign stored decisions with the pinned decision key.

Some runs signed their decisions with a per-run key that was never published, so `ratchet verify` cannot
check them. This adds the pinned key's signature to each envelope, keeps the run's own signature, and
records on the decision when and why it was countersigned:

    uv run python -m ratchet.improve.countersign --db ratchet_v3_full [--id ID ...]

Only decisions whose stored fields equal their signed payload are countersigned. Any other is reported and
left unchanged, because the pinned key must never vouch for an edited record.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ratchet_cases.types import canonical_json

from ratchet.improve.receipt import ReceiptError, Signer, countersign

NOTE = "countersigned by the pinned decision key; the run signed with a per-run key that was not published"


def payload_matches(doc: Mapping[str, Any]) -> bool:
    """True when every field of the signed payload equals the stored decision's field."""
    envelope = json.loads(base64.b64decode(str(doc["envelope"])))
    payload = json.loads(base64.b64decode(envelope["payload"]))
    return all(canonical_json(doc.get(k)) == canonical_json(v) for k, v in payload.items())


def countersign_doc(doc: Mapping[str, Any], signer: Signer, now: datetime) -> dict[str, Any] | None:
    """The fields to update on `doc`, or None when `signer` has already countersigned it."""
    raw = base64.b64decode(str(doc["envelope"]))
    signed = countersign(raw, signer)
    if signed == raw:
        return None
    return {
        "envelope": base64.b64encode(signed).decode("ascii"),
        "countersigned": {"keyid": signer.keyid, "at": now.isoformat(), "note": NOTE},
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ratchet.improve.countersign")
    ap.add_argument("--uri", default=os.environ.get("RATCHET_MONGODB_URI", ""))
    ap.add_argument("--db", required=True)
    ap.add_argument("--id", action="append", default=[], help="decision _id (default: every decision)")
    args = ap.parse_args(argv)
    if not args.uri:
        print("error: pass --uri or set RATCHET_MONGODB_URI", file=sys.stderr)
        return 2
    try:
        signer = Signer.for_decisions()
    except ReceiptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    from pymongo import MongoClient
    from pymongo.collection import Collection

    col: Collection[dict[str, Any]] = MongoClient(args.uri)[args.db]["decisions"]
    query: dict[str, Any] = {"envelope": {"$exists": True}}
    if args.id:
        query["_id"] = {"$in": args.id}
    now = datetime.now(UTC)
    done = skipped = refused = 0
    for doc in col.find(query):
        if not payload_matches(doc):
            print(f"REFUSED {doc['_id']}: stored decision differs from its signed payload")
            refused += 1
            continue
        update = countersign_doc(doc, signer, now)
        if update is None:
            skipped += 1
            continue
        col.update_one({"_id": doc["_id"]}, {"$set": update})
        print(f"COUNTERSIGNED {doc['_id']}")
        done += 1
    print(
        f"countersigned {done}, already countersigned {skipped}, refused {refused} (key {signer.keyid[:12]})"
    )
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
