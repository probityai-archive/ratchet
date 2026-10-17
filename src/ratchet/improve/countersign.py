"""Countersign stored decisions with the pinned decision key.

Some runs signed their decisions with a per-run key that was never published, so `ratchet verify` cannot
check them. This adds the pinned key's signature to each envelope, keeps the run's own signature, and
records on the decision when and why it was countersigned:

    uv run python -m ratchet.improve.countersign --db ratchet_v3_full [--id ID ...]
    uv run python -m ratchet.improve.countersign --jsonl RUN/decisions.jsonl [--jsonl ...]

Only decisions whose stored fields equal their signed payload are countersigned. Any other is reported and
left unchanged, because the pinned key must never vouch for an edited record.

A JSONL decision log (one decision per line, bare or as `{"decision": {...}}`) is countersigned only when
each decision's own signature verifies under the run's published per-run key: `signer.json` beside the file
(`{"keyid": ..., "public_key": <hex>}`) or `--signer-json`. A file without that key is refused whole, since
the pinned key must not vouch for a signature nobody can check. The file is rewritten atomically, the
previous version kept as `<file>.bak`, and a second run changes nothing.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ratchet.canonical import canonical_json
from ratchet.improve.receipt import ReceiptError, Signer, countersign, key_id, verify

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


@dataclass
class FileReport:
    """What one JSONL file's countersigning did; `refused` holds (line number, reason)."""

    path: Path
    read: int = 0
    countersigned: int = 0
    already: int = 0
    refused: list[tuple[int, str]] = field(default_factory=list)


def run_key(signer_json: Path) -> dict[str, Ed25519PublicKey]:
    """The run's published per-run key from `signer.json`; refused unless its keyid names that key."""
    raw = json.loads(signer_json.read_text(encoding="utf-8"))
    key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(str(raw["public_key"])))
    if key_id(key) != raw.get("keyid"):
        raise ReceiptError(f"{signer_json}: keyid does not name the public key it carries")
    return {key_id(key): key}


def _decision(record: dict[str, Any]) -> dict[str, Any]:
    inner = record.get("decision")
    return inner if isinstance(inner, dict) else record


def countersign_jsonl(
    path: Path, signer: Signer, trusted_run_key: Mapping[str, Ed25519PublicKey] | None, now: datetime
) -> FileReport:
    """Countersign every verifiable decision of a JSONL log in place (atomic rewrite, `.bak` kept)."""
    report = FileReport(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    changed = False
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            out.append(line)
            continue
        report.read += 1
        record = json.loads(line)
        doc = _decision(record)
        reason = _refusal(doc, trusted_run_key)
        if reason is not None:
            report.refused.append((n, reason))
            out.append(line)
            continue
        update = countersign_doc(doc, signer, now)
        if update is None:
            report.already += 1
            out.append(line)
            continue
        doc.update(update)
        report.countersigned += 1
        changed = True
        out.append(json.dumps(record, sort_keys=True))
    if changed:
        _atomic_write(path, "\n".join(out) + "\n")
    return report


def _refusal(doc: Mapping[str, Any], trusted_run_key: Mapping[str, Ed25519PublicKey] | None) -> str | None:
    if not doc.get("envelope"):
        return "no signed envelope"
    if not trusted_run_key:
        return "no published per-run key (signer.json) to check the original signature"
    try:
        verify(base64.b64decode(str(doc["envelope"])), trusted_run_key)
    except (ReceiptError, ValueError) as exc:
        return f"original signature does not verify under the per-run key: {exc}"
    if not payload_matches(doc):
        return "stored decision differs from its signed payload"
    return None


def _atomic_write(path: Path, text: str) -> None:
    """Keep the current file as `<file>.bak`, then replace it in one rename."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    bak = path.with_name(path.name + ".bak")
    bak.write_bytes(path.read_bytes())
    os.replace(tmp, path)


def _jsonl_main(paths: Sequence[str], signer_json: str | None, signer: Signer) -> int:
    now = datetime.now(UTC)
    refused = 0
    for name in paths:
        path = Path(name)
        key_file = Path(signer_json) if signer_json else path.with_name("signer.json")
        trusted = run_key(key_file) if key_file.is_file() else None
        rep = countersign_jsonl(path, signer, trusted, now)
        for n, reason in rep.refused:
            print(f"REFUSED {path}:{n}: {reason}")
        print(
            f"{path}: read {rep.read}, countersigned {rep.countersigned}, already {rep.already}, "
            f"refused {len(rep.refused)}"
        )
        refused += len(rep.refused)
    return 1 if refused else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ratchet.improve.countersign")
    ap.add_argument("--uri", default=os.environ.get("RATCHET_MONGODB_URI", ""))
    ap.add_argument("--db")
    ap.add_argument("--id", action="append", default=[], help="decision _id (default: every decision)")
    ap.add_argument("--jsonl", action="append", default=[], help="a JSONL decision log to countersign")
    ap.add_argument("--signer-json", help="the run's per-run key (default: signer.json beside each file)")
    args = ap.parse_args(argv)
    if bool(args.db) == bool(args.jsonl):
        print("error: pass --db or --jsonl", file=sys.stderr)
        return 2
    if args.jsonl:
        try:
            return _jsonl_main(args.jsonl, args.signer_json, Signer.for_decisions())
        except ReceiptError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
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
