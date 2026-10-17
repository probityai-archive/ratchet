"""Every stored decision record re-derives under the canonicaliser it was signed with and under RFC 8785.

Decisions signed before the RFC 8785 change carry payloads written by the earlier canonicaliser (sorted keys,
no whitespace, Python number spelling). Their signatures cover those exact bytes, and nothing re-serialises
a stored payload: verification reads the bytes as signed. This test reads a directory of decision logs
(`*_decisions.jsonl`, one bare decision or one `{"decision": ...}` per line), and for every record checks:

- the payload equals the earlier canonicaliser's output for its own value (so the record is what that
  canonicaliser signed, and was not edited after signing);
- RFC 8785 re-derives the same JSON value, and its bytes differ from the stored ones only where a number
  is spelled differently (an integral float such as `1.0`, or an exponent such as `1e-07`).

It needs the records, which are not in this repository: point RATCHET_RECORDS_DIR at them and run
`pytest -m records tests/records`.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import pytest

from ratchet.canonical import canonical_json, canonicalizer

pytestmark = pytest.mark.records
earlier_canonical_json = canonicalizer(None)


def _spelled_differently(value: Any) -> bool:
    """True when the value holds a number whose Python spelling is not its ECMAScript spelling."""
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return json.dumps(value) != canonical_json(value).decode("ascii")
    if isinstance(value, dict):
        return any(_spelled_differently(v) for v in value.values())
    if isinstance(value, list):
        return any(_spelled_differently(v) for v in value)
    return False


def _records() -> list[tuple[str, dict[str, Any]]]:
    root = os.environ.get("RATCHET_RECORDS_DIR", "")
    if not root:
        raise RuntimeError("set RATCHET_RECORDS_DIR to the directory of *_decisions.jsonl files")
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(Path(root).glob("*_decisions.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                out.append((f"{path.name}:{n}", row.get("decision", row)))
    return out


def test_every_record_rederives_under_both_canonicalisers() -> None:
    records = _records()
    assert records, "no decision records found"
    differ = 0
    for where, record in records:
        envelope = base64.b64decode(record["envelope"], validate=True)
        doc = json.loads(envelope)
        assert earlier_canonical_json(doc) == envelope, f"{where}: envelope is not as signed"
        payload = base64.b64decode(doc["payload"], validate=True)
        value = json.loads(payload)
        assert earlier_canonical_json(value) == payload, f"{where}: payload is not as signed"
        assert "canonicalization" not in value, f"{where}: a record before RFC 8785 names no canonicaliser"
        rfc = canonical_json(value)
        assert json.loads(rfc) == value, f"{where}: RFC 8785 changed the value"
        if rfc != payload:
            differ += 1
            assert _spelled_differently(value), f"{where}: bytes differ beyond number spelling"
    print(f"records {len(records)}; RFC 8785 bytes differ (number spelling only) in {differ}")
