"""Trigger probe: the function runs under node with a stub context (unit) and on the sandbox (sandbox)."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from typing import Any

import pytest
from pymongo.database import Database

from ratchet.atlas_features import triggers
from ratchet.atlas_features.common import PROBE_DB

HARNESS = """
const m = require(process.argv[1]);
const calls = [];
globalThis.context = { services: { get: () => ({ db: () => ({ collection: () => ({
  updateOne: async (q, u, o) => { calls.push({ q, u, o }); return { upsertedId: "x" }; } }) }) }) } };
(async () => {
  const out = await m.handler({ fullDocument: JSON.parse(process.argv[2]) });
  console.log(JSON.stringify({ out, calls, sanitized: m.sanitize(process.argv[3]) }));
})();
"""


def run_function(failure: dict[str, Any], text: str = "") -> dict[str, Any]:
    node = shutil.which("node")
    assert node, "node is required to test the trigger function"
    proc = subprocess.run(
        [node, "-e", HARNESS, str(triggers.FUNCTION_SOURCE), json.dumps(failure), text],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return dict(json.loads(proc.stdout))


def test_sanitize_removes_ids_emails_and_amounts() -> None:
    text = "Charge ch_204 for order 7781 refunded $1,250.00; mail bob@example.com, order #4421, 900 dollars"
    out = run_function({"failure_id": "f", "summary": "x"}, text)["sanitized"]
    for leaked in ("ch_204", "7781", "1,250", "bob@", "4421", "900"):
        assert leaked not in out
    assert out.count("<") >= 5


def test_function_upserts_one_document_keyed_by_failure_id() -> None:
    res = run_function({"failure_id": "f-1", "case_family": "x", "summary": "refund for ch_9 twice"})
    call = res["calls"][0]
    assert call["q"] == {"_id": "fm:f-1"}
    assert call["o"] == {"upsert": True}
    assert "$setOnInsert" in call["u"]
    assert call["u"]["$setOnInsert"]["summary"] == "refund for <id> twice"
    assert res["out"] == {"id": "fm:f-1", "inserted": True}


def test_function_skips_events_without_a_failure_id() -> None:
    res = run_function({"summary": "no id"})
    assert res["calls"] == []
    assert "skipped" in res["out"]


def test_trigger_body_listens_to_inserts_only_with_full_document() -> None:
    cfg = triggers.trigger_body("s", "f")["config"]
    assert cfg["operation_types"] == ["INSERT"]
    assert cfg["full_document"] is True
    assert cfg["database"] == PROBE_DB


@pytest.mark.sandbox
def test_trigger_writes_one_sanitized_memory_for_a_duplicate_insert(
    sandbox_db: Database[dict[str, Any]],
) -> None:
    src, mem = sandbox_db[triggers.SOURCE], sandbox_db[triggers.MEMORY]
    failure_id = f"test-{uuid.uuid4().hex[:8]}"
    doc = {"failure_id": failure_id, "summary": "Charge ch_204 refunded $50 twice for bob@example.com"}
    try:
        src.insert_one(dict(doc))
        found = triggers.wait_for(mem, f"fm:{failure_id}", timeout_s=30)
        assert found is not None, (
            "the trigger wrote no memory within 30 s (is the app deployed? run the probe)"
        )
        src.insert_one(dict(doc))
        time.sleep(5)
        stored = list(mem.find({"failure_id": failure_id}))
        assert len(stored) == 1
        assert "ch_204" not in stored[0]["summary"]
        assert "bob@" not in stored[0]["summary"]
    finally:
        mem.delete_many({"failure_id": failure_id})
        src.delete_many({"failure_id": failure_id})
