"""Automated embedding: definitions (unit) and a text-only round trip on the sandbox (sandbox)."""

from __future__ import annotations

from typing import Any

import pytest
from pymongo.database import Database

from ratchet.atlas_features import autoembed
from ratchet.atlas_features.dataset import QUERIES


def test_definition_stores_text_only() -> None:
    field = autoembed.auto_definition("voyage-4")["fields"][0]
    assert field == {"type": "autoEmbed", "modality": "text", "path": "summary", "model": "voyage-4"}


def test_query_leg_sends_text_and_no_vector() -> None:
    stage = autoembed.auto_leg("voyage-4", "refund twice")[0]["$vectorSearch"]
    assert stage["query"] == "refund twice"
    assert "queryVector" not in stage
    assert stage["index"] == "auto_voyage_4"


@pytest.mark.sandbox
def test_text_query_finds_gold_in_the_top_three(sandbox_db: Database[dict[str, Any]]) -> None:
    coll = sandbox_db[autoembed.COLLECTION]
    autoembed.ensure(coll)
    picked = [q for q in QUERIES if q.kind == "paraphrase"]
    for q in picked:
        got = autoembed._ids(coll, [*autoembed.auto_leg("voyage-4", q.text), {"$limit": 3}])
        assert q.gold in got, (q.id, got)
