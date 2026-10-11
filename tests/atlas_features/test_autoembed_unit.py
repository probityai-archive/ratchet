"""Automated embedding probe against fake collections."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from atlas_fakes import FakeClient, FakeCollection, FakeDb, op_failure

from ratchet.atlas_features import autoembed
from ratchet.atlas_features.common import ProbeResult
from ratchet.atlas_features.dataset import DOCS, QUERIES


def gold_for(pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blob = repr(pipeline)
    for q in QUERIES:
        if q.text in blob:
            return [{"_id": q.gold}]
    return [{"_id": "od1"}]


def test_index_names_are_valid_identifiers() -> None:
    assert autoembed.index_name("voyage-4-large") == "auto_voyage_4_large"


def test_hybrid_pipeline_feeds_the_text_leg_and_the_auto_leg() -> None:
    stage = autoembed.auto_rank_fusion("voyage-4", "refund")[0]["$rankFusion"]
    assert set(stage["input"]["pipelines"]) == {"vec", "txt"}
    assert stage["input"]["pipelines"]["vec"][0]["$vectorSearch"]["query"] == "refund"


def test_ensure_seeds_text_only_documents_and_creates_three_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autoembed, "wait_queryable", lambda *_a, **_k: 0.0)
    coll = FakeCollection("c", FakeDb())
    waits = autoembed.ensure(coll)  # type: ignore[arg-type]
    assert len(coll.docs) == len(DOCS)
    assert all("embedding" not in d for d in coll.docs.values())
    assert len(coll.created_indexes) == 3
    assert set(waits) == {"auto_voyage_4", "auto_voyage_4_large", "fusion_txt"}
    coll.search_indexes = [{"name": n} for n in waits]
    coll.created_indexes.clear()
    autoembed.ensure(coll)  # type: ignore[arg-type]
    assert coll.created_indexes == []


def test_score_mode_computes_recall_and_captures_a_server_error() -> None:
    good = FakeCollection("c", FakeDb(), gold_for)
    out = autoembed.score_mode(good, lambda t: autoembed.auto_leg("voyage-4", t))  # type: ignore[arg-type]
    assert out["ok"] is True
    assert out["recall_at_3"] == 1.0
    assert out["recall_at_1"] == 1.0
    bad = FakeCollection("c", FakeDb(), lambda _p: op_failure(7, "boom"))
    err = autoembed.score_mode(bad, lambda t: autoembed.auto_leg("voyage-4", t))  # type: ignore[arg-type]
    assert err == {"ok": False, "error": "code 7: boom"}


def test_sync_check_reports_seconds_for_insert_and_update(monkeypatch: pytest.MonkeyPatch) -> None:
    coll = FakeCollection("c", FakeDb(), lambda _p: [{"_id": "zzprobe"}])
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    out = autoembed.sync_check(coll, "voyage-4", timeout_s=5)  # type: ignore[arg-type]
    assert out["insert_searchable_s"] is not None
    assert out["update_searchable_s"] is not None
    assert "zzprobe" not in coll.docs


def test_wait_found_gives_up_at_the_timeout() -> None:
    coll = FakeCollection("c", FakeDb(), lambda _p: [])
    assert autoembed._wait_found(coll, "voyage-4", "q", "zzprobe", timeout_s=0.0) is None  # type: ignore[arg-type]


def test_run_and_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = FakeDb()
    db[autoembed.COLLECTION].aggregate_fn = gold_for
    monkeypatch.setattr(autoembed, "connect", lambda: FakeClient())
    monkeypatch.setattr(autoembed, "probe_db", lambda _c: db)
    monkeypatch.setattr(autoembed, "wait_queryable", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(autoembed, "sync_check", lambda *_a, **_k: {"insert_searchable_s": 1.0})
    monkeypatch.setattr(ProbeResult, "save", lambda self, *_a, **_k: tmp_path / "x.json")
    result = autoembed.run()
    assert result.works is True
    assert set(result.data["recall"]) == set(autoembed.MODELS)
    assert autoembed.main() == 0
    assert "autoembed: works=True" in capsys.readouterr().out
