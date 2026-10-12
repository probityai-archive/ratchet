from __future__ import annotations

import base64
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from ratchet_cases.types import canonical_json

from ratchet.improve.countersign import countersign_doc, countersign_jsonl, main, payload_matches, run_key
from ratchet.improve.receipt import (
    DECISION_TYPE,
    HUMAN_APPROVAL_TYPE,
    ReceiptError,
    Signer,
    countersign,
    verify,
)

PAYLOAD = {"admitted": False, "parent": "p", "candidate": "c"}


def _stored(signer: Signer) -> dict[str, object]:
    env = signer.sign(DECISION_TYPE, canonical_json(PAYLOAD))
    return {
        **PAYLOAD,
        "_id": "p->c",
        "envelope": base64.b64encode(env).decode("ascii"),
        "keyid": signer.keyid,
    }


def test_countersign_keeps_the_run_signature_and_adds_the_pinned_one(decision_key: Signer) -> None:
    run_key = Signer.generate()
    original = run_key.sign(DECISION_TYPE, canonical_json(PAYLOAD))
    signed = countersign(original, decision_key)
    assert [s["keyid"] for s in json.loads(signed)["signatures"]] == [run_key.keyid, decision_key.keyid]
    assert verify(signed, {decision_key.keyid: decision_key.public_key}) == PAYLOAD
    assert verify(signed, {run_key.keyid: run_key.public_key}) == PAYLOAD
    assert countersign(signed, decision_key) == signed


def test_countersign_refuses_a_non_decision_envelope(decision_key: Signer) -> None:
    other = Signer.generate().sign(HUMAN_APPROVAL_TYPE, b"{}")
    with pytest.raises(ReceiptError, match="only a decision envelope"):
        countersign(other, decision_key)


def test_countersign_doc_makes_a_run_signed_decision_verify_under_the_pinned_key(
    decision_key: Signer,
) -> None:
    doc = _stored(Signer.generate())
    assert payload_matches(doc)
    update = countersign_doc(doc, decision_key, datetime(2026, 10, 11, tzinfo=UTC))
    assert update is not None and update["countersigned"]["keyid"] == decision_key.keyid
    env = base64.b64decode(update["envelope"])
    assert verify(env, {decision_key.keyid: decision_key.public_key})["candidate"] == "c"
    assert countersign_doc({**doc, **update}, decision_key, datetime.now(UTC)) is None


def test_an_edited_decision_does_not_match_its_payload() -> None:
    doc = _stored(Signer.generate())
    assert not payload_matches({**doc, "admitted": True})


# -- JSONL decision logs ----------------------------------------------------------------------------------


def _log(tmp_path: Path, run_key: Signer, *, wrapped: bool = False, n: int = 2) -> Path:
    """A run directory with `n` decisions signed by `run_key` and its published signer.json."""
    run = tmp_path / "run"
    run.mkdir()
    lines = []
    for i in range(n):
        payload = {**PAYLOAD, "candidate": f"c{i}"}
        env = run_key.sign(DECISION_TYPE, canonical_json(payload))
        doc = {**payload, "decision_id": f"p->c{i}", "envelope": base64.b64encode(env).decode("ascii")}
        lines.append(json.dumps({"decision": doc, "decision_id": f"p->c{i}"} if wrapped else doc))
    (run / "decisions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pub = run_key.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    (run / "signer.json").write_text(
        json.dumps({"keyid": run_key.keyid, "public_key": pub}), encoding="utf-8"
    )
    return run / "decisions.jsonl"


def _docs(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r.get("decision", r) for r in rows]


@pytest.mark.parametrize("wrapped", [False, True])
def test_a_jsonl_log_is_countersigned_and_the_original_kept(
    decision_key: Signer, tmp_path: Path, wrapped: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    run_key = Signer.generate()
    path = _log(tmp_path, run_key, wrapped=wrapped)
    before = path.read_bytes()
    assert main(["--jsonl", str(path)]) == 0
    assert f"{path}: read 2, countersigned 2, already 0, refused 0" in capsys.readouterr().out
    assert path.with_name("decisions.jsonl.bak").read_bytes() == before
    for doc in _docs(path):
        env = base64.b64decode(str(doc["envelope"]))
        assert verify(env, {decision_key.keyid: decision_key.public_key})["candidate"] == doc["candidate"]
        assert verify(env, {run_key.keyid: run_key.public_key})["candidate"] == doc["candidate"]
        assert doc["countersigned"]["keyid"] == decision_key.keyid  # type: ignore[index]
    assert decision_key.private_key.private_bytes_raw().hex() not in capsys.readouterr().out


def test_a_second_run_changes_nothing(decision_key: Signer, tmp_path: Path) -> None:
    path = _log(tmp_path, Signer.generate())
    assert main(["--jsonl", str(path)]) == 0
    after_first, bak = path.read_bytes(), path.with_name("decisions.jsonl.bak").read_bytes()
    rep = countersign_jsonl(path, decision_key, run_key(path.with_name("signer.json")), datetime.now(UTC))
    assert (rep.read, rep.countersigned, rep.already, rep.refused) == (2, 0, 2, [])
    assert path.read_bytes() == after_first
    assert path.with_name("decisions.jsonl.bak").read_bytes() == bak


def test_a_decision_whose_signature_does_not_verify_is_refused(
    decision_key: Signer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _log(tmp_path, Signer.generate())
    rows = path.read_text(encoding="utf-8").splitlines()
    doc = json.loads(rows[0])
    env = json.loads(base64.b64decode(doc["envelope"]))
    env["signatures"][0]["sig"] = base64.b64encode(b"\0" * 64).decode("ascii")
    doc["envelope"] = base64.b64encode(canonical_json(env)).decode("ascii")
    rows[0] = json.dumps(doc)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert main(["--jsonl", str(path)]) == 1
    out = capsys.readouterr().out
    assert f"REFUSED {path}:1: original signature does not verify" in out
    assert "read 2, countersigned 1, already 0, refused 1" in out
    first, second = _docs(path)
    assert "countersigned" not in first and "countersigned" in second


def test_a_log_without_its_per_run_key_is_refused_whole(decision_key: Signer, tmp_path: Path) -> None:
    path = _log(tmp_path, Signer.generate())
    path.with_name("signer.json").unlink()
    before = path.read_bytes()
    assert main(["--jsonl", str(path)]) == 1
    assert path.read_bytes() == before and not path.with_name("decisions.jsonl.bak").exists()


def test_a_signer_file_whose_keyid_does_not_name_its_key_is_refused(tmp_path: Path) -> None:
    path = _log(tmp_path, Signer.generate())
    raw = json.loads(path.with_name("signer.json").read_text(encoding="utf-8"))
    path.with_name("signer.json").write_text(json.dumps({**raw, "keyid": "0" * 64}), encoding="utf-8")
    with pytest.raises(ReceiptError, match="keyid does not name"):
        run_key(path.with_name("signer.json"))


def test_the_rewrite_is_atomic(decision_key: Signer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _log(tmp_path, Signer.generate())
    before = path.read_bytes()

    def fail(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        countersign_jsonl(path, decision_key, run_key(path.with_name("signer.json")), datetime.now(UTC))
    assert path.read_bytes() == before


def test_main_needs_exactly_one_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert main(["--db", "x", "--jsonl", "y"]) == 2
    assert "pass --db or --jsonl" in capsys.readouterr().err
