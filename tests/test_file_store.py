"""The signed append-only file store: what it keeps, what it refuses, and the loop running on it."""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ratchet_cases.types import Cost, Event, Outcome, Trace

from ratchet.improve.activate import ActivationError, envelope_digest
from ratchet.store import IntegrityError
from ratchet.store.file_store import LOG, FileActivator, FileStore, matches
from ratchet.types import Harness

SEED = Harness.build(parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=["a"],
                     created_by="seed")  # fmt: skip
CHILD = SEED.child(system_prompt="p\nmore", created_by="improver")


def _trace(run: str = "r1", passed: bool = True) -> Trace:
    events = [Event(0, "model", {"cached": False}, 1.0), Event(1, "final", {"x": 2.5}, 3.0)]
    return Trace(run, "case-1", SEED.version, "m", events, Outcome(passed, [], "d", Cost(wall_s=12.0)))


@pytest.fixture
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def test_records_round_trip_and_survive_a_reopen(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    s = FileStore(tmp_path, key=key)
    assert s.put_harness(SEED) and s.put_harness(CHILD) and not s.put_harness(SEED)
    assert s.put_trace(_trace()) and not s.put_trace(_trace())
    s.put_eval({"type": "eval", "n": 1.0}, eval_id="e1")
    s.put_decision({"type": "decision", "admitted": True, "candidate": CHILD.version}, decision_id="d1")
    assert s.checkpoint("run-1", {"step": 1}) == 1 and s.checkpoint("run-1", {"step": 2}) == 2
    again = FileStore(tmp_path, trusted=key.public_key())
    assert again.get_harness(CHILD.version) == CHILD
    assert again.get_trace("r1", "case-1") is not None
    assert [t.run_id for t in again.traces({"harness_version": SEED.version, "passed": True})] == ["r1"]
    assert again.evals({"type": "eval"})[0]["_id"] == "e1"
    assert again.decisions({"admitted": True})[0]["candidate"] == CHILD.version
    assert again.resume("run-1") == {"step": 2}
    [root] = again.lineage()
    assert root.version == SEED.version and [c.version for c in root.children] == [CHILD.version]
    with pytest.raises(PermissionError):
        again.put_eval({"type": "eval"})


def test_other_content_under_a_stored_id_is_refused(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    s = FileStore(tmp_path, key=key)
    s.put_eval({"type": "eval", "n": 1}, eval_id="e1")
    s.put_eval({"type": "eval", "n": 1}, eval_id="e1")
    with pytest.raises(IntegrityError, match="different content"):
        s.put_eval({"type": "eval", "n": 2}, eval_id="e1")
    assert s.put_trace(_trace())
    with pytest.raises(IntegrityError, match="different content"):
        s.put_trace(_trace(passed=False))


def _lines(tmp_path: Path) -> list[bytes]:
    return (tmp_path / LOG).read_bytes().splitlines(keepends=True)


@pytest.mark.parametrize("damage", ["edit", "drop", "swap", "foreign", "partial"])
def test_a_damaged_log_refuses_to_open(tmp_path: Path, key: Ed25519PrivateKey, damage: str) -> None:
    s = FileStore(tmp_path, key=key)
    for i in range(3):
        s.put_eval({"type": "eval", "i": i}, eval_id=f"e{i}")
    lines = _lines(tmp_path)
    if damage == "edit":
        doc = json.loads(lines[1])
        body = json.loads(base64.b64decode(doc["payload"]))
        body["doc"]["i"] = 99
        doc["payload"] = base64.b64encode(json.dumps(body).encode()).decode()
        lines[1] = json.dumps(doc).encode() + b"\n"
    elif damage == "drop":
        del lines[1]
    elif damage == "swap":
        lines[0], lines[1] = lines[1], lines[0]
    elif damage == "foreign":
        other = FileStore(tmp_path / "other", key=Ed25519PrivateKey.generate())
        other.put_eval({"type": "eval"}, eval_id="x")
        lines[2] = _lines(tmp_path / "other")[0]
    else:
        lines[2] = lines[2][:-10]
    (tmp_path / LOG).write_bytes(b"".join(lines))
    with pytest.raises(IntegrityError):
        FileStore(tmp_path, trusted=key.public_key())


def test_two_writers_share_one_log(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    a, b = FileStore(tmp_path, key=key), FileStore(tmp_path, key=key)

    def write(store: FileStore, tag: str) -> None:
        for i in range(25):
            store.put_eval({"type": "eval", "w": tag, "i": i}, eval_id=f"{tag}{i}")

    threads = [threading.Thread(target=write, args=(s, t)) for s, t in ((a, "a"), (b, "b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(FileStore(tmp_path, trusted=key.public_key()).evals()) == 50


def test_filters_follow_mongodb_equality_and_in(tmp_path: Path) -> None:
    doc: dict[str, Any] = {"type": "generation", "candidates": [{"cluster_id": "c1"}, {"cluster_id": "c2"}],
                           "context": {"change": {"id": "g1"}}, "tags": ["x", "y"]}  # fmt: skip
    assert matches(doc, {"candidates.cluster_id": {"$in": ["c2", "c9"]}})
    assert not matches(doc, {"candidates.cluster_id": {"$in": ["c9"]}})
    assert matches(doc, {"context.change.id": "g1", "type": "generation", "tags": "y"})
    assert not matches(doc, {"context.missing": "g1"})


def test_watch_yields_records_appended_after_it_starts(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    s = FileStore(tmp_path, key=key)
    s.put_eval({"type": "eval"}, eval_id="before")
    stream = s.watch("evals", timeout_s=2.0, poll_ms=10)
    writer = FileStore(tmp_path, key=key)
    writer.put_decision({"type": "decision"}, decision_id="d")
    writer.put_eval({"type": "eval"}, eval_id="after")
    first = next(stream)
    assert (first.operation, first.collection, first.doc_id) == ("insert", "evals", "after")


def _decision(store: FileStore, signer_key: Ed25519PrivateKey, parent: str | None, candidate: str) -> str:
    from ratchet.verify import dsse

    env = dsse.sign("application/vnd.ratchet.decision+json", b"{}", [signer_key])
    doc = {"admitted": True, "parent": parent, "candidate": candidate,
           "envelope": base64.b64encode(env).decode()}  # fmt: skip
    store.put_decision(doc, decision_id=f"{parent}->{candidate}")
    return envelope_digest(doc["envelope"])


def test_activation_moves_only_from_the_expected_pointer(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    store = FileStore(tmp_path, key=key)
    act = FileActivator(store)
    assert act.bootstrap(SEED.version) == (SEED.version, 1) == act.bootstrap("other")
    digest = _decision(store, key, SEED.version, CHILD.version)
    did = f"{SEED.version}->{CHILD.version}"
    with pytest.raises(ActivationError, match="stale"):
        act.activate(candidate=CHILD.version, expected=SEED.version, expected_revision=7, decision_id=did,
                     receipt_digest=digest, idempotency_key="k1")  # fmt: skip
    with pytest.raises(ActivationError, match="digest"):
        act.activate(candidate=CHILD.version, expected=SEED.version, expected_revision=1, decision_id=did,
                     receipt_digest="0" * 64, idempotency_key="k1")  # fmt: skip
    done = act.activate(candidate=CHILD.version, expected=SEED.version, expected_revision=1, decision_id=did,
                        receipt_digest=digest, idempotency_key="k1")  # fmt: skip
    assert (done.version, done.revision) == (CHILD.version, 2)
    again = FileActivator(FileStore(tmp_path, key=key)).activate(
        candidate=CHILD.version, expected=SEED.version, expected_revision=1, decision_id=did,
        receipt_digest=digest, idempotency_key="k1")  # fmt: skip
    assert again.replayed and FileActivator(FileStore(tmp_path, key=key)).active() == (CHILD.version, 2)


def test_the_loop_runs_on_the_file_store(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    """The improver and the referee each on a signed file store of their own."""
    from improve.fakes import FakeRunCase, FakeSimulator, LocalRunner
    from improve.test_loop import CASES, RoutingChat
    from improve.test_loop import SEED as LOOP_SEED

    from ratchet.improve.loop import Lab, evolve
    from ratchet.improve.receipt import Signer, verify
    from ratchet.improve.referee import RefereeCore, RefereeParts

    signer = Signer.generate()
    improver_store = FileStore(tmp_path / "improver", key=key)
    referee_store = FileStore(tmp_path / "referee", key=signer.private_key)
    referee = RefereeCore(
        RefereeParts(
            cases=[c for c in CASES if c.split == "heldout"],
            run_case=FakeRunCase(),
            model=None,
            simulator=FakeSimulator(),
            runner=LocalRunner(),
            signer=signer,
            store=referee_store,
            activator=FileActivator(referee_store),
            suite_digest="suite",
            workers=4,
        )
    )
    lab = Lab(
        cases=[c for c in CASES if c.split != "heldout"],
        run_case=FakeRunCase(),
        simulator=FakeSimulator(),
        runner=LocalRunner(),
        referee=referee,
        improver=RoutingChat(),
        workers=4,
        max_evals=4,
        suite_digest="suite",
        max_clusters=1,
    )
    result = evolve(improver_store, model=None, generations=1, lab=lab, seed=LOOP_SEED)
    held = FileStore(tmp_path / "referee", trusted=signer.public_key)
    admitted = held.decisions({"admitted": True})
    assert admitted and result.generations[0].promoted is not None
    for d in admitted:
        payload = verify(base64.b64decode(d["envelope"]), {signer.keyid: signer.public_key})
        assert payload["candidate"] == d["candidate"]
    assert FileActivator(held).active()[0] == result.generations[0].promoted
    assert not FileStore(tmp_path / "improver", trusted=key.public_key()).decisions()
