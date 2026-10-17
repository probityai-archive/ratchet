"""`ratchet verify` and `show decision --envelope-out` on a real signed decision: the improvement loop runs
(fake world and model, real guard checks, real Atlas), its receipts are stored, and the CLI verifies them."""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from improve.fakes import FakeRunCase, FakeSimulator, LocalRunner
from improve.test_loop import CASES, SEED, RoutingChat
from rich.console import Console

from ratchet.cli import LiveBackend, main
from ratchet.improve.loop import EvolveResult, Lab, evolve
from ratchet.improve.receipt import Signer
from ratchet.improve.referee import RefereeCore, RefereeParts
from ratchet.improve.replay import GuardRunner
from ratchet.improve.sim import Simulator
from ratchet.store.activation import MongoActivator
from ratchet.store.atlas import Store
from ratchet.view.matrix import CaseInfo

pytestmark = pytest.mark.integration
SIGNER = Signer.from_seed_hex("77" * 32)
PUB = SIGNER.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


class FakeWorldBackend(LiveBackend):
    """The real store and command wiring; the fake world's simulator, runner and case kinds."""

    def case_info(self, case_id: str) -> CaseInfo | None:
        case = next((c for c in CASES if c.id == case_id), None)
        return None if case is None else CaseInfo(case.kind, "refund", case.split)

    def simulator(self) -> Simulator:
        return FakeSimulator()

    def runner(self) -> AbstractContextManager[GuardRunner]:
        return _local()


@contextmanager
def _local() -> Iterator[GuardRunner]:
    yield LocalRunner()


@pytest.fixture
def evolved(atlas_db: tuple[str, str, Store]) -> Iterator[tuple[str, str, EvolveResult, Store]]:
    """The loop on the main database; the referee on `<db>_heldout`, where its decisions are. The CLI reads
    both, as an auditor does."""
    uri, db, store = atlas_db
    held = Store(uri, f"{db}_heldout")
    referee = RefereeCore(
        RefereeParts(
            cases=[c for c in CASES if c.split == "heldout"],
            run_case=FakeRunCase(),
            model=None,
            simulator=FakeSimulator(),
            runner=LocalRunner(),
            signer=SIGNER,
            store=held,
            activator=MongoActivator(held._client, f"{db}_heldout"),
            suite_digest="ab" * 32,
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
        max_clusters=1,
        suite_digest="ab" * 32,
    )
    result = evolve(store, model=None, generations=1, lab=lab, seed=SEED)
    yield uri, db, result, held
    held._client.drop_database(f"{db}_heldout")
    held.close()


def run(argv: list[str], uri: str, db: str) -> tuple[int, str]:
    buf = io.StringIO()
    env = {"RATCHET_MONGODB_URI": uri, "RATCHET_DB": db}
    code = main(argv, backend=FakeWorldBackend(env), console=Console(file=buf, width=150))
    return code, buf.getvalue()


def admitted_decision(store: Store) -> dict[str, Any]:
    return next(d for d in store.decisions() if d.get("admitted"))


def test_verify_recomputes_from_the_store_and_keeps_recomputed_apart_from_authenticated_only(
    evolved: tuple[str, str, EvolveResult, Store],
) -> None:
    uri, db, result, store = evolved
    assert result.generations[0].promoted
    decision = admitted_decision(store)
    code, out = run(["verify", decision["candidate"], "--pubkey", PUB], uri, db)
    assert code == 0, out
    assert "VERIFIED" in out
    assert "reason code supported: admitted_receipt_verified" in out
    assert "RECOMPUTED (10 of 10)" in out
    for item in ("report_digests", "gate:heldout", "gate:replay", "gate:overfit_lint", "gate:authority"):
        assert item in out
    assert "AUTHENTICATED ONLY (signed, not re-derived)  fresh_model_execution" in out
    assert "6 of 6 named reports checked; every recomputable item recomputed" in out
    assert "checked by the dsse crate" in out


def test_a_rejection_receipt_verifies_too(evolved: tuple[str, str, EvolveResult, Store]) -> None:
    uri, db, _, store = evolved
    rejected = next(d for d in store.decisions() if not d.get("admitted"))
    code, out = run(["verify", rejected["candidate"], "--pubkey", PUB], uri, db)
    assert code == 0, out
    assert "rejection_receipt_verified" in out


def test_a_stored_report_edited_after_signing_is_contradicted(
    evolved: tuple[str, str, EvolveResult, Store],
) -> None:
    uri, db, _, store = evolved
    decision = admitted_decision(store)
    run_one = next(e for e in store.evals() if e.get("role") == "candidate" and e.get("type") == "eval")
    store.db["evals"].update_one({"_id": run_one["_id"]}, {"$set": {"wall_s": 999.0}})
    _, out = run(["verify", decision["candidate"], "--pubkey", PUB], uri, db)
    assert "recomputation INCOMPLETE" in out or "NOT SUPPLIED" in out or "PARTIAL" in out
    assert "RECOMPUTED (10 of 10)" not in out


def test_a_tampered_stored_decision_is_caught_before_any_recomputation(
    evolved: tuple[str, str, EvolveResult, Store],
) -> None:
    uri, db, _, store = evolved
    decision = admitted_decision(store)
    store.db["decisions"].update_one({"_id": decision["_id"]}, {"$set": {"admitted": False}})
    code, out = run(["verify", decision["candidate"], "--pubkey", PUB], uri, db)
    assert code == 1
    assert "TAMPERED" in out
    assert "RECOMPUTED (" not in out


def test_missing_evidence_prints_as_not_supplied(evolved: tuple[str, str, EvolveResult, Store]) -> None:
    uri, db, _, store = evolved
    decision = admitted_decision(store)
    store.db["evals"].delete_many({"type": "eval"})
    _, out = run(["verify", decision["candidate"], "--pubkey", PUB], uri, db)
    assert "NOT SUPPLIED" in out
    assert "recomputation INCOMPLETE" in out


def test_envelope_out_writes_the_signed_bytes_and_verify_accepts_the_file(
    evolved: tuple[str, str, EvolveResult, Store], tmp_path: Path
) -> None:
    uri, db, _, store = evolved
    decision = admitted_decision(store)
    target = tmp_path / "receipt.dsse"
    code, out = run(["show", "decision", decision["candidate"], "--envelope-out", str(target)], uri, db)
    assert code == 0
    assert str(target) in out
    assert target.read_bytes() == base64.b64decode(decision["envelope"])
    code, out = run(["verify", str(target), "--pubkey", PUB], uri, db)
    assert code == 0
    assert "VERIFIED" in out
    wrong = Signer.from_seed_hex("88" * 32).public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    assert run(["verify", str(target), "--pubkey", wrong], uri, db)[0] == 1
