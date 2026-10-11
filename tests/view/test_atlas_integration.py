"""The CLI against a real Atlas store: the commands read what the store really holds, and `live` follows a
real change stream."""

from __future__ import annotations

import base64
import io
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from rich.console import Console

from ratchet.cli import LiveBackend, main
from ratchet.improve.receipt import DECISION_TYPE, Signer
from ratchet.improve.replay import GuardRunner
from ratchet.improve.sim import RecordedViews, Simulator
from ratchet.store import Store
from ratchet.types import Harness, Trace, canonical_json
from ratchet.view.matrix import CaseInfo

from .fakes import RULES, FakeRunner, info, make_trace, sample_guards, sample_traces

pytestmark = pytest.mark.integration
SIGNER = Signer.from_seed_hex("55" * 32)


class AtlasBackend(LiveBackend):
    """The real store and the real command wiring, with the case suite and the sandbox swapped for fakes
    (the recorded views in the traces stand in for a rebuilt world)."""

    def case_info(self, case_id: str) -> CaseInfo | None:
        return info(case_id)

    def simulator(self) -> Simulator:
        return RecordedViews()

    def runner(self) -> AbstractContextManager[GuardRunner]:
        return _fake_runner()


@contextmanager
def _fake_runner() -> Iterator[GuardRunner]:
    yield FakeRunner(RULES)


def seed_atlas(store: Store) -> tuple[Harness, Harness]:
    guards = {g.id: g for g in sample_guards()}
    seed = Harness.build(
        parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=["a"], created_by="seed"
    )
    good = seed.child(guards=[guards["g-retry"]], created_by="improver")
    store.put_harness(seed)
    store.put_harness(good)
    for t in sample_traces():
        store.put_trace(Trace(t.run_id, t.case_id, seed.version, t.model, t.events, t.outcome))
    body = {
        "type": "decision",
        "parent": seed.version,
        "candidate": good.version,
        "candidate_digest": good.digest,
        "admitted": True,
        "checks": [{"name": "replay", "passed": True, "evidence": {}}],
        "context": {"generation": 1, "id": "c-g-retry", "kind": "guard", "change": asdict(guards["g-retry"])},
        "decided_at": "2026-10-11T12:00:00+00:00",
    }
    env = SIGNER.sign(DECISION_TYPE, canonical_json(body))
    store.put_decision(
        {**body, "envelope": base64.b64encode(env).decode(), "keyid": SIGNER.keyid},
        decision_id=f"{seed.version}->{good.version}",
    )
    other = {"generation": 1, "id": "c-g-all", "kind": "guard", "change": asdict(guards["g-all-refunds"])}
    rejected = {**body, "candidate": "rej00000001", "admitted": False, "context": other}
    store.put_decision(rejected, decision_id=f"{seed.version}->rej00000001")
    store.put_eval({
        "type": "eval", "harness_version": seed.version, "train": {"passed": 1, "total": 5},
        "heldout": {"passed": 2, "total": 6}, "results": [], "cost": {"model_calls": 4},
    })  # fmt: skip
    return seed, good


def run(argv: list[str], uri: str, db: str) -> tuple[int, str]:
    buf = io.StringIO()
    env = {"RATCHET_MONGODB_URI": uri, "RATCHET_DB": db}
    code = main(argv, backend=AtlasBackend(env), console=Console(file=buf, width=140, height=35))
    return code, buf.getvalue()


def test_lineage_trace_decision_and_verify_read_the_real_store(atlas_db: tuple[str, str, Store]) -> None:
    uri, db, store = atlas_db
    seed, good = seed_atlas(store)
    code, out = run(["show", "lineage"], uri, db)
    assert code == 0
    assert seed.version in out
    assert good.version in out
    assert "rej00000001" in out
    code, out = run(["show", "trace", "r-t1/refund.t1"], uri, db)
    assert code == 0
    assert "charge refunded twice" in out
    code, out = run(["show", "decision", good.version], uri, db)
    assert code == 0
    assert "ADMITTED" in out
    pub = SIGNER.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    code, out = run(["verify", good.version, "--pubkey", pub], uri, db)
    assert code == 0
    assert "VERIFIED" in out
    assert "independent verifier" in out


def test_replay_and_report_run_over_real_traces(atlas_db: tuple[str, str, Store], tmp_path: Path) -> None:
    uri, db, store = atlas_db
    seed, _ = seed_atlas(store)
    code, out = run(["replay", "--delay", "0", "--full"], uri, db)
    assert code == 0
    assert f"parent {seed.version}" in out
    assert "0 model calls" in out
    assert "ADMITTED" in out
    assert "REJECTED" in out
    target = tmp_path / "report.html"
    code, _ = run(["report", "-o", str(target)], uri, db)
    assert code == 0
    assert "Replay matrix: parent" in target.read_text()


def test_live_prints_a_trace_the_moment_it_is_written(atlas_db: tuple[str, str, Store]) -> None:
    uri, db, store = atlas_db
    seed_atlas(store)

    stop = threading.Event()

    def write_until_seen() -> None:
        """The stream opens when the command has read the lineage; keep writing until it has printed one."""
        n = 0
        while not stop.wait(1.0):
            n += 1
            store.put_trace(
                make_trace(f"run-live-{n}", "refund.t1", False, [], reasons=["arrived while watching"])
            )

    writer = threading.Thread(target=write_until_seen)
    writer.start()
    code, out = run(["live", "--collections", "traces", "--max-events", "1", "--timeout", "20"], uri, db)
    stop.set()
    writer.join()
    assert code == 0
    assert "refund.t1" in out
    assert "arrived while watching" in out


class JsonModel:
    """Answers every request with the same JSON object and no tool call: the worker ends at once, and the
    improver's replies parse but propose nothing."""

    name = "static"

    def complete(self, messages: object, tools: object) -> object:
        from ratchet.agent.model import scripted_turn

        return scripted_turn('{"summary": "nothing to add", "candidates": []}')


class EndToEnd(LiveBackend):
    def model(self) -> object:
        return JsonModel()


def test_evolve_runs_the_real_loop_against_atlas_and_the_views_read_it_back(
    atlas_db: tuple[str, str, Store], monkeypatch: pytest.MonkeyPatch
) -> None:
    uri, db, store = atlas_db
    monkeypatch.setenv("RATCHET_SIGNING_KEY", "66" * 32)
    buf = io.StringIO()
    env = {"RATCHET_MONGODB_URI": uri, "RATCHET_DB": db}
    code = main(
        ["evolve", "--generations", "1", "--workers", "4"],
        backend=EndToEnd(env),
        console=Console(file=buf, width=140),
    )
    out = buf.getvalue()
    assert code == 0, out
    assert "harness lineage" in out
    assert "head:" in out
    assert store.lineage()
    buf2 = io.StringIO()
    assert main(["show", "lineage"], backend=LiveBackend(env), console=Console(file=buf2, width=140)) == 0
    assert "ACTIVE rev 1" in buf2.getvalue()
