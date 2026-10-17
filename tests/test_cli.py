from __future__ import annotations

import base64
import io
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from rich.console import Console
from view.fakes import (
    RULES,
    FakeRunner,
    FakeStore,
    decision_doc,
    eval_doc,
    info,
    make_trace,
    scenario,
)
from view.test_live import Scripted, change
from view.test_telemetry import FakeSink

from ratchet import cli
from ratchet.canonical import canonical_json
from ratchet.cli import CliError, LiveBackend, main
from ratchet.improve.receipt import DECISION_TYPE, Signer
from ratchet.improve.sim import RecordedViews
from ratchet.types import Harness

SIGNER = Signer.from_seed_hex("33" * 32)


class FakeBackend:
    def __init__(self, store: FakeStore | None = None, sink: FakeSink | None = None) -> None:
        self.seed_h: Harness | None
        if store is None:
            self.store_obj, self.seed_h, _ = scenario()
        else:
            self.store_obj, self.seed_h = store, next(iter(store.harnesses.values()), None)
        self.sink = sink or FakeSink()
        self.env: dict[str, str] = {}
        self.evolved: list[tuple[int, int]] = []
        self.resumed: tuple[str, str, str] = ("", "", "")

    @contextmanager
    def open_store(self) -> Iterator[FakeStore]:
        yield self.store_obj

    def model(self) -> object:
        return object()

    def seed(self) -> Harness:
        assert self.seed_h is not None
        return self.seed_h

    def cases(self, split: str, family: str | None, ids: Sequence[str]) -> list[Any]:
        pool = [
            SimpleNamespace(id="refund.t1", split="train", family="refund", kind="trap"),
            SimpleNamespace(id="refund.t3", split="heldout", family="refund", kind="trap"),
            SimpleNamespace(id="mail.t3", split="train", family="mail", kind="trap"),
        ]
        out = [
            c
            for c in pool
            if (split == "all" or c.split == split) and (family is None or c.family == family)
            and (not ids or c.id in ids)
        ]  # fmt: skip
        if not out:
            raise CliError("no cases match that selection")
        return out

    def case_info(self, case_id: str) -> Any:
        return info(case_id)

    def get_case(self, case_id: str) -> Any:
        return SimpleNamespace(id=case_id)

    def committed_calls(self, case: Any, saved: Any) -> dict[str, bool] | None:
        return {"c1": True}

    def resume_case(self, harness: Harness, case: Any, model: Any, store: Any, run_id: str) -> Any:
        self.resumed = (harness.version, case.id, run_id)
        return make_trace(
            run_id, case.id, True, [("payments_refund", {"charge_id": "ch1"})], version=harness.version
        )

    def evaluate(
        self, harness: Harness, cases: Sequence[Any], model: Any, store: Any, workers: int, repeats: int
    ) -> Any:
        v = harness.version
        traces = [
            make_trace(f"run-{c.id}", c.id, False, [("refund", {})], version=v, reasons=["boom"])
            for c in cases
        ]
        rows = [
            {"case_id": c.id, "split": c.split, "kind": "trap", "passed": False, "interventions": 0,
             "reasons": ["boom"]}
            for c in cases
        ]  # fmt: skip
        payload = eval_doc(v, (0, len(cases)), (0, 0), rows)
        return SimpleNamespace(
            to_payload=lambda: payload,
            traces=traces,
            harness_version=v,
            results=[SimpleNamespace(**r) for r in rows],
        )

    def evolve(
        self,
        store: Any,
        model: Any,
        generations: int,
        cases: Sequence[Any],
        workers: int,
        repeats: int,
        log: Any,
    ) -> Any:
        self.evolved.append((generations, len(cases)))
        log("generation 1 started")
        seed = self.seed()
        child = seed.child(system_prompt="more careful", created_by="improver")
        sid, cid = seed.version, child.version
        row = {"case_id": "refund.t1", "split": "train", "kind": "trap", "must_succeed": True,
               "run_id": "run-p", "interventions": 0}  # fmt: skip
        store.put_harness(seed)
        store.put_trace(make_trace("run-p", "refund.t1", False, [], version=sid))
        store.put_eval({**eval_doc(sid, (0, 1), (0, 1), [row]), "role": "parent"})
        store.put_eval({**eval_doc(cid, (1, 1), (1, 1), []), "role": "candidate"})
        store.put_decision(
            decision_doc(sid, cid, True, change={"description": "x"}), decision_id=f"{sid}->{cid}"
        )
        store.put_harness(child)
        store.put_eval(
            {"type": "generation", "generation": 1, "parent": sid, "level1": {"promoted": cid}},
            eval_id="gen-1",
        )
        won = {"heldout_passed": 1, "heldout_total": 1}
        gen = SimpleNamespace(generation=1, parent=sid, promoted=cid, note="",
                              level1={"parent_score": {"heldout_passed": 0}, "child_score": won})  # fmt: skip
        idle = SimpleNamespace(generation=2, parent=cid, promoted=None, level1={},
                               note="no training failures left")  # fmt: skip
        return SimpleNamespace(harness=child, generations=[gen, idle])

    @contextmanager
    def runner(self) -> Iterator[FakeRunner]:
        yield FakeRunner(RULES)

    def simulator(self) -> RecordedViews:
        return RecordedViews()

    def telemetry(self) -> FakeSink:
        return self.sink


def run(argv: list[str], backend: Any = None) -> tuple[int, str]:
    buf = io.StringIO()
    console = Console(file=buf, width=140, height=35, force_terminal=False)
    code = main(argv, backend=backend or FakeBackend(), console=console)
    return code, buf.getvalue()


# -- run ----------------------------------------------------------------------------------------------------


def test_run_evaluates_stores_and_prints() -> None:
    backend = FakeBackend()
    code, out = run(["run", "--split", "train"], backend)
    assert code == 0
    assert "running" in out
    assert "refund.t1" in out
    assert "refund.t3" not in out
    store = backend.store_obj
    assert store is not None
    assert len(store.trace_list) > 11
    assert store.eval_docs[-1]["role"] == "run"


def test_run_with_raindrop_sends_a_cohort_and_prints_the_cap() -> None:
    backend = FakeBackend()
    code, out = run(["run", "--raindrop", "--cohort", "candidate", "--experiment", "e1"], backend)
    assert code == 0
    assert backend.sink.evaluations[0]["cohort"] == "candidate"
    assert backend.sink.evaluations[0]["experiment"] == "e1"
    assert backend.sink.closed
    assert "120 of 900 monthly events used" in out


def test_a_raindrop_failure_never_fails_a_run() -> None:
    backend = FakeBackend(sink=FakeSink(fail=True))
    code, out = run(["run", "--raindrop"], backend)
    assert code == 0
    assert "raindrop: RuntimeError: raindrop down" in out
    refused = FakeBackend(sink=FakeSink(refuse="no room"))
    assert "raindrop: not sent: no room" in run(["run", "--raindrop"], refused)[1]


def test_run_rejects_an_unknown_harness_and_empty_selection() -> None:
    code, out = run(["run", "--harness", "nope"])
    assert code == 1
    assert "no harness version 'nope'" in out
    code, out = run(["run", "--family", "nothing"])
    assert code == 1
    assert "no cases match" in out


def test_run_a_stored_harness_by_version() -> None:
    backend = FakeBackend()
    _, _, good = scenario()
    store = backend.store_obj
    assert store is not None
    version = next(v for v in store.harnesses if v != backend.seed().version)
    code, out = run(["run", "--harness", version, "--case", "refund.t1"], backend)
    assert code == 0
    assert good.version == version
    assert "refund.t1" in out


# -- evolve -------------------------------------------------------------------------------------------------


def test_evolve_prints_generations_lineage_and_head() -> None:
    backend = FakeBackend()
    code, out = run(["evolve", "--generations", "2"], backend)
    assert code == 0
    assert backend.evolved == [(2, 3)]
    assert "generation 1 started" in out
    assert "promoted" in out
    assert "held-out 0 -> 1/1" in out
    assert "nothing admitted: no training failures left" in out
    assert "harness lineage" in out
    assert "head:" in out


def test_evolve_with_raindrop_sends_each_generation() -> None:
    backend = FakeBackend()
    code, out = run(["evolve", "--raindrop", "--experiment", "demo"], backend)
    assert code == 0
    sink = backend.sink
    assert [(e["cohort"], e["experiment"]) for e in sink.evaluations] == [
        ("baseline", "demo"),
        ("candidate", "demo"),
    ]
    assert len(sink.decisions) == 1
    assert sink.closed
    assert "monthly events used" in out


def test_evolve_with_a_dead_raindrop_still_records_the_decision() -> None:
    backend = FakeBackend(sink=FakeSink(fail=True))
    code, out = run(["evolve", "--raindrop"], backend)
    assert code == 0
    assert "admission is unaffected" in out
    assert backend.store_obj is not None
    assert any(d["admitted"] for d in backend.store_obj.decision_docs)


# -- replay -------------------------------------------------------------------------------------------------


def test_replay_defaults_to_the_latest_generation_and_prints_the_matrix() -> None:
    code, out = run(["replay", "--delay", "0", "--full"])
    assert code == 0
    assert "replay matrix" in out
    assert "ADMITTED" in out
    assert "REJECTED" in out
    assert "0 model calls" in out
    assert "rej00000002: procedure candidate: no guard to replay" in out
    assert "rcpt:" in out


def test_replay_named_candidate() -> None:
    code, out = run(["replay", "rej00000001", "--delay", "0"])
    assert code == 0
    assert "g-all-refunds" in out
    assert "g-retry" not in out


def test_replay_errors_are_plain() -> None:
    code, out = run(["replay"], FakeBackend(FakeStore()))
    assert code == 1
    assert "no candidates recorded yet" in out
    code, out = run(["replay", "rej00000002"])
    assert code == 1
    assert "nothing to replay" in out
    code, out = run(["replay", "nope"])
    assert code == 1
    assert "cannot tell which harness" in out


def test_replay_one_matrix_per_parent_and_a_column_cap() -> None:
    backend = FakeBackend()
    store = backend.store_obj
    assert store is not None
    seed = backend.seed()
    other = store.add_harness(seed.child(system_prompt="other"))
    store.trace_list += [
        t.__class__(t.run_id + "-o", t.case_id, other.version, t.model, t.events, t.outcome)
        for t in store.trace_list[:3]
    ]
    store.decision_docs.append(
        decision_doc(
            other.version,
            "kid00000001",
            False,
            change={"id": "g-any-mail", "description": "d", "source": "s"},
        )
    )
    code, out = run(
        ["replay", "--delay", "0", "--max-columns", "1", "c-g-all-refunds", "kid00000001"], backend
    )
    assert code == 0
    assert out.count("replay matrix") == 2
    code, out = run(["replay", "--delay", "0", "--max-columns", "1"], backend)
    assert "showing the first 1 of 2 guards" in out


def test_replay_warns_when_the_parent_has_no_traces() -> None:
    backend = FakeBackend()
    assert backend.store_obj is not None
    backend.store_obj.trace_list = []
    code, out = run(["replay"], backend)
    assert code == 1
    assert "no traces recorded for parent harness" in out


# -- show ---------------------------------------------------------------------------------------------------


def test_show_lineage_trace_and_decision() -> None:
    backend = FakeBackend()
    code, out = run(["show", "lineage"], backend)
    assert code == 0
    assert "rej00000001" in out
    assert "refund.t1" in run(["show", "trace", "r-t1/refund.t1"], backend)[1]
    assert "refund.t2" in run(["show", "trace", "r-t2"], backend)[1]
    code, out = run(["show", "decision", "rej00000001"], backend)
    assert code == 0
    assert "REJECTED" in out
    assert "why rejected" in out


def test_show_errors() -> None:
    assert "needs an id" in run(["show", "trace"])[1]
    assert "needs an id" in run(["show", "decision"])[1]
    assert "no trace 'zzz'" in run(["show", "trace", "zzz"])[1]
    assert "no trace 'a/b'" in run(["show", "trace", "a/b"])[1]
    code, out = run(["show", "decision", "zzz"])
    assert code == 1
    assert "no decision 'zzz'" in out


# -- verify -------------------------------------------------------------------------------------------------


def signed_backend(tamper: bool = False) -> tuple[FakeBackend, str]:
    backend = FakeBackend()
    store = backend.store_obj
    assert store is not None
    doc = decision_doc("p0", "c0", True)
    body = {
        k: doc[k]
        for k in ("parent", "candidate", "candidate_digest", "admitted", "checks", "context", "decided_at")
    }
    body["type"] = "decision"
    env = SIGNER.sign(DECISION_TYPE, canonical_json(body))
    doc.update(body)
    doc["envelope"] = base64.b64encode(env).decode()
    doc["keyid"] = SIGNER.keyid
    if tamper:
        doc["admitted"] = False
    store.decision_docs.append(doc)
    return backend, SIGNER.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def test_verify_a_stored_decision() -> None:
    backend, pub = signed_backend()
    code, out = run(["verify", "c0", "--pubkey", pub], backend)
    # the signature is good, but a hand-signed body is not a full admission predicate
    assert code == 1
    assert "✓ DSSE envelope signature valid" in out
    assert "NOT ESTABLISHED" in out
    assert "reason code not_established" in out
    backend.env = {"RATCHET_SIGNING_KEY": "33" * 32}
    assert run(["verify", "p0->c0"], backend)[0] == 1


def test_verify_catches_tampering_wrong_keys_and_missing_decisions(tmp_path: Path) -> None:
    backend, pub = signed_backend(tamper=True)
    code, out = run(["verify", "c0", "--pubkey", pub], backend)
    assert code == 1
    assert "TAMPERED" in out
    wrong = Signer.from_seed_hex("44" * 32).public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    assert run(["verify", "c0", "--pubkey", wrong], backend)[0] == 1
    code, out = run(["verify", "nope", "--pubkey", pub], backend)
    assert code == 1
    assert "no decision 'nope'" in out
    env_file = tmp_path / "r.dsse"
    env_file.write_bytes(
        SIGNER.sign(DECISION_TYPE, canonical_json({"parent": "p", "candidate": "c", "admitted": True}))
    )
    code, out = run(["verify", "--envelope", str(env_file), "--pubkey", pub], backend)
    assert code == 0
    assert str(env_file) in out


# -- live and report ----------------------------------------------------------------------------------------


def test_live_prints_the_lineage_then_follows() -> None:
    store, _, _ = scenario()
    scripted = Scripted(
        {"traces": [[change("traces", {"case_id": "fresh.case", "passed": True, "harness_version": "h"})]]}
    )
    scripted.harnesses, scripted.decision_docs = store.harnesses, store.decision_docs
    backend = FakeBackend(scripted)
    code, out = run(["live", "--collections", "traces", "--max-events", "1"], backend)
    assert code == 0
    assert "live: traces" in out
    assert "harness lineage" in out
    assert "fresh.case" in out


def test_report_writes_one_html_file(tmp_path: Path) -> None:
    target = tmp_path / "r.html"
    code, out = run(["report", "-o", str(target), "--title", "Demo"])
    assert code == 0
    html = target.read_text()
    assert "<title>Demo</title>" in html
    assert "Replay matrix: parent" in html
    assert "wrote" in out
    bare = FakeBackend(FakeStore())
    code, out = run(["report", "-o", str(tmp_path / "e.html")], bare)
    assert code == 0
    assert "report has no replay matrix" in out


# -- main ---------------------------------------------------------------------------------------------------


def test_main_reports_unexpected_errors_and_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom(FakeBackend):
        def open_store(self) -> Any:
            raise ValueError("kaput")

    code, out = run(["show", "lineage"], Boom())
    assert code == 1
    assert "ratchet: ValueError: kaput" in out
    monkeypatch.setenv("RATCHET_DEBUG", "1")
    assert run(["show", "lineage"], Boom())[0] == 1

    class Stop(FakeBackend):
        def open_store(self) -> Any:
            raise KeyboardInterrupt

    assert run(["show", "lineage"], Stop())[0] == 130


def test_the_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_default_console_and_backend_report_the_missing_store_uri(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("RATCHET_MONGODB_URI", raising=False)
    assert main(["show", "lineage"]) == 1
    assert "RATCHET_MONGODB_URI is not set" in capsys.readouterr().err


# -- the real wiring ----------------------------------------------------------------------------------------


def test_live_backend_selects_cases_from_the_real_suite() -> None:
    b = LiveBackend({})
    train = b.cases("train", "refund_lost_reply", [])
    assert {c.id for c in train} == {"refund_lost_reply.v1", "refund_lost_reply.v2"}
    assert [c.id for c in b.cases("all", None, ["ordinary.v1"])] == ["ordinary.v1"]
    with pytest.raises(CliError):
        b.cases("train", "no_such_family", [])
    ci = b.case_info("refund_lost_reply.v1")
    assert ci is not None
    assert (ci.kind, ci.family) == ("trap", "refund_lost_reply")
    assert b.case_info("nope") is None
    assert b.seed().created_by == "seed"
    assert hasattr(b.simulator(), "walk")


def test_live_backend_telemetry_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAINDROP_WRITE_KEY", raising=False)

    def no_secret_tool(*a: Any, **k: Any) -> Any:
        raise OSError("no secret-tool")

    monkeypatch.setattr("ratchet.telemetry.raindrop.subprocess.run", no_secret_tool)
    with pytest.raises(CliError, match="RAINDROP_WRITE_KEY"):
        LiveBackend({}).telemetry()


def test_live_backend_runs_the_real_worker_with_a_scripted_model() -> None:
    from ratchet.agent.model import ScriptedModel, scripted_turn

    b = LiveBackend({})
    cases = b.cases("train", None, ["ordinary.v1"])
    model = ScriptedModel([scripted_turn("I looked at the ticket and have nothing to add.")])
    report = b.evaluate(b.seed(), cases, model, None, 1, 1)
    assert [r.case_id for r in report.results] == ["ordinary.v1"]
    assert report.traces[0].events[-1].kind == "final"


def test_live_backend_guard_runner_opens_and_closes_the_runtime() -> None:
    with LiveBackend({}).runner() as runner:
        assert runner.check_batch.__name__ == "check_batch"


def test_live_backend_model_comes_from_the_environment() -> None:
    env = {"RATCHET_BASE_URL": "http://localhost:1/v1", "RATCHET_API_KEY": "k", "RATCHET_MODEL": "m"}
    assert LiveBackend(env).model().name == "m"
    assert cli.backend_env(LiveBackend(env)) == env
    assert cli.backend_env(FakeBackend()) == {}


def test_audit_cases_run_only_when_named_and_never_enter_the_matrix() -> None:
    b = LiveBackend({})
    assert all(c.split in ("train", "heldout") for c in b.cases("all", None, []))
    assert {c.split for c in b.cases("audit", None, [])} == {"audit"}
    audit = next(c for c in b.cases("audit", None, []))
    ci = b.case_info(audit.id)
    assert ci is not None
    assert ci.split == "audit"


# -- resume -------------------------------------------------------------------------------------------------


def checkpoint(status: str = "running", version: str | None = None) -> dict[str, Any]:
    call = {"name": "payments_refund", "args": {"charge_id": "ch1"}, "call_id": "c1"}
    result = {"result": {"call_id": "c1", "ok": False, "error": "timeout"}}
    return {
        "status": status, "run_id": "run-x", "case_id": "refund.t1", "harness_version": version or "",
        "events": [{"seq": 0, "kind": "tool_result", "payload": result, "ts": 1.0}],
        "executed": [call], "pending": [{"name": "tickets_get", "args": {}, "call_id": "c2"}],
    }  # fmt: skip


def test_resume_reports_what_was_not_repeated() -> None:
    backend = FakeBackend()
    assert backend.store_obj is not None
    backend.store_obj.checkpoints["run-x"] = checkpoint(version=backend.seed().version)
    code, out = run(["resume", "run-x"], backend)
    assert code == 0, out
    assert "RESUMING" in out
    assert "reply lost (timeout)" in out
    assert "asked for but not yet run when it stopped: tickets_get" in out
    assert "NOT REPEATED" not in out or "RE-ISSUED" in out  # the fake run repeats the refund by design
    assert "RE-ISSUED BY THE MODEL" in out
    assert backend.resumed[2] == "run-x"
    assert any(t.run_id == "run-x" for t in backend.store_obj.trace_list)


def test_resume_refuses_a_missing_or_finished_run() -> None:
    backend = FakeBackend()
    assert backend.store_obj is not None
    code, out = run(["resume", "nope"], backend)
    assert code == 1
    assert "no checkpoint" in out
    backend.store_obj.checkpoints["run-x"] = checkpoint("final", backend.seed().version)
    code, out = run(["resume", "run-x"], backend)
    assert code == 1
    assert "already finished" in out
    backend.store_obj.checkpoints["run-x"] = checkpoint("running", "0" * 12)
    code, out = run(["resume", "run-x"], backend)
    assert code == 1
    assert "not in the store" in out


def test_show_decision_envelope_out_and_verify_a_file(tmp_path: Path) -> None:
    backend, pub = signed_backend()
    target = tmp_path / "r.dsse"
    code, out = run(["show", "decision", "c0", "--envelope-out", str(target)], backend)
    assert code == 0
    assert "wrote the signed envelope" in out
    code, out = run(["verify", str(target), "--pubkey", pub], backend)
    assert code == 0
    assert "VERIFIED" in out
    assert backend.store_obj is not None
    backend.store_obj.decision_docs[-1]["envelope"] = None
    code, out = run(["show", "decision", "c0", "--envelope-out", str(target)], backend)
    assert code == 1
    assert "no signed envelope" in out


def test_verify_a_jsonl_log_shows_the_pinned_countersignature(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from ratchet.improve.countersign import countersign_doc

    pub = SIGNER.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    run_key = Signer.from_seed_hex("55" * 32)
    rows = []
    for cand in ("c1", "c2"):
        body = {"parent": "p", "candidate": cand, "admitted": False}
        env = run_key.sign(DECISION_TYPE, canonical_json(body))
        rows.append({**body, "decision_id": f"p->{cand}", "keyid": run_key.keyid,
                     "envelope": base64.b64encode(env).decode()})  # fmt: skip
    rows[0].update(countersign_doc(rows[0], SIGNER, datetime(2026, 10, 12, tzinfo=UTC)) or {})
    log = tmp_path / "decisions.jsonl"
    log.write_text("\n".join(json.dumps({"decision": r}) for r in rows) + "\n", encoding="utf-8")
    code, out = run(["verify", "p->c1", "--jsonl", str(log), "--pubkey", pub], FakeBackend())
    assert code == 0 and "countersigned by the pinned key on 2026-10-12" in out
    assert run(["verify", "--jsonl", str(log), "--pubkey", pub], FakeBackend())[0] == 1
    assert run(["verify", "nope", "--jsonl", str(log), "--pubkey", pub], FakeBackend())[0] == 1


def test_demo_double_refund_runs_the_scenario_with_the_store_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ratchet.demo import scenario

    seen: list[tuple[str, str | None, bool]] = []
    monkeypatch.setattr(
        scenario, "main", lambda db, uri=None, reset=False: seen.append((db, uri, reset)) or 0
    )
    monkeypatch.setenv(cli.URI_ENV, "mongodb://example:1/")
    assert main(["demo", "double-refund", "--reset"]) == 0
    monkeypatch.delenv(cli.URI_ENV)
    assert main(["demo", "double-refund"]) == 0
    assert seen == [("ratchet_demo", "mongodb://example:1/", True), ("ratchet_demo", None, False)]


def test_demo_names_only_the_double_refund_scenario() -> None:
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["demo", "admit"])
