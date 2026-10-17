"""A referee results file, verified against its bundles: the receipts, the per-case held-out tables and the
held-out grades recomputed from the disclosed cases."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from improve.fakes import FakeSimulator, LocalRunner
from improve.test_loop import MemStore
from improve.test_referee_policies import (
    FIX,
    HELDOUT_IDS,
    MODEL,
    SEED,
    Reps,
    _core,
    _original,
    _request,
    _train,
)
from improve.test_referee_policies import TRAIN as FAKE_TRAIN
from ratchet_cases.cases import get, suite_digest

from ratchet.canonical import canonical_json
from ratchet.improve.activate import MemoryActivator
from ratchet.improve.evaluate import evaluate
from ratchet.improve.evaluator import evaluator_identity
from ratchet.improve.overfit import Literals
from ratchet.improve.receipt import Signer
from ratchet.improve.referee import RefereeCore, RefereeParts, Request
from ratchet.verify import dsse
from ratchet.verify.results import RESULTS_TYPE, main, verify_results

from .runs_scenario import CANDIDATE, CASES, HELDOUT, PARENT, TRAIN, _run

SUITE = "5" * 64  # a well-formed suite digest for the fake world, whose cases no published suite holds


def _real(tmp_path: Path, signer: Signer) -> Path:
    """A referee over two of the suite's held-out cases, with the worker on a scripted model."""
    store = MemStore()
    core = RefereeCore(
        RefereeParts(
            cases=[get(i) for i in HELDOUT],
            run_case=_run,
            model=None,
            simulator=FakeSimulator(),
            runner=LocalRunner(),
            signer=signer,
            store=store,
            activator=MemoryActivator(store.decision),
            suite_digest=suite_digest(),
            workers=1,
            results_dir=tmp_path / "results",
        )
    )
    train = [get(TRAIN)]
    core.start(evaluator_identity(train, {}, suite_digest()))
    core.parent_runs(1, PARENT, evaluate(PARENT, train, _run, None, workers=1))
    request = Request(CANDIDATE, evaluate(CANDIDATE, train, _run, None, workers=1), [], {},
                      Literals(set(), set(), set()), [TRAIN], {})  # fmt: skip
    core.decide_many(1, [request])
    return Path(core.results("real")["path"])


def _resign(path: Path, signer: Signer, edit: Any) -> None:
    doc = json.loads(dsse.parse(path.read_bytes()).payload)
    edit(doc)
    path.write_bytes(signer.sign(RESULTS_TYPE, canonical_json(doc)))


def test_a_results_file_verifies_and_every_held_out_run_is_graded_again(tmp_path: Path) -> None:
    signer = Signer.generate()
    path = _real(tmp_path, signer)
    report = verify_results(path.read_bytes(), signer.public_key, path.parent, CASES)
    assert report.ok, report.reason
    (detail,) = report.recomputed
    assert detail["cases"] == len(HELDOUT)
    assert detail["heldout_grades"]["status"] == "recomputed"
    assert detail["heldout_grades"]["regraded"] == detail["heldout_grades"]["heldout_rows"] > 0


def test_a_changed_table_a_changed_report_and_a_foreign_receipt_are_each_caught(tmp_path: Path) -> None:
    signer = Signer.generate()
    path = _real(tmp_path, signer)
    root = path.parent
    pristine = path.read_bytes()

    def flip(doc: dict[str, Any]) -> None:
        outcomes = doc["decisions"][0]["heldout"]["outcomes"]
        first = next(iter(outcomes))
        outcomes[first][0] = "pass" if outcomes[first][0] == "fail" else "fail"

    _resign(path, signer, flip)
    assert verify_results(path.read_bytes(), signer.public_key, root, CASES).reason.startswith(
        "heldout_table_mismatch"
    )
    path.write_bytes(pristine)
    entry = json.loads(dsse.parse(pristine).payload)["decisions"][0]
    report_file = root / entry["bundle"] / "reports" / "candidate_run.json"
    doc = json.loads(report_file.read_text())
    doc["wall_s"] += 1
    report_file.write_text(json.dumps(doc))
    got = verify_results(pristine, signer.public_key, root, CASES)
    assert got.decision == "contradicted" and got.reason.startswith("receipt_does_not_verify")
    _resign(path, signer, lambda d: d["decisions"][0].update(receipt_sha256="0" * 64))
    assert verify_results(path.read_bytes(), signer.public_key, root, CASES).reason.startswith(
        "receipt_not_the_listed_one"
    )
    _resign(path, signer, lambda d: d["decisions"][0].update(bundle="../../outside"))
    assert verify_results(path.read_bytes(), signer.public_key, root, CASES).reason.startswith(
        "bundle_outside_results"
    )


def test_the_envelope_itself_must_be_ours(tmp_path: Path) -> None:
    signer = Signer.generate()
    path = _real(tmp_path, signer)
    assert (
        verify_results(path.read_bytes(), Signer.generate().public_key, path.parent).reason
        == "signature_invalid"
    )
    assert verify_results(b"{", signer.public_key, path.parent).decision == "no_verdict"
    other = signer.sign("application/json", b"{}")
    assert verify_results(other, signer.public_key, path.parent).reason == "unsupported_payload_type"
    wrong = signer.sign(RESULTS_TYPE, canonical_json({"type": "x"}))
    assert verify_results(wrong, signer.public_key, path.parent).reason == "unsupported_results_version"


def test_ruling_two_repeatability_and_discordance_are_recomputed(tmp_path: Path) -> None:
    signer = Signer.generate()
    core = _core(tmp_path, signer, MemStore(), suite_digest=SUITE)
    core.start(evaluator_identity(FAKE_TRAIN, MODEL, SUITE))
    core.deterministic(SEED, _request(FIX, "det:fix"), _train(SEED), repeat_heldout=2)
    path = Path(core.results("det")["path"])
    report = verify_results(path.read_bytes(), signer.public_key, path.parent)
    assert report.ok, report.reason
    ids = {d.get("id") for d in report.recomputed}
    assert {"repeatability", "parent_discordance"} <= ids
    trace_file = path.parent / "runs" / "repeat_b.jsonl"
    lines = trace_file.read_text().splitlines()
    doc = json.loads(lines[0])
    doc["events"][0]["payload"]["content"] = "changed"
    trace_file.write_text("\n".join([json.dumps(doc), *lines[1:]]) + "\n")
    assert verify_results(path.read_bytes(), signer.public_key, path.parent).reason == (
        "repeatability_digests_b_mismatch"
    )


def _v3(tmp_path: Path, signer: Signer) -> Path:
    originals = {("src", FIX.version): _original(FIX)}
    core = _core(
        tmp_path,
        signer,
        MemStore(),
        repetitions=Reps(),
        decisions=lambda db, v: originals.get((db, v)),
        suite_digest=SUITE,
    )
    core.stochastic(SEED, [f"src:{FIX.version}"], HELDOUT_IDS[:3], 7, workers=8)
    return Path(core.results("v3")["path"])


def test_v3_counts_are_recounted_from_the_signed_execution_records(tmp_path: Path) -> None:
    signer = Signer.generate()
    path = _v3(tmp_path, signer)
    report = verify_results(path.read_bytes(), signer.public_key, path.parent)
    assert report.ok, report.reason
    (detail,) = report.recomputed
    assert detail["cases"] == 3 and detail["executions"] == 2 * 3 * 40
    assert report.authenticated_only == ["v3_identity_controls"]
    _resign(path, signer, lambda d: d["decisions"][0]["per_case"][0].update(arm=41))
    assert verify_results(path.read_bytes(), signer.public_key, path.parent).reason.startswith(
        "v3_count_mismatch"
    )


def test_v3_execution_records_must_be_the_signed_ones(tmp_path: Path) -> None:
    signer = Signer.generate()
    path = _v3(tmp_path, signer)
    arms = next((path.parent / "cohorts").iterdir()) / "arms" / "parent.json"
    records = json.loads(arms.read_text())
    records[0]["passed"] = not records[0]["passed"]
    arms.write_text(json.dumps(records))
    assert verify_results(path.read_bytes(), signer.public_key, path.parent).reason.startswith(
        "arm_records_not_signed"
    )


def _pem(tmp_path: Path, signer: Signer) -> Path:
    pem = tmp_path / "key.pub"
    pem.write_bytes(signer.public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))
    return pem


def test_the_command_line_exit_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    signer = Signer.generate()
    path = _real(tmp_path, signer)
    pem = _pem(tmp_path, signer)
    case_files = []
    for cid, doc in CASES.items():
        f = tmp_path / f"{cid}.json"
        f.write_text(json.dumps(doc))
        case_files += ["--case", f"{cid}={f}"]
    assert main([str(path), "--key", str(pem), *case_files]) == 0
    assert "1 of 1 decisions verified" in capsys.readouterr().out
    assert main([str(path), "--key", str(pem), "--case-suite", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    (tmp_path / "other").mkdir()
    assert main([str(path), "--key", str(_pem(tmp_path / "other", Signer.generate()))]) == 1
    assert main([str(path), "--key", str(pem), "--case", "bad"]) == 2
    copy = tmp_path / "copy"  # the directory verifies on its own, wherever it is put
    shutil.copytree(path.parent, copy)
    assert main([str(copy / path.name), "--key", str(pem)]) == 0


def test_v3_executions_on_suite_cases_are_graded_again_from_the_case_documents(tmp_path: Path) -> None:
    signer, store = Signer.generate(), MemStore()
    brief = PARENT.child(system_prompt=f"{PARENT.system_prompt}\n\nBe brief.", created_by="improver")
    original = {
        "parent": PARENT.version,
        "candidate": brief.version,
        "context": {"kind": "prompt_edit", "change": {"append": "Be brief."}},
        "checks": [
            {"name": "replay", "passed": True, "evidence": {}},
            {"name": "overfit_lint", "passed": True, "evidence": {"hits": []}},
        ],
    }
    core = RefereeCore(
        RefereeParts(
            cases=[get(i) for i in HELDOUT],
            run_case=_run,
            model=None,
            simulator=FakeSimulator(),
            runner=LocalRunner(),
            signer=signer,
            store=store,
            activator=MemoryActivator(store.decision),
            suite_digest=suite_digest(),
            results_dir=tmp_path / "results",
            repetitions=Reps(),
            decisions=lambda db, v: original if (db, v) == ("src", brief.version) else None,
        )
    )
    out = core.stochastic(PARENT, [f"src:{brief.version}"], [HELDOUT[0]], 11, workers=8)
    assert out["invalid"] == 0
    path = Path(core.results("v3-suite")["path"])
    report = verify_results(path.read_bytes(), signer.public_key, path.parent, CASES)
    assert report.ok, report.reason
    (detail,) = report.recomputed
    assert detail["regraded"] == detail["executions"] == 2 * 40
    trace_file = next((path.parent / "cohorts").iterdir()) / "traces" / "parent.jsonl"
    lines = trace_file.read_text().splitlines()
    doc = json.loads(lines[0])
    doc["outcome"]["final_state_digest"] = "0" * 64
    trace_file.write_text("\n".join([json.dumps(doc), *lines[1:]]) + "\n")
    got = verify_results(path.read_bytes(), signer.public_key, path.parent, CASES)
    assert got.reason.startswith("execution_grade_mismatch") and got.reason.endswith(":final_state")
