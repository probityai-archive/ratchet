from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from ratchet.canonical import canonical_json
from ratchet.improve.receipt import Signer
from ratchet.verify import DEFAULT_GATES
from ratchet.verify.adapter import AUTHENTICATED_ONLY
from ratchet.verify.cli import main

from . import runs_scenario
from .helpers import HARNESSES, RECOMPUTED, REPORTS, RUN_RECORD_ITEMS, payload, sign


def pem(signer: Signer) -> bytes:
    return signer.public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)


def dump(path: Path, doc: dict[str, object]) -> str:
    path.write_bytes(json.dumps(doc, indent=2).encode())
    return str(path)


@pytest.fixture
def files(tmp_path: Path, signer: Signer) -> dict[str, Path]:
    key = tmp_path / "k.pub"
    key.write_bytes(pem(signer))
    receipt = tmp_path / "r.json"
    receipt.write_bytes(sign(payload(), signer))
    return {"key": key, "receipt": receipt, "dir": tmp_path}


def evidence_args(files: dict[str, Path]) -> list[str]:
    d = files["dir"]
    args: list[str] = []
    for i, (name, doc) in enumerate(REPORTS.items()):
        args += ["--report", f"{name}={dump(d / f'report{i}.json', doc)}"]
    for role, doc in HARNESSES.items():
        args += ["--harness", f"{role}={dump(d / f'{role}.json', doc)}"]
    for i, (name, doc) in enumerate(RECOMPUTED.items()):
        args += ["--recomputed", f"{name}={dump(d / f'rc{i}.json', doc)}"]
    return args


def run(files: dict[str, Path], *extra: str) -> int:
    return main([str(files["receipt"]), "--key", str(files["key"]), *extra])


def test_gate_evidence_without_run_records_exits_zero_and_lists_both_kinds(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(files, *evidence_args(files), "--require-reports", "--require-admitted")
    out = capsys.readouterr().out
    assert code == 0
    assert "admitted; 5 of 5 named reports checked" in out
    assert f"RECOMPUTED (10 of {10 + len(RUN_RECORD_ITEMS)})" in out
    assert f"NOT SUPPLIED: {', '.join(RUN_RECORD_ITEMS)}" in out
    assert "AUTHENTICATED ONLY (not re-derived)" in out
    assert run(files, *evidence_args(files), "--require-complete") == 1


def scenario_args(d: Path, signer: Signer) -> list[str]:
    """A receipt over real runs, with every report, harness and trace file it names."""
    s = runs_scenario.scenario()
    receipt = d / "scenario.json"
    receipt.write_bytes(sign(runs_scenario.body_for(s), signer))
    key = d / "k.pub"
    key.write_bytes(pem(signer))
    args = [str(receipt), "--key", str(key)]
    for name, doc in s.reports.items():
        args += ["--report", f"{name}={dump(d / f'{name}.json', doc)}"]
        lines = d / f"{name}.jsonl"
        lines.write_text("".join(json.dumps(t) + "\n" for t in s.traces[name]))
        args += ["--traces", f"{name}={lines}"]
    for role, doc in s.harnesses.items():
        args += ["--harness", f"{role}={dump(d / f'{role}.json', doc)}"]
    for name, doc in runs_scenario.RECOMPUTED.items():
        args += ["--recomputed", f"{name}={dump(d / f'{name}.json', doc)}"]
    return args


def test_a_receipt_over_real_runs_with_its_traces_recomputes_every_item(
    tmp_path: Path, signer: Signer, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([*scenario_args(tmp_path, signer), "--require-reports", "--require-complete"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "every recomputable item recomputed" in out and "NOT SUPPLIED" not in out
    assert "AUTHENTICATED ONLY (not re-derived): model_turns_as_recorded" in out


def test_missing_evidence_is_never_printed_as_recomputed(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(files) == 0
    out = capsys.readouterr().out
    assert "recomputation INCOMPLETE" in out and "every recomputable item recomputed" not in out
    assert "NOT SUPPLIED: report_digests, gate:heldout" in out
    assert "AUTHENTICATED ONLY" in out


def test_require_complete_fails_when_evidence_is_missing(files: dict[str, Path]) -> None:
    assert run(files, "--require-complete") == 1


def test_json_output(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert run(files, *evidence_args(files), "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["admission_verified"] is True and out["recomputation_complete"] is False
    assert out["authenticated_only"] == list(AUTHENTICATED_ONLY)


def test_traces_flag_needs_name_and_file(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert run(files, "--traces", "no-equals-sign") == 2
    assert run(files, "--traces", f"x={files['dir'] / 'absent.jsonl'}") == 2
    assert capsys.readouterr().err.count("ratchet-verify:") == 2


def test_unverifiable_receipt_exits_one(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    files["receipt"].write_bytes(sign(payload(), Signer.generate()))
    assert run(files) == 1
    assert "contradicted untrusted_key" in capsys.readouterr().out


def test_rejection_exits_zero_unless_admission_is_required(files: dict[str, Path], signer: Signer) -> None:
    checks = [{"name": g, "passed": g != "replay", "evidence": {}} for g in DEFAULT_GATES]
    files["receipt"].write_bytes(sign(payload(admitted=False, checks=checks), signer))
    assert run(files) == 0
    assert run(files, "--require-admitted") == 1


def test_report_mismatch_exits_one(files: dict[str, Path]) -> None:
    path = files["dir"] / "p.json"
    path.write_bytes(canonical_json({"type": "eval", "other": 1}))
    assert run(files, "--report", f"parent_run_1={path}") == 1


def test_approval_and_human_key_flags_are_read(files: dict[str, Path], signer: Signer) -> None:
    approval = files["dir"] / "approval.json"
    approval.write_bytes(b"{}")
    human = files["dir"] / "human.pub"
    human.write_bytes(pem(signer))
    assert run(files, "--approval", str(approval), "--human-key", str(human)) == 0


@pytest.mark.parametrize("junk", [b"{", b"[]"])
def test_malformed_receipt_exits_two(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str], junk: bytes
) -> None:
    files["receipt"].write_bytes(junk)
    assert run(files) == 2
    assert "no_verdict" in capsys.readouterr().out


def test_unreadable_inputs_exit_two(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(files["dir"] / "missing.json"), "--key", str(files["key"])]) == 2
    assert main([str(files["receipt"]), "--key", str(files["dir"] / "none.pub")]) == 2
    assert run(files, "--report", "no-equals-sign") == 2
    assert run(files, "--report", f"x={files['dir'] / 'absent.json'}") == 2
    files["dir"].joinpath("bad.json").write_bytes(b"not json")
    assert run(files, "--report", f"x={files['dir'] / 'bad.json'}") == 2
    assert run(files, "--approval", str(files["dir"] / "no-approval.json")) == 2
    assert capsys.readouterr().err.count("ratchet-verify:") == 6


def test_module_entry_point(files: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ratchet.verify", str(files["receipt"]), "--key", str(files["key"])])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("ratchet.verify", run_name="__main__")
    assert exit_info.value.code == 0
