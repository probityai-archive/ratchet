"""A tool addition or procedure edit is admitted only on an admitted k-session bundle plus exercised runs."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from ratchet_cases.types import Cost, Event, Outcome, Trace

from ratchet.canonical import canonical_json
from ratchet.improve import sessions
from ratchet.improve.admit import ExtendedChange, check_structure
from ratchet.improve.evaluate import CaseResult, EvalReport
from ratchet.improve.receipt import DECISION_TYPE, Signer
from ratchet.types import Harness, Procedure
from ratchet.verify import Options, verify_decision

PARENT = Harness.build(
    parent=None,
    system_prompt="p",
    procedures=[Procedure("p1", "refund", "a refund", ["a"])],
    guards=[],
    tool_allowlist=["lookup"],
    created_by="seed",
)
WITH_TOOL = PARENT.child(tool_allowlist=["lookup", "refund_status"], created_by="improver")
EDITED = PARENT.child(procedures=[Procedure("p1", "refund", "a refund", ["a", "b"])], created_by="improver")


def _bundle(root: Path, candidate: Harness, gain: float) -> Path:
    arms = {a: PARENT.version for a in sessions.UNCHANGED} | {"candidate": candidate.version}
    protocol = sessions.Protocol.for_k(f"ext-{candidate.version}", [f"case{i}" for i in range(6)], arms)

    def execute(slot: sessions.Slot) -> sessions.Outcome:
        rate = 0.2 + 0.1 * int(slot.case_id[-1]) + (gain if slot.arm == "candidate" else 0.0)
        return sessions.Outcome(random.Random(f"{slot.seed}/{slot.case_id}").random() < rate)

    sessions.run_protocol(protocol, execute, root, endpoints_for=lambda s: ["http://a", "http://b"])
    return root


def _runs(*events: Event) -> list[EvalReport]:
    trace = Trace("run-1", "c", "v", "m", list(events), Outcome(True, [], "d", Cost()))
    result = CaseResult("c", "heldout", False, "trap", True, [], "run-1", Cost())
    return [EvalReport("v", [result], 1.0, [trace])]


CALLED = Event(1, "tool_call", {"call": {"name": "refund_status", "args": {}, "call_id": "x"}}, 1.0)
USED = Event(1, "procedure", {"name": "refund", "how": "selected", "found": True}, 1.0)


def test_an_added_tool_is_admitted_on_an_admitted_bundle_when_the_candidate_called_it(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "b", WITH_TOOL, gain=0.6)
    assert not check_structure(PARENT, WITH_TOOL).passed  # additive-only stays the default
    check = check_structure(PARENT, WITH_TOOL, extended=ExtendedChange(bundle), cand_runs=_runs(CALLED))
    assert check.passed and check.evidence["extended"]["change_class"] == "exercised_under_k_sessions"
    unused = check_structure(PARENT, WITH_TOOL, extended=ExtendedChange(bundle), cand_runs=_runs())
    assert not unused.passed and unused.evidence["extended"]["tools_not_called"] == ["refund_status"]


def test_an_edited_procedure_must_have_been_used(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "b", EDITED, gain=0.6)
    assert check_structure(PARENT, EDITED, extended=ExtendedChange(bundle), cand_runs=_runs(USED)).passed
    assert not check_structure(
        PARENT, EDITED, extended=ExtendedChange(bundle), cand_runs=_runs(CALLED)
    ).passed


@pytest.mark.parametrize("problem", ["no gain", "another candidate", "edited bundle"])
def test_a_bundle_that_does_not_admit_this_candidate_refuses(tmp_path: Path, problem: str) -> None:
    if problem == "no gain":
        bundle = _bundle(tmp_path / "b", WITH_TOOL, gain=0.0)
    else:
        bundle = _bundle(tmp_path / "b", EDITED if problem == "another candidate" else WITH_TOOL, gain=0.6)
    if problem == "edited bundle":
        log = bundle / "executions.jsonl"
        log.write_text(log.read_text().replace('"passed":true', '"passed":false'))
    check = check_structure(PARENT, WITH_TOOL, extended=ExtendedChange(bundle), cand_runs=_runs(CALLED))
    assert not check.passed and check.evidence["extended"]["refused"]


def test_the_verifier_recomputes_the_bundle_an_extended_change_was_admitted_on(tmp_path: Path) -> None:
    from verify.helpers import payload

    bundle = _bundle(tmp_path / "b", WITH_TOOL, gain=0.6)
    extended = check_structure(PARENT, WITH_TOOL, extended=ExtendedChange(bundle), cand_runs=_runs(CALLED))
    body = payload(parent=PARENT.version, candidate=WITH_TOOL.version, candidate_digest=WITH_TOOL.digest)
    body["checks"] = [
        c
        if c["name"] != "authority"
        else {"name": "authority", "passed": True, "evidence": extended.evidence}
        for c in body["checks"]
    ]
    signer = Signer.generate()
    envelope = signer.sign(DECISION_TYPE, canonical_json(body))
    harnesses = {"parent": asdict(PARENT), "candidate": asdict(WITH_TOOL)}
    only = Options(required_gates=("authority",))
    r = verify_decision(envelope, signer.public_key, harnesses=harnesses, sessions=bundle, options=only)
    assert r.ok, r.reason
    authority = next(i for i in r.recomputed if i["id"] == "gate:authority")
    assert authority["status"] == "partial" and "recomputed k-session verdict" in authority["covered_by"]
    missing = verify_decision(envelope, signer.public_key, harnesses=harnesses, options=only)
    assert next(i for i in missing.recomputed if i["id"] == "gate:authority")["status"] == "not_supplied"


def test_ratchet_verify_reads_the_bundle_with_sessions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from cryptography.hazmat.primitives import serialization
    from verify.helpers import payload

    from ratchet.verify.cli import main

    bundle = _bundle(tmp_path / "b", WITH_TOOL, gain=0.6)
    extended = check_structure(PARENT, WITH_TOOL, extended=ExtendedChange(bundle), cand_runs=_runs(CALLED))
    body = payload(parent=PARENT.version, candidate=WITH_TOOL.version, candidate_digest=WITH_TOOL.digest)
    body["checks"] = [
        c
        if c["name"] != "authority"
        else {"name": "authority", "passed": True, "evidence": extended.evidence}
        for c in body["checks"]
    ]
    for c in body["checks"]:
        if c["name"] == "replay":
            c["evidence"] = {"new_guards": [], "reports": []}  # a tool grant adds no guard to replay
    signer = Signer.generate()
    (tmp_path / "r.json").write_bytes(signer.sign(DECISION_TYPE, canonical_json(body)))
    pem = signer.public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    (tmp_path / "k.pem").write_bytes(pem)
    for role, h in (("parent", PARENT), ("candidate", WITH_TOOL)):
        (tmp_path / f"{role}.json").write_text(json.dumps(asdict(h)))
    argv = [
        str(tmp_path / "r.json"),
        "--key",
        str(tmp_path / "k.pem"),
        "--json",
        "--harness",
        f"parent={tmp_path / 'parent.json'}",
        "--harness",
        f"candidate={tmp_path / 'candidate.json'}",
    ]

    def authority(extra: list[str]) -> dict[str, Any]:
        main([*argv, *extra])
        out = json.loads(capsys.readouterr().out)
        return next(i for i in out["recomputed"] if i["id"] == "gate:authority")

    assert authority(["--sessions", str(bundle)])["status"] == "partial"
    assert authority([])["status"] == "not_supplied"
