"""Run the agent-change-admission conformance corpus through `ratchet.verify`.

The corpus ships in the pinned `agent-evidence-vectors` package as
`corpora/vectors-agent-change-admission`. When the pinned package does not carry it the test fails and
says so, because an absent corpus and a passing one must never print the same result.

Every member in the bare DSSE carriage must get exactly the decision and reason the manifest declares.
Members in the in-toto carriage are signed under `application/vnd.in-toto+json`, which this verifier
does not read yet; each must be refused as `unsupported_payload_type`, and the count is asserted, so the
day the verifier learns that carriage this test fails and is updated to the manifest's answers.
"""

from __future__ import annotations

import base64
import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from ratchet_cases.types import canonical_json

from ratchet.improve.receipt import DECISION_TYPE, Signer
from ratchet.verify import PREDICATE_URI
from ratchet.verify.receipt import NO_VERDICT, Options, verify_decision

CORPUS = "vectors-agent-change-admission"
ISSUER_SEED_LABEL = b"agent-change-admission/v1 test issuer key"


ROOT = Path(str(resources.files("agent_evidence_vectors"))) / "corpora" / CORPUS


def _manifest() -> dict[str, Any]:
    path = ROOT / "MANIFEST.json"
    if not path.is_file():
        pytest.fail(f"{path} is absent: the pinned agent-evidence-vectors does not ship {CORPUS}")
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def _members(carriage: str) -> list[dict[str, Any]]:
    return [v for v in _manifest()["vectors"] if v["carriage"] == carriage]


def _verify(member: dict[str, Any]) -> tuple[str, str]:
    directory = ROOT / member["path"]
    case = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    policy = case["policy"]
    keys = [Ed25519PublicKey.from_public_bytes(base64.b64decode(k)) for k in policy["trusted_keys"].values()]
    reports = {name: (directory / file).read_bytes() for name, file in case["reports"].items()}
    result = verify_decision(
        (directory / case["envelope"]).read_bytes(),
        keys,
        reports=reports,
        options=Options(
            required_gates=tuple(policy["required_gates"]), require_reports=policy["require_reports"]
        ),
    )
    reason = "malformed_envelope" if result.decision == NO_VERDICT else result.reason
    return result.decision, reason


def test_corpus_pins_the_predicate_this_verifier_checks() -> None:
    manifest = _manifest()
    assert manifest["suite"] == "agent-change-admission/v1"
    assert manifest["predicateType"] == PREDICATE_URI


@pytest.mark.parametrize("member", _members("dsse-bare") if (ROOT / "MANIFEST.json").is_file() else [None])
def test_bare_carriage_member_gets_the_declared_verdict(member: dict[str, Any] | None) -> None:
    if member is None:
        _manifest()
        return
    expected = (member["expected"]["decision"], member["expected"]["reason"])
    assert _verify(member) == expected, member["id"]


def test_in_toto_carriage_is_refused_by_payload_type_until_supported() -> None:
    intoto = _members("dsse-intoto")
    assert len(intoto) == 4
    assert {_verify(m) for m in intoto} == {("not_established", "unsupported_payload_type")}


# Members edited after signing or signed by another key under the issuer keyid: the signer cannot
# reproduce them.
EDITED_AFTER_SIGNING = {
    "tampered-payload",
    "tampered-signature",
    "malformed-signature-not-base64",
    "wrong-key-claims-pinned-keyid",
}


def test_the_real_signer_writes_every_issuer_signed_bare_envelope_byte_for_byte() -> None:
    signer = Signer.from_seed_hex(hashlib.sha256(ISSUER_SEED_LABEL).hexdigest())
    reproduced: list[str] = []
    differ: list[str] = []
    for member in _members("dsse-bare"):
        envelope = (ROOT / member["path"] / "envelope.json").read_bytes()
        try:
            doc = json.loads(envelope)
            payload = json.loads(base64.b64decode(doc["payload"]))
        except (ValueError, KeyError):
            continue
        if doc["payloadType"] != DECISION_TYPE or [s["keyid"] for s in doc["signatures"]] != [signer.keyid]:
            continue
        rewritten = signer.sign(DECISION_TYPE, canonical_json(payload))
        (reproduced if rewritten == envelope else differ).append(member["id"])
    assert set(differ) == EDITED_AFTER_SIGNING
    assert len(reproduced) == 23
