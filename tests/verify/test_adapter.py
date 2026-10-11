"""The `ratchet_decision/v1` adapter driven directly through `probity_verify.adjudicate`.

These tests show what the pinned verifier itself contributes: artifact binding, refusal of malformed controls
without a verdict, and the consumer-selected adapter rule.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from probity_verify import CaseError, adjudicate  # type: ignore[import-untyped]

from ratchet.improve.receipt import Signer
from ratchet.types import sha256_hex
from ratchet.verify import adapter
from ratchet.verify.keys import raw_b64
from ratchet.verify.receipt import Options, _build

from .helpers import HARNESSES, RECOMPUTED, REPORTS, payload, sign

ADAPTERS = {adapter.CLAIM_TYPE: adapter.adjudicate}


def _report_file(root: Path, name: str) -> Path:
    return root / "reports" / f"{sha256_hex(name.encode())[:16]}.json"


class Built:
    def __init__(self, root: Path, signer: Signer) -> None:
        self.root = root
        evidence = {"reports": REPORTS, "harnesses": HARNESSES, "recomputed": RECOMPUTED}
        self.case, self.policy = _build(
            root, sign(payload(), signer), [signer.public_key], evidence, None, Options()
        )
        self.assessment = next(iter(self.policy["assessments"].values()))

    def run(self) -> dict[str, Any]:
        result: dict[str, Any] = adjudicate(
            copy.deepcopy(self.case), copy.deepcopy(self.policy), self.root, adapters=ADAPTERS
        )
        return result


@pytest.fixture
def built(tmp_path: Path, signer: Signer) -> Built:
    root = Path(os.path.realpath(tmp_path))
    return Built(root, signer)


def test_clean_case_is_supported_and_binds_the_policy(built: Built) -> None:
    result = built.run()
    assert (result["decision"], result["reason"]) == ("supported", "admitted_receipt_verified")
    assert len(result["policy_sha256"]) == 64
    assert result["scope"]["artifacts"]["envelope"]["status"] == "bound"
    assert "does not establish" in result["scope"]["limit"]


def test_policy_digest_changes_with_the_pinned_key(built: Built, other_signer: Signer) -> None:
    before = built.run()["policy_sha256"]
    built.assessment["trusted_keys"] = {other_signer.keyid: raw_b64(other_signer.public_key)}
    after = built.run()
    assert after["policy_sha256"] != before
    assert after["reason"] == "untrusted_key"


def test_the_adapter_is_only_available_when_the_consumer_selects_it(built: Built) -> None:
    with pytest.raises(CaseError):
        adjudicate(built.case, built.policy, built.root)


def test_envelope_altered_after_the_case_was_written_is_refused_without_a_verdict(built: Built) -> None:
    path = built.root / "envelope.json"
    path.write_bytes(path.read_bytes().replace(b"DSSE", b"dsse", 1) + b" ")
    with pytest.raises(CaseError, match="binding_mismatch"):
        built.run()


def test_report_altered_after_the_case_was_written_is_refused_without_a_verdict(built: Built) -> None:
    path = _report_file(built.root, "parent_run_1")
    path.write_bytes(path.read_bytes()[::-1])
    with pytest.raises(CaseError, match="binding_mismatch"):
        built.run()


def test_symlinked_envelope_is_refused(built: Built, tmp_path: Path) -> None:
    target = built.root / "elsewhere.json"
    (built.root / "envelope.json").rename(target)
    (built.root / "envelope.json").symlink_to(target)
    with pytest.raises(CaseError, match="symlink traversal refused"):
        built.run()


def test_missing_envelope_is_not_established(built: Built) -> None:
    (built.root / "envelope.json").unlink()
    result = built.run()
    assert (result["decision"], result["reason"]) == ("not_established", "envelope_unavailable")


def test_missing_report_file_counts_as_not_supplied(built: Built) -> None:
    (_report_file(built.root, "parent_run_1")).unlink()
    result = built.run()
    assert result["decision"] == "supported"
    assert result["scope"]["reports"]["verified"] == [
        "candidate_confirm_run",
        "candidate_run",
        "parent_run_2",
        "replay:g-new",
    ]
    built.assessment["require_reports"] = True
    assert built.run()["reason"] == "report_not_supplied:parent_run_1"


def test_consumer_binding_must_match_the_envelope(built: Built) -> None:
    built.assessment["bindings"] = {"envelope": "0" * 64}
    result = built.run()
    assert (result["decision"], result["reason"]) == ("contradicted", "consumer_binding_mismatch")


def mutate_assessment(key: str, value: Any) -> Callable[[Built], None]:
    def apply(b: Built) -> None:
        b.assessment[key] = value

    return apply


def drop_assessment_field(key: str) -> Callable[[Built], None]:
    def apply(b: Built) -> None:
        del b.assessment[key]

    return apply


def set_case(key: str, value: Any) -> Callable[[Built], None]:
    def apply(b: Built) -> None:
        b.case[key] = value

    return apply


def set_policy(key: str, value: Any) -> Callable[[Built], None]:
    def apply(b: Built) -> None:
        b.policy[key] = value

    return apply


def unknown_role(b: Built) -> None:
    b.case["artifacts"]["stray"] = b.case["artifacts"]["envelope"]


def bad_report_role(b: Built) -> None:
    b.case["artifacts"]["report:"] = b.case["artifacts"]["envelope"]


def no_envelope_role(b: Built) -> None:
    del b.case["artifacts"]["envelope"]


def unknown_assessment(b: Built) -> None:
    b.policy["assessments"] = {"other": b.assessment}


def keyid_not_hash_of_key(b: Built) -> None:
    (keyid,) = b.assessment["trusted_keys"]
    other = Signer.generate()
    b.assessment["trusted_keys"] = {keyid: raw_b64(other.public_key)}


@pytest.mark.parametrize(
    "mutate",
    [
        mutate_assessment("claim_type", "ratchet_decision/v2"),
        mutate_assessment("trusted_keys", {}),
        mutate_assessment("trusted_keys", {"k" * 64: "AAAA"}),
        mutate_assessment("trusted_keys", {"nothex": "AAAA"}),
        mutate_assessment("trusted_keys", {"a" * 64: "***"}),
        mutate_assessment("required_gates", []),
        mutate_assessment("required_gates", ["a", "a"]),
        mutate_assessment("required_gates", [1]),
        mutate_assessment("require_reports", "yes"),
        mutate_assessment("require_complete", 1),
        mutate_assessment("eval_roles", {"p1": "a"}),
        mutate_assessment("trusted_human_keys", {"nothex": "AAAA"}),
        mutate_assessment("bindings", {"envelope": "0" * 64, "report": "1" * 64}),
        mutate_assessment("bindings", {"envelope": "short"}),
        mutate_assessment("bindings", []),
        drop_assessment_field("predicate_type"),
        set_policy("schema_version", "other/v1"),
        set_case("schema_version", "other/v1"),
        set_case("artifacts", []),
        unknown_role,
        bad_report_role,
        no_envelope_role,
        unknown_assessment,
        keyid_not_hash_of_key,
    ],
)
def test_malformed_controls_are_refused_without_a_verdict(
    built: Built, mutate: Callable[[Built], None]
) -> None:
    mutate(built)
    with pytest.raises(CaseError):
        built.run()


def test_adapter_called_directly_checks_its_own_selection(built: Built) -> None:
    """The kernel selects by claim_type, but the adapter does not rely on that."""
    built.assessment["claim_type"] = "ratchet_decision/v2"
    with pytest.raises(CaseError, match="unsupported claim_type"):
        adapter.adjudicate(built.case, built.policy, built.root)
    built.policy["assessments"] = {"other": built.assessment}
    with pytest.raises(CaseError, match="no consumer assessment"):
        adapter.adjudicate(built.case, built.policy, built.root)
