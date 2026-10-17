"""The `ratchet_decision/v1` claim adapter that `probity-verify` runs for a Ratchet admission receipt.

Split of work, stated exactly:

probity-verify (pinned) does: reads every artifact through anchored, non-following file descriptors and binds
its bytes to a SHA-256 and length; refuses a malformed case or policy without a verdict; snapshots and hashes
the consumer policy and reports `policy_sha256`; rejects an adapter that mutates its inputs; validates the
`probity-decision/v1` envelope this module returns.

This adapter does: parses the DSSE envelope, verifies the Ed25519 signature against the key the policy pins,
checks the payload type and predicate, the payload's required fields and its internal consistency, and runs
the steps in `ratchet.verify.evidence`: report digests, and the admission gates recomputed from the stored
evidence whenever that evidence is supplied.

The verdict reports two lists and never merges them. RECOMPUTED: derived here from bytes (signature, digests,
the admission predicate, replay when its recorded inputs are supplied). AUTHENTICATED ONLY: stated by the
evaluator and covered by its signature, never re-derived (fresh model execution, capture fidelity, sandbox
enforcement, hidden grades). An item whose inputs were not supplied is printed as not supplied.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# probity-verify ships no py.typed marker; the pinned commit is the contract for these private helpers.
from probity_verify.common import (  # type: ignore[import-untyped]
    CaseError,
    _artifact,
    _digest,
    _object,
    _string,
)

from ratchet.canonical import CANONICALIZATION
from ratchet.verify import dsse, evidence

CLAIM_TYPE = "ratchet_decision/v1"
CASE_SCHEMA = "ratchet-decision-case/v1"
POLICY_SCHEMA = "ratchet-decision-policy/v1"
DECISION_PAYLOAD_TYPE = "application/vnd.ratchet.decision+json"
PREDICATE_URI = "https://github.com/probityai/agent-evidence-vocabulary/blob/main/docs/predicates/agent-change-admission-v1.md"
DEFAULT_GATES = ("replay", "heldout", "no_regression", "authority", "benign_twins", "overfit_lint")
# Stochastic policy v3 (docs/CONTRACT.md): its own gate names, and the exact thresholds a v3 receipt must
# carry. A v3 decision never carries a zero-regression gate.
V3_POLICY = {
    "name": "stochastic", "version": "v3", "runs_per_arm": 40, "panels": 2, "panel_size": 20,
    "high_pass_min": 30, "low_pass_max": 10, "recovery_panel_margin": 8, "min_stable_recoveries": 2,
    "max_stable_regressions": 0, "max_case_decline": 8, "max_aggregate_decline_points": 5.0,
}  # fmt: skip
V3_GATES = ("replay", "stochastic_v3", "cohort_qualified", "effective_change", "benign_twins", "overfit_lint",
            "authority")  # fmt: skip
REPORT_ROLE = "report:"
HARNESS_ROLE = "harness:"
RECOMPUTED_ROLE = "recomputed:"
DEFAULT_EVAL_ROLES = {
    "p1": "parent_run_1",
    "p2": "parent_run_2",
    "c1": "candidate_run",
    "c2": "candidate_confirm_run",
}
AUTHENTICATED_ONLY = (
    "fresh_model_execution",
    "capture_fidelity",
    "sandbox_enforcement",
    "hidden_grades",
)
REPORT_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,200}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

LIMIT = (
    "RECOMPUTED covers only the items listed as recomputed. AUTHENTICATED ONLY items rest on the evaluator's "
    "signature: this verifier did not re-derive them and does not establish that the evaluations ran, that "
    "the reports are true, or that the key holder is honest."
)


def _trusted_keys(value: Any) -> dict[str, Ed25519PublicKey]:
    if not isinstance(value, dict) or not value:
        raise CaseError("trusted_keys: expected a nonempty object of keyid to base64 raw public key")
    keys: dict[str, Ed25519PublicKey] = {}
    for keyid, encoded in value.items():
        _digest(keyid, "trusted_keys keyid")
        try:
            key = Ed25519PublicKey.from_public_bytes(
                base64.b64decode(_string(encoded, "trusted_keys key"), validate=True)
            )
        except ValueError as exc:
            raise CaseError("trusted_keys: not a base64 Ed25519 public key") from exc
        if dsse.keyid_of(key) != keyid:
            raise CaseError("trusted_keys: keyid is not the SHA-256 of its key")
        keys[keyid] = key
    return keys


def _gates(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(g, str) or not g for g in value):
        raise CaseError("required_gates: expected a nonempty list of names")
    if len(set(value)) != len(value):
        raise CaseError("required_gates: duplicate name")
    return list(value)


def _eval_roles(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(DEFAULT_EVAL_ROLES):
        raise CaseError("eval_roles: expected exactly p1, p2, c1, c2")
    return {k: _string(v, f"eval_roles.{k}") for k, v in value.items()}


def _human_keys(value: Any) -> dict[str, Ed25519PublicKey]:
    return _trusted_keys(value) if value else {}


def _assessment(case_id: str, policy: Any) -> dict[str, Any]:
    policy = _object(policy, "policy", {"schema_version", "assessments"})
    if policy["schema_version"] != POLICY_SCHEMA:
        raise CaseError("unsupported ratchet decision policy schema_version")
    if not isinstance(policy["assessments"], dict) or case_id not in policy["assessments"]:
        raise CaseError("case_id has no consumer assessment")
    a: dict[str, Any] = _object(
        policy["assessments"][case_id],
        "assessment",
        {
            "claim_type",
            "bindings",
            "trusted_keys",
            "trusted_human_keys",
            "predicate_type",
            "required_gates",
            "eval_roles",
            "require_reports",
            "require_complete",
        },
    )
    if a["claim_type"] != CLAIM_TYPE:
        raise CaseError("unsupported claim_type")
    for flag in ("require_reports", "require_complete"):
        if not isinstance(a[flag], bool):
            raise CaseError(f"{flag}: expected boolean")
    return a


_PREFIX_NAME = {REPORT_ROLE, RECOMPUTED_ROLE}


def _valid_role(role: str) -> bool:
    if role in {"envelope", "approval", "harness:parent", "harness:candidate"}:
        return True
    prefix = next((p for p in _PREFIX_NAME if role.startswith(p)), None)
    return prefix is not None and bool(REPORT_NAME.fullmatch(role[len(prefix) :]))


def _roles(case: Any) -> tuple[str, dict[str, Any]]:
    case = _object(case, "case", {"schema_version", "case_id", "artifacts"})
    if case["schema_version"] != CASE_SCHEMA:
        raise CaseError("unsupported ratchet decision case schema_version")
    artifacts = case["artifacts"]
    if not isinstance(artifacts, dict) or "envelope" not in artifacts:
        raise CaseError("artifacts: expected an envelope")
    for role in artifacts:
        if not _valid_role(role):
            raise CaseError(f"artifacts: unknown role {role!r}")
    return _string(case["case_id"], "case_id"), artifacts


class _Verdict:
    """Accumulates checks and finishes the `probity-decision/v1` envelope."""

    def __init__(self, case_id: str, scope: dict[str, Any]) -> None:
        self.case_id = case_id
        self.scope = scope
        self.checks: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []
        scope["recomputed"] = self.items

    def met(self, name: str, **extra: Any) -> None:
        self.checks.append({"id": name, "status": "met", **extra})

    def recomputed(self, name: str) -> None:
        self.items.append({"id": name, "status": "recomputed"})

    def finish(self, decision: str, reason: str, failed: str | None = None) -> dict[str, Any]:
        if failed is not None:
            self.checks.append(
                {"id": failed, "status": "failed" if decision == "contradicted" else "unavailable"}
            )
        pending = [i["id"] for i in self.items if i["status"] != "recomputed"]
        self.scope["authenticated_only"] = list(AUTHENTICATED_ONLY)
        self.scope["recomputation_complete"] = decision == "supported" and not pending
        self.scope["not_recomputed"] = pending
        return {
            "schema_version": "probity-decision/v1",
            "case_id": self.case_id,
            "claim_type": CLAIM_TYPE,
            "decision": decision,
            "reason": reason,
            "scope": self.scope,
            "checks": self.checks,
        }


def _field(payload: dict[str, Any], name: str, kind: type) -> str | None:
    value = payload.get(name)
    if not isinstance(value, kind) or not value:
        return f"payload_field_invalid:{name}"
    return None


def _digest_map(value: Any, nonempty: bool) -> bool:
    return (
        isinstance(value, dict)
        and (bool(value) or not nonempty)
        and all(
            isinstance(k, str) and REPORT_NAME.fullmatch(k) and isinstance(v, str) and HEX64.fullmatch(v)
            for k, v in value.items()
        )
    )


def _evaluator_problem(evaluator: Any) -> str | None:
    if (
        not isinstance(evaluator, dict)
        or not isinstance(evaluator.get("version"), str)
        or not evaluator["version"]
    ):
        return "payload_field_invalid:evaluator"
    components = evaluator.get("components")
    suite = components.get("suite_file") if isinstance(components, dict) else None
    if not isinstance(suite, str) or not HEX64.fullmatch(suite):
        return "payload_field_invalid:evaluator.components.suite_file"
    return None


def _schema_problem(payload: dict[str, Any]) -> str | None:
    """The first field the signed payload lacks or malforms, named in the shape the IMPROVE lane emits."""
    # A decision signed before RFC 8785 names no canonicaliser; a later one names exactly this one.
    if "canonicalization" in payload and payload["canonicalization"] != CANONICALIZATION:
        return "payload_field_invalid:canonicalization"
    for name in ("parent", "candidate"):
        if problem := _field(payload, name, str):
            return problem
    digest = payload.get("candidate_digest")
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        return "payload_field_invalid:candidate_digest"
    if problem := _evaluator_problem(payload.get("evaluator")):
        return problem
    exceptions = payload.get("exceptions")
    if not isinstance(exceptions, list) or any(not isinstance(e, str) for e in exceptions):
        return "payload_field_invalid:exceptions"
    if not isinstance(payload.get("admitted"), bool):
        return "payload_field_invalid:admitted"
    checks = payload.get("checks")
    if not isinstance(checks, list) or any(
        not isinstance(c, dict) or not isinstance(c.get("name"), str) or not isinstance(c.get("passed"), bool)
        for c in checks
    ):
        return "payload_field_invalid:checks"
    if not _digest_map(payload.get("corpus_digests"), nonempty=False):
        return "payload_field_invalid:corpus_digests"
    if not _digest_map(payload.get("report_digests"), nonempty=True):
        return "payload_field_invalid:report_digests"
    return None


def _gate_problem(payload: dict[str, Any], required: list[str]) -> tuple[str, str] | None:
    """(decision, reason) when the gate record is incomplete or contradicts the admitted flag."""
    passed = {c["name"]: c["passed"] for c in payload["checks"]}
    missing = [g for g in required if g not in passed]
    if missing:
        return "not_established", f"gate_missing:{missing[0]}"
    failed = [g for g in required if not passed[g]]
    if payload["admitted"] and failed:
        return "contradicted", f"admitted_with_failed_gate:{failed[0]}"
    if not payload["admitted"] and not failed:
        return "contradicted", "rejected_with_all_gates_passed"
    return None


def _supplied(data: dict[str, bytes | None], prefix: str) -> dict[str, bytes]:
    return {
        role[len(prefix) :]: raw for role, raw in data.items() if role.startswith(prefix) and raw is not None
    }


def _pipeline(ctx: evidence.Ctx) -> None:
    evidence.check_report_digests(ctx)
    evidence.recompute_eval_gates(ctx)
    evidence.recompute_authority(ctx)
    evidence.recompute_replay(ctx)
    evidence.recompute_overfit(ctx)


def _run_evidence(v: _Verdict, ctx: evidence.Ctx, require_complete: bool) -> dict[str, Any]:
    try:
        _pipeline(ctx)
    except evidence.Refuse as stop:
        v.scope.update(ctx.scope)
        return v.finish(stop.decision, stop.reason, stop.failed)
    v.scope.update(ctx.scope)
    v.items.extend(ctx.items)
    for item in ctx.items:
        if item["status"] == "recomputed":
            v.met(item["id"])
    pending = [i["id"] for i in v.items if i["status"] != "recomputed"]
    if require_complete and pending:
        return v.finish("not_established", f"recomputation_incomplete:{pending[0]}", "recomputation")
    payload = ctx.payload
    return v.finish(
        "supported", "admitted_receipt_verified" if payload["admitted"] else "rejection_receipt_verified"
    )


def adjudicate(case: Any, policy: Any, artifact_root: Path) -> dict[str, Any]:
    """Derive every check from the bound envelope bytes and the separately pinned policy."""
    case_id, artifacts = _roles(case)
    a = _assessment(case_id, policy)
    keys = _trusted_keys(a["trusted_keys"])
    human_keys = _human_keys(a["trusted_human_keys"])
    required = _gates(a["required_gates"])
    eval_roles = _eval_roles(a["eval_roles"])
    bindings = a["bindings"]
    if not isinstance(bindings, dict) or set(bindings) - {"envelope"}:
        raise CaseError("bindings: expected only an envelope digest")
    pinned_envelope = _digest(bindings.get("envelope"), "bindings.envelope")

    data: dict[str, bytes | None] = {}
    bound: dict[str, dict[str, Any]] = {}
    for role, metadata in artifacts.items():
        data[role], bound[role] = _artifact(artifact_root, metadata, f"artifacts.{role}")
    scope: dict[str, Any] = {
        "artifacts": bound,
        "pinned_keyids": sorted(keys),
        "predicate_type": a["predicate_type"],
        "required_gates": required,
        "limit": LIMIT,
    }
    v = _Verdict(case_id, scope)

    envelope_bytes = data["envelope"]
    if envelope_bytes is None:
        return v.finish("not_established", "envelope_unavailable", "envelope")
    if bound["envelope"]["sha256"] != pinned_envelope:
        return v.finish("contradicted", "consumer_binding_mismatch", "envelope")
    try:
        env = dsse.parse(envelope_bytes)
    except dsse.EnvelopeError as exc:
        raise CaseError(f"malformed envelope: {exc}") from exc
    v.met("envelope")

    if env.payload_type != DECISION_PAYLOAD_TYPE:
        return v.finish("not_established", "unsupported_payload_type", "payload_type")
    v.met("payload_type")

    signed = dsse.verify_signature(env, keys)
    if signed.keyid is None:
        return v.finish("contradicted", signed.reason or "signature_invalid", "signature")
    scope["keyid"] = signed.keyid
    v.met("signature", keyid=signed.keyid)
    v.recomputed("signature")

    try:
        payload = json.loads(env.payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    if not isinstance(payload, dict):
        return v.finish("not_established", "payload_not_json_object", "payload")
    if payload.get("predicateType") != a["predicate_type"]:
        return v.finish("not_established", "unsupported_predicate", "predicate")
    v.met("predicate")
    if problem := _schema_problem(payload):
        return v.finish("not_established", problem, "payload_schema")
    v.met("payload_schema")
    v.recomputed("payload_predicate")
    scope["evaluator_version"] = payload["evaluator"]["version"]
    scope["suite_digest"] = payload["evaluator"]["components"]["suite_file"]
    scope.update(
        {k: payload[k] for k in ("parent", "candidate", "candidate_digest", "admitted", "exceptions")}
    )

    policy = payload.get("policy")
    if isinstance(policy, dict) and policy.get("name") == "ruling-2":
        scope["policy"] = "ruling-2"  # the default gates; the evaluator detail rides in the policy record
    elif policy is not None:
        if policy != V3_POLICY:
            return v.finish("not_established", "unsupported_policy", "policy")
        if any(c["name"] == "no_regression" for c in payload["checks"]):
            return v.finish("contradicted", "v3_claims_zero_regression", "policy")
        required = list(V3_GATES)
        scope["policy"] = "stochastic-v3"
    if gate := _gate_problem(payload, required):
        return v.finish(gate[0], gate[1], "gates")
    v.met("gates", count=len(required))
    v.recomputed("admission_flag")

    ctx = evidence.Ctx(
        payload=payload,
        claimed={c["name"]: c["passed"] for c in payload["checks"]},
        required=required,
        reports=_supplied(data, REPORT_ROLE),
        harnesses=_supplied(data, HARNESS_ROLE),
        recomputed=_supplied(data, RECOMPUTED_ROLE),
        approval=data.get("approval"),
        eval_roles=eval_roles,
        human_keys=human_keys,
        require_reports=a["require_reports"],
    )
    return _run_evidence(v, ctx, a["require_complete"])
