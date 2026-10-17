# Verifying a decision

A Ratchet admission receipt is checked by our pinned public package, `probity-verify`, and by a key that was
pinned before the receipt existed. Ratchet's own code is not the only thing that can say yes.

```sh
python -m ratchet.verify receipt.json \
  --report parent_run_1=parent_run_1.json --report parent_run_2=parent_run_2.json \
  --report candidate_run=candidate_run.json --report candidate_confirm_run=candidate_confirm_run.json \
  --harness parent=parent.json --harness candidate=candidate.json \
  --recomputed overfit_lint=overfit.json --require-admitted
```

`--key` defaults to `keys/ratchet-decisions.pub`. Exit 0 verifies, 1 does not, 2 means the input was malformed and no
verdict was reached.

## Pins

| Package | Pin | Where |
|---|---|---|
| `probity-verify` | `==0.1.1`, PyPI, wheel SHA-256 `e5788644f64d4faf73d999f9b8371f761141493403f642659ba0ad36a15ef6cf` | `pyproject.toml`, `uv.lock` |
| `agent-evidence-vectors` | `==0.19.0`, PyPI | `pyproject.toml`, `uv.lock` |

`probity-verify` 0.1.1 has the `adapters=` argument this build uses. `tests/verify/test_pinned_dependencies.py` asserts both pins and runs the
vector harness's own self-test, so a drifted pin fails a test.

Clean-environment result for the earlier pins (`probity-verify==0.1.0`, `agent-evidence-vectors==0.17.9`),
2026-10-11, Python 3.12.14, a fresh venv holding only these two packages:

- both install from the pins above; `probity_verify` exports `adjudicate` and `CaseError`;
- `agent-evidence-vectors --self-test`: 20 checks, 0 failed;
- `agent-evidence-vectors --verifier "aee-verify --json"` with `aee-verify@v0.17.7` from `go install`: 285 vectors,
  285 pass, 0 fail, 0 suite refusals;
- `probity-verify`'s own corpora run against the pinned package: `vectors-source-coverage` executed 6 of 6,
  `vectors-operand-lineage` 10 of 10.

## Who checks what

| Check | probity-verify | `ratchet.verify` |
|---|---|---|
| Artifact bytes bound to SHA-256 and length; anchored, non-following file reads; symlinks refused | yes | |
| Malformed case or policy refused without a verdict; adapter must be selected by the caller | yes | |
| Policy hashed into the verdict (`policy_sha256`, so the verdict names the pinned key and options) | yes | |
| Adapter may not mutate its inputs; `probity-decision/v1` envelope validated | yes | |
| DSSE parse, PAE, Ed25519 signature under the pinned key; payload type; predicate | | `dsse.py`, `adapter.py` |
| Payload fields; `admitted` agrees with the gate list | | `adapter.py` |
| Report digests against supplied stored reports | | `evidence.py` |
| Admission gates recomputed from stored evidence | | `recompute.py` |

`probity-verify` has no DSSE or Ratchet-payload support of its own. The adapter `ratchet_decision/v1`
(`src/ratchet/verify/adapter.py`) is the extension point its `adapters=` argument provides. Every PAE, signature
and signature check, for signing and for verifying, goes through `probityai.dsse`, the binding of the `dsse` crate;
`dsse.py` adds only the structural read and the rule that a key is tried only when a signature entry names it.
Two other implementations check it: the tests keep the earlier pure-Python reader (`tests/verify/python_dsse.py`)
and require it to agree with the binding on honest, malformed and forged envelopes, and `securesystemslib`
accepts the same envelopes. A countersigned decision can be held to both of its keys (`threshold=2`).

## Recomputed and authenticated only

Every report prints the two lists apart and says what was not supplied. `ok` never implies a complete
recomputation: print `summary`, or read `recomputed`, `authenticated_only` and `recomputation_complete`.

RECOMPUTED, derived by this verifier from bytes:

| Item | Needs |
|---|---|
| `signature` | the envelope and the pinned key |
| `payload_predicate` | the envelope |
| `admission_flag`: `admitted` equals the conjunction of the required gates | the envelope |
| `report_digests`: each supplied report hashes to the digest the receipt names | stored reports |
| `gate:heldout`, `gate:no_regression`, `gate:benign_twins` | the evaluation reports the receipt names: P1, P2, C1 and C2, or P1, P2 and C1 for a refusal made after C1 (the gate runs C2 only when C1 clears every gate, so such a receipt names no C2; the held-out gate then recomputes as failed, since admission needs both candidate runs) |
| `gate:authority` | parent and candidate harness documents; a human approval when authority grew by declared exception; for a tool grant or procedure edit, the k-session bundle (`--sessions DIR`), recomputed and checked against the parent and candidate. That item reads `partial`: that the candidate's runs called the tool or used the procedure is read from the signed evidence |
| `gate:replay` | the replay result recomputed from the recorded inputs, per added guard (`recomputed: replay:<guard id>`) |
| `gate:overfit_lint` | the screen result recomputed from the training literals (`recomputed: overfit_lint`) |

Each item reads `recomputed`, `partial` or `not_supplied`. A disagreement between a recomputed gate and the
receipt's claim is `contradicted` (`gate_recompute_mismatch:<gate>`), in both directions: a receipt that claims a
failure the evidence does not show is as wrong as one that claims a pass it does not support.

AUTHENTICATED ONLY, asserted by the evaluator and covered by its signature, never re-derived here:
`fresh_model_execution`, `capture_fidelity`, `sandbox_enforcement`, `hidden_grades`. A recomputed replay or
overfit result is supplied by the caller who ran it from the recorded inputs; this verifier compares it with the
receipt and does not run the sandboxed guard itself.

The gate rules follow `docs/CONTRACT.md`, ruling 2: at least two stable recoveries (failing in P1 and P2, passing
in C1 and C2); every case passing in P1 or P2 passes in C1 and C2; only FRESH passes count; zero must-succeed
regressions; zero benign-twin interventions; the candidate appends only (tools a subset, parent guards and
procedures unchanged and in order, parent prompt a prefix, unique ids, digest equals its content address).

## What the receipt must carry

`Decision.to_payload()` of the admission lane:

| Field | Rule |
|---|---|
| `predicateType` | exactly the pinned predicate URI (below) |
| `parent`, `candidate` | non-empty; equal the harness versions when harnesses are supplied |
| `candidate_digest` | 64 hex; equals the candidate harness digest |
| `admitted` | bool; equals the conjunction of the required gates |
| `checks` | list of `{name, passed, evidence}`; every required gate present |
| `evaluator` | `version` non-empty and `components.suite_file` 64 hex (the frozen suite digest) |
| `corpus_digests` | guard id to 64 hex replay snapshot digest |
| `canonicalization` | `rfc8785`; absent only on a decision signed before RFC 8785, whose report digests are over sorted-keys JSON |
| `report_digests` | non-empty; name to 64 hex, the SHA-256 of the report's canonical JSON under `canonicalization` |
| `exceptions` | list of strings; `authority grew under ...` marks a human-approved exception |

Evaluation reports are found by name: `parent_run_1`, `parent_run_2`, `candidate_run`,
`candidate_confirm_run` (the consumer can map others with `Options.eval_roles`). A receipt that names the first three but no `candidate_confirm_run` records a refusal made before C2 ran; a receipt that names C2 needs C2 supplied. Replay reports are named
`replay:<guard id>`. These are the names `admit()` writes today.

Known drift: the admission lane's held-out gate still uses the margin `max(2, D + 1)`; ruling 2 requires two stable
recoveries and no loss of a protected case. Supplying the four evaluation reports exposes any disagreement as
`gate_recompute_mismatch:heldout`.

## Reasons

| Decision | Reason |
|---|---|
| supported | `admitted_receipt_verified`, `rejection_receipt_verified` |
| contradicted | `untrusted_key`, `signature_invalid`, `consumer_binding_mismatch`, `admitted_with_failed_gate:<g>`, `rejected_with_all_gates_passed`, `report_digest_mismatch:<n>`, `report_not_named:<n>`, `gate_recompute_mismatch:<g>`, `authority_recompute_failed_without_exception`, `admitted_with_refused_extended_change`, `sessions_bundle_refused:<why>`, `sessions_verdict_not_admitted`, `sessions_bundle_not_the_named_one`, `sessions_bundle_compares_other_harnesses`, `harness_not_the_decided_candidate`, `harness_not_the_decided_parent`, `replay_recompute_mismatch:<guard>:<field>`, `replay_snapshot_mismatch:<guard>`, `replay_new_guards_mismatch`, `replay_check_passed_with_failed_guard`, `report_malformed:<detail>`, `evidence_not_json:<role>` |
| not_established | `unsupported_payload_type`, `unsupported_predicate`, `payload_not_json_object`, `payload_field_invalid:<field>`, `gate_missing:<g>`, `report_not_supplied:<n>`, `authority_exception_unverified:<why>`, `recomputation_incomplete:<item>`, `envelope_unavailable` |
| no_verdict | malformed envelope, `invalid_report_name:<n>`, `invalid_harness_role`, `no_trusted_key`, or a malformed case or policy |

`Options(require_reports=True)` demands every named report; `Options(require_complete=True)` demands every
recomputable item, which is the setting an activation step should use.

## The pinned key

`keys/ratchet-decisions.pub` (PEM, Ed25519) is committed and is the only key the verifier trusts by default. The
receipt names a `keyid` (SHA-256 of the raw public key) but never carries a key, and a signature naming an
untrusted keyid is never tried. The seed lives in the system keyring (`service ratchet account
decision-signing-key`, 64 hex characters) and is never committed or printed; `RATCHET_SIGNING_KEY` overrides it
for a run. Rotation: pass several keys to `verify_decision`.

Every run signs with the pinned key or refuses to start (`Signer.for_decisions`); there is no fallback to a
fresh key, because a decision signed by an unpublished key can never be checked. Decisions recorded before
that rule by a per-run key are countersigned with `python -m ratchet.improve.countersign --db <db>`: the pinned
key's signature is added beside the run's own (DSSE carries several), only when the stored decision still
equals its signed payload, and the decision records `countersigned` with the date. `ratchet verify` then names
the pinned key and the run key it was first signed with.

## Proposed predicate and vocabulary term

Proposal for `agent-evidence-vocabulary`; it documents the receipt and does not verify one.

Predicate type the receipt carries:

```
https://github.com/probityai/agent-evidence-vocabulary/blob/main/docs/predicates/agent-change-admission-v1.md
```

Proposed term, in the registry's shape:

```yaml
posture_and_coverage:
  admission_decision:
    definition: >
      Whether a proposed change to an agent harness was admitted against a frozen comparison, with every
      gate recorded as a named check carrying the counts and identifiers it rests on. A consumer MUST
      recompute the gates it can from the digest-committed reports and MUST treat the gates it cannot
      recompute (fresh model execution, capture fidelity, sandbox enforcement, hidden grades) as
      authenticated only.
    values: [admitted, rejected]
    source_path: predicate.admitted
    status: proposed
    issuers_in_production: []
    review_by: '2027-04-11'
```

Field mapping: `witnessScope` is `SELF` (the evaluator signs its own run), `claimSubject` is `agent`,
`doesNotAssert` lists the authenticated-only items, and `report_digests` is the coverage denominator.

## What a verified receipt does not establish

That the evaluations ran, that the stored reports are true, that the capture was faithful, that the sandbox
enforced its limits, or that the key holder is honest. Replay shows a guard's behaviour on recorded inputs, not
that the counterfactual continuation succeeds. Authority is an enforced policy, not a safety proof.
