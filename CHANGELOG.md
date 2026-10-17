# Changelog

## Unreleased

Hardening of what a receipt rests on: canonical bytes, signatures, the admission rule's statistics and the
store. Every change below is in this repository; the releases it waits for are named at the end.

- Every record Ratchet signs or digests is RFC 8785 canonical JSON, through `probityai.jcs` (the `jcs-admit`
  crate). Decisions name their canonicaliser (`canonicalization: rfc8785`) and stored traces and records name
  theirs (`digest_canonicalization`). A record that names none was written earlier: its digests are recomputed
  over the sorted-keys JSON it was signed with, so every stored decision still verifies. On the 105 stored
  public-benchmark decisions, RFC 8785 differs from the signed bytes only in number spelling, in 20 of them.
- Every DSSE envelope is signed and verified by `probityai.dsse` (the `dsse` crate); the two Python
  implementations are gone. The earlier pure-Python reader stays in the tests as an oracle that must agree with
  the binding. `verify(..., threshold=2)` holds a countersigned decision to both of its keys.
- The k-session rule (ledger D37) as `ratchet.improve.sessions`: the protocol, a runner for one host or one
  Slurm allocation (`ratchet.improve.sessions_live`), a bundle, and `python -m ratchet.improve.sessions verify`,
  which recomputes the t statistics, the harm screen and the log checks from the bundle alone. Its statistics
  match the reference implementation on 24 fixed cohorts.
- `pymongo` and `raindrop-ai` are the `atlas` and `telemetry` extras. Without them, records go to a signed
  append-only file store (`RATCHET_STORE_DIR`).
- `probity-verify==0.1.1` and `agent-evidence-vectors==0.19.0`.

Waits for: `probityai` 0.1.0 on PyPI (built from its merged source until then).

## 0.1.2 (2026-10-13)

Fixes in how a stored decision is checked and shown, and a demo image that proves its own health on two
kinds of machine. The suite, the gates and the decision rule are unchanged.

- A rejected decision no longer verifies as CONTRADICTED when its guard fired on a passing trace. The signer
  lists those false fires as `run_id@seq` under `false_fires`, but the checker compared them with a field
  named `false_positives` that no signed decision carries. So every rejected decision with a false fire
  failed `ratchet verify` even when the recomputed fires were identical. The checker now reads the field the
  signer writes, and its test fixtures use the same field.
- Every stored decision in the 2026-10-12 database backup was verified again after the fix: 123 rejected
  and 1 admitted, from the 14 backed-up Atlas databases, with replay recomputed from the stored traces
  wherever the backup holds them. Of the 123 rejected decisions, 26 verify and none is contradicted. In 15
  of the 26 the guard replay was recomputed in full, false fires included; the other 11 verify on their
  signature and stored reports. The remaining 97 (91 from `ratchet_l2`, 3 each from `ratchet_improve_live`
  and `ratchet_improve_smoke`) are not established, which is not a contradiction: the backup lacks evidence
  their receipts name, such as the recorded replay of a guard. The one admitted decision also verifies, with
  its replay recomputed.
- `ratchet show decision` counted one failing step several times when a failed run gave several reasons
  that all point at the same step, so a guard could be reported as missing "8 of 8" targets when there were
  3. `show decision` now counts each failing step once. A decision records its targets as it was signed,
  so the repeats stay in the signed record and the check that recomputes it.
- The demo image has a health check. `docker ps` shows it healthy once the store answers search-index
  calls, and `docker/demo/check.sh` waits for that before it runs the demo.
- CI builds and runs the demo image on arm64 as well as x86-64 (`ratchet demo double-refund`, then the
  reviewer sequence in `docs/RUNNING.md`, in a container with no network).

Requires `probity-ratchet-cases==0.2.0`.

## 0.1.1 (2026-10-12)

The harness depends on `probity-ratchet-cases` 0.2.0 in place of `ratchet-cases`.

## 0.1.0

First release.
