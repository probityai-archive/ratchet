# Architecture

## The loop

```
failures --> diagnose --> propose --> replay --> fresh evaluation --> admit --> sign --> activate
   ^                                                                                          |
   +------------------------------------- the new harness runs the next cases <------------------+
```

1. The worker runs cases under the active harness and records every event as a trace.
2. The improver clusters the failures and proposes guards, procedures and prompt sections. It sees
   training material only, plus lessons retrieved from earlier attempts on similar failures.
3. Each guard is replayed over the recorded traces. No model is called.
4. Candidates that survive run on the training cases in the loop's process and on the held-out cases at
   the referee, which also runs the parent twice on them and the candidate a second time to confirm.
5. The referee applies the admission rule and signs the decision with a key the loop never holds.
6. The referee activates: it moves the active-version pointer only from the expected version, after it
   checks the stored receipt.

## The case suite (`probity-ratchet-cases/`)

The simulated company and the cases ship as their own package, [`probity-ratchet-cases`](../probity-ratchet-cases/README.md),
which the harness depends on at an exact version. It has no runtime dependencies.

| Module | Job |
|---|---|
| `ratchet_cases.world` | The simulated company: billing provider, payments, orders, subscriptions, tickets, approvals, mail, policy, and a fault layer that can lose a reply after the action committed |
| `ratchet_cases.cases` | Eight case families, trap cases, benign twins and must-succeed cases, split into train, held-out and audit, with graders that read the final state of every system. `DIGESTS.json` pins the suite digest of every release |
| `ratchet_cases.run` | A bare reference loop with no guards, the trace file format, and regrading by replay |

## Packages (`src/ratchet`)

| Package | Job |
|---|---|
| `agent` | The harness bundle, the worker loop, the guard runtime, the chat-completions model client |
| `store` | Atlas persistence: versions, runs, traces, evaluations, decisions, failure memory, checkpoints, change streams, and the database roles that keep the improver away from held-out cases |
| `improve` | Diagnosis, proposal, replay, the admission rule, receipts, activation, the version archive, retrieval of similar failures, and the search over the improver's own instructions |
| `verify` | Independent checking of a receipt through the pinned verifier and a pinned key |
| `export` | Admitted guards as Claude Code hooks, and replay of a guard over Claude Code session transcripts |
| `view` | The terminal views and the static report: replay matrix, lineage, trace, decision, live change stream |
| `bench` | Adapters that run the same loop on public agent benchmarks, plus cross-model transfer runs; the measured results are in [RESULTS.md](RESULTS.md) |
| `telemetry` | An optional sink that sends runs, evaluations and decisions to Raindrop |
| `atlas_features` | Probes that run an Atlas capability on the cluster before the product claims it; the results are in [the probe page](results/atlas-probes.md) |

## The guard sandbox

A guard is untrusted model output. Its source defines `check(call, view, history)` and returns a reason
string to block or `None` to allow. Each guard runs in its own child process (`python -I -S`, empty
environment) that applies, before it reads the source:

- resource limits for CPU, memory and time,
- a new network namespace with no network,
- a seccomp filter,
- a Landlock file-system ruleset.

The child reports which layers took effect, and the runtime refuses to start one that lacks a required layer.
Every evaluation re-executes the module source, so a guard carries no state from one call to the next and a
replay gives the same answer as the live run. A guard that fails to compile, raises, returns something other
than a string, runs too long or kills its process blocks the call, and the error is recorded.

## What the store keeps

| Collection | Holds |
|---|---|
| `harness_versions` | Every version, its parent and its digest. A read recomputes the digest and rejects a mismatch |
| `runs`, `traces`, `trace_chunks` | One trace per document, with events embedded. A trace over 4 MB splits into ordered chunks, and its header is written last, so a crash leaves no half-visible trace |
| `evals` | Evaluation reports, each labelled FRESH, RECORDED or CACHED |
| `decisions` | Every admission decision, rejections included, with checks, evidence and the signed envelope |
| `failures` | Failure summaries with embeddings, searched by vector similarity |
| `checkpoints` | The state after each committed world call, so a killed run resumes without repeating a side effect |
| `embedding_cache` | Embeddings keyed by content |
| `cases` (held-out database) | Held-out cases, in a separate database that the improver's role has no privilege on |
| held-out database, every collection | The referee's records: held-out runs and traces, referee-side replays, decisions and activation |

Every write is idempotent. A second write of the same document is a no-op, and a second write of different
content under the same id raises an error instead of replacing history.

## What the improver is allowed to see

| Material | Improver | Evaluator |
|---|---|---|
| Training cases and traces | yes | yes |
| Replay results on training traces | yes | yes |
| Held-out cases and per-case outcomes | no | yes |
| Final audit cohort | no, until the harness is frozen | yes |
| Admin and evaluator credentials | no | yes |

The improver's database user holds the working database and nothing on held-out cases. The boundary is a
server-side role, so a bug in Ratchet's code cannot widen it.

## The referee

The loop (`ratchet.improve.loop`) holds training cases only. Everything that reads a held-out case or its
trace belongs to the referee (`ratchet.improve.referee`): the parent's held-out runs, each candidate's
held-out run and confirming run, the referee-side replay over every passing and benign-twin trace, the
admission rule, the decision signature and activation. `ProcessReferee` runs it as its own Python process,
built there by a factory (`module:function`) from a JSON config, and the two speak JSON lines: the referee
parses data and never unpickles anything the loop sends. The loop gets back counts (held-out passes, the
admission flag, the names of failed gates, the number of stable recoveries, the receipt digest) and keeps
them as `verdict` records, which the version archive reads; it never receives a held-out case id, outcome or
trace. With Atlas the referee writes to `<db>_heldout`, which the improver's role cannot read
(`tests/store/test_store_roles.py`); with the file store it writes beside it (`RATCHET_REFEREE_STORE_DIR`,
signed with the decision key). `ratchet show`, `verify` and `report` read both stores, as an auditor does.

## Receipts

A decision is RFC 8785 canonical JSON (`ratchet.canonical.canonical_json`, through the `probityai.jcs`
binding of the `jcs-admit` crate), wrapped in a DSSE envelope and signed with Ed25519. Decisions signed before
the RFC 8785 change keep their payload bytes as signed; nothing re-serialises a stored payload, so their
signatures still verify, and `tests/records` re-derives every stored record under both canonicalisers. The receipt names the
candidate, the parent, the policy and evaluator versions, the corpus snapshot and every report digest.
`ratchet verify` and `python -m ratchet.verify` check it against a public key that was pinned before the
receipt existed. [VERIFY.md](VERIFY.md) lists the pins.
