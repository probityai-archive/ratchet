# Ruling 2 on a batch-invariant gpt-oss-120b endpoint

Ruling 2 compares exactly two parent runs (P1, P2) and two candidate runs (C1, C2), so it needs an endpoint
that returns the same trace for the same input. The default vLLM server does not: at temperature 0 with a
fixed seed, two identical runs disagreed on 7 of 19 held-out cases ([transfer.md](transfer.md)). This page
covers 2026-10-11, 16:13 to 17:46 PT. It records the repeatability evidence for vLLM's batch-invariant
mode and the two signed ruling-2 decisions made on it, for the candidates the live v3 cohort had refused
([v3-live.md](v3-live.md)).

## Endpoint

Two servers ran gpt-oss-120b (MXFP4, snapshot `b5c939de`) on vLLM 0.26.0 with `VLLM_BATCH_INVARIANT=1`,
each on one B200 on HiPerGator: Slurm jobs 45570883 and 45572252. The records are
[gpt-oss-120b-det-45570883.json](../endpoints/gpt-oss-120b-det-45570883.json) and
[gpt-oss-120b-det-45572252.json](../endpoints/gpt-oss-120b-det-45572252.json). Batch invariance moves
attention to FlashAttention 2 and the MXFP4 experts to an emulation backend (amd-quark 0.13). vLLM's
prefix cache stayed on (hit rate 97%) and requests were batched freely: up to 37 at once.

## Repeatability

| Check | Result |
|---|---|
| One case's first worker request, 8 copies sent at once among 8 unrelated requests, 2 rounds, on each server | 1 distinct output on each server, the same 315-token output on both |
| All 19 held-out cases, parent harness, twice on server 45570883 | normalized trace digest identical on 19 of 19 |
| The same, twice on server 45572252 | identical on 19 of 19 |
| The 19 digests compared across the two servers | identical on 19 of 19 |
| P1 against P2 held-out outcomes, each server | 0 discordant cases |

The normalized trace is every event's kind and content (model text, reasoning, tool calls and results)
plus the outcome, without timestamps, run ids or tool-call ids (`ratchet.improve.run_det.normalized`).
The parent passed 5 of 19 held-out cases and 5 of 18 training cases on every run.

Throughput: one request generates about 7.5 tokens per second whatever the batch size, so throughput
grows with concurrency. With 37 workers, a role took 432 to 485 seconds whether it had 19 cases or 37:
2.4 cases per minute at 19 cases and 4.6 to 5.1 at 37. The longest case sets each role's wall time.

## Decisions

Database `ratchet_r2_det` holds the traces, each role's evaluation and the decisions. The candidates
were rebuilt from their original decisions in `ratchet_improve_luna` and `ratchet_improve_oss` and
checked by digest. Each receipt is signed with the
Ratchet decision key (keyid `b61d7e39...d6d0c`) and names `policy: {name: ruling-2, evaluator:
deterministic, endpoint_config_digest}`. Both verify against `keys/ratchet-decisions.pub` with
`python -m ratchet.verify`. That check recomputes the signature, the predicate and the admission flag.

| Candidate | Decision id | Endpoint config digest | Admitted | Failed gates | Receipt SHA-256 |
|---|---|---|---|---|---|
| `ad50dac8194c` (guard) | `det:dfdfffeeb388->ad50dac8194c:15cfde558bb5` | `15cfde558bb5...` | no | heldout | `eac682df...0905` |
| `b972467a3b9a` | `det:dfdfffeeb388->b972467a3b9a:c419414172d4` | `c419414172d4...` | no | heldout, no_regression | `3b61caf8...162d` |

- `ad50dac8194c` produced exactly the parent's held-out outcomes: 5 of 19 in C1, the same five cases. It
  has no stable recovery.
- `b972467a3b9a` passed 8 of 19 held-out cases in C1. It turned four parent failures into passes
  (approval_revoked.v4, injection.v3, injection.v4, order_modification.v5) and failed policy_changed.v4,
  which the parent passed in both runs. A broken protected case refuses the candidate after C1, so C2 was
  not run and none of the four counts as a stable recovery.

These verdicts agree with v3's refusals of the same two candidates, now without a nondeterministic
evaluator. v3 could not judge them because its cohort failed qualification.
