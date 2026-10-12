# Improver-instruction search: seed, challenger and searched text

Three versions of the improver's instructions (the seed, the loop's own challenger and the text the instruction search returned) proposing harness changes for the same frozen failure clusters, with every candidate judged by the admission rule.

Run of record `level2-run2b`, 2026-10-11 21:11 to 23:37 UTC (wall 8,797.1 s), module
`ratchet.improve.instruction_search`. Worker model and improver model both `gpt-oss-120b` at
`http://localhost:8801/v1`, 128 workers, Atlas database `ratchet_l2`, comparison record
`level2-run2b-comparison`, decisions `level2-run2b:*`. The run file `private/improve/l2/level2-run2b.json`
is not committed; every number below is read from it and checked against the Atlas records.

Result: 0 of 84 candidate slots admitted (0 of 28 per arm). The search returned the seed instructions
unchanged after 62 metric calls (budget 60). No guard from any arm reached a fresh evaluation, because every
guard replay on the frozen batch ended incomplete. Those replays never ran the guards: a defect in the guard
runtime, since fixed, failed every check (see [Guard replay](#guard-replay)). This run does not measure the
three texts' guards and needs a rerun for that.

## What was compared

1. The parent harness `dfdfffeeb388` ran fresh twice. Run P1 covered the training and held-out cases:
   4 of 18 training cases and 4 of 19 held-out cases passed. Run P2 covered the held-out cases: 3 of 19 passed.
2. The improver grouped P1's training failures into 17 failure clusters (17 improver calls). The clusters
   were split by case family. Families A were `ordinary`, `policy_changed`, `reconciliation` and
   `refund_lost_reply`.
   - Batch A, every member in families A: 2 clusters. Only these reach the search and the challenger's derivation.
   - Frozen batch B, no member in families A: 7 clusters (digest `b243c1a8eb71`). All three arms are judged here.
   - Mixed, members on both sides: 8 clusters, excluded.
3. Challenger. The seed instructions proposed for batch A (8 slots); those candidates went through
   admission (6 evaluated fresh, 0 of 8 admitted), and one call of the loop's instruction drafter wrote a new
   text from those outcomes: `d3f51c15e6b6`, parent `942dc26006bf`.
4. Search. On batch A only, the search scores an instruction text by the share of requested candidate slots
   that are guards which clear replay over the recorded training traces and contain no copied training
   literal. Replay calls no worker model. Budget 60 metric calls; it made 62, with 62 proposer calls and 15
   reflection calls. It returned `942dc26006bf`, the seed.
5. Each arm then proposed 4 candidates for each of the 7 frozen clusters: 28 slots and 7 proposal calls per
   arm, no repair rounds. A guard went to fresh evaluation only if it cleared replay; a procedure or prompt
   edit went to fresh evaluation if it passed the copied-literal screen. Every candidate evaluated fresh was
   decided by the real `admit()` and stored as a signed decision.

## Per arm

| Arm | Instructions | Proposed / slots | Guard / procedure / prompt edit | Screened out | Replay cleared / failed / incomplete | Evaluated fresh | Admitted / slots |
|---|---|---|---|---|---|---|---|
| seed | `942dc26006bf` | 28 / 28 | 14 / 7 / 7 | 0 | 0 / 0 / 14 | 14 | 0 / 28 |
| challenger | `d3f51c15e6b6` | 28 / 28 | 16 / 7 / 5 | 0 | 0 / 0 / 16 | 12 | 0 / 28 |
| search | `942dc26006bf` | 28 / 28 | 14 / 7 / 7 | 0 | 0 / 0 / 14 | 14 | 0 / 28 |

No proposal was invalid or a duplicate. In every arm, replay incomplete plus evaluated fresh equals 28: every
guard ended incomplete, and every procedure and prompt edit was evaluated fresh.

The seed and search arms used the same text, so they are two independent proposal rounds from the seed
instructions.

### Why the evaluated candidates were refused

Every evaluated candidate was refused after its first fresh run; no candidate reached the confirming run.
Counts are refused candidates over candidates evaluated fresh. The checks are described in
[ADMISSION.md](../ADMISSION.md).

| Arm | Held-out recoveries (`heldout`) | No regression (`no_regression`) | Benign twins (`benign_twins`) |
|---|---|---|---|
| seed | 14 / 14 | 14 / 14 | 0 / 14 |
| challenger | 12 / 12 | 12 / 12 | 1 / 12 |
| search | 14 / 14 | 14 / 14 | 2 / 14 |

Held-out flips in the first fresh run, summed over the candidates evaluated fresh. Fail to pass: a held-out
case that failed in both parent runs passed. Pass to fail: a held-out case that passed in either parent run
failed.

| Arm | Candidates | Fail to pass | Pass to fail | Most fail-to-pass on one candidate | Candidates with 2 or more fail-to-pass and no pass-to-fail |
|---|---|---|---|---|---|
| seed | 14 | 33 | 50 | 8 | 0 / 14 |
| challenger | 12 | 32 | 40 | 5 | 0 / 12 |
| search | 14 | 31 | 48 | 4 | 1 / 14 |

## Guard replay

All 44 guards proposed for the frozen batch (14 seed, 16 challenger, 14 search) ended with replay verdict
INCOMPLETE. On each of them every check returned an error: 3,271 of 3,271 checks, each recorded with the
error text `None`. None of the 44 hit its target failure. An error can never clear a guard, so none was
evaluated fresh.

Earlier in the same run, the challenger's derivation on batch A replayed its 4 guards to complete verdicts
(2 cleared, 2 failed). In the search, 0 of its 62 evaluations cleared a guard: 32 evaluations of the seed
text and 2 of each of 15 proposed texts all scored 0 on both batch-A clusters, so no proposed text scored
above the seed.

Cause. The guard runtime started each sandbox from the sandbox's source file on disk, read when the sandbox
started. The run's process loaded the runtime at 21:11 UTC. By 21:24 UTC the checkout it ran
from had moved to a sandbox version that opens with a new startup message (commit `341721f`). From then on every
sandbox the run started spoke a protocol its runtime did not know; the runtime took the new message for a
failed start and recorded the absent error field, `None`, as the error of every check. No guard code ran.
Replaying three of these guards over their recorded traces reproduces it: with the run's runtime and the newer
sandbox file every check errors with `None`; with a consistent runtime the same guards give complete verdicts
(2 rejected, 1 incomplete on a real `AttributeError` in the guard). Commit `b164c59` fixes it: the runtime now
reads the sandbox source once, when it is loaded, and a startup message outside the protocol is an
infrastructure error that quotes the message.

What it touched in this run, from the Atlas records and the run file:

| Replay | Affected | Evidence |
|---|---|---|
| Frozen-batch screen, all arms | 44 of 44 guards, 3,271 of 3,271 checks | every error text is `None` |
| Challenger derivation, screen | 0 of 4 guards | complete verdicts |
| Challenger derivation, replay check at admission | 2 of 2 guard decisions, 276 of 276 checks | all checks errored, 21:24 to 21:25 UTC |
| Search, 62 evaluations | not measurable per check | the search ran from 21:25 to 22:56 UTC, after the change; its record keeps only the count of cleared guards (0 in all 62) |

The two derivation decisions also failed the held-out, no-regression and benign-twin checks. This run
therefore compares the three texts on procedures and prompt edits only; it does not measure their guards or
the search.

## Admission decisions in Atlas

`ratchet_l2` holds 91 signed decisions, 0 admitted.

| Experiment | Decisions | Admitted |
|---|---|---|
| `level2-run2b` challenger derivation (batch A) | 6 | 0 |
| `level2-run2b` seed | 14 | 0 |
| `level2-run2b` challenger | 12 | 0 |
| `level2-run2b` search | 14 | 0 |
| `level2-run1` challenger derivation (batch A) | 27 | 0 |
| `level2-run1` seed | 5 | 0 |
| `level2-run1` challenger | 5 | 0 |
| `level2-run1` search | 4 | 0 |
| `level2-run2` challenger derivation (batch A), stopped | 4 | 0 |
| Total | 91 | 0 |

## Cost

| Run | Worker model calls | Worker tokens in / out | Improver model calls | Improver tokens in / out |
|---|---|---|---|---|
| `level2-run2b` | 16,291 | 53,194,382 / 2,654,940 | 118 | 522,618 / 199,755 |
| `level2-run1` | 14,547 | 46,769,835 / 2,364,503 | 105 | 424,382 / 186,885 |

## Earlier runs

`level2-run1`, 2026-10-11 20:12 to 21:45 UTC (wall 5,575.2 s), same models and endpoint, 32 workers.
Families A were `approval_revoked`, `concurrent_worker`, `injection` and `order_modification`. Parent
`dfdfffeeb388`: P1 4 of 18 training and 5 of 19 held-out cases passed, P2 6 of 19 held-out. The improver formed 19
clusters: 9 in batch A, 2 in frozen batch B (digest `2c71fac4fa18`), 8 mixed and excluded. Each arm had 8
slots and 2 proposal calls. The search made 63 metric calls (budget 60) and returned `eb0ed1cfd7ba`, which
scored 0.1944 on batch A against the seed's 0.1389. The challenger was `a5d88dbad0c8`.

| Arm | Instructions | Proposed / slots | Guard / procedure / prompt edit | Screened out | Replay cleared / failed / incomplete | Evaluated fresh | Admitted / slots |
|---|---|---|---|---|---|---|---|
| seed | `942dc26006bf` | 8 / 8 | 4 / 2 / 2 | 3 | 2 / 0 / 0 | 5 | 0 / 8 |
| challenger | `a5d88dbad0c8` | 8 / 8 | 4 / 2 / 2 | 2 | 2 / 1 / 0 | 5 | 0 / 8 |
| search | `eb0ed1cfd7ba` | 7 / 8 | 4 / 1 / 2 | 1 | 2 / 2 / 0 | 4 | 0 / 8 |

The same defect reached `level2-run1` at its end: the replay check at admission errored on all 151 checks
for each of the 6 guard decisions of its frozen batch (2 per arm), decided from 21:42 to 21:45 UTC. Each of
those 6 also failed the held-out, no-regression and benign-twin checks, so its 0 admitted stands. Its screen
replays (27 guards) and its 9 derivation decisions with a guard ran before the change and are complete.

`level2-run2` swapped the family split (families A `ordinary`, `policy_changed`, `reconciliation` and
`refund_lost_reply`), and its improver formed 2 batch-A and 10 frozen clusters. It stopped at its 7,200 s
time cap at 21:11 UTC, after its search had finished and before any frozen-batch decision; it recorded 4
decisions, all from the challenger's derivation. Its search, on batch-A clusters with the same ids as
`level2-run2b`'s, cleared at least one guard in 40 of its 62 evaluations and returned `963b2b1b98d9`.
`level2-run2b` reran it with 128 workers and an 18,000 s cap.
