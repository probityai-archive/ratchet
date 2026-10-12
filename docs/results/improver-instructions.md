# Improver-instruction search: seed, challenger and searched text

Three versions of the improver's instructions (the seed, the loop's own challenger and the text the instruction search returned) proposing harness changes for the same frozen failure clusters, with every candidate judged by the admission rule.

Run of record `level2-run3`, 2026-10-12, module `ratchet.improve.instruction_search`, wall 8,963.9 s. Its
decisions were signed from 05:57:18 to 08:17:32 UTC. It reruns `level2-run2b` with the same settings after
the guard-runtime fix (pull request #2, merge `0bea97e`), because `level2-run2b`'s guard comparison is void
(see [Earlier runs](#earlier-runs)).

- Code: a snapshot of `main` at `8f18ac4`, which contains the fix. It ran on the Hetzner box
  `quint-testrunner-2`.
- Models: worker and improver are both `gpt-oss-120b`, served by vLLM on HiPerGator and reached through a
  tunnel at `127.0.0.1:18955`.
- Settings, read from the run file: database `ratchet_l2`, families A `ordinary`, `policy_changed`,
  `reconciliation` and `refund_lost_reply`, 4 candidates per cluster, a search budget of 60 metric calls,
  experiment `level2-run3`. These are the values `level2-run2b` used. The run file does not record the
  worker count.
- Store: a local `mongodb/mongodb-atlas-local:8.3.9-20260928T091948Z` container on the box, with the
  settings in `docker/compose.yaml`. The hosted Atlas cluster could not be reached when the run started.
  Every database the run wrote was dumped at the end.
- Records: the run file `private/improve/l2/level2-run3.json` and the dump
  `private/improve/l2/local-atlas-dump.archive.gz` are not committed. Every number below comes from them.

Result: 0 of 120 candidate slots were admitted (0 of 40 per arm). This time the guards were replayed: the
frozen-batch screen ran 4,266 guard checks, and 4 of them errored. Each of those 4 was an `AttributeError`
raised by the guard's own code. Every candidate that reached a fresh evaluation was refused after its first
fresh run, and each of them failed the held-out and no-regression checks.

## What was compared

1. The parent harness `dfdfffeeb388` ran fresh twice. Run P1 covered the training and held-out cases:
   2 of 18 training cases and 4 of 19 held-out cases passed. Run P2 covered the held-out cases: 5 of 19
   passed.
2. The improver grouped P1's training failures into 20 failure clusters (20 improver calls). The clusters
   were split by case family:
   - Batch A, every member in families A: 3 clusters. Only these reach the search and the challenger's
     derivation.
   - Frozen batch B, no member in families A: 10 clusters (digest `d6cdeedd071d`). All three arms are judged
     here.
   - Mixed, members on both sides: 7 clusters, excluded.

   These clusters come from this run's own parent runs. They are not the clusters of `level2-run2b`, which
   had 7 frozen clusters.
3. Challenger. The seed instructions proposed for batch A (12 slots). Those candidates went through
   admission: 9 were evaluated fresh and 0 of 12 were admitted. One call of the loop's instruction drafter
   then wrote a new text from those outcomes: `0ee2bb1c3dd2`, parent `942dc26006bf`.
4. Search. On batch A only, the search scores an instruction text by the share of requested candidate slots
   that are guards which clear replay over the recorded training traces and contain no copied training
   literal. Replay calls no worker model. With a budget of 60 metric calls, it made 63, with 63 proposer
   calls and 10 reflection calls. It returned `2df30522be1e`, a text that differs from the seed. The run log
   line "search 2df30522be1e after 63 metric calls" matches the run file. The run file does not keep the
   per-evaluation search scores.
5. Each arm then proposed 4 candidates for each of the 10 frozen clusters, giving 40 slots and 10 proposal
   calls per arm, with no repair rounds. A guard went to fresh evaluation only if it cleared replay. A
   procedure or prompt edit went to fresh evaluation if it passed the copied-literal screen. Every candidate
   evaluated fresh was decided by the real `admit()` and stored as a signed decision.

## Per arm

| Arm | Instructions | Proposed / slots | Guard / procedure / prompt edit | Screened out | Replay cleared / failed / incomplete | Evaluated fresh | Admitted / slots |
|---|---|---|---|---|---|---|---|
| seed | `942dc26006bf` | 40 / 40 | 20 / 10 / 10 | 0 | 7 / 13 / 0 | 27 | 0 / 40 |
| challenger | `0ee2bb1c3dd2` | 40 / 40 | 22 / 10 / 8 | 1 | 8 / 12 / 2 | 25 | 0 / 40 |
| search | `2df30522be1e` | 41 / 40 | 21 / 10 / 10 | 0 | 2 / 18 / 1 | 22 | 0 / 40 |

No proposal was invalid or a duplicate. The search arm's proposer returned 5 candidates, 3 of them guards,
for cluster `f661df4135` instead of 4, and all 5 were processed. That is why it shows 41 proposed for 40
slots. The challenger's screened-out candidate was a procedure that copied the training literal `T-1201`.

Guards that cleared replay, as a share of guards proposed: seed 7 of 20, challenger 8 of 22, search 2 of
21. Frozen clusters with at least one cleared guard: seed 6 of 10, challenger 6 of 10, search 2 of 10.

Evaluated fresh is the cleared guards plus the procedures and prompt edits that passed the screen: seed
7 + 20 = 27, challenger 8 + 17 = 25, search 2 + 20 = 22.

### Why the evaluated candidates were refused

Every evaluated candidate was refused after its first fresh run (C1). None reached the confirming run.
Counts are refused candidates over candidates evaluated fresh. One candidate can fail more than one check.
The checks are described in [ADMISSION.md](../ADMISSION.md).

| Arm | Replay at admission (`replay`) | Held-out recoveries (`heldout`) | No regression (`no_regression`) | Benign twins (`benign_twins`) |
|---|---|---|---|---|
| seed | 4 / 27 | 27 / 27 | 27 / 27 | 3 / 27 |
| challenger | 3 / 25 | 25 / 25 | 25 / 25 | 5 / 25 |
| search | 0 / 22 | 22 / 22 | 22 / 22 | 1 / 22 |

The replay check at admission replays a guard over all 18 training traces, not only the cluster's own
traces. Of the 17 guards that cleared the frozen-batch screen, 7 were refused there: seed 4 of 7 and
challenger 3 of 8, each because it fired on a benign twin trace. Search had 0 of 2 refused. Each of the 7
still hit its target failures.

Held-out flips in C1, summed over the candidates evaluated fresh. Fail to pass is a held-out case that
failed in both parent runs and passed in C1. Pass to fail is a held-out case that passed in either parent
run and failed in C1.

| Arm | Candidates | Fail to pass | Pass to fail | Most fail-to-pass on one candidate | Candidates with 2 or more fail-to-pass and no pass-to-fail |
|---|---|---|---|---|---|
| seed | 27 | 60 | 156 | 4 | 0 / 27 |
| challenger | 25 | 57 | 133 | 5 | 0 / 25 |
| search | 22 | 50 | 112 | 6 | 0 / 22 |

## Guard replay

The guards were replayed this time, unlike in `level2-run2b`.

| Replay | Guards | Checks | Errored checks | Verdicts |
|---|---|---|---|---|
| Frozen-batch screen, seed | 20 | 1,352 | 0 | 7 cleared, 13 rejected |
| Frozen-batch screen, challenger | 22 | 1,491 | 3 | 8 cleared, 12 rejected, 2 incomplete |
| Frozen-batch screen, search | 21 | 1,423 | 1 | 2 cleared, 18 rejected, 1 incomplete |
| Challenger derivation, batch-A screen | 6 | 368 | 4 | 3 cleared, 1 rejected, 2 incomplete |
| Replay check at admission, frozen batch | 17 | 2,394 | 0 | 10 cleared, 7 rejected |
| Replay check at admission, derivation | 3 | 402 | 0 | 2 cleared, 1 rejected |

Every errored check was an exception in the guard's own source: `AttributeError: 'str' object has no
attribute 'get'` (2), `AttributeError: 'NoneType' object has no attribute 'get'` (2), and
`AttributeError: 'Event' object has no attribute 'get'` (4, in the derivation). Each errored guard is one
of the incomplete ones. No check errored with `None`. In `level2-run2b`, by contrast, 3,271 of 3,271
frozen-batch checks errored with `None`.

## Admission decisions

The local store's `ratchet_l2.decisions` holds 83 decisions, 0 admitted. All 83 carry key id
`b61d7e39c111`. All 83 verify under the pinned key `keys/ratchet-decisions.pub`, and each stored decision
matches its signed payload field for field.

| Experiment | Decisions | Admitted |
|---|---|---|
| `level2-run3` challenger derivation (batch A) | 9 | 0 |
| `level2-run3` seed | 27 | 0 |
| `level2-run3` challenger | 25 | 0 |
| `level2-run3` search | 22 | 0 |
| Total | 83 | 0 |

The decisions of the earlier runs are in the hosted Atlas database `ratchet_l2`, not in this store. That
database holds 91 decisions, 0 admitted: 46 from `level2-run2b`, 41 from `level2-run1` and 4 from the
stopped `level2-run2`.

## Cost

| Run | Worker model calls | Worker tokens in / out | Improver model calls | Improver tokens in / out |
|---|---|---|---|---|
| `level2-run3` | 29,164 | 94,354,423 / 4,669,305 | 127 | 583,266 / 237,448 |
| `level2-run2b` | 16,291 | 53,194,382 / 2,654,940 | 118 | 522,618 / 199,755 |
| `level2-run1` | 14,547 | 46,769,835 / 2,364,503 | 105 | 424,382 / 186,885 |

## Earlier runs

### `level2-run2b`: guard comparison void

`level2-run2b` ran 2026-10-11 from 21:11 to 23:37 UTC (wall 8,797.1 s) with the same models, a different
endpoint (`localhost:8801`), 128 workers and the hosted Atlas database. Its guard comparison is void. A
defect in the guard runtime, fixed since, made every frozen-batch guard check error with `None`: 3,271 of
3,271 checks across all 44 guards (14 seed, 16 challenger, 14 search). No guard code ran.

Cause: the guard runtime started each sandbox from the sandbox's source file on disk, read when the sandbox
started. The run's process loaded the runtime at 21:11 UTC. By 21:24 UTC the checkout it ran from had
moved to a sandbox version that opens with a new startup message (commit `341721f`). From then on the
runtime took that message for a failed start and recorded the absent error field, `None`, as the error of
every check. Commit `b164c59` (pull request #2) fixes it. The runtime now reads the sandbox source once,
when it is loaded, and treats a startup message outside the protocol as an infrastructure error that
quotes the message. The same defect hit the replay check at admission for the challenger's 2 derivation
guard decisions (276 of 276 checks errored), and it falls inside the search's window (21:25 to 22:56 UTC).

What the run recorded, kept as history. Its procedures and prompt edits were evaluated normally. Its
guards, and the search, were not measured.

- Parent P1: 4 of 18 training and 4 of 19 held-out cases passed. P2: 3 of 19 held-out passed.
- Clusters: 17 in all, 2 in batch A, 7 frozen (digest `b243c1a8eb71`) and 8 mixed.
- Challenger: `d3f51c15e6b6`.
- Search: returned the seed `942dc26006bf` unchanged after 62 metric calls.

| Arm | Instructions | Proposed / slots | Guard / procedure / prompt edit | Replay cleared / failed / incomplete | Evaluated fresh | Admitted / slots |
|---|---|---|---|---|---|---|
| seed | `942dc26006bf` | 28 / 28 | 14 / 7 / 7 | 0 / 0 / 14 (void) | 14 | 0 / 28 |
| challenger | `d3f51c15e6b6` | 28 / 28 | 16 / 7 / 5 | 0 / 0 / 16 (void) | 12 | 0 / 28 |
| search | `942dc26006bf` | 28 / 28 | 14 / 7 / 7 | 0 / 0 / 14 (void) | 14 | 0 / 28 |

### `level2-run1`

`level2-run1` ran 2026-10-11 from 20:12 to 21:45 UTC (wall 5,575.2 s) with the same models and endpoint as
`level2-run2b`, 32 workers, and families A `approval_revoked`, `concurrent_worker`, `injection` and
`order_modification`.

- Parent `dfdfffeeb388`: P1 passed 4 of 18 training and 5 of 19 held-out cases. P2 passed 6 of 19 held-out.
- Clusters: 19 in all, 9 in batch A, 2 frozen (digest `2c71fac4fa18`) and 8 mixed.
- Budget: 8 slots and 2 proposal calls per arm.
- Search: 63 metric calls on a budget of 60. It returned `eb0ed1cfd7ba`, which scored 0.1944 on batch A
  against the seed's 0.1389.
- Challenger: `a5d88dbad0c8`.

| Arm | Instructions | Proposed / slots | Guard / procedure / prompt edit | Screened out | Replay cleared / failed / incomplete | Evaluated fresh | Admitted / slots |
|---|---|---|---|---|---|---|---|
| seed | `942dc26006bf` | 8 / 8 | 4 / 2 / 2 | 3 | 2 / 0 / 0 | 5 | 0 / 8 |
| challenger | `a5d88dbad0c8` | 8 / 8 | 4 / 2 / 2 | 2 | 2 / 1 / 0 | 5 | 0 / 8 |
| search | `eb0ed1cfd7ba` | 7 / 8 | 4 / 1 / 2 | 1 | 2 / 2 / 0 | 4 | 0 / 8 |

The same defect reached `level2-run1` at its end. The replay check at admission errored on all 151 checks
for each of the 6 guard decisions of its frozen batch (2 per arm), decided from 21:42 to 21:45 UTC. Each
of those 6 also failed the held-out, no-regression and benign-twin checks, so its 0 admitted stands. Its
screen replays (27 guards) and its 9 derivation decisions with a guard ran before the change and are
complete.

### `level2-run2`

`level2-run2` used the families A of `level2-run2b`. It stopped at its 7,200 s time cap at 21:11 UTC, after
its search finished and before any frozen-batch decision. It recorded 4 decisions, all from the
challenger's derivation.
