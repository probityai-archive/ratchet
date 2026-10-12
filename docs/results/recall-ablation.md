# Recall ablation: the proposer with and without prior fixes

The proposer with and without prior fixes recalled from a history run, over six paired repetitions with the same cases, models and proposal budget, with every count given against its denominator.

Run of record `recall_abl3`, 2026-10-12. The run files `private/improve/recall/a/recall-ablation-v3a.json`
and `private/improve/recall/b/recall-ablation-v3b.json`, and the database dumps beside them, are not
committed. Every number below comes from them.

## Design

One history run H is one generation of the plain loop, written to a history database. The failure
clusters of H and the outcomes of their candidates are stored as memory. In each repetition two arms then
run one generation each from the same seed harness `dfdfffeeb388`, with the same cases, models and budget:

- Arm A proposes with no prior fixes.
- Arm B proposes with prior fixes recalled from H (up to 5) added to each proposal prompt.

Budget per generation: at most 4 failure clusters, 4 candidates per cluster, and 3 candidates evaluated
fresh. Cases: the 16 cases of the families `refund_lost_reply`, `approval_revoked`, `concurrent_worker`
and `reconciliation`, 8 training and 8 held-out. Each arm reruns the parent and forms its own failure
clusters from its own runs. The two arms of a repetition therefore share cases and budget but not always
clusters. The cluster ids each arm formed are listed under [Per repetition](#per-repetition).

Each cluster is placed in a stratum by what H holds for it:

- `admitted_before`: H admitted a candidate for it.
- `rejected_only_before`: H decided candidates for it and admitted none.
- `no_history`: H has no record of it.

Settings, read from the run files: worker `gpt-5.6-luna`, improver `gpt-6-astra`, both through the local
subscription proxy (reached on the box at `127.0.0.1:18987`), 12 workers, `max_clusters` 4, `candidates`
4, `max_evals` 3, prefix `recall_abl3`.

The script differs from the one that produced the void run in three ways. It signs decisions with
`Signer.for_decisions`, so only the pinned decision key can sign. It reads the database URI and the Voyage
key from the environment. It imports the cases from `ratchet_cases`.

## Two halves, two histories

The six repetitions ran in two halves. Each half ran on its own Hetzner burst box, against its own local
`mongodb/mongodb-atlas-local:8.3.9-20260928T091948Z` container, from a snapshot of `main` at `8f18ac4`.

| Half | Repetitions | Box | History run H | Clusters in H | H candidates / decided / admitted |
|---|---|---|---|---|---|
| a | 1 to 3 | `box-burst-1791785120-23376` | its own | 4 | 14 / 3 / 0 |
| b | 4 to 6 | `box-burst-1791786190-15389` | its own | 2 | 8 / 3 / 1 |

Each half built its own history, so arm B in repetitions 1 to 3 recalls from a different history than arm
B in repetitions 4 to 6. History a admitted nothing, so its clusters fall in `rejected_only_before`.
History b admitted one prompt edit for cluster `d077c6d2f4` (promoted harness `3b73d7e37b23`), so that
cluster falls in `admitted_before` in repetitions 4 to 6. The strata are therefore not comparable across
the halves.

## Totals over the 6 repetitions

| Arm | Stratum | Clusters | Slots | Proposed | Guards | Replay cleared | Evaluated and decided | Admitted |
|---|---|---|---|---|---|---|---|---|
| A | all | 16 | 64 | 64 | 32 | 30 | 18 | 3 |
| A | no_history | 9 | 36 | 36 | 18 | 16 | 0 | 0 |
| A | rejected_only_before | 4 | 16 | 16 | 8 | 8 | 9 | 2 |
| A | admitted_before | 3 | 12 | 12 | 6 | 6 | 9 | 1 |
| B | all | 12 | 48 | 48 | 24 | 19 | 18 | 4 |
| B | no_history | 5 | 20 | 20 | 10 | 6 | 0 | 0 |
| B | rejected_only_before | 4 | 16 | 16 | 8 | 7 | 9 | 2 |
| B | admitted_before | 3 | 12 | 12 | 6 | 6 | 9 | 2 |

Admitted over evaluated and decided: A 3 of 18, B 4 of 18. Admitted over slots: A 3 of 64, B 4 of 48.

## Per repetition

Stratum `all`:

| Rep | Arm | Clusters | Slots | Proposed | Guards | Replay cleared | Evaluated and decided | Admitted |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 2 | 8 | 8 | 4 | 4 | 3 | 0 |
| 1 | B | 1 | 4 | 4 | 2 | 2 | 3 | 0 |
| 2 | A | 4 | 16 | 16 | 8 | 7 | 3 | 1 |
| 2 | B | 4 | 16 | 16 | 8 | 7 | 3 | 1 |
| 3 | A | 1 | 4 | 4 | 2 | 2 | 3 | 1 |
| 3 | B | 1 | 4 | 4 | 2 | 2 | 3 | 1 |
| 4 | A | 4 | 16 | 16 | 8 | 8 | 3 | 0 |
| 4 | B | 1 | 4 | 4 | 2 | 2 | 3 | 0 |
| 5 | A | 1 | 4 | 4 | 2 | 2 | 3 | 0 |
| 5 | B | 1 | 4 | 4 | 2 | 2 | 3 | 1 |
| 6 | A | 4 | 16 | 16 | 8 | 7 | 3 | 1 |
| 6 | B | 4 | 16 | 16 | 8 | 4 | 3 | 1 |

Stratum `no_history` (repetitions with no such cluster are left out):

| Rep | Arm | Clusters | Slots | Proposed | Guards | Replay cleared | Evaluated and decided | Admitted |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1 | 4 | 4 | 2 | 2 | 0 | 0 |
| 2 | A | 2 | 8 | 8 | 4 | 3 | 0 | 0 |
| 2 | B | 2 | 8 | 8 | 4 | 4 | 0 | 0 |
| 4 | A | 3 | 12 | 12 | 6 | 6 | 0 | 0 |
| 6 | A | 3 | 12 | 12 | 6 | 5 | 0 | 0 |
| 6 | B | 3 | 12 | 12 | 6 | 2 | 0 | 0 |

Stratum `rejected_only_before` (half a only):

| Rep | Arm | Clusters | Slots | Proposed | Guards | Replay cleared | Evaluated and decided | Admitted |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1 | 4 | 4 | 2 | 2 | 3 | 0 |
| 1 | B | 1 | 4 | 4 | 2 | 2 | 3 | 0 |
| 2 | A | 2 | 8 | 8 | 4 | 4 | 3 | 1 |
| 2 | B | 2 | 8 | 8 | 4 | 3 | 3 | 1 |
| 3 | A | 1 | 4 | 4 | 2 | 2 | 3 | 1 |
| 3 | B | 1 | 4 | 4 | 2 | 2 | 3 | 1 |

Stratum `admitted_before` (half b only):

| Rep | Arm | Clusters | Slots | Proposed | Guards | Replay cleared | Evaluated and decided | Admitted |
|---|---|---|---|---|---|---|---|---|
| 4 | A | 1 | 4 | 4 | 2 | 2 | 3 | 0 |
| 4 | B | 1 | 4 | 4 | 2 | 2 | 3 | 0 |
| 5 | A | 1 | 4 | 4 | 2 | 2 | 3 | 0 |
| 5 | B | 1 | 4 | 4 | 2 | 2 | 3 | 1 |
| 6 | A | 1 | 4 | 4 | 2 | 2 | 3 | 1 |
| 6 | B | 1 | 4 | 4 | 2 | 2 | 3 | 1 |

Clusters each arm formed, and the admitted candidate:

| Rep | Arm A clusters | Arm B clusters | Shared | A admitted | B admitted |
|---|---|---|---|---|---|
| 1 | `7c61c5915e`, `d077c6d2f4` | `d077c6d2f4` | 1 | none | none |
| 2 | `645efb4d48`, `acdf16e71a`, `d077c6d2f4`, `e0b813af1a` | `645efb4d48`, `c7b9c63232`, `d077c6d2f4`, `e0b813af1a` | 3 | prompt edit | procedure |
| 3 | `d077c6d2f4` | `d077c6d2f4` | 1 | prompt edit | prompt edit |
| 4 | `645efb4d48`, `d077c6d2f4`, `e0b813af1a`, `f6b4da2c1d` | `d077c6d2f4` | 1 | none | none |
| 5 | `d077c6d2f4` | `d077c6d2f4` | 1 | none | procedure |
| 6 | `645efb4d48`, `acdf16e71a`, `d077c6d2f4`, `e0b813af1a` | `10eaafde9d`, `645efb4d48`, `d077c6d2f4`, `e0b813af1a` | 3 | prompt edit | procedure |

## What the evaluation budget reached

With 3 fresh evaluations per generation, the 3 decided candidates of every one of the 12 arm runs were the
guard, the procedure and the prompt edit for cluster `d077c6d2f4`. In that cluster the agent sent the
refund notification to a placeholder address rather than the customer of record. Every run formed it, and
it has history in both halves. No candidate for any other cluster was evaluated, so `no_history` has 0
decisions in both arms. Every admitted candidate on this page is for `d077c6d2f4`.

In all 12 runs the decided guard was refused, and each of the 12 failed the replay check at admission as
well as the held-out and benign-twin checks.

## Paired comparison

Admitted per repetition, A then B: 0 and 0, 1 and 1, 1 and 1, 0 and 0, 0 and 1, 1 and 1. Five pairs are
equal. One pair (repetition 5) differs, and it favours B. One differing pair out of six cannot separate
the arms: under no difference, a single differing pair favours B with probability one half.

On the one cluster both arms evaluated in every repetition, B admitted 4 of 18 decided candidates and A
admitted 3 of 18. A's three admissions were all prompt edits. B's four were three procedures and one
prompt edit.

Guards that cleared replay: A 30 of 32, B 19 of 24. The arms proposed for different clusters (A 16
clusters, B 12), so this count is not paired cluster for cluster.

## Guard replay

The guards were replayed. Across the 12 arm runs, 56 guards ran 3,662 replay checks (A 2,082, B 1,580),
and none errored.

## The void run

The first run of this ablation, prefix `recall_abl2` (`private/improve/recall-ablation.json`), is void and
reports no result here. It ran before the guard-runtime fix (pull request #2, merge `0bea97e`). All 30 of
its guards ended INCOMPLETE, with every check, 1,994 of 1,994, erroring with `None`, so no guard code ran.
It also stopped after 5 of its 12 planned arm runs.

## Decisions

The script drops each arm's database once its counts are written, so the dumps keep only the two history
databases. Each holds 3 decisions: half a 0 admitted, half b 1 admitted. All 6 carry key id
`b61d7e39c111`. All 6 verify under the pinned key `keys/ratchet-decisions.pub`, and each stored decision
matches its signed payload field for field. The 36 arm decisions survive only as candidate statuses in the
run files.
