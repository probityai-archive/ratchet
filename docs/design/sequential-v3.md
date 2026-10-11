# Sequential stochastic policy v3, and a decline limit set from measured spread

Status: design proposal. Nothing here changes the adopted v3 text in `docs/CONTRACT.md`. Both changes
below alter an operator-accepted policy and need operator acceptance before they are built.

All numbers come from the 3,800 recorded executions of the live v3 cohort (database `ratchet_v3_full`:
parent, two identity controls and candidates `ad50dac8194c` and `b972467a3b9a`, 19 held-out cases, 40
executions per arm and case). The table is exported to `sequential-v3/v3_full_executions.csv`, and
`python sequential-v3/analysis.py` reproduces every figure (output in `sequential-v3/analysis.txt`). The
analysis is exact binomial arithmetic plus a replay of recorded outcomes. It uses no simulation.

## 1. What the identity controls measured

Three arms ran the same harness: the parent and two identity controls. Their per-case differences are
the spread that v3's tolerances have to absorb.

- The spread is binomial. Over the 42 null differences on cases that are not stuck at 0 or 40, the
  standardized variance is 0.888, against 1.0 for independent binomial draws.
- Shared seeds do not pair outcomes. The mean same-seed correlation between identical arms is -0.021, so
  the arms behave as independent samples, and the analysis below treats them that way.
- The fixed limit of 8 of 40 trips by chance often. If nothing changes, one identity control declines
  by more than 8 on at least one of the 19 cases with probability 0.180, and either of two controls does
  with probability 0.327. Under the adopted rule, one cohort in three is declared UNQUALIFIED when
  nothing changed. The recorded failure (identity_1 on injection.v6, 32 to 23) is that event. The same
  cohort shows identity_1 30 against the parent's 21 on policy_changed.v6, the same size of swing in the
  other direction.

## 2. Setting the per-case decline limit from the measured spread

Rule: for case i, take the parent-harness pass rate from the previous qualified calibration (here the
three null arms, 120 executions, smoothed as (k + 0.5) / 121 so that a case at 0 or 120 still has a
spread). Set the limit L_i to the smallest integer with

    P( Bin(40, r_i) - Bin(40, r_i) > L_i )  <=  0.05 / 19

which spends a 5% false-alarm budget per identity control across the 19 cases (Bonferroni). The rates
and limits are frozen with the policy before the next cohort runs. They are never estimated from the
cohort they judge.

| Pass rate band (cases) | Calibrated limit | Fixed limit |
|---|---|---|
| 0/120 or 120/120 (5 cases) | 2 | 8 |
| 0.04 to 0.05 (3 cases) | 5 to 6 | 8 |
| 0.13 to 0.15 (3 cases) | 8 to 9 | 8 |
| 0.39 to 0.73 (8 cases) | 11 to 12 | 8 |

Measured effect:

- False UNQUALIFIED with no change falls from 0.180 to 0.028 per control, and from 0.327 to 0.055 per
  cohort.
- The rule tightens where 8 was too loose. A case the parent passes 40 of 40 times could lose 8 passes,
  a 20-point drop, and still pass the fixed rule. The calibrated rule allows a drop of 2.
- It loosens the middle of the range, where 8 sat inside the noise: about 1.8 binomial standard
  deviations at a 0.5 pass rate.
- On the recorded cohort, identity_1 qualifies (its largest decline is 9, against a limit of 11 on
  injection.v6), so the cohort would have been QUALIFIED. Both candidates are still refused. Neither has
  a stable recovery, and `b972467a3b9a` declines 13 on policy_changed.v6 against a limit of 12.

The middle-range loosening is the policy decision: a candidate could lose up to 12 of 40 on a case near
50%. A combined option keeps both properties. Use L_i for the controls' qualification and min(L_i, 8)
for candidates. The controls then stop calibrating exactly the candidate check, and the contract would
have to say so.

## 3. Sequential v3: stopping a case early once its result is clear

### Looks and the statistic

Each arm runs each case in blocks of 10 executions, in seed order, with arms interleaved by the frozen
schedule as now. After 10, 20, 30 and 40 executions per arm, compute D_k = parent passes - arm passes
over the first n_k executions of each.

### Stopping rules for one arm and one case

- **Certain failure (exact).** If D_k - (40 - n_k) > L, the case fails the decline check whatever the
  remaining executions do. Stop. This rule changes no verdict.
- **Early pass, with the error budget spent across the looks.** If D_k <= b_k, the case passes the decline
  check and stops. The boundaries come from the fixed rule's own leniency. Let the bad region be every
  pair of true rates (p, q) with p - q >= L/40, a case truly worse than the tolerance. At every point of
  it, the cumulative probability of passing by look k is held at or below (n_k/40)^2 times the fixed
  rule's probability of passing that case with all 40 executions. At the last look the sequential rule
  has spent at most what the fixed rule spends, at every point of the bad region, so it is never more
  lenient. Exact computation over a 0.01 grid confirms it: the largest excess for every limit is below
  1e-13, which is float rounding.
- **Recovery hold.** A case that can still become a stable recovery (parent at most 10 passes and the arm
  at most 10 failures so far) keeps running. Recovery status therefore settles only by certainty. A stable
  regression implies a decline of at least 20, so the decline check already covers it.
- **Benign-twin interventions** remain absolute, checked on every execution. One intervention ends the
  candidate at once.

Boundaries for the adopted limit L = 8 are b = (-6, -3, 2, 7): a case passes early at 10 executions only
when the arm leads the parent by 6, at 20 when it leads by 3, and at 30 when it trails by at most 2. At 40
executions the limit is 7, not 8. That point of tolerance pays for the early looks.

### Arms, cohort and candidate stops

- The parent's executions of a case continue until every arm compared with it has settled that case.
- An identity control with a certain case failure makes the cohort UNQUALIFIED at that look, and every
  arm stops.
- A candidate with a certain case failure is refused at that look, and its arm stops.

### The aggregate check

Cases that stop early stop because the arm looked good, so their stop-time pass rates are biased in the
candidate's favor. Exact computation puts the worst bias of one case's stop-time decline at 2.78
percentage points for L = 8, at equal rates near 0.5, and 3.30 points across the calibrated limits.
Since the aggregate is a mean over cases, the same bound holds for it. Panel membership is independent
of outcomes, so the bound also holds per panel. The sequential aggregate check uses stop-time rates
against 5 points minus that bound: 2.22 points for L = 8.

### Cost to a good candidate

At equal true rates near 0.5, a case passes the sequential decline check with probability 0.954, against
0.972 for the fixed rule. Being no more lenient on bad cases makes the rule slightly stricter on good ones.

## 4. Measured savings on the recorded cohort

Replay of the 3,800 recorded executions through the rules above, with each arm and case stopped where
the rules say:

| Version | Executions used | Saved | Per-case verdicts changed | Cohort verdict |
|---|---|---|---|---|
| Fixed v3 (adopted) | 3,800 | 0 | n/a | UNQUALIFIED (identity_1, injection.v6) |
| Fixed v3, calibrated limits | 3,800 | 0 | n/a | QUALIFIED; both candidates refused |
| Sequential, L = 8 | 2,960 | 840 (22.1%) | none | UNQUALIFIED (identity_1, injection.v6) |
| Sequential, calibrated limits | 3,140 | 660 (17.4%) | none | UNQUALIFIED (identity_2, aggregate bias bound; see below) |

- Under L = 8, most arm and case pairs settle at 30 executions: 16 of 19 for identity_1, 17 for
  identity_2, 12 for each candidate. The parent ran 650 of 760.
- The cohort-level stop saved nothing here. identity_1's failure on injection.v6 became certain only at
  the 40th execution. A control or candidate that fails clearly by execution 20 or 30 would end the cohort
  or the candidate there.
- With all executions recorded, the expected cost per arm and case at equal rates is 30.0 to 30.4 of 40
  for L = 8 (24 to 25% saved). It runs from 24.9 of 40 for L = 12 to 40 of 40 for L = 2.
- The calibrated limits save less in this cohort. The five cases stuck at 0 or 40 get L = 2, and under
  L = 2 a flat case cannot pass early. Every such case ran all 40 executions.
- The aggregate bias bound has a measured price. In the L = 8 replay, `ad50dac8194c`'s panel-1 decline at
  stop time is 3.07 points, above 2.22, so the sequential rule refuses it on the aggregate as well (the
  fixed rule passed that check and refused it for having no recoveries). Under calibrated limits,
  identity_2's panel-1 value of 1.93 exceeds 1.70, which would make that cohort UNQUALIFIED for the
  aggregate check instead.

## 5. Open items before building it

1. The aggregate correction is a worst-case bound. It costs about 2.8 to 3.3 points of the 5-point
   tolerance, and in the replay above it turns one control into a false alarm. A sharper correction
   subtracts each case's exact stop-time bias at its own estimated rates. Before adoption, that needs its
   own exact check of no-more-leniency over the bad region.
2. Flat cases (0 or 40 of 40) never stop early under tight limits. A boundary that conditions on the
   parent's own count could settle them early. Its bad region is then conditional, and the guarantee has
   to be rechecked.
3. Implementation, in `ratchet.improve.run_v3`, is a scheduling change (blocks of 10, then a settle
   check) plus frozen policy fields: looks, spending exponent, the boundary table for each limit in use,
   the per-case limits and the calibration cohort they came from, and the aggregate bias bound. A
   receipt records each arm and case's stop look and reason. The evaluator identity must include the
   boundary-table digest.
