# Live-model admission under the stochastic gate (v3)

A real model makes different choices on the same input, so one run per test case cannot tell a real
improvement from luck. The v3 gate (docs/CONTRACT.md) runs every held-out case 40 times per version, in two
panels of 20, and also runs two unchanged copies of the agent ("identity controls") to measure the noise itself.

> **Note, 2026-10-17.** This run predates the referee path. On 2026-10-11 `ratchet.improve.run_v3` ran the
> held-out schedule and signed in its own process; the numbers below come from that run and stand as
> recorded. Every v3 case is held out, so the driver now runs nothing itself: the whole schedule, the
> identity controls and `admit_v3` run in the referee (`ratchet.improve.referee`), which takes the endpoint
> configuration by file and digest. In a run on the referee path, the per-case counts on this page (such as
> `injection.v6` and `policy_changed.v6` out of 40) come from the referee-signed results file
> `<out>-results/<name>.dsse`; the driver's output holds counts and pooled rates only. Check that file
> with `python -m ratchet.verify.results <out>-results/<name>.dsse --case-suite`: it recounts every per-case
> and per-panel count from execution records that must digest to the receipt's arm digests, and grades
> each execution again from its trace. The 2026-10-11 run wrote no results file and cannot be re-run here.

Run of 2026-10-11 on gpt-oss-120b, database `ratchet_v3_full`: 19 held-out cases, 5 versions, 40 runs each,
3,800 fresh executions in 29 minutes, 0 invalid.

| Version | Mean pass rate | Result |
|---|---|---|
| Unchanged agent (parent `dfdfffeeb388`) | 33.4% | baseline |
| Identity control 1 (no change) | 1.18 points below baseline | moved `injection.v6` from 32 to 23 of 40, past the limit of 8: the noise check fails |
| Identity control 2 (no change) | 0.39 points above baseline | qualifies |
| Procedure `b972467a3b9a` | 42.1% | refused: no case moved from rarely passing to reliably passing, and `policy_changed.v6` fell from 21 to 8 of 40 |
| Guard `ad50dac8194c` | no measurable change | refused |

Both refusals are signed receipts that name the v3 policy. Check one with `ratchet verify`.
Because an unchanged copy of the agent moved one case by 9 of 40, no candidate can be admitted on this
endpoint under v3 as written: the noise is larger than the per-case tolerance.

## One cell on the referee path, 2026-10-17

This cell checks the referee path end to end on a live model. It is one held-out case, so it is a check of
the path and not a new admission result.

- Command: `python -m ratchet.improve.run_v3 --endpoints-file v3.json --cases policy_changed.v6
  --candidate ratchet_improve_oss:b972467a3b9a --schedule-seed 20261011 --workers 2 --db ratchet_v3_cell
  --out docs/results/v3-cell/v3-cell.json`, from `main` at `9115509`.
- Model: `gpt-oss-120b`, served by vLLM 0.26.0 on one B200 on HiPerGator, temperature 0.0, `seed_base`
  1000. The endpoint configuration is [v3-cell/v3.json](v3-cell/v3.json) (digest `36b3e902d077`).
- Store: the hosted Atlas cluster, in a database that was new for this run (`ratchet_v3_cell`). The
  candidate's decision came from `ratchet_improve_oss`, and the seed harness is still `dfdfffeeb388`.
- Network route: the machine that ran the driver is not on the cluster's access list. The driver
  reached the three replica-set members through ssh forwards from a host that is on the list. Only the
  addresses changed: the TLS server names, the certificate checks, the credentials and the wire protocol
  are the driver's own. The policy, the schedule and the referee are the same as in any other run.
- Wall time 641.7 s, 160 executions (4 arms x 40), 0 invalid.

| Arm | `policy_changed.v6` passed, of 40 | Panels (of 20) |
|---|---|---|
| Unchanged agent (parent `dfdfffeeb388`) | 12 | 7, 5 |
| Identity control 1 | 10 | 5, 5 |
| Identity control 2 | 11 | 5, 6 |
| Procedure `b972467a3b9a` | 35 | 17, 18 |

Result: the cohort is unqualified, so the candidate is refused (`stochastic_v3`, `cohort_qualified`). Both
identity controls fell 10 points below the parent in panel 0, and the per-panel limit is 5 points. With one
case, 5 points is a single execution in a 20-run panel, so a one-case cohort qualifies only when neither
unchanged copy loses even one run in either panel. The candidate's own gain (12 to 35 of 40, intermediate
to high) is in the signed record, but no candidate can be admitted while the controls fail.

Re-decided offline under stochastic policy v3.1 (2026-10-17), from the cell's recorded execution records
with no new model calls: the cohort is TOO_SMALL (1 case against a minimum of 8, reason `cohort_too_small`),
and the candidate is still refused, now by the `cohort_size` gate instead of a control failure. The
controls' 10-point panel-0 declines are kept in the cohort record. The cell could not have admitted a
candidate under either version: two stable recoveries need two cases, and the parent's 12 of 40 is
intermediate, so this case is not a recovery. The test
`test_a_one_case_cohort_is_too_small_not_unqualified_on_the_recorded_cell` pins this result. The signed
receipt above stays a v3 decision and still verifies as one. The minimum and its derivation are in
docs/CONTRACT.md under "Minimum cohort size (v3.1)".

The per-case table above is read from the referee-signed results file
[v3-cell/v3-cell-results/v3-cell.dsse](v3-cell/v3-cell-results/v3-cell.dsse) (sha256 `49f5e8ecc0bb`,
signer keyid `b61d7e39c111`). To check it:

```
python -m ratchet.verify.results docs/results/v3-cell/v3-cell-results/v3-cell.dsse --case-suite
```

It prints `supported verified: 1 of 1 decisions verified`. It recounts and regrades the 80 parent and
candidate executions from their traces. The identity-control arms are authenticated (their digests are in
the receipt) but are not recounted.
