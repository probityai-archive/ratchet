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
