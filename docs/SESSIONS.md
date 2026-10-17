# The k-session rule

A candidate harness is compared with its parent in k independent sessions. Each session runs four fresh
arms: the parent, two identity arms (the parent's bytes under their own seed streams) and the candidate,
each n times on every case, all randomly interleaved. The rule is ledger D37, with k = 8 and n = 5 by
default. `ratchet.improve.sessions` implements the protocol, runs it, writes a bundle and recomputes the
verdict from the bundle alone.

## Protocol

- **Schedule.** Every (arm, case, rep) runs once per session, in an order drawn from the session's own
  predeclared seed (`sha256("<label>/<session>/schedule")`).
- **Endpoints.** Each execution's endpoint is drawn uniformly at random from the session's pool and
  recorded, so routing cannot follow the candidate.
- **Seeds.** The seed of (session, arm, rep) is the first 8 bytes of `sha256("<label>/<session>/<arm>/<rep>")`,
  so no seed is shared between arms. Seeds can exceed 2**53 and are written as decimal strings.
- **Lost executions.** A lost execution is rerun once, in a slot drawn at random from the rest of its
  session's schedule. No outcome is read before the session ends. An execution lost twice makes the
  comparison inconclusive.

## Test

For session s and unchanged arm j, D_sj is the sum over cases of (candidate passes minus arm j passes) / n,
and t_j = mean_s(D_sj) / (sd_s(D_sj) / sqrt(k)). The candidate is admitted only when all of these hold:

- the smallest t_j reaches t(0.95, k - 1), which is 1.894578605061305 for k = 8;
- the harm screen passes: per case, a one-sided exact test of the candidate's kn runs against the unchanged
  arms' 3kn, at 0.05.

The comparison is inconclusive, never a rejection, when any of these holds:

- a sample SD is zero;
- an execution is invalid;
- a log check fires:
  - **E**, the endpoint mix, at chi-square p < 0.01;
  - **T\***, completion time at p < 0.01 while the unchanged arms' outcomes trend with dispatch time at
    p < 0.05;
  - **P**, the pairing check, at z >= 2.7130518884727213.

There is no padding controller.

The statistics are pure Python (`ratchet/improve/_dist.py`). `tests/improve/test_sessions.py` checks every
statistic and every verdict against the reference implementation that produced the simulation tables, on 24
fixed cohorts (`tests/improve/data/d37_oracle.json`).

## Running it

On one host, or inside one Slurm allocation whose job writes its servers' URLs to a file:

```sh
uv run python -m ratchet.improve.sessions_live --label my-run --cases a,b,c \
  --harness parent=parent.json --harness candidate=candidate.json \
  --endpoints-file endpoints.txt --model gpt-oss-120b --workers 32 --out bundle/
```

The sessions run one after another. The endpoints file is read again before every session, and
`--session-start-cmd CMD` runs `CMD <session>` first, which is where a deployment is restarted so that each
session gets a fresh serving process. The harness files are the JSON documents the exporters write, each
checked against its own digest.

## The bundle, and checking it offline

| file | contents |
| --- | --- |
| `manifest.json` | the protocol (label, cases, arm harness versions, k, n, both thresholds), its digest, and per session the endpoint pool and the schedule digest |
| `executions.jsonl` | every attempt: session, schedule position, arm, case, rep, seed, endpoint, attempt, dispatch and completion seconds, outcome, run id |
| `verdict.json` | the verdict with every statistic it rests on |

```sh
python -m ratchet.improve.sessions verify bundle/
```

`verify` rebuilds each session's schedule from its seed and recorded pool, refuses a log that departs from it
(another seed, endpoint or position, a repeated attempt, a rerun of a valid execution), recomputes the
verdict, and compares it with `verdict.json` field for field. It exits 0 when they match, 1 when they do not,
and 2 when it refuses the bundle. A bundle cannot declare a laxer threshold: the protocol's t and z
constants must equal the test's own quantiles.
