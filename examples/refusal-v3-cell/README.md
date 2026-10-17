# A signed refusal you can check offline

One live cell of the v3 gate refused a candidate that looked better. On `policy_changed.v6` the candidate
passed 35 of 40 runs and its parent passed 12 of 40. Both identity controls (unchanged copies of the agent)
fell 10 points below the parent in panel 0, and the allowed decline is 5 points. So the cohort was
unqualified, and Ratchet signed a refusal of the candidate.

```sh
uv run python examples/refusal-v3-cell/verify.py
```

```text
regraded 80 executions from their traces
decision verified, candidate refused, reason: identity controls breached
```

The script checks the pinned digest of the signed results file, runs `python -m ratchet.verify.results ... --case-suite`,
and reads the signed decision. CI runs it on every pull request.

"Decision verified" is not "candidate accepted". It says the signed record is genuine and its counts match the
executions. The decision itself is a refusal. This is single-cell evidence from one run by the operator: one case,
one model (gpt-oss-120b), one schedule. It checks the referee path; it is not an admission result.

The records stay where the results page cites them, so no copy can drift:

| Record | Path |
| --- | --- |
| Signed decision (sha256 `49f5e8ecc0bb`) | `docs/results/v3-cell/v3-cell-results/v3-cell.dsse` |
| 80 parent and candidate execution records | `docs/results/v3-cell/v3-cell-results/cohorts/cd7b60eb83174597/traces/` |
| Endpoint configuration and run output | `docs/results/v3-cell/v3.json`, `docs/results/v3-cell/v3-cell.json` |
| Write-up | `docs/results/v3-live.md` |
