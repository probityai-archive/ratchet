# Ratchet: the 3-minute live demo

One narrative spine: a double refund slips through, a patch to stop it is proposed, and the patch is
checked against every recorded action before anything is adopted. All eight case families and both
benchmark results sit in the evidence, but the spoken story is one incident.

The presenter speaks in their own words. The "claim" column is what each beat has to establish, and the
quoted opening is the approved wording.

## Rules for the whole run

1. Every screen is labelled. `demo.py` prints `[LIVE <time>]` or `[RECORDED <time>]` above each command.
   Say the label out loud when it is RECORDED ("this is the stored run from 2:05 PM").
2. Nothing in the 3 minutes waits for a model call. Fresh evaluations were launched before the slot, and
   the screens read their stored, timestamped results.
3. If the live proposer produced no usable candidate, show its real rejection and load the earlier
   measured candidate. A human-written fixture may demonstrate a gate, and is never called autonomous
   discovery.
4. Do not compare to other teams. Say what Ratchet visibly does.
5. Numbers come from the stored reports on screen. If a number is not on the screen, do not say it.

## Setup

One terminal, 30 pt font, dark theme, full screen. A second terminal pane on the side holds the
background evolution run. In the main pane:

```sh
cd <repo> && source .venv/bin/activate
uv run python docs/demo/derive_ids.py     # every id and number below, read from Atlas and the stored results
set -a; . docs/demo/ids.env; set +a
alias d='python3 docs/demo/demo.py run'   # d matrix, d trace, ...
```

| When | Command | Why |
|---|---|---|
| T-35 min | `uv run python docs/demo/derive_ids.py` | writes `docs/demo/ids.env`; names every value that does not exist yet and stops, so nothing is typed by hand |
| T-30 min | `python3 docs/demo/demo.py preflight` | environment, store reachable, ids resolved, every step has a recording under 3 hours old |
| T-25 min | `python3 docs/demo/demo.py snapshot` | stores the output of every step as the RECORDED fallback; the resume step is skipped because resuming finishes the run (record it once with `demo.py snapshot resume` on a spare killed run) |
| T-6 min | `python3 docs/demo/demo.py evolve-bg 1` | evolution run for the 0:50 beat; log in `docs/demo/out/evolve.log` |
| T-4 min | `ratchet run --harness $CLEARED_CAND --split heldout --workers 8` in the side pane | keeps the change stream busy for the 1:20 beat; its numbers are status only, not admission evidence |
| T-1 min | open the Raindrop cohort view for the experiment in a browser tab; `clear` | cohort view for the 2:10 beat |

`demo.py list` shows each step, its window and the age of its recording. `demo.py recorded <id>` forces
the stored output.

## Second by second

### 0:00 to 0:30  The rejected candidate  (`d matrix`)

Command run: `ratchet replay $REJECTED_CAND $CLEARED_CAND`

| Second | Screen | Presenter claim |
|---|---|---|
| 0:00 | The matrix reveals row by row: trap failures first, then past successes, benign twins | "Passing held-out tests is necessary, but it does not tell you which recorded actions a patch would block." |
| 0:09 | Pointer on guard B: a red `BLOCK` in a past-success or benign-twin row | "Ratchet replays the proposed guard against recorded history: here it stops the bad retry, but also blocks a legitimate refund," |
| 0:18 | VERDICT row: B `REJECTED` in red | "so we reject it." |
| 0:22 | Guard A column: every target failure hit, zero blocked past successes, verdict green | "Only a change that clears replay, fresh evaluation, and authority checks gets a signed admission record." |

The screen must show: the parent version, the trace count, the replay time and "0 model calls" from the
footer, guard A's `n/n targets` and `0 blocked`, and guard B's false fire. Read the trace count and
duration off the footer. Do not quote a figure that is not printed there.

Fallback: `d matrix` switches itself to the stored matrix (`--delay 0`, static) and prints the RECORDED
label with its time.

### 0:30 to 0:50  The grader reads the ledger  (`d trace`)

Command run: `ratchet show trace $BAD_TRACE`

| Second | Screen | Presenter claim |
|---|---|---|
| 0:30 | Trace timeline from Atlas | "This is the failing run, read back from Atlas." |
| 0:35 | The `refund` call, then the `ERROR timeout` result | The refund committed; the timeout hid it. |
| 0:40 | The retried `refund` call and the grader reasons in red | "The agent retried. The grader checks the ledger, not the agent's explanation." |

The screen must show: a `tool_call` for the refund, a `tool_result` error, a second `tool_call`, and the
grader's reasons naming the ledger.

Fallback: stored trace output, labelled.

### 0:50 to 1:20  Evolution and the new guard  (`d evolve`, then `d replay-new`)

Commands run: `python3 docs/demo/demo.py follow $EVOLVE_LOG`, then `ratchet replay`

| Second | Screen | Presenter claim |
|---|---|---|
| 0:50 | The evolve log scrolls: failure clusters, retrieved similar failures, a proposed guard | "This generation started six minutes ago. It read prior failures from Atlas and proposed a guard." (Say "live-generated" only if the log shows the proposal.) |
| 1:02 | `ratchet replay` with no arguments: the latest generation's candidates in the matrix | "Replay runs in seconds with no model call: target hits, false fires, and every recorded trace checked." |
| 1:15 | Verdict list: `N/N checked, CLEARED` for a cleared guard, or `k/N checked, INCOMPLETE` with the first counterexample for one that stopped early | "Clearance needs the whole history. One counterexample ends a search." |

Mark every item on screen as live-generated or precomputed. The `[LIVE live-generated ...]` tag is
printed for these two steps. If the log shows no proposal yet, use the stored run and say so.

Fallback: `tail` of the stored log and the stored matrix, labelled RECORDED.

### 1:20 to 1:50  Fresh evaluation and admission  (`d eval-status`, then `d admitted`)

Commands run: `ratchet live --timeout 10 --max-events 8`, then `ratchet show decision $ADMITTED_DECISION`

| Second | Screen | Presenter claim |
|---|---|---|
| 1:20 | Change-stream lines landing as the side-pane evaluation writes traces | "The evaluator is running fresh cases right now, writing into Atlas." |
| 1:30 | Decision panel: gates table with replay, fresh, no_regression, benign_twins, overfit_lint, the authority gate, cost | "This is a finished paired evaluation, timestamped." |
| 1:38 | The paired table and the `fresh` row: stable recoveries against the number needed, protected cases kept, the parent's own run-to-run disagreement D, the confirming run | Read the printed counts: the stable recoveries against the number needed, and the protected cases that passed in both candidate runs (`STABLE_RECOVERIES`, `STABLE_NEEDED`, `PROTECTED_N` in `ids.env`). The parent ran twice, so its noise is on screen. |
| 1:46 | `benign_twins` and `no_regression` rows | "Benign twins untouched, protected cases still pass. Cost is recorded, not a gate." |

The screen must show: ADMITTED, the `decided` timestamp, the receipt id, and every gate PASS.

Fallback: `d admitted` is a stored read of a decision, so it rarely fails. `eval-status` falls back to a
stored change-stream capture, labelled.

### 1:50 to 2:10  Three boundaries  (`d authority`, `d denied`, `d verify`)

Commands run: `ratchet show decision $AUTHORITY_DECISION`, `python3 docs/demo/denied_read.py`,
`ratchet verify $ADMITTED_DECISION`

| Second | Screen | Presenter claim |
|---|---|---|
| 1:50 | Decision panel: REJECTED, `authority` FAIL listing what grew (tools granted, guard removed or altered) | "A change that grows the agent's authority is rejected, whatever its score. This is an enforced policy, not a safety proof." |
| 1:58 | `denied_read.py`: improver identity, held-out cases, `DENIED (code 13)` and the server's message | "The improver's database identity has no privilege on the held-out cases. Different mechanism, same record." |
| 2:04 | `ratchet verify`: signature valid, key matches, stored decision equals the signed one | "The receipt verifies against a pinned public key, not a key carried in the file." |

Fallback: `authority` and `verify` are stored reads. `denied` falls back to the stored output of the same
script, labelled.

### 2:10 to 2:35  Lineage and breadth  (`d lineage`, `d results`, then the Raindrop tab)

Commands run: `ratchet show lineage`, `cat $RESULTS_FILE`

| Second | Screen | Presenter claim |
|---|---|---|
| 2:10 | Lineage tree from Atlas: admitted versions and rejected candidates as nodes | "Every version, every rejection, in Atlas." |
| 2:16 | `RESULTS_FILE`: AgentDojo and tau-bench retail rows with task counts, version pins and task ids location | State the subset sizes as printed. The synthetic world is separately labelled. |
| 2:27 | Browser tab: Raindrop cohort view of matched baseline and candidate runs (opened by hand; no URL is derived) | "Same runs, matched cohorts, in Raindrop." Simulated and benchmark cohorts are labelled as such. |

Fallback: lineage and results are stored reads. If the browser tab fails, open
`docs/demo/recorded/raindrop.png` (a screenshot taken at preflight) and say "stored screenshot, taken at
<time>".

### 2:35 to 2:50  Level 2  (`d level2`)

Command run: `cat $LEVEL2_FILE`

| Second | Screen | Presenter claim |
|---|---|---|
| 2:35 | The level-2 comparison: improver instructions on proposal tasks they were not tuned on, same budget for each arm | "The improver's own instructions go through the same kind of test." |
| 2:42 | Per arm: proposed, invalid, duplicates, replay-cleared, evaluated fresh, admitted, and the yield with its denominator | State the result exactly as printed, positive or negative. |

Fallback: stored file read.

### 2:50 to 3:00  Resume  (`d resume`)

Command run: `$RESUME_CMD`

| Second | Screen | Presenter claim |
|---|---|---|
| 2:50 | An interrupted case resumes from its checkpoint; the ledger shows the refund exactly once | "A run killed after a committed refund resumes without refunding twice." |
| 2:56 | Closing frame: the lineage head | "Every adopted version comes with its evidence. Every rejected change stays in the record." |

Fallback: stored output of the same resume.

## Questions to prepare for (1 to 2 minutes of Q&A)

| Question | Answer, with the artifact |
|---|---|
| Does replay prove the patch works? | No. It proves the guard's behaviour on recorded inputs: it fires on the failures and on no recorded success. Fresh evaluation tests the behaviour. Neither is a universal proof, and the decision says which evidence was checked. |
| Why trust the held-out numbers? | The parent runs twice fresh, so its own disagreement is measured and printed. The candidate must recover cases the parent fails both times, keep every case the parent ever passed, and repeat that on a second run. Held-out cases sit in a database the improver cannot read, and a final audit cohort is untouched until the harness is frozen. |
| What does the signature prove? | That this decision, with these report digests and this corpus snapshot, was signed by the pinned key. It does not prove the evaluator was honest. |
| What was built today? | The repository's README lists new work and reused dependencies separately. |
| Does it work outside the simulated company? | The AgentDojo and tau-bench retail runs on screen, with subset sizes, and a Claude Code `PreToolUse` export of admitted guards. |

## What the script depends on

`docs/demo/derive_ids.py` reads these. If one does not exist yet, it names it and stops.

| Needed | Source | Failing message names |
|---|---|---|
| An admitted decision with a `fresh` gate and a receipt that `ratchet verify` accepts | Atlas `decisions` | `ADMITTED_DECISION` and why each admitted one was refused |
| A replay-rejected guard candidate beside an admitted one, same parent | Atlas `decisions`, `harness_versions` | `CLEARED_CAND`, `REJECTED_CAND`, `REPLAY_TRACES`, `BAD_TRACE` |
| A rejected decision that failed an authority gate | Atlas `decisions` | `AUTHORITY_DECISION` |
| A checkpoint still `running` (kill a run mid-case) | Atlas `checkpoints` | `RESUME_RUN`, `RESUME_CMD` |
| Held-out case counts and result tables | `docs/results/agentdojo.md`, `docs/results/tau-retail.md` | `AGENTDOJO_CASES`, `TAU_CASES` |
| A level-2 result with a recorded claim | `docs/results/level2.json` or `private/improve/l2.json` (or `--level2`) | `LEVEL2_FILE`, the yields |
| `RATCHET_IMPROVER_URI` for `denied_read.py` | environment | checked by `demo.py preflight` |

`ratchet show decision ID --envelope-out FILE` writes a receipt for the standalone verifier in
[docs/VERIFY.md](../VERIFY.md).
