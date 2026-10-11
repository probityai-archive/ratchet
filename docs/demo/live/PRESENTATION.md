# Present Ratchet in three minutes

Private speaker notes for Round 1 of the MongoDB "Harness Engineering & Model Wrangling" hackathon.
Speaker: Sankalp Gilda, team Probity. Three minutes of demo, then one to two minutes of judge questions.

Read the quoted passages aloud. Follow the cues in brackets without reading them. Wait for each real result
on screen before describing it. Never read the panels or IDs aloud; point at the one word that matters
(REJECTED, ADMITTED, VERIFIED, FAIL). If the network fails, switch to [the backup](BACKUP.md).

## Before going on stage

1. In the repository root, run `docs/demo/live/run_demo.sh --warm`. It runs all seven steps with the output
   hidden and prints each step's time. It must end with `warm: all 7 steps ok`.
2. Make the terminal font large: the panels are about 80 columns wide.
3. Start `docs/demo/live/run_demo.sh`. It shows a title and waits. Each Enter clears the screen and runs the
   next step under a one-line header.
4. Keep the backup clip (`docs/demo/out/20261011T232221Z/ratchet-169.mp4`, local file) open in a paused player behind the terminal.

### Measured step times

Measured 2026-10-11 against Atlas from this laptop (4 cores, load average about 6), in the `--warm` pass, which
is the same command set the stage run uses. A first cold run of the same commands is in the second column.

| Step | Command | Warm pass | Cold run |
|---|---|---|---|
| 1 | `ratchet show trace` (the double refund) | 2.9 s | 2.2 s |
| 2 | `ratchet replay` (two rules, 8 conversations) | 4.3 s | 4.2 s |
| 3 | `ratchet show decision` (admitted) | 2.6 s | 2.2 s |
| 4 | `ratchet verify` | 3.8 s | 7.6 s |
| 5 | `ratchet show decision` (new tool refused) | 2.8 s | 1.5 s |
| 6 | `ratchet show decision b972467a3b9a`, `RATCHET_DB=ratchet_v3_full` | 2.1 s | 2.5 s |
| 7 | `cat docs/results/hooks-replay.md` | 0.0 s | 0.0 s |

Talk through each wait with the line written under "while it runs". Step 4 is the slowest; its line is the longest.

## 0:00-0:25 - The person and the problem

[Title screen is showing. Do not press Enter yet.]

> I'm Sankalp Gilda, team Probity. Yesterday we built the gate that checks each action. Today we built how the
> gate's rules are made and proven.
>
> When an AI agent makes a mistake, someone writes a fix: a new safety rule, a new prompt. Nobody proves the
> fix works, or that it doesn't break something that already worked. Ratchet does that proof, and signs it.

## 0:25-0:45 - How it works: one support case

[Enter: step 1, the recorded conversation.] While it runs:

> Here's how it works, on one support case.

[Wait for the grader lines at the bottom. Point at "refund issued twice".]

> The agent refunds forty-five dollars. The payment provider's reply is lost, so the agent tries again. The
> customer is refunded twice.

## 0:45-1:10 - Case one: REJECTED

[Enter: step 2, the replay table.] While it runs:

> Ratchet writes two candidate safety rules and reruns eight recorded conversations through both. No AI calls.

[Wait for VERDICT. Point at column A, the red BLOCK cells, then REJECTED.]

> Rule A sounds sensible: stop large refunds. But it misses every double refund and blocks refunds that should
> go through. Rejected.

## 1:10-1:45 - Case two: ADMITTED

[Point at column B, then ADMITTED.]

> Rule B stops all four failures and blocks nothing that worked.

[Enter: step 3, the decision.] While it runs:

> Then it faces cases the AI that wrote it never saw.

[Wait for ADMITTED. Point at the PASS column.]

> It fixed two failing cases and kept the cases that must pass. Admitted, with a signed receipt.

[Enter: step 4, verify.] While it runs:

> Anyone can check that receipt. A separate verifier, sharing no code with the signer, recomputes the decision.

[Wait for VERIFIED.]

> Verified.

## 1:45-1:55 - A change that reaches for power

[Enter: step 5.] While it runs:

> One more.

[Wait for REJECTED. Point at "authority FAIL".]

> This change tried to give the agent a new tool. Refused, automatically.

## 1:55-2:30 - Case three: now a real model, REFUSED

[Enter: step 6.] While it runs:

> Now a real model: gpt-oss-120b, three thousand eight hundred runs.

[Wait for REJECTED. Point at "33.4% -> 42.1%".]

> An AI-written change lifts the success rate from thirty-three point four to forty-two point one percent.
> That's the share of nineteen unseen support cases solved, averaged over forty runs each. Most tools ship that.
>
> But one customer case fell from twenty-one to eight passes out of forty. Ratchet refused it and signed the
> refusal. And the agent with no change at all swung one case from thirty-two to twenty-three. Without a strict
> gate, luck looks like progress.

## 2:30-2:42 - Real data

[Enter: step 7. Point at the table.]

> Real data: one safety rule rerun over six hundred twenty-nine thousand actions from thirteen thousand real
> coding-agent sessions. Seven fires, each a forced overwrite of the main branch it would have stopped. Zero
> errors.

## 2:42-3:00 - MongoDB and the ask

[Stay on the table.]

> All of it lives in MongoDB Atlas: eighty-two signed decisions, Vector Search over past failures, lineage of
> every version, and the unseen test cases locked away from the improver by a database role.
>
> If your team ships agent changes, come talk to me. Ratchet is on PyPI today.

Spoken words: 370, about 140 a minute, with the command waits (about 18 s in total) carrying the rest of the three minutes.

## Answers to judge questions

| Question | Answer |
| --- | --- |
| Is the admission real or scripted? | The walkthrough uses a fixed, repeatable agent so the example runs the same way every time; the gates, replay, signature and verification are the real ones; the real-model result is next, from 3,800 runs. |
| Why not an LLM judge? | A judge's opinion is one more noisy sample. Ratchet reruns the actual conversations and checks outcomes against the case's own grader: did the customer get refunded twice. A rule is admitted on what it blocks and what it lets through, not on a score someone wrote. |
| What about noise in a real model? | We measure it. Every unseen case runs 40 times per version, in two panels of 20, and two unchanged copies of the agent run beside the change. One unchanged copy moved a case from 32 to 23 of 40, so the gate refused to call anything an improvement on that model. That is the gate doing its job. |
| What is MongoDB doing here? | Atlas is the system of record: every version, trace and signed decision (82 stored: 1 admitted, 81 rejected). Vector Search finds past failures like the current one, `$graphLookup` walks the lineage of every change, change streams drive the loop, and the unseen test cases sit behind a database role the improver cannot read. |
| What did you reuse, and what did you build today? | The repository's first commit is today; everything in `src/ratchet`, the tests, examples and docs was written at the event: the improvement loop, replay, the admission gates, the 40-run noise gate, the Atlas store and search, signing, and the CLI. Reused: our earlier public verifier `probity-verify`, pinned to one commit, which independently checks each receipt; plus pinned public benchmarks (tau-bench, AgentDojo). Public at github.com/probityai/ratchet and on PyPI as `probity-ratchet`. |
| How does this differ from evals? | An eval gives a score. Ratchet gives a decision: admit or refuse a specific change, with the gates that decided it, cases the change's author never saw, a check that nothing that worked now breaks, and a signed receipt anyone can recheck later. |
| How does it do on public benchmarks? | On tau-bench retail the baseline harness passed 70 of 115. Run repeatedly on the same 40 unseen tasks, the same harness scored 25, 31 and 26: the swing is exactly why a single run cannot prove a change. |
| Does it work on real agents, not only the walkthrough? | Yes: the 629,192-action replay is over 13,712 real coding-agent sessions, with 7 fires and 0 errors, and the 3,800-run refusal is on gpt-oss-120b. |
| What does it cost? | Replay makes no model calls: 8 conversations replay in under a second. The real-model gate is the expensive part: 3,800 runs took 29 minutes. Each decision records model calls before and after the change. |
| What is next? | Wire Ratchet in front of yesterday's Probity Gate, so the gate's own rules are admitted this way; and run the 40-run gate on more models, so a real change can clear it. |

## Sources

| Fact | Source |
| --- | --- |
| Walkthrough ids and counts | `docs/demo/ids.env` |
| 3,800 runs, 33.4% to 42.1%, 21 to 8, 32 to 23 | [v3-live.md](../../results/v3-live.md), decision `b972467a3b9a` in `ratchet_v3_full` |
| 629,192 actions, 13,712 sessions, 7 fires, 0 errors | [hooks-replay.md](../../results/hooks-replay.md) |
| tau-bench retail | [tau-retail.md](../../results/tau-retail.md) |
| Atlas features | [atlas-probes.md](../../results/atlas-probes.md) |
