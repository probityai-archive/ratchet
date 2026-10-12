# Terminal-Bench 2.0: Ratchet results

Run records live under `private/bench/terminal/` (never committed); every number below is read from them and
printed with its denominator. Commands: `python -m ratchet.bench.terminal oracle|trial|evolve|rates`.

## Protocol

- Task suite: the 89-task Terminal-Bench 2.0 repository at commit `2fd12b88aafdd04a52c298e3940bcb189f9766d6`,
  run with `terminal-bench==0.2.18` pinned in the bench extra. Each task runs in its own prebuilt image
  (`docker_image` in its `task.toml`) with the task's CPU and memory limits, on shared-core Hetzner boxes.
- The agent runs in Ratchet's worker loop under a Ratchet harness, with two tools: `run_command(cmd,
  timeout_s)` and `write_file(path, content)`. Its time limit is the task's own agent timeout, in wall time.
- Grading is the task's own `tests/test.sh`, copied in after the agent finishes and run unchanged in the same
  container; a run passes when the reward it writes is 1. A run that crashed counts as a failed run.
- The two phases are fixed by task metadata: phase 1 is every task marked `hard` with a 900 s agent timeout
  (12 tasks), phase 2 every task marked `hard` with an 1800 s agent timeout (10 tasks). Inside each phase a
  fixed seed (20261011) chose the held-out tasks; the assignment and its digest
  (`97264170d0b16b2bf9767d369ce4ef6b116b4d6ceafb13c87dc0d871784047df`) are committed in
  `src/ratchet/bench/terminal/split.json`, before any run.

| Phase | Train | Held-out |
|---|---|---|
| 1 | cancel-async-tasks, configure-git-webserver, fix-code-vulnerability, gpt2-codegolf, make-doom-for-mips, password-recovery, sparql-university, torch-pipeline-parallelism | model-extraction-relu-logits, polyglot-rust-c, torch-tensor-parallelism, write-compressor |
| 2 | extract-moves-from-video, feal-differential-cryptanalysis, feal-linear-cryptanalysis, llm-inference-batching-scheduler, make-mips-interpreter, protein-assembly | dna-assembly, mcmc-sampling-stan, path-tracing, path-tracing-reverse |

- Phase 1 evolves from the seed harness on phase-1 train tasks, with phase-1 held-out tasks as the admission
  rule's held-out cases. Phase 2 starts from the harness phase 1 admitted, trains on phase-1 and phase-2 train
  tasks (so the phase-1 train tasks stay protected), uses phase-2 held-out tasks as held-out cases, and keeps
  the phase-1 held-out tasks as an audit cohort that no decision reads; after phase 2 they are re-run to test
  that phase-1 gains did not erode.
- Models: agent and improver `gpt-5.5` through an OpenAI-compatible endpoint, server-default sampling.

## Reference-solution check (no model)

Each task's reference solution (`solution/solve.sh`) is run through the world's own `run_command` tool in a
fresh container and graded by the task's tests, to prove the container, the tools and the grader before any
agent run.

| Run | Tasks | Reward 1 |
|---|---|---|
| `oracle/20261011T204924` (both phases) | 22 | 18 |
| `oracle/20261011T210920` (re-run of the 4 failures, with outputs kept) | 4 | 0 |
| `oracle/20261012T002605` (re-check of llm-inference-batching-scheduler after the grader fix) | 1 | 1 |

Why the four failed:

- `make-doom-for-mips` (phase 1): the solution installs Debian packages without refreshing the package index;
  the image's index is stale and the archive answers 404, so `clang` is missing (solution exit 2). An agent can
  refresh the index, so the task stays solvable.
- `protein-assembly` (phase 2): the solution's DNA optimiser raised `NoSolutionError` (solution exit 1) in both
  runs, so `/app/gblock.txt` was never written.
- `mcmc-sampling-stan` (phase 2): the solution's CRAN build of `RcppParallel` and `StanHeaders` failed, so
  `rstan` was never installed (solution exit 1) in both runs.
- `llm-inference-batching-scheduler` (phase 2): the grader's fault, fixed. Its `test.sh` has its `#!` line
  below the canary comment; the grader started it from `sh`, which stopped at `source: not found`. The grader
  now starts tests from bash, as the suite's runner does. In the re-check `oracle/20261012T002605` the
  reference solution exited 0 and all 6 of the task's tests passed (reward 1), so 19 of the 22 reference
  solutions are now shown to pass.

## Smoke run: seed harness on 3 real tasks

Harness `d7ff14d17d68` (seed: a generic terminal-agent prompt, both tools, no procedures, no guards),
`runs/d7ff14d17d68/smoke-trial-1`, all runs FRESH.

| Task | Result | Model calls | Wall |
|---|---|---|---|
| cancel-async-tasks | pass | 4 | 137 s |
| sparql-university | pass | 7 | 205 s |
| password-recovery | pass | 15 | 332 s |

3/3 runs passed over 3 tasks; 26 model calls, 105,728 input and 8,093 output tokens, 23 tool calls.

## Phase 1 evolution

Run `evolve/phase-1/run-1`: 2 generations at most, 6 workers, at most 3 candidate evaluations, 4 failure
clusters and 3 candidates per cluster per generation. Evaluator version `306de2f2ca7e`.

### Run 1, generation 1 (split `97264170...`, 4 held-out tasks)

| Harness | Change | Train passed | Held-out passed | Endpoint retries | Decision |
|---|---|---|---|---|---|
| `d7ff14d17d68` (parent run 1) | seed | 4/8 | 3/4 | 0 | parent |
| `d7ff14d17d68` (parent run 2) | seed | not run | 1/4 | 0 | parent |
| `805ee4ba2227` | one guard | 5/8 | 1/4 | 1 | rejected: no stable recovery; broke protected held-out tasks model-extraction-relu-logits and torch-tensor-parallelism |
| `54f0cd991cd9` | one guard | 5/8 | 1/4 | 1 | rejected: same two protected held-out tasks broken, no stable recovery |
| `637e96b5f8d1` | one procedure | 4/8 | 3/4 | 6 | rejected: nothing broken, but no stable recovery (polyglot-rust-c failed in all three runs) |

The improver proposed 9 candidates from 3 failure clusters; 6 were replay-tested, 4 cleared replay, 3 were
evaluated, 0 admitted, 3 rejected with signed decisions. The admission rule counts stable recoveries on
held-out cases only and needs two (a case failing in both parent runs and passing in both candidate runs).
Only one of the four held-out tasks, polyglot-rust-c, failed in both parent runs, so no candidate could
reach two. Run 2 below uses a wider held-out split for that reason.

Run 1 was killed at 16:00 PT (19:00 ET) on 2026-10-11, in generation 2, when the session running it was
killed. Generation 1 above is complete; the run's records hold no generation-2 evaluation.

Every decision in the runs on this page was signed with a fresh per-run key, not the pinned decision key
`b61d7e39c111` (per-run keyids in the records: run 1 `34e9eed03d0a`, run 1b `ec3488dd60c7`, run 2
`c59009a9ab99`, run 2b `b45171c900b1`), so these decisions are to be countersigned with the pinned key.

### Run 1b: run 1 rerun from scratch

Run `evolve/phase-1/run-1b` has the same configuration as run 1 (its `config.json` differs only in the start
time), with agent and improver `gpt-5.5`. `box_run.sh` ran it on the Hetzner box quint-testrunner-3, with
model calls going through the local model proxy. It finished both generations in 7,068 s and kept the seed
harness `d7ff14d17d68`: 0 of the 5 evaluated candidates were admitted.

In the tables below, a candidate's second run is made only when its first run clears every gate, so every
candidate was decided on one run. Re-runs counts runs repeated under the evaluator's predeclared limit of
two (`infra_retries` in the record). A run still lost after two re-runs is invalid evidence and counts as
not passed.

#### Run 1b, generation 1 (split `97264170...`, 4 held-out tasks)

| Harness | Change | Train passed | Held-out passed | Re-runs | Decision |
|---|---|---|---|---|---|
| `d7ff14d17d68` (parent run 1) | seed | 5/8 | 2/4 | 0 | parent |
| `d7ff14d17d68` (parent run 2) | seed | not run | 2/4 | 1 | parent |
| `131a1ced3042` | one guard | 5/8 | 2/4 | 1 | rejected: no recovery; regressed train task torch-pipeline-parallelism |
| `ff2c8af6b6a9` | one procedure | 5/8 | 2/4 | 0 | rejected: one recovery (torch-tensor-parallelism), two needed; broke protected held-out task model-extraction-relu-logits |
| `a31f462c573c` | one guard | 4/8 | 1/4 | 2 | rejected: one recovery (torch-tensor-parallelism), two needed; broke protected held-out tasks model-extraction-relu-logits and write-compressor; regressed train task torch-pipeline-parallelism |

The improver proposed 9 candidates from 3 failure clusters; the 6 guards were replay-tested and all 6
cleared replay, 3 candidates were evaluated, 0 admitted, 3 rejected with signed decisions. Two held-out
tasks, polyglot-rust-c and torch-tensor-parallelism, failed in both parent runs, so two stable recoveries
were reachable; no candidate passed both.

#### Run 1b, generation 2

| Harness | Change | Train passed | Held-out passed | Re-runs | Decision |
|---|---|---|---|---|---|
| `d7ff14d17d68` (parent run 1) | seed | 4/8 | 1/4 | 0 | parent |
| `d7ff14d17d68` (parent run 2) | seed | not run | 1/4 | 4 (2 runs lost) | parent |
| `42f6f52e00ec` | one guard (current instructions) | 4/8 | 0/4 | 7 (1 run lost) | rejected: no recovery; broke protected held-out tasks model-extraction-relu-logits and write-compressor; train task cancel-async-tasks counted as regressed (its run was lost) |
| `11aa4041b0af` | one guard (challenger instructions) | 4/8 | 1/4 | 7 (2 runs lost) | rejected: invalid evidence on 3 of 4 held-out tasks (polyglot-rust-c and write-compressor lost in parent run 2, model-extraction-relu-logits lost in the candidate run); no recovery; protected model-extraction-relu-logits counted as broken |

In generation 2 the improver wrote candidates under two instruction versions, the current `942dc26006bf`
and a challenger `c3c9b46a02e0`, 9 each over the same 3 failure clusters. Of the 18 proposed, 9 guards
were replay-tested (5 and 4), 6 cleared replay (5 and 1), 2 were evaluated (one per version), 0 admitted,
2 rejected with signed decisions. The challenger was not kept. The 5 runs lost in this generation each
ended in `OSError: [Errno 28] No space left on device` on the box. Of the run's 22 re-runs (4 in
generation 1, 18 in generation 2), 2 follow a recorded model-endpoint failure (HTTP 408); the other 20 have
no endpoint-failure trace.

### Run 2 (split `d7018648...`, 6 phase-1 held-out tasks)

Split `src/ratchet/bench/terminal/split-heldout-6-5.json` (6 of 12 phase-1 and 5 of 10 phase-2 tasks held
out, same seed), committed before its first run. Run `evolve/phase-1/run-2-heldout-6`, same settings as run 1
on this split, evaluator version `dd34d9132969`.

Run 2 started at 22:59:37 UTC on 2026-10-11, less than a minute before the session that launched it was
killed at 16:00 PT (23:00 UTC), and its launcher stopped then. The job on its box went on writing records.
Generation 1 finished at 23:31 UTC: parent run 1 passed 3/6 train and 3/6 held-out, parent run 2 3/6 held-out; the
improver proposed 3 candidates from 1 failure cluster, the 2 guards were replay-tested and neither cleared,
and the 1 procedure was evaluated (3/6 train, 4/6 held-out) and rejected with one recovery where two are
needed. The generation-2 parent runs were recorded at 23:55 UTC. The records stop there, with no
generation-2 candidate evaluation.

### Run 2b: run 2 rerun from scratch

Run `evolve/phase-1/run-2-heldout-6b` has the same configuration as run 2 (its `config.json` differs only in
the start time), with agent and improver `gpt-5.5`. `box_run.sh` ran it on a Hetzner burst box
(box-burst-1791764228-29835), with model calls going through the local model proxy. It finished both
generations and kept the seed harness `d7ff14d17d68`: 0 of the 5 evaluated candidates were admitted.

#### Run 2b, generation 1 (split `d7018648...`, 6 held-out tasks)

| Harness | Change | Train passed | Held-out passed | Re-runs | Decision |
|---|---|---|---|---|---|
| `d7ff14d17d68` (parent run 1) | seed | 4/6 | 3/6 | 0 | parent |
| `d7ff14d17d68` (parent run 2) | seed | not run | 3/6 | 0 | parent |
| `ea25f4252325` | one guard | 4/6 | 4/6 | 0 | rejected: one recovery (polyglot-rust-c), two needed; broke protected held-out task torch-tensor-parallelism |
| `03a68486b8d1` | one procedure | 5/6 | 2/6 | 1 | rejected: no recovery; broke protected held-out tasks model-extraction-relu-logits and torch-tensor-parallelism |
| `50c9aaeef488` | one guard | 4/6 | 4/6 | 0 | rejected: nothing broken, but no recovery (make-doom-for-mips and polyglot-rust-c failed again) |

The improver proposed 6 candidates from 2 failure clusters; the 4 guards were replay-tested, 3 cleared
replay and 1 was INCOMPLETE, 3 candidates were evaluated, 0 admitted, 3 rejected with signed decisions. Two
held-out tasks, make-doom-for-mips and polyglot-rust-c, failed in both parent runs.

#### Run 2b, generation 2

| Harness | Change | Train passed | Held-out passed | Re-runs | Decision |
|---|---|---|---|---|---|
| `d7ff14d17d68` (parent run 1) | seed | 2/6 | 3/6 | 0 | parent |
| `d7ff14d17d68` (parent run 2) | seed | not run | 1/6 | 0 | parent |
| `74c28046f2ff` | one guard (current instructions) | 3/6 | 3/6 | 0 | rejected: nothing broken, but no recovery |
| `1dff6f1439ff` | one guard (challenger instructions) | 4/6 | 4/6 | 0 | rejected: nothing broken; one recovery (polyglot-rust-c), two needed |

The improver wrote under the current instructions `942dc26006bf` and a challenger `64bd1f039e28`, 9
candidates each over the same 3 failure clusters. Of the 18 proposed, 12 guards were replay-tested (6 and
6), 8 cleared replay (4 and 4), 2 were evaluated (one per version), 0 admitted, 2 rejected with signed
decisions. The challenger was not kept. Three held-out tasks, make-doom-for-mips, polyglot-rust-c and
torch-tensor-parallelism, failed in both parent runs. The run's single re-run (generation 1) followed a
recorded model-endpoint failure (HTTP 408).

## Seed harness on phase 2 (transfer baseline)

`runs/d7ff14d17d68/phase2-trial-1`: 5/10 runs passed over 10 tasks (extract-moves-from-video,
feal-differential-cryptanalysis, feal-linear-cryptanalysis, llm-inference-batching-scheduler,
mcmc-sampling-stan passed). Four runs were lost to the model endpoint (HTTP 429 after 12 attempts) and re-run
under the evaluator's predeclared limit of two re-runs; the counted run is the completed one.

Endpoint note: the agent's time limit is checked before each model call; a model call already in flight
(including its rate-limit retries) is not cut off, so a run can exceed the task's agent timeout in wall time.
