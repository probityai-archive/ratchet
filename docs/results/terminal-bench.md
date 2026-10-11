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

Why the four fail:

- `make-doom-for-mips` (phase 1): the solution installs Debian packages without refreshing the package index;
  the image's index is stale and the archive answers 404, so `clang` is missing (solution exit 2). An agent can
  refresh the index, so the task stays solvable.
- `protein-assembly` (phase 2): the solution's DNA optimiser raised `NoSolutionError` (solution exit 1) in both
  runs, so `/app/gblock.txt` was never written.
- `mcmc-sampling-stan` (phase 2): the solution's CRAN build of `RcppParallel` and `StanHeaders` failed, so
  `rstan` was never installed (solution exit 1) in both runs.
- `llm-inference-batching-scheduler` (phase 2): the grader's fault, fixed. Its `test.sh` has its `#!` line
  below the canary comment; the grader started it from `sh`, which stopped at `source: not found`. The grader
  now starts tests from bash, as the suite's runner does. The re-run of this task is pending a free box.

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

Generation 1 parent (`d7ff14d17d68`, the seed): train 4/8, held-out 3/4 in run 1 and 1/4 in run 2.
