# Changelog

Every version maps to one suite digest, recorded in `src/ratchet_cases/cases/DIGESTS.json`. While the
version is 0.x, a change to a shipped case, a grader rule or a fault's meaning is a minor version.

## 0.2.0 (2026-10-12)

The 52 shipped cases are unchanged: the suite digest is still
`7d4c38e0d18c39c7d6cb955a28cb9bbc134cd1f78958c2c4ac5b2f04dd9b6679`. This release adds two suites that
run inside other benchmarks.

- `ratchet_cases.agentdojo`: benign twins for AgentDojo v1.2.2. Importing the module
  (`-ml ratchet_cases.agentdojo`) registers the benchmark version `ratchet-benign-twins-v0.1` with the
  workspace, travel, banking and slack suites. Each user task is the benign twin of one injection task:
  the slot an attack would fill holds an instruction the user asked for, graded by AgentDojo's own utility
  checks. Install with `pip install 'ratchet-cases[agentdojo]'`, which pins `agentdojo==0.1.35`.
- `ratchet-cases agentdojo-report <logdir>`: the false intervention rate per pipeline (twins failed over
  twins run) from AgentDojo run logs, read with the standard library only.
- `ratchet_cases.tau2`: the `retail_faults` task set for tau2-bench. For each of the seven retail write
  tools, one task in two variants, a timeout after the call commits and one before it does: 14 tasks,
  graded on the live databases against the gold ones rather than on a replay. Registered through the
  `tau2.plugins` entry point. tau2-bench is installed from its repository, not from PyPI.
- The sdist carries the AgentDojo and tau2-bench tests.

## 0.1.0 (2026-10-12)

First release: 52 cases in eight families, graded on final state, with deterministic tool faults, exact
graders, scripted solvers, the `ratchet-cases` command line and `SUITE_DIGEST`.
