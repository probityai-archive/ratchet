# ratchet-cases

Fifty-two agent tasks in a simulated company, each graded on the final state of every system it touches, never on what the agent says.

It is for anyone testing a tool-using agent, or a guard or defense around one, who needs to know both that the agent avoids the traps and that it still finishes legitimate work.

## Install

```sh
pip install ratchet-cases==0.2.0
```

Python 3.10 or later. No runtime dependencies.

## Try it in 30 seconds

Three scripted solvers, no model: one with every careful habit, one that trusts everything it reads, one that refuses everything.

```sh
ratchet-cases selftest
```

```text
reference trap 26/26, benign_twin 21/21, must_succeed 5/5
naive     trap 0/26, benign_twin 21/21, must_succeed 5/5
refuser   trap 0/26, benign_twin 0/21, must_succeed 0/5
selftest passed on 52 case(s); suite digest 7d4c38e0d18c39c7d6cb955a28cb9bbc134cd1f78958c2c4ac5b2f04dd9b6679
```

The refuser row is the point: an agent that blocks every action never falls for a trap and fails every benign twin, the case that looks like a trap but asks for ordinary work.

## Run your agent

```sh
ratchet-cases run --adapter openai --model <model> --base-url <url> --out runs/mine   # RATCHET_API_KEY for the key
ratchet-cases run --adapter python:mypkg.agent:act --out runs/mine                    # act(messages, tools) -> assistant message
ratchet-cases grade runs/mine/traces.jsonl
ratchet-cases report runs/ --metric false_intervention_rate
```

`grade` rebuilds each final state by replaying the recorded tool calls and grades it again. `false_intervention_rate` is the share of benign twins the agent failed.

## What is in it

Eight families: a refund that commits while its reply is lost, an approval revoked between check and execution, a policy that changes mid-session, prompt injection in tool-returned content, a concurrent worker that already handled the ticket, a reconciliation whose totals match while its accounts do not, an order modification with a late addition, and ordinary work that must succeed. The seven trap families each carry traps and three benign twins; 5 must-succeed cases are ordinary in-policy work. Faults are declared and deterministic: a timeout after or before commit, or another actor's change, at a fixed point in the agent's call sequence.

Splits: 18 train, 19 held-out, 15 audit. **All three splits are public.** The held-out and audit cases were published with the source on 2026-10-11, so they are no longer sealed and a result on them is not a held-out result. A sealed cohort, run by a steward, is separate planned work.

## tau2-bench retail faults

`ratchet_cases.tau2` puts a deterministic timeout, after or before the call commits, on the first call to each of tau2-bench's seven retail write tools: 14 tasks, each graded on the live databases against the gold ones rather than on a replay of the trajectory. Install tau2-bench from its repository first; it is not on PyPI under its own name.

```python
from ratchet_cases import tau2

print([tau2.run(task, tau2.Policy.GIVE_UP).passed for task in tau2.faulted_tasks()])
```

An agent that gives up after a timeout passes the seven after-commit tasks and fails the seven before-commit ones. A tau2-bench that carries `Task.faults` and loads `tau2.plugins` entry points runs the same set as `retail_faults`.

## Version and digest

Version 0.2.0. `ratchet_cases.SUITE_DIGEST` is computed from the shipped cases at import, and `DIGESTS.json` records the digest of every release, so `ratchet-cases==X.Y.Z` pins the exact cases. Any change to a case, a grader rule or a fault's meaning is a new minor version while the version is 0.x.

## Links

| Page | What is in it |
|---|---|
| [Source](https://github.com/probityai/ratchet/tree/main/ratchet-cases) | The package in the Ratchet repository |
| [Ratchet](https://github.com/probityai/ratchet) | The harness that turns failures on these cases into proven guards |
| [Releases](https://github.com/probityai/ratchet/releases) | Wheels, provenance attestations and digests |

License: Apache-2.0.
