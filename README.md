# Ratchet

Ratchet turns observed agent failures into tested, signed guard changes.

When an agent fails, Ratchet writes a guard: a small rule that checks each tool call before it runs. It adopts the guard only after a replay (rerunning recorded past conversations through it) and fresh tests on held-out cases (test cases the improver never sees). Each decision is a receipt, a signed record anyone can verify offline.

It is for teams running tool-using agents that touch money, accounts or tickets, and need a fix to a failing agent to be proven before it ships.

## Install

```sh
git clone --branch probity-ratchet-v0.1.2 https://github.com/probityai/ratchet && cd ratchet
uv sync
```

Python 3.12 or later. The guard sandbox needs Linux.

```sh
pip install probity-ratchet-cases==0.2.0 && ratchet-cases selftest   # the case suite alone: probity-ratchet-cases/README.md
```

## Try it in 30 seconds

No database, no model key. This guard stops force pushes to `main`. It is a single standard-library file, exported from an admitted guard, and it is a Claude Code hook (a script run before each tool call). It denies the call:

```sh
echo '{"hook_event_name": "PreToolUse", "tool_name": "Bash", "cwd": ".", "tool_input": {"command": "git push --force origin main"}}' \
  | python3 examples/claude-code/hooks/ratchet_no_force_push_main.py; echo "exit $?"
```

```text
blocked by guard no-force-push-main: force push to main rewrites shared history; push a branch and open a pull request instead
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "blocked by guard no-force-push-main: force push to main rewrites shared history; push a branch and open a pull request instead"}}exit 2
```

Before export, the guard was replayed over six recorded Claude Code sessions: it fired on the three calls it was written for and on none of the other twelve (`examples/claude-code/replay.json`). The hook's header names the receipt (the signed decision) that admitted it; `uv run python -m ratchet.export.claude_code --verify examples/claude-code/hooks/ratchet_no_force_push_main.py` checks that receipt.

A signed refusal you can check offline is in `examples/refusal-v3-cell/`: `uv run python examples/refusal-v3-cell/verify.py` regrades 80 executions and prints `decision verified, candidate refused, reason: identity controls breached`. "Decision verified" is not "candidate accepted": it says the signed record is genuine, and here the record is a refusal. It is single-cell evidence from one run by the operator.

## What it does

A support agent works in a simulated company: a billing provider, tickets, approvals and mail. Its eight case families are seven kinds of trap, each with benign twins (matching cases where the action is allowed, so a rule that blocks them fails), plus ordinary requests that must succeed. Every case is graded on the final state of every system, never on the agent's words.

When the agent fails, the improver proposes a guard or a procedure (written steps for the agent). A change is adopted only if four checks pass: replay; fresh paired tests on held-out cases against the parent harness (the code around the model: prompt, tools and rules); no growth of the agent's authority; and a signed receipt, which verifies offline against a pinned public key. The gates are in [docs/ADMISSION.md](docs/ADMISSION.md). Every version, trace, evaluation and decision, rejections included, lives in MongoDB Atlas.

## Status

Version 0.1.2, built on 2026-10-11 at the MongoDB harness hackathon. The suite, graders and rule are frozen before each run. Measured results, with denominators, are in [docs/RESULTS.md](docs/RESULTS.md).

## Docs

| Page | What is in it |
|---|---|
| [Running](docs/RUNNING.md) | Local Atlas, environment and every command |
| [Admission rule](docs/ADMISSION.md) | The gates in plain words |
| [k-session rule](docs/SESSIONS.md) | The replicated rule, runner and offline check |
| [Architecture](docs/ARCHITECTURE.md) | Packages, guard sandbox, storage |
| [Verifying a decision](docs/VERIFY.md) | Check a receipt offline |
| [Results](docs/RESULTS.md) | Every measured result and its records |
| [Claude Code hook](examples/claude-code/README.md) | Prove a guard on your own sessions, then install it |
| [Built today and reused](docs/REUSED.md) | What was written at the event and what is pinned |
| [Signed refusal example](examples/refusal-v3-cell/README.md) | Check a refused candidate offline |
| [Build contract](docs/CONTRACT.md) | Types, lane interfaces and the full admission rule |
