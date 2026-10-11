# Ratchet

Ratchet turns observed agent failures into executable guards, and checks each guard against recorded history and fresh evaluations before the agent's harness adopts it.

It is for teams running tool-using agents that touch money, accounts or tickets, and need a fix to a failing agent to be proven before it ships.

## Install

```sh
git clone --branch v0.1.0 https://github.com/probityai/ratchet && cd ratchet
uv sync
```

Python 3.12 or later. The guard sandbox needs Linux.

## Try it in 30 seconds

No database, no model key. This guard stops force pushes to `main`. It is a single standard-library file, exported from an admitted guard, and it denies the call before it runs:

```sh
echo '{"hook_event_name": "PreToolUse", "tool_name": "Bash", "cwd": ".", "tool_input": {"command": "git push --force origin main"}}' \
  | python3 examples/claude-code/hooks/ratchet_no_force_push_main.py; echo "exit $?"
```

```text
blocked by guard no-force-push-main: force push to main rewrites shared history; push a branch and open a pull request instead
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "blocked by guard no-force-push-main: force push to main rewrites shared history; push a branch and open a pull request instead"}}exit 2
```

Before it was exported, the guard was replayed over six recorded Claude Code sessions: it fired on the three calls it was written for and on none of the other twelve (`examples/claude-code/replay.json`). The hook's header names the signed decision that admitted it; `uv run python -m ratchet.export.claude_code --verify examples/claude-code/hooks/ratchet_no_force_push_main.py` checks that decision.

## What it does

A support agent works in a simulated company: a billing provider, tickets, approvals and mail. Its eight case families are seven kinds of trap, each with benign twins that look like the trap and are fine, plus ordinary requests that must succeed. Every case is graded on the final state of every system, never on the agent's words.

When the agent fails, the improver proposes a guard (code that blocks a tool call) or a procedure. A change is adopted only if all of these hold:

1. **Replay.** Each guard fires on every failure it targets and on no recorded success, using the exact inputs the guard saw live. No model call.
2. **Fresh evaluation.** The parent and the candidate each run twice on held-out cases. The candidate must recover cases the parent fails both times and keep every case the parent ever passed.
3. **No growth of authority.** Existing guards and prompt bytes stay, and the tool list can only shrink.
4. **A signed record.** The decision, its evidence digests and its checks are signed, and verify offline against a pinned public key.

Every version, trace, evaluation and decision, rejections included, is stored in MongoDB Atlas. The exact rule is in [docs/ADMISSION.md](docs/ADMISSION.md).

## Status

Version 0.1.0, built on 2026-10-11 at the MongoDB harness hackathon in San Francisco. The suite, graders and rule are frozen and digested before each run. Measured results, with denominators, are in [docs/RESULTS.md](docs/RESULTS.md).

## Docs

| Page | What is in it |
|---|---|
| [Running](docs/RUNNING.md) | Local Atlas, environment, `ratchet run`, `evolve`, `replay`, `show`, `verify`, `live`, `report` |
| [Admission rule](docs/ADMISSION.md) | The gates in plain words |
| [Architecture](docs/ARCHITECTURE.md) | Packages, the guard sandbox, what is stored where |
| [Verifying a decision](docs/VERIFY.md) | Check a receipt with the pinned verifier and key |
| [Results](docs/RESULTS.md) | Every measured result and where its raw records are |
| [Claude Code hook](examples/claude-code/README.md) | Prove a guard on your own sessions, then install it |
| [Build contract](docs/CONTRACT.md) | Types, lane interfaces and the full admission rule |

## Built today and reused

The repository's first commit is from 2026-10-11, and everything under `src/ratchet`, `tests`, `examples` and `docs` was written at the event. The rest is a pinned dependency or a service.

| Reused | Pin |
|---|---|
| Runtime | `pymongo>=4.10`, `httpx>=0.27`, `cryptography>=44`, `rich>=15`, `raindrop-ai==0.0.75` |
| Earlier work of the same authors | `probity-verify` at commit `b920366`, `agent-evidence-vectors==0.17.7` |
| Benchmark extra | `agentdojo==0.1.35`, `litellm==1.102.1`, `terminal-bench==0.2.18`, `tau-bench` at `59a200c`, `tau2` at `4ce7c03`, `tool-sandbox` at `c8571d7` |
| Instruction search extra | `gepa==0.1.4` |
| Services | MongoDB Atlas (`mongodb-atlas-local:8.3.9-20260928T091948Z` for development), Voyage embeddings, Raindrop |

The integration that writes and checks Ratchet's own admission records is new. The verifier it calls is not.

License: Apache-2.0.
