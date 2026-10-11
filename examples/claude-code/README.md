# Ratchet guard as a Claude Code hook

One guard, proven against session transcripts, then installed into Claude Code as a PreToolUse hook that
denies the call before it runs. The guard here blocks force pushes to, and deletion of, `main` or `master`.

| File | What it is |
| --- | --- |
| `no_force_push_main.py` | The guard: `check(call, view, history) -> str | None` |
| `transcripts/` | Six synthetic Claude Code sessions (one in a subagent file), written for this example |
| `expected_fires.txt` | The three tool calls the guard must block, by tool use id |
| `replay.json` | The replay over `transcripts/`: 15 calls, 3 fires, all expected, 0 unexpected, 0 errors |
| `harness.json`, `receipt.json`, `public_key.txt` | The harness holding the guard, the signed decision that admitted it, and the signer's public key |
| `hooks/ratchet_no_force_push_main.py` | The generated hook: one self-contained file, Python 3.12+, standard library only |
| `settings.json` | The `.claude/settings.json` fragment that installs it |
| `build.py` | Rebuilds every file above: `uv run python examples/claude-code/build.py` |

## Prove it on your own sessions first

```bash
uv run python -m ratchet.export.replay_transcripts --guard examples/claude-code/no_force_push_main.py \
    ~/.claude/projects --out private/replay.json
```

The command prints counts only (sessions, calls, fires, guard errors, per tool); the fire sites and the
guard's reasons go to the `--out` file. Each call is judged with the inputs the live hook builds: the call,
the view `{"cwd", "git_branch"}` and the events before it. Pass `--expect ids.txt` to require that the
guard fires on exactly those calls.

On one developer machine (13,712 sessions, 629,192 tool calls) the previous version of this guard fired
on 7 calls in 4 sessions, with no guard error, at 1,599 checks per second. 6 were force pushes onto
`main`. The seventh was text inside a quoted heredoc, which the current version reads as data.

## Install

```bash
mkdir -p .claude/hooks
cp examples/claude-code/hooks/ratchet_no_force_push_main.py .claude/hooks/
```

Then merge `settings.json` into `.claude/settings.json`. `onFailure: "block"` makes Claude Code deny the
call if the hook cannot even start. The hook itself denies on every error: a guard exception, a sandbox
layer that cannot be applied, unreadable input, or any edit to the file.

Try it without Claude Code:

```bash
echo '{"hook_event_name": "PreToolUse", "tool_name": "Bash", "cwd": ".",
       "tool_input": {"command": "git push --force origin main"}}' \
  | python3 .claude/hooks/ratchet_no_force_push_main.py; echo "exit $?"
```

It prints the deny decision and exits 2. Claude Code shows the agent: `blocked by guard
no-force-push-main: force push to main rewrites shared history; push a branch and open a pull request
instead`.

## What runs at each tool call

The hook runs the guard in the Ratchet sandbox child (`python -I -S`, empty environment, CPU and memory
limits, a new network namespace, seccomp, Landlock). The child is embedded byte for byte, so the hook needs
no Ratchet install. On Linux all four layers are required. Elsewhere, export with `--require ''`; the
header then records `sandbox_require: none`.

The first call starts a session server from the hook file itself, on a socket in
`$XDG_RUNTIME_DIR/ratchet-hook/`. It keeps one sandboxed child per session and reads each transcript
incrementally. Later calls cost one short request. The server exits after 15 idle minutes, or when its
socket file is deleted. Measured on a 141 MB transcript with a hook process per call:

| Guard | First call | Later calls (median) |
| --- | --- | --- |
| This guard (never reads its history, so the transcript is never read) | 0.50 s | 0.15 s |
| A guard that reads its history | 5.5 s | 0.21 s |

## Provenance

The hook's header names the guard, the harness version and digest, the SHA-256 of `receipt.json`, and the
key that signed it. The exporter refuses a guard unless that receipt verifies under a trusted key and
admits exactly this harness.

```bash
uv run python -m ratchet.export.claude_code --verify .claude/hooks/ratchet_no_force_push_main.py
sha256sum examples/claude-code/receipt.json     # equals decision_receipt_sha256 in the header
```

This example's decision records one check, the transcript replay above, admitted by a human. Ratchet's
held-out, no-regression and authority gates run on its simulated-company cases and do not apply to this
coding-agent guard.

To export your own admitted guard:

```bash
uv run python -m ratchet.export.claude_code --harness harness.json --guard-id <id> \
    --receipt receipt.json --trusted-key <public key hex> --out .claude/hooks/ratchet_<id>.py
```

## The same guards at an MCP gateway

```bash
uv run python -m ratchet.export.mcp_gateway --harness harness.json --receipt receipt.json \
    --trusted-key <public key hex> -- <mcp server command> [args...]
```

For a remote server, put the gateway in front of its Streamable HTTP endpoint and point the client at
`http://127.0.0.1:8808/`:

```bash
uv run python -m ratchet.export.mcp_gateway --harness harness.json --receipt receipt.json \
    --trusted-key <public key hex> --upstream https://example.org/mcp --listen 127.0.0.1:8808
```

Every `tools/call` is checked against every guard of the admitted harness before it reaches the server. A
blocked call is answered with an error result carrying the reason. Everything else, including streamed
responses, passes through unchanged.
