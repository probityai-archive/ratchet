# Guard replay over real coding-agent history

One Ratchet guard, `no-force-push-main`, replayed in Ratchet's sandboxed guard runtime over the operator's own
Claude Code session transcripts. No model call is made: each recorded tool call is checked as if the guard had
been installed before it ran.

| Measure | Value |
|---|---|
| Sessions | 13,712 |
| Tool calls checked | 629,192 |
| Guard fires | 7 |
| Guard errors | 0 |
| Check time | 393 s (about 1,600 checks per second) |

Every fire gives the reason "force push to main rewrites shared history; push a branch and open a pull request
instead". Unreadable transcript files (1) and malformed lines (118) are counted and skipped.
