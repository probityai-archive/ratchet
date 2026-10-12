# Guard replay over real coding-agent history

One Ratchet guard, `no-force-push-main`, tested by replay over the operator's own Claude Code session transcripts.
A guard is a small rule that checks each tool call before it runs. Replay means rerunning recorded past
conversations through the rule, in Ratchet's sandboxed guard runtime. No model call is made: each recorded tool
call is checked as if the guard had been installed before it ran.

| Measure | Value |
|---|---|
| Sessions | 13,712 |
| Tool calls checked | 629,192 |
| Guard fires | 7: 6 real force-pushes to a main branch, 1 false alarm |
| Checker crashes | 0 |
| Check time | 393 s (about 1,600 checks per second) |

Every fire gives the reason "force push to main rewrites shared history; push a branch and open a pull request
instead". Six fires were real force-pushes to a main branch. One was a false alarm: the rule matched the text
of a force-push command inside a document an agent was writing, and nothing was pushed. Unreadable transcript files (1) and malformed lines (118) are counted and skipped.
