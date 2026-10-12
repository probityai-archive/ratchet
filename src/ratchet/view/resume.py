"""What a resumed run did and did not do twice.

A killed run leaves a checkpoint in Atlas: the events so far, the calls it had executed and the calls the
model had asked for but not yet run. Resuming rebuilds the world by re-applying the executed calls to a
fresh copy (the simulated company), verifies its digest against the checkpoint, and continues from the next
model request. Nothing the run had already committed is sent to the company again.

This module reads the checkpoint and the finished trace and says so in plain words, call by call.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ratchet_cases.types import ToolCall, Trace
from ratchet_cases.world import AGENT, World
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

READ_MARKS = ("_get", "_list")


@dataclass(frozen=True)
class Effect:
    """One call the run executed before it was interrupted."""

    call_id: str
    name: str
    args: dict[str, Any]
    reads: bool
    outcome: str  # "ok", "reply lost (timeout)", "timed out, not committed" or the tool's error
    committed: bool | None = None  # what the system's audit log shows; None when it could not be checked

    @property
    def committed_but_unconfirmed(self) -> bool:
        return not self.reads and self.outcome.startswith("reply lost")


@dataclass(frozen=True)
class Reissue:
    """A call the resumed run made again with the same name and arguments as a committed one."""

    effect: Effect
    call_id: str
    executed: bool
    outcome: str


@dataclass(frozen=True)
class ResumeReport:
    run_id: str
    case_id: str
    harness_version: str
    interrupted_after_events: int
    carried: list[Effect]
    pending: list[dict[str, Any]]
    reissued: list[Reissue]
    passed: bool | None = None
    reasons: tuple[str, ...] = ()

    @property
    def carried_writes(self) -> list[Effect]:
        """State-changing calls that were committed, or whose commit could not be checked."""
        return [e for e in self.carried if not e.reads and e.committed is not False]

    @property
    def uncommitted(self) -> list[Effect]:
        """Writes the audit log shows changed nothing (a timeout before commit, or a refused call)."""
        return [e for e in self.carried if not e.reads and e.committed is False]


def _is_read(name: str) -> bool:
    return any(mark in name for mark in READ_MARKS)


def _result_of(events: Sequence[Mapping[str, Any]], call_id: str) -> str:
    for e in events:
        if e.get("kind") == "tool_result":
            res = (e.get("payload") or {}).get("result") or {}
            if res.get("call_id") == call_id:
                if res.get("ok"):
                    return "ok"
                err = str(res.get("error") or "error")
                return "reply lost (timeout)" if err == "timeout" else err
    return "no result recorded"


def audit_commits(
    saved: Mapping[str, Any], fresh_world: Callable[[], World], actor: str = AGENT
) -> dict[str, bool] | None:
    """For each executed call, whether the system's audit log shows the call changed anything.

    The executed calls are re-applied, in order, to a fresh copy of the company; a call committed when the
    audit log of some system gained an entry by the agent while it ran. A timeout is not evidence either
    way: only the audit log is. None when the rebuilt world does not match the checkpoint's world digest,
    since then the rebuilt ledger proves nothing about the run that was interrupted."""
    world = fresh_world()
    found: dict[str, bool] = {}
    for raw in saved.get("executed") or []:
        before = _agent_entries(world, actor)
        world.call(ToolCall(str(raw["name"]), dict(raw.get("args") or {}), str(raw["call_id"])))
        found[str(raw["call_id"])] = _agent_entries(world, actor) > before
    return found if world.digest() == saved.get("world_digest") else None


def _agent_entries(world: World, actor: str) -> int:
    audit = world.view().to_json()["audit"]
    return sum(1 for system, log in audit.items() if system != "world" for e in log if e["actor"] == actor)


def _outcome(raw: str, reads: bool, committed: bool | None) -> str:
    if raw != "reply lost (timeout)" or reads:
        return raw
    if committed is None:
        return "timeout, commit not checked"
    return "reply lost (timeout)" if committed else "timed out, not committed"


def carried_effects(saved: Mapping[str, Any], commits: Mapping[str, bool] | None = None) -> list[Effect]:
    """The calls the run had executed. `commits` (from `audit_commits`) decides what a timeout means."""
    events = saved.get("events") or []
    out = []
    for c in saved.get("executed") or []:
        call_id, name = str(c.get("call_id")), str(c.get("name"))
        reads = _is_read(name)
        committed = None if commits is None else commits.get(call_id)
        out.append(
            Effect(
                call_id,
                name,
                dict(c.get("args") or {}),
                reads,
                _outcome(_result_of(events, call_id), reads, committed),
                committed,
            )
        )
    return out


def _calls(trace: Trace, after_seq: int) -> list[tuple[str, str, dict[str, Any]]]:
    out = []
    for e in trace.events:
        if e.kind == "tool_call" and e.seq >= after_seq:
            c = e.payload.get("call", e.payload)
            out.append((str(c.get("call_id")), str(c.get("name")), dict(c.get("args") or {})))
    return out


def _closing(trace: Trace, call_id: str) -> tuple[bool, str]:
    for e in trace.events:
        if e.kind == "guard_block" and e.payload.get("call_id") == call_id:
            return False, f"blocked by {e.payload.get('source')}"
        if e.kind == "tool_result" and (e.payload.get("result") or {}).get("call_id") == call_id:
            res = e.payload["result"]
            return True, "ok" if res.get("ok") else str(res.get("error") or "error")
    return False, "no result recorded"


def reissued(carried: Sequence[Effect], trace: Trace, after_seq: int) -> list[Reissue]:
    """Calls after the interruption that repeat a committed write: same tool, same arguments."""
    writes = [e for e in carried if not e.reads]
    out: list[Reissue] = []
    for call_id, name, args in _calls(trace, after_seq):
        same = next((w for w in writes if w.name == name and canon(w.args) == canon(args)), None)
        if same is not None:
            ran, outcome = _closing(trace, call_id)
            out.append(Reissue(same, call_id, ran, outcome))
    return out


def canon(args: Mapping[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)


def build_report(
    saved: Mapping[str, Any], trace: Trace | None = None, commits: Mapping[str, bool] | None = None
) -> ResumeReport:
    carried = carried_effects(saved, commits)
    after = len(saved.get("events") or [])
    return ResumeReport(
        run_id=str(saved.get("run_id")),
        case_id=str(saved.get("case_id")),
        harness_version=str(saved.get("harness_version")),
        interrupted_after_events=after,
        carried=carried,
        pending=[dict(p) for p in saved.get("pending") or []],
        reissued=[] if trace is None else reissued(carried, trace, after),
        passed=None if trace is None else trace.outcome.passed,
        reasons=() if trace is None else tuple(trace.outcome.reasons),
    )


def _args_text(args: Mapping[str, Any]) -> str:
    text = ", ".join(f"{k}={v}" for k, v in sorted(args.items()))
    return text if len(text) <= 70 else text[:69] + "…"


def before(report: ResumeReport) -> RenderableType:
    """Printed before the run continues: what the checkpoint already holds."""
    head = Text.assemble(
        (" RESUMING ", "bold black on bright_cyan"),
        f"  {report.run_id}  case {report.case_id}  harness {report.harness_version}",
        (f"  after {report.interrupted_after_events} recorded events", "dim"),
    )
    t = Table(box=None, header_style="bold", pad_edge=False)
    for col in ("committed before the interruption", "arguments", "result"):
        t.add_column(col, no_wrap=col != "arguments")
    for e in report.carried:
        style = "yellow" if e.committed_but_unconfirmed else ("dim" if e.reads else "white")
        t.add_row(
            Text(e.name, style="bold" if not e.reads else "dim"),
            Text(_args_text(e.args), style=style),
            Text(e.outcome, style="bold yellow" if e.committed_but_unconfirmed else style),
        )
    parts: list[RenderableType] = [head, Text(), t]
    if report.pending:
        names = ", ".join(str(p.get("name")) for p in report.pending)
        parts += [Text(), Text(f"asked for but not yet run when it stopped: {names}", style="cyan")]
    return Group(*parts)


def after(report: ResumeReport) -> RenderableType:
    """Printed once the run has finished: which committed effects were not repeated."""
    writes = report.carried_writes
    again = {r.effect.call_id: r for r in report.reissued}
    lines: list[RenderableType] = []
    if not writes and not report.uncommitted:
        lines.append(Text("The run had committed no state-changing call before it stopped.", style="dim"))
    for e in writes:
        r = again.get(e.call_id)
        if r is None:
            lines.append(
                Text.assemble(
                    ("  NOT REPEATED  ", "bold black on green"),
                    f"  {e.name}({_args_text(e.args)})",
                    (f"  was {e.outcome}; the resume did not send it again", "green"),
                )
            )
        else:
            how = "and it ran again" if r.executed else f"but it did not run ({r.outcome})"
            lines.append(
                Text.assemble(
                    (
                        "  RE-ISSUED BY THE MODEL  ",
                        "bold white on red" if r.executed else "bold black on yellow",
                    ),
                    f"  {e.name}({_args_text(e.args)})",
                    (f"  the model asked for it again {how}: {r.outcome}", "red" if r.executed else "yellow"),
                )
            )
    retried = {r.effect.call_id: r for r in report.reissued}
    for e in report.uncommitted:
        r = retried.get(e.call_id)
        tail = "it has not been retried" if r is None else f"the model retried it: {r.outcome}"
        lines.append(
            Text.assemble(
                ("  NOT COMMITTED  ", "bold black on cyan"),
                f"  {e.name}({_args_text(e.args)})",
                (f"  {e.outcome}; the audit log shows no change, so a retry is not a repeat; {tail}", "cyan"),
            )
        )
    verdict = (
        Text("")
        if report.passed is None
        else Text.assemble(
            ("  case graded  ", "bold"),
            ("PASS", "bold green") if report.passed else ("FAIL", "bold red"),
            ("" if report.passed else "  " + "; ".join(report.reasons[:2]), "red"),
        )
    )
    return Panel(Group(*lines, Text(), verdict), title="committed effects", border_style="green")
