"""Replay a guard over Claude Code session transcripts before it is installed as a hook.

Every tool call in every transcript is shown to the guard with the inputs the live hook would build
(`ratchet.export.hookrt`): the call, the coding view {"cwd", "git_branch"} and the events before the call.
Checks run in batches through the sandboxed guard runtime, the same isolation as a live run. The report
counts calls, fires and guard errors per session and tool; with an expected set of tool use ids it also
says whether the guard fired on exactly those calls.

Transcripts are private. The command line prints counts only; fire details (session, call id, the guard's
reason) go to the `--out` file alone.

    python -m ratchet.export.replay_transcripts --guard guard.py ~/.claude/projects --out private/r.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, overload

from ratchet.agent.guards import GuardDecision, GuardItem, GuardRuntime
from ratchet.export.hookrt import Transcript, parse_transcript
from ratchet.types import Event, Guard, ToolCall, sha256_hex


class Prefix(Sequence[Event]):
    """The first `length` events of a session, without copying them. Consecutive calls share one event
    list, so the sandbox sends each child only the events it has not seen."""

    def __init__(self, events: list[Event], length: int) -> None:
        self._events = events
        self._length = length

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, index: int) -> Event: ...
    @overload
    def __getitem__(self, index: slice) -> list[Event]: ...
    def __getitem__(self, index: int | slice) -> Event | list[Event]:
        if isinstance(index, slice):
            return self._events[slice(*index.indices(self._length))]
        if not -self._length <= index < self._length:
            raise IndexError(index)
        return self._events[index % self._length]


@dataclass(frozen=True)
class Fire:
    session: str
    call_id: str
    tool: str
    line: int
    reason: str


@dataclass(frozen=True)
class Site:
    session: str
    call_id: str
    tool: str
    line: int


def session_items(guard: Guard, transcript: Transcript) -> list[GuardItem]:
    events = [Event(e["seq"], e["kind"], e["payload"], e["ts"]) for e in transcript.events]
    return [
        GuardItem(guard, ToolCall(**site.call), site.view, Prefix(events, site.history_len))
        for site in transcript.calls
    ]


@dataclass
class TranscriptReplayReport:
    guard_id: str
    guard_sha256: str
    sessions: int = 0
    calls: int = 0
    lines: int = 0
    malformed_lines: int = 0
    unreadable: int = 0
    fires: list[Fire] = field(default_factory=list)
    errors: list[Fire] = field(default_factory=list)
    calls_by_tool: Counter[str] = field(default_factory=Counter)
    fires_by_tool: Counter[str] = field(default_factory=Counter)
    parse_s: float = 0.0
    check_s: float = 0.0
    expected: list[str] | None = None

    @property
    def missed(self) -> list[str]:
        fired = {f.call_id for f in self.fires}
        return sorted(set(self.expected or ()) - fired)

    @property
    def unexpected(self) -> list[str]:
        return sorted({f.call_id for f in self.fires} - set(self.expected or ()))

    @property
    def passed(self) -> bool | None:
        """With an expected set: fired on exactly those calls, with no guard error. Else None."""
        if self.expected is None:
            return None
        return not self.missed and not self.unexpected and not self.errors

    @property
    def checks_per_s(self) -> float:
        return self.calls / self.check_s if self.check_s > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        """Counts only: safe to print or log for private transcripts."""
        out: dict[str, Any] = {
            "guard_id": self.guard_id,
            "guard_sha256": self.guard_sha256,
            "sessions": self.sessions,
            "calls": self.calls,
            "lines": self.lines,
            "malformed_lines": self.malformed_lines,
            "unreadable_files": self.unreadable,
            "fires": len(self.fires),
            "errors": len(self.errors),
            "calls_by_tool": dict(self.calls_by_tool.most_common()),
            "fires_by_tool": dict(self.fires_by_tool.most_common()),
            "parse_s": round(self.parse_s, 3),
            "check_s": round(self.check_s, 3),
            "checks_per_s": round(self.checks_per_s, 1),
        }
        if self.expected is not None:
            out.update(passed=self.passed, missed=len(self.missed), unexpected=len(self.unexpected))
        return out

    def to_payload(self) -> dict[str, Any]:
        """Summary plus every fire and error, with session labels and guard reasons."""
        out = self.summary()
        out["fire_sites"] = [vars(f) for f in self.fires]
        out["error_sites"] = [vars(f) for f in self.errors]
        if self.expected is not None:
            out.update(expected=self.expected, missed_ids=self.missed, unexpected_ids=self.unexpected)
        return out


def discover(paths: Iterable[Path]) -> Iterator[tuple[str, Path]]:
    """(label, file) for each transcript: files as given, directories searched for *.jsonl. The label is
    the path relative to the directory it was found under."""
    for path in paths:
        if path.is_dir():
            for found in sorted(path.rglob("*.jsonl")):
                yield found.relative_to(path).as_posix(), found
        else:
            yield path.name, path


def _record(report: TranscriptReplayReport, sites: list[Site], decisions: list[GuardDecision]) -> None:
    for site, decision in zip(sites, decisions, strict=True):
        if decision.error is not None:
            report.errors.append(Fire(site.session, site.call_id, site.tool, site.line, decision.error[:300]))
        elif decision.blocked:
            report.fires.append(Fire(site.session, site.call_id, site.tool, site.line, str(decision.reason)))
            report.fires_by_tool[site.tool] += 1


def replay_transcripts(
    guard: Guard,
    sessions: Iterable[tuple[str, Path]],
    runtime: GuardRuntime,
    *,
    chunk: int = 20000,
    expect: Collection[str] | None = None,
) -> TranscriptReplayReport:
    """Replay `guard` over every tool call in `sessions`, in batches of about `chunk` checks."""
    report = TranscriptReplayReport(
        guard.id,
        sha256_hex(guard.source.encode("utf-8")),
        expected=None if expect is None else sorted(expect),
    )
    items: list[GuardItem] = []
    sites: list[Site] = []

    def flush() -> None:
        started = time.perf_counter()
        decisions = runtime.batch(items)
        report.check_s += time.perf_counter() - started
        _record(report, sites, decisions)
        items.clear()
        sites.clear()

    for label, path in sessions:
        started = time.perf_counter()
        try:
            with path.open(encoding="utf-8") as handle:
                transcript = parse_transcript(handle)
        except (OSError, UnicodeDecodeError):
            report.unreadable += 1
            continue
        report.sessions += 1
        report.lines += transcript.lines
        report.malformed_lines += transcript.malformed
        items.extend(session_items(guard, transcript))
        report.parse_s += time.perf_counter() - started
        for site in transcript.calls:
            tool = site.call["name"]
            sites.append(Site(label, site.call["call_id"], tool, site.line))
            report.calls_by_tool[tool] += 1
            report.calls += 1
        if len(items) >= chunk:
            flush()
    if items:
        flush()
    return report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ratchet.export.replay_transcripts", description=__doc__.split("\n")[0]
    )
    p.add_argument(
        "paths", nargs="+", type=Path, help="transcript files, or directories searched for *.jsonl"
    )
    p.add_argument(
        "--guard", type=Path, required=True, help="guard source file defining check(call, view, history)"
    )
    p.add_argument("--guard-id", help="defaults to the guard file's stem")
    p.add_argument("--expect", type=Path, help="file of tool use ids the guard must fire on, one per line")
    p.add_argument("--out", type=Path, help="write the full report, with fire sites, to this JSON file")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--chunk", type=int, default=20000)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 when the replay meets the expectation (or, without one, ran with no guard error)."""
    args = _parser().parse_args(argv)
    guard = Guard(args.guard_id or args.guard.stem, "", args.guard.read_text(encoding="utf-8"))
    expect = None
    if args.expect is not None:
        expect = [x.strip() for x in args.expect.read_text(encoding="utf-8").splitlines() if x.strip()]
    with GuardRuntime(workers=args.workers) as runtime:
        report = replay_transcripts(guard, discover(args.paths), runtime, chunk=args.chunk, expect=expect)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_payload(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.summary(), indent=2))
    ok = report.passed if report.passed is not None else not report.errors
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
