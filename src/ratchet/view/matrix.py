"""The replay matrix: every recorded trace against every candidate guard, with no model call.

Rows are the traces the parent harness recorded, grouped by what they prove:
  trap failures    traces that failed on a trap; a good guard fires on its own targets here
  past successes   traces that passed; any guard that fires here would have broken something that worked
  benign twins     traces of cases that look like a trap but are fine; a fire here is a false alarm
  other failures   failures that are not traps; shown, never held against a guard
Columns are candidate guards. A cell is the guard's verdict on the whole recorded run: it fires at some
tool call (and the matrix says which) or it does not. The verdict under each column is the replay rule of
the admission check, plus the decision itself when one is recorded.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from ratchet_cases.types import Trace
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ratchet.improve.replay import GuardInput, GuardRunner, WalkCache
from ratchet.types import Guard
from ratchet.view.data import DecisionInfo

RowKind = Literal["trap_failure", "past_success", "benign_twin", "other_failure"]
GROUPS: tuple[RowKind, ...] = ("trap_failure", "past_success", "benign_twin", "other_failure")
GROUP_TITLE: dict[RowKind, str] = {
    "trap_failure": "TRAP FAILURES",
    "past_success": "PAST SUCCESSES",
    "benign_twin": "BENIGN TWINS",
    "other_failure": "OTHER FAILURES",
}
MAX_COLUMNS = 8


@dataclass(frozen=True)
class CaseInfo:
    """What the case suite says about a case, so a trace can be filed under the right group."""

    kind: str
    family: str
    split: str = "train"


@dataclass(frozen=True)
class MatrixRow:
    run_id: str
    case_id: str
    family: str
    kind: RowKind


@dataclass(frozen=True)
class MatrixColumn:
    guard: Guard
    candidate: str
    letter: str
    decision: DecisionInfo | None = None


@dataclass(frozen=True)
class Cell:
    """`late` marks a guard that fires on a target failure only after the step that caused it: too late."""

    fired: bool
    seq: int | None = None
    error: str | None = None
    late: bool = False


@dataclass(frozen=True)
class ColumnVerdict:
    """`replay` is the replay gate in the words the contract uses: "37/37 checked, CLEARED", "k/N checked,
    INCOMPLETE" or "37/37 checked, REJECTED"."""

    label: str
    ok: bool
    reasons: list[str]
    replay: str = ""


@dataclass
class Matrix:
    parent: str
    rows: list[MatrixRow]
    columns: list[MatrixColumn]
    cells: dict[tuple[int, int], Cell]
    targets: list[set[int]]
    horizons: list[dict[int, int]]
    elapsed_s: float
    checks: int
    unfaithful: set[int] = field(default_factory=set)
    recorded_at: str = ""
    model_calls: int = field(default=0, init=False)

    def rows_of(self, kind: RowKind) -> list[int]:
        return [i for i, r in enumerate(self.rows) if r.kind == kind]

    def fired(self, row: int, col: int) -> bool:
        return self.cells[(row, col)].fired

    def hits(self, col: int) -> tuple[int, int]:
        """(target trap failures the guard fires on, target trap failures)."""
        tgt = self.targets[col]
        return sum(self.fired(r, col) for r in tgt), len(tgt)

    def count(self, kind: RowKind, col: int) -> int:
        return sum(self.fired(r, col) for r in self.rows_of(kind))

    def errors(self, col: int) -> int:
        return sum(1 for r in range(len(self.rows)) if self.cells[(r, col)].error)

    def replay_reasons(self, col: int) -> list[str]:
        out: list[str] = []
        hit, total = self.hits(col)
        late = sum(1 for r in self.targets[col] if self.cells[(r, col)].late)
        if total == 0:
            out.append("no trap failure recorded to prove it on")
        elif hit < total:
            out.append(f"misses {total - hit} of {total} target failures")
            if late:
                out.append(f"{late} of them only after the failing step")
        blocked, fired = self.count("past_success", col), self.count("benign_twin", col)
        if blocked:
            out.append(f"blocks {blocked} past {'success' if blocked == 1 else 'successes'}")
        if fired:
            out.append(f"fires on {fired} benign {'twin' if fired == 1 else 'twins'}")
        if self.errors(col):
            out.append(f"{self.errors(col)} guard error(s)")
        if self.unfaithful:
            out.append(f"{len(self.unfaithful)} trace(s) whose rebuilt world differs from the record")
        return out

    def coverage(self, col: int) -> tuple[int, int]:
        """(traces fully checked, traces). A trace where the guard errored, or whose rebuilt world differs
        from the record, was not checked."""
        total = len(self.rows)
        unchecked = self.unfaithful | {r for r in range(total) if self.cells[(r, col)].error}
        return total - len(unchecked), total

    def replay_verdict(self, col: int) -> str:
        checked, total = self.coverage(col)
        if checked < total:
            return "INCOMPLETE"
        return "REJECTED" if self.replay_reasons(col) else "CLEARED"

    def verdict(self, col: int) -> ColumnVerdict:
        reasons = self.replay_reasons(col)
        checked, total = self.coverage(col)
        replay = f"{checked}/{total} checked, {self.replay_verdict(col)}"
        d = self.columns[col].decision
        if d is None:
            label = self.replay_verdict(col)
            return ColumnVerdict(label, label == "CLEARED", reasons, replay)
        extra = d.reasons(skip=("replay",)) if not d.admitted else []
        return ColumnVerdict("ADMITTED" if d.admitted else "REJECTED", d.admitted, reasons + extra, replay)


def classify(trace: Trace, info: CaseInfo | None) -> RowKind:
    if info is not None and info.kind == "benign_twin":
        return "benign_twin"
    if trace.outcome.passed:
        return "past_success"
    return "trap_failure" if info is not None and info.kind == "trap" else "other_failure"


def pick_rows(
    traces: Sequence[Trace], info: Callable[[str], CaseInfo | None]
) -> tuple[list[Trace], list[MatrixRow]]:
    """The newest trace per case (audit-cohort cases left out), ordered by group then case id."""
    newest: dict[str, Trace] = {}
    for t in traces:
        ci = info(t.case_id)
        if ci is None or ci.split != "audit":  # the audit cohort is never shown beside a decision
            newest[t.case_id] = t
    rows: list[tuple[int, str, Trace, MatrixRow]] = []
    for t in newest.values():
        ci = info(t.case_id)
        kind = classify(t, ci)
        row = MatrixRow(t.run_id, t.case_id, ci.family if ci else "?", kind)
        rows.append((GROUPS.index(kind), t.case_id, t, row))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows], [r[3] for r in rows]


def _targets(guard: Guard, rows: Sequence[MatrixRow]) -> tuple[set[int], dict[int, int]]:
    """The trap failures a guard is meant to stop, and the step by which it had to stop each one.

    A guard names its targets as `run@seq` failure ids; one that names none (or none in these rows) is judged
    against every trap failure, with no step bound."""
    traps = [i for i, r in enumerate(rows) if r.kind == "trap_failure"]
    named: dict[str, int] = {}
    for fid in guard.origin_failure_ids:
        run, _, seq = fid.rpartition("@")
        if run and seq.isdigit():
            named[run] = max(named.get(run, 0), int(seq))
    own = {i: named[rows[i].run_id] for i in traps if rows[i].run_id in named}
    return (set(own), own) if own else (set(traps), {})


def compute_matrix(
    parent: str,
    columns: Sequence[MatrixColumn],
    traces: Sequence[Trace],
    info: Callable[[str], CaseInfo | None],
    walks: WalkCache,
    runner: GuardRunner,
) -> Matrix:
    """Run every column's guard over every trace. No model is involved: guards read recorded state only."""
    started = time.perf_counter()
    picked, rows = pick_rows(traces, info)
    cells: dict[tuple[int, int], Cell] = {}
    target_sets: list[set[int]] = []
    horizons: list[dict[int, int]] = []
    checks = 0
    for ci, col in enumerate(columns):
        inputs: list[GuardInput] = []
        where: list[tuple[int, int]] = []
        for ri, (trace, row) in enumerate(zip(picked, rows, strict=True)):
            failed = row.kind in ("trap_failure", "other_failure")
            for s in walks.walk(trace).steps:
                if failed or s.executed:
                    inputs.append(GuardInput(s.call, s.view, s.history))
                    where.append((ri, s.seq))
        verdicts = runner.check_batch(col.guard, inputs) if inputs else []
        if len(verdicts) != len(inputs):
            raise RuntimeError(f"guard runtime returned {len(verdicts)} verdicts for {len(inputs)} inputs")
        checks += len(inputs)
        found: dict[int, Cell] = {}
        for (ri, seq), v in zip(where, verdicts, strict=True):
            prev = found.get(ri)
            if v.error is not None:
                found[ri] = prev if prev and prev.fired else Cell(False, seq, v.error[:200])
            elif v.block is not None and not (prev and prev.fired):
                found[ri] = Cell(True, seq)
        targets, horizon = _targets(col.guard, rows)
        horizons.append(horizon)
        target_sets.append(targets)
        for ri in range(len(rows)):
            cell = found.get(ri, Cell(False))
            if cell.fired and ri in horizon and cell.seq is not None and cell.seq > horizon[ri]:
                cell = Cell(False, cell.seq, late=True)
            cells[(ri, ci)] = cell
    stamp = max((e.ts for t in picked for e in t.events), default=None)
    m = Matrix(
        parent, rows, list(columns), cells, target_sets, horizons, 0.0, checks,
        {i for i, t in enumerate(picked) if not walks.walk(t).faithful},
        "" if stamp is None else datetime.fromtimestamp(stamp, UTC).isoformat(timespec="seconds"),
    )  # fmt: skip
    m.elapsed_s = time.perf_counter() - started
    return m


def letters(n: int) -> list[str]:
    return [chr(ord("A") + i) if i < 26 else str(i + 1) for i in range(n)]


# -- rendering --------------------------------------------------------------------------------------------

LineKind = Literal["group", "row", "more"]


@dataclass(frozen=True)
class Line:
    kind: LineKind
    group: RowKind
    row: int = -1
    count: int = 0


def _bad_row(m: Matrix, r: int) -> bool:
    kind = m.rows[r].kind
    if kind in ("past_success", "benign_twin"):
        return any(m.fired(r, c) or m.cells[(r, c)].error for c in range(len(m.columns)))
    return False


def _missed(m: Matrix, r: int) -> bool:
    return any(r in m.targets[c] and not m.fired(r, c) for c in range(len(m.columns)))


def layout(m: Matrix, budget: int, full: bool = False) -> list[Line]:
    """Which lines to show. Everything when it fits; otherwise the rows that tell the story first (every
    row a guard wrongly fired on, then trap failures a guard missed, then the rest) and a count of the
    rows left out under each group."""
    groups = [(g, m.rows_of(g)) for g in GROUPS if m.rows_of(g)]
    total = len(groups) + sum(len(rs) for _, rs in groups)
    chosen: set[int] = {r for _, rs in groups for r in rs}
    if not full and total > budget:
        room = max(0, budget - 2 * len(groups))
        bad = [r for g, rs in groups if g in ("past_success", "benign_twin") for r in rs if _bad_row(m, r)]
        traps = sorted(m.rows_of("trap_failure"), key=lambda r: (not _missed(m, r), r))
        others = [r for g, rs in groups if g == "other_failure" for r in rs]
        chosen = set()
        for r in [*bad, *traps, *others]:
            if len(chosen) < room:
                chosen.add(r)
    out: list[Line] = []
    for g, rs in groups:
        out.append(Line("group", g))
        out += [Line("row", g, r) for r in rs if r in chosen]
        hidden = sum(1 for r in rs if r not in chosen)
        if hidden:
            out.append(Line("more", g, count=hidden))
    return out


def _cell_text(m: Matrix, r: int, c: int) -> Text:
    cell, kind = m.cells[(r, c)], m.rows[r].kind
    at = f" @{cell.seq}" if cell.seq is not None else ""
    if cell.error:
        return Text("! error", style="bold yellow")
    if kind == "trap_failure":
        if cell.fired:
            return Text(f"● BLOCK{at}", style="bold green")
        if cell.late:
            return Text(f"◐ late{at}", style="bold yellow")
        return Text("○ missed", style="bold red") if r in m.targets[c] else Text("-", style="dim")
    if kind in ("past_success", "benign_twin"):
        if cell.fired:
            return Text(f" ✗ BLOCK{at} ", style="bold white on red")
        return Text("· pass", style="green")
    return Text(f"● BLOCK{at}", style="cyan") if cell.fired else Text("·", style="dim")


def _group_text(m: Matrix, g: RowKind, c: int) -> Text:
    rows = m.rows_of(g)
    n = sum(m.fired(r, c) for r in rows)
    if g == "trap_failure":
        hit, total = m.hits(c)
        return Text(f"{hit}/{total} targets", style="bold green" if hit == total and total else "bold red")
    if g == "other_failure":
        return Text(f"{n} blocked", style="cyan" if n else "dim")
    return Text(f"{n} blocked", style="bold red") if n else Text("0 blocked", style="bold green")


def _verdict_cell(v: ColumnVerdict) -> Text:
    if v.label == "ADMITTED":
        return Text(" ADMITTED ", style="bold black on green")
    if v.label == "CLEARED":
        return Text(" CLEARED ", style="bold black on cyan")
    if v.label == "INCOMPLETE":
        return Text(" INCOMPLETE ", style="bold black on yellow")
    return Text(" REJECTED ", style="bold white on red")


def _label(m: Matrix, r: int) -> Text:
    row = m.rows[r]
    return Text(f" {row.case_id}", style="white")


def build_table(m: Matrix, lines: Sequence[Line], width: int, show_verdict: bool = True) -> Table:
    n = len(m.columns)
    cw = max(10, min(14, (width - 34) // max(n, 1)))
    t = Table(
        box=None, show_edge=False, pad_edge=False, expand=False, header_style="bold white on blue",
        row_styles=[], collapse_padding=True,
    )  # fmt: skip
    t.add_column(" recorded trace", min_width=28, no_wrap=True)
    for col in m.columns:
        t.add_column(col.letter, justify="center", min_width=cw, max_width=cw, no_wrap=True)
    for ln in lines:
        if ln.kind == "group":
            t.add_section()
            t.add_row(
                Text(f"{GROUP_TITLE[ln.group]} ({len(m.rows_of(ln.group))})", style="bold bright_white"),
                *[_group_text(m, ln.group, c) for c in range(n)],
            )
        elif ln.kind == "row":
            t.add_row(_label(m, ln.row), *[_cell_text(m, ln.row, c) for c in range(n)])
        else:
            t.add_row(Text(f"   ... {ln.count} more rows (--full shows all)", style="dim"), *[""] * n)
    if show_verdict:
        t.add_section()
        verdicts = [_verdict_cell(m.verdict(c)) for c in range(n)]
        t.add_row(Text("VERDICT", style="bold bright_white"), *verdicts)
    return t


def legend(m: Matrix) -> Table:
    t = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1))
    t.add_column(no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_column(overflow="fold")
    for c, col in enumerate(m.columns):
        v = m.verdict(c)
        mark = Text("✓", style="bold green") if v.ok else Text("✗", style="bold red")
        head = Text.assemble((f"{col.letter} ", "bold"), (col.guard.id, "cyan"))
        why = "; ".join(v.reasons) if v.reasons else "fires on its targets and on nothing that worked"
        if col.decision and col.decision.receipt:
            why += f"  [{col.decision.receipt}]"
        body = Text.assemble(
            (f"{v.label}: {why}", "green" if v.ok else ("yellow" if v.label == "INCOMPLETE" else "red")),
            (f"   replay {v.replay}", "dim"),
        )
        t.add_row(mark, head, body)
    return t


def footer(m: Matrix) -> Text:
    n = len(m.rows)
    rate = m.checks / m.elapsed_s if m.elapsed_s > 0 else float(m.checks)
    return Text.assemble(
        (f"{n} traces, {m.elapsed_s:.2f} s, {m.model_calls} model calls", "bold bright_white"),
        (f"   {len(m.columns)} guards, {m.checks:,} guard checks ({rate:,.0f}/s) on recorded state", "dim"),
    )


def header(m: Matrix) -> Text:
    groups = "  ".join(f"{len(m.rows_of(g))} {GROUP_TITLE[g].lower()}" for g in GROUPS if m.rows_of(g))
    stamp = f" RECORDED {m.recorded_at} " if m.recorded_at else " RECORDED "
    return Text.assemble(
        (" RATCHET ", "bold black on bright_cyan"),
        (" replay matrix ", "bold"),
        (f" parent {m.parent}  ", "dim"),
        (groups, "dim"),
        ("  ", ""),
        (stamp, "black on grey70"),
    )


class MatrixView:
    """A renderable matrix. `reveal` shows only the first N lines, which is how the recording animates."""

    def __init__(self, m: Matrix, *, height: int = 35, width: int = 120, full: bool = False) -> None:
        self.m = m
        overhead = 8 + len(m.columns)
        self.lines = layout(m, max(8, height - overhead), full)
        self.width = width
        self.reveal: int | None = None

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        shown = self.lines if self.reveal is None else self.lines[: self.reveal]
        done = self.reveal is None or self.reveal >= len(self.lines)
        body: list[RenderableType] = [header(self.m), Text(), build_table(self.m, shown, self.width, done)]
        if done:
            body += [Text(), Panel(legend(self.m), border_style="dim", padding=(0, 1)), footer(self.m)]
        yield Group(*body)


def show_matrix(console: Console, m: Matrix, *, delay: float = 0.04, full: bool = False) -> None:
    """Print the matrix; on a terminal, reveal it line by line so a screen recording shows the replay run."""
    view = MatrixView(m, height=console.height, width=console.width, full=full)
    if not console.is_terminal or delay <= 0 or not view.lines:
        console.print(view)
        return
    step = min(delay, 2.5 / len(view.lines))
    with Live(view, console=console, refresh_per_second=30, transient=False) as live:
        for i in range(1, len(view.lines) + 1):
            view.reveal = i
            live.refresh()
            time.sleep(step)
        view.reveal = None
        live.refresh()
