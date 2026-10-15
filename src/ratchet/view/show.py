"""One trace as a timeline, and one decision as a table of its gates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ratchet_cases.types import Event, Trace
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ratchet.improve.stochastic import RUNS as V3_RUNS
from ratchet.view.admission import (
    HELDOUT_CHECKS,
    RUNS,
    FreshView,
    ReplayRow,
    fresh_view,
    procedures_not_applicable,
    replay_rows,
)
from ratchet.view.data import DecisionInfo, eval_mode

KIND_STYLE = {
    "model": "blue",
    "tool_call": "bold white",
    "tool_result": "white",
    "guard_block": "bold red",
    "procedure": "magenta",
    "checkpoint": "dim",
    "final": "bold green",
}


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _args(args: Mapping[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(", ", "="))[1:-1].replace('"', "")


def event_summary(e: Event) -> str:
    p = e.payload
    if e.kind == "model":
        calls = [str(c.get("name") if isinstance(c, dict) else c) for c in p.get("tool_calls") or []]
        said = clip(str(p.get("content") or ""), 70)
        return (said + "  " if said else "") + ("-> " + ", ".join(calls) if calls else "")
    if e.kind == "tool_call":
        c = p.get("call", p)
        return f"{c.get('name')}({clip(_args(c.get('args') or {}), 80)})"
    if e.kind == "tool_result":
        r = p.get("result", p)
        if r.get("ok"):
            return "ok " + clip(json.dumps(r.get("data"), sort_keys=True, ensure_ascii=False), 80)
        return f"ERROR {r.get('error')}"
    if e.kind == "guard_block":
        return f"[{p.get('source')}] {clip(str(p.get('message') or ''), 90)}"
    if e.kind == "procedure":
        return f"{p.get('name')} ({p.get('how')})"
    if e.kind == "checkpoint":
        return f"after {p.get('executed')} world call(s)"
    return f"{p.get('reason')}: {clip(str(p.get('content') or ''), 80)}"


def trace_label(t: Trace) -> str:
    """A trace read back from the store is a RECORDED display, stamped with when it ran; it says whether the
    run itself was FRESH or CACHED (a byte-identical response played back)."""
    cached = any(e.kind == "model" and e.payload.get("cached") is True for e in t.events)
    ran = (
        datetime.fromtimestamp(t.events[-1].ts, UTC).isoformat(timespec="seconds") if t.events else "unknown"
    )
    return f"RECORDED {ran} (run was {'CACHED' if cached else 'FRESH'})"


def trace_view(t: Trace) -> RenderableType:
    o = t.outcome
    badge = Text(
        " PASS " if o.passed else " FAIL ", style="bold white on green" if o.passed else "bold white on red"
    )
    head = Text.assemble(
        badge, f"  {t.case_id}", ("  harness ", "dim"), (t.harness_version, "cyan"),
        ("  run ", "dim"), t.run_id, ("  model ", "dim"), t.model,
    )  # fmt: skip
    cost = o.cost
    label = trace_label(t)
    sub = Text(
        f"{label}   {cost.model_calls} model calls, {cost.tool_calls} tool calls, "
        f"{cost.input_tokens}+{cost.output_tokens} tokens, {cost.wall_s:.1f} s",
        style="dim",
    )
    table = Table(box=None, show_header=True, header_style="bold", pad_edge=False)
    table.add_column("seq", justify="right", style="dim")
    table.add_column("event", no_wrap=True)
    table.add_column("what", overflow="fold")
    for e in t.events:
        table.add_row(
            str(e.seq),
            Text(e.kind, style=KIND_STYLE.get(e.kind, "")),
            Text(event_summary(e), style=KIND_STYLE.get(e.kind, "")),
        )
    body: list[RenderableType] = [head, sub, Text(), table]
    if o.reasons:
        body += [Text(), Text("grader:", style="bold"), *[Text(f"  - {r}", style="red") for r in o.reasons]]
    return Group(*body)


def evidence_line(check: Mapping[str, Any]) -> str:
    name, ev = check.get("name"), check.get("evidence") or {}
    if name == "replay":
        rows = replay_rows(ev)
        guards = [r for r in rows if not r.guard_id.startswith("(")]
        return "; ".join(f"{r.guard_id}: {r.summary}" for r in rows) or f"{len(guards)} new guards"
    if name in HELDOUT_CHECKS:
        v = fresh_view(ev)
        if v is None:
            return ""
        return (
            f"{len(v.stable)} stable recoveries (need {v.needed}), {len(v.protected)} protected, "
            f"{len(v.broken)} broken, parent discordance D={v.discordance}"
            + (
                "; C2 not run (refused after C1)"
                if v.refused_after_c1
                else "; C2 pending"
                if v.confirm_pending
                else ""
            )
        )
    if name == "no_regression":
        return (
            f"{ev.get('must_succeed_passed_in_either_parent_run')} must-succeed cases passed before; "
            f"regressed {len(ev.get('regressed') or [])}"
        )
    if name == "benign_twins":
        line = f"{ev.get('interventions')} intervention(s)"
        hit = ev.get("intervened")
        if isinstance(hit, list) and hit:
            line += f" on {len(hit)} benign twin{'' if len(hit) == 1 else 's'}"
        runs = ev.get("benign_twin_runs")
        return f"{line} over {runs} twin runs" if runs is not None else line
    if name == "stochastic_v3":
        return stochastic_line(ev.get("comparison") or {})
    if name == "cohort_qualified":
        return cohort_line(ev.get("cohort") or {})
    if name == "overfit_lint":
        return f"{len(ev.get('hits') or [])} copied literal(s); screened {ev.get('screened')}"
    if name == "structure":
        added = f"+{ev.get('added_guards', 0)} guards, +{ev.get('added_procedures', 0)} procedures"
        bad = ev.get("violations") or []
        return f"{added}; breaks: {', '.join(bad)}" if bad else f"{added}; additive only, tools a subset"
    if name == "cost":
        p, c = ev.get("parent") or {}, ev.get("candidate") or {}
        # one count per agent turn, whether a model or the demo's scripted agent took it
        return f"agent turns {p.get('model_calls')} -> {c.get('model_calls')} (recorded, not a gate)"
    return ""


def _declines(cases: list[Mapping[str, Any]]) -> str:
    bad = [c for c in cases if not c.get("decline_ok", True)]
    return "; ".join(f"{c['case_id']} fell {c['parent']} -> {c['arm']}" for c in bad)


def stochastic_line(comparison: Mapping[str, Any]) -> str:
    """Pooled pass rate of parent and candidate, stable recoveries, and every case past the decline limit."""
    cases = list(comparison.get("cases") or [])
    if not cases:
        return ""
    parent = sum(c["parent"] for c in cases) / (V3_RUNS * len(cases))
    arm = sum(c["arm"] for c in cases) / (V3_RUNS * len(cases))
    stable = sum(1 for c in cases if c.get("stable_recovery"))
    line = f"pass rate {parent:.1%} -> {arm:.1%} ({len(cases)} cases x {V3_RUNS} runs)"
    line += f"; {stable} stable recoveries"
    declines = _declines(cases)
    return f"{line}; over the decline limit: {declines}" if declines else line


def cohort_line(cohort: Mapping[str, Any]) -> str:
    """Which identity control (an unchanged arm) failed, and on which case: the endpoint's noise floor."""
    failed = [c for c in cohort.get("controls") or [] if not c.get("qualifies")]
    parts = [f"{c['arm']}: {_declines(list(c.get('cases') or []))}" for c in failed]
    status = str(cohort.get("status", ""))
    return f"{status}, unchanged control moved: {'; '.join(parts)}" if parts else status


_OUTCOME_STYLE = {
    "pass": "green",
    "fail": "red",
    "invalid": "bold yellow",
    "pending": "dim",
    "not run": "dim",
}
_ROLE_TEXT = {
    "broken": ("PROTECTED CASE BROKE", "bold white on red"),
    "invalid": ("NO VALID EVIDENCE", "bold black on yellow"),
    "stable": ("STABLE RECOVERY", "bold green"),
    "protected": ("protected", "cyan"),
    "unrecovered": ("still failing", "dim"),
}


def fresh_table(v: FreshView) -> RenderableType:
    """Case by case: the parent twice (P1, P2), the candidate twice (C1, C2), and what each case became."""
    t = Table(box=None, header_style="bold", pad_edge=False, title="fresh runs, paired", title_justify="left")
    t.add_column("held-out case", no_wrap=True)
    for r in RUNS:
        t.add_column(r, justify="center")
    t.add_column("role")
    for row in v.rows:
        text, style = _ROLE_TEXT[row.role]
        t.add_row(
            row.case_id,
            *[Text(o, style=_OUTCOME_STYLE.get(o, "")) for o in row.outcomes],
            Text(text, style=style),
        )
    need = f"need {v.needed}"
    sums = Text.assemble(
        (
            f"stable recoveries {len(v.stable)} ({need}) ",
            "bold green" if len(v.stable) >= v.needed else "bold red",
        ),
        (f"  protected {len(v.protected)} ", "cyan"),
        (f"  broken {len(v.broken)} ", "bold red" if v.broken else "green"),
        (f"  parent discordance D={v.discordance}", "dim"),
        ("  (reported, not a threshold)", "dim"),
    )
    notes: list[RenderableType] = []
    if v.refused_after_c1:
        notes.append(
            Text("C2 (the confirming run) was not made: the candidate was refused after C1", style="dim")
        )
    elif v.confirm_pending:
        notes.append(Text("C2 (the confirming run) has not run yet", style="yellow"))
    if v.invalid:
        notes.append(Text(f"{len(v.invalid)} case(s) lack valid FRESH evidence: not counted", style="yellow"))
    if not v.same_manifest:
        notes.append(Text("the four runs did not share one case manifest", style="bold red"))
    return Group(t, sums, *notes)


def replay_table(rows: Sequence[ReplayRow], procedures: Sequence[str]) -> RenderableType:
    t = Table(
        box=None, header_style="bold", pad_edge=False, title="replay, per added guard", title_justify="left"
    )
    for col in ("guard", "coverage", "verdict", "targets hit", "false fires", "benign twin fires", "errors"):
        t.add_column(col, no_wrap=True)
    style = {"CLEARED": "bold green", "REJECTED": "bold red", "INCOMPLETE": "bold yellow"}
    for r in rows:
        hit = "-" if r.targets_total is None else f"{r.targets_hit}/{r.targets_total}"
        t.add_row(
            r.guard_id,
            r.coverage,
            Text(r.verdict, style=style.get(r.verdict, "bold red")),
            hit,
            Text(str(r.false_fires), style="red" if r.false_fires else "green"),
            Text(str(r.benign_fires), style="red" if r.benign_fires else "green"),
            Text(str(r.errors), style="yellow" if r.errors else "green"),
        )
    extra = [
        Text(f"procedure {p}: replay NOT APPLICABLE (natural-language trigger)", style="dim")
        for p in procedures
    ]
    return Group(t, *extra)


def decision_view(d: DecisionInfo) -> RenderableType:
    badge = Text(
        " ADMITTED " if d.admitted else " REJECTED ",
        style="bold white on green" if d.admitted else "bold white on red",
    )
    head = Text.assemble(badge, f"  {d.parent} -> {d.candidate}", (f"  {d.what}", "white"))
    table = Table(box=None, header_style="bold", pad_edge=False)
    table.add_column("gate")
    table.add_column("result")
    table.add_column("evidence", overflow="fold")
    detail: list[RenderableType] = []
    for c in d.checks:
        ok = bool(c.get("passed"))
        table.add_row(
            str(c["name"]),
            Text("PASS" if ok else "FAIL", style="bold green" if ok else "bold red"),
            evidence_line(c),
        )
        ev = c.get("evidence") or {}
        if c["name"] in HELDOUT_CHECKS and (v := fresh_view(ev)) is not None:
            detail += [Text(), fresh_table(v)]
        if c["name"] == "replay" and (rows := replay_rows(ev)):
            detail += [Text(), replay_table(rows, procedures_not_applicable(ev))]
    why = [Text(f"  - {r}", style="red") for r in d.reasons()]
    foot = Text.assemble(
        ("receipt ", "dim"), (d.receipt or "none", "magenta"),
        ("   key ", "dim"), ((d.keyid or "none")[:16], "magenta"),
        ("   decided ", "dim"), d.decided_at,
    )  # fmt: skip
    parts: list[RenderableType] = [head, Text(), table, *detail]
    if why:
        parts += [Text(), Text("why rejected:", style="bold"), *why]
    parts += [Text(), foot, Text(f"check the signature: ratchet verify {d.id}", style="dim")]
    return Panel(Group(*parts), border_style="green" if d.admitted else "red")


def run_view(payload: Mapping[str, Any]) -> RenderableType:
    """An evaluation report (the payload `EvalReport.to_payload` stores) as a per-case table."""
    tr, he = payload.get("train") or {}, payload.get("heldout") or {}
    results = payload.get("results") or []
    false_int = sum(int(r.get("interventions", 0)) for r in results if r.get("kind") != "trap")
    head = Text.assemble(
        ("harness ", "dim"), (str(payload.get("harness_version")), "bold cyan"),
        ("   train ", "dim"), (f"{tr.get('passed')}/{tr.get('total')}", "bold"),
        ("   held-out ", "dim"), (f"{he.get('passed')}/{he.get('total')}", "bold"),
        ("   false interventions ", "dim"), (str(false_int), "bold red" if false_int else "bold green"),
        ("   ", ""), (eval_mode(payload), "bold magenta"), (f"   {payload.get('wall_s')} s", "dim"),
    )  # fmt: skip
    table = Table(box=None, header_style="bold", pad_edge=False)
    for col in ("case", "split", "kind", "mode", "result", "blocks", "why"):
        table.add_column(col, no_wrap=col != "why", overflow="fold")
    for r in results:
        ok = bool(r.get("passed"))
        why = "" if ok else clip(str((r.get("reasons") or [""])[0]), 80)
        table.add_row(
            str(r.get("case_id")),
            str(r.get("split")),
            str(r.get("kind")),
            str(r.get("mode") or "FRESH"),
            Text("PASS" if ok else "FAIL", style="bold green" if ok else "bold red"),
            str(r.get("interventions", 0)),
            Text(why, style="red"),
        )
    return Group(table, Text(), head)
