"""One trace as a timeline, and one decision as a table of its gates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ratchet.types import Event, Trace
from ratchet.view.data import DecisionInfo

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
    sub = Text(
        f"{cost.model_calls} model calls, {cost.tool_calls} tool calls, "
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
        reps = ev.get("reports") or []
        fp = sum(len(r.get("false_positives") or []) for r in reps)
        hit = sum(int(r.get("targets_hit", 0)) for r in reps)
        tot = sum(int(r.get("targets_total", 0)) for r in reps)
        return f"{len(reps)} new guard(s): fires on {hit}/{tot} target failures, {fp} false fire(s)"
    if name == "heldout":
        confirm = "pending" if ev.get("confirm_pending") else ev.get("confirm_fresh_passed")
        return (
            f"parent fresh {ev.get('parent_fresh_passed')}, candidate {ev.get('candidate_fresh_passed')} "
            f"of {ev.get('cases')}, needs {ev.get('required')}; confirming run {confirm}"
        )
    if name == "no_regression":
        return (
            f"{ev.get('must_succeed_passed_in_either_parent_run')} must-succeed cases passed before; "
            f"regressed {len(ev.get('regressed') or [])}"
        )
    if name == "benign_twins":
        return f"{ev.get('heldout_benign_twin_runs')} twin runs, {ev.get('interventions')} intervention(s)"
    if name == "overfit_lint":
        return f"{len(ev.get('hits') or [])} copied literal(s); screened {ev.get('screened')}"
    if name == "authority":
        grew = [k for k in ("tools_granted", "guards_removed", "guards_altered") if ev.get(k)]
        return (
            "authority grew: " + ", ".join(grew)
            if grew
            else "tools subset of parent, no guard removed or altered"
        )
    if name == "cost":
        p, c = ev.get("parent") or {}, ev.get("candidate") or {}
        return f"model calls {p.get('model_calls')} -> {c.get('model_calls')} (recorded, not a gate)"
    return ""


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
    for c in d.checks:
        ok = bool(c.get("passed"))
        table.add_row(
            str(c["name"]),
            Text("PASS" if ok else "FAIL", style="bold green" if ok else "bold red"),
            evidence_line(c),
        )
    why = [Text(f"  - {r}", style="red") for r in d.reasons()]
    foot = Text.assemble(
        ("receipt ", "dim"), (d.receipt or "none", "magenta"),
        ("   key ", "dim"), ((d.keyid or "none")[:16], "magenta"),
        ("   decided ", "dim"), d.decided_at,
    )  # fmt: skip
    parts: list[RenderableType] = [head, Text(), table]
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
        ("   ", ""), (f"{payload.get('wall_s')} s", "dim"),
    )  # fmt: skip
    table = Table(box=None, header_style="bold", pad_edge=False)
    for col in ("case", "split", "kind", "result", "blocks", "why"):
        table.add_column(col, no_wrap=col != "why", overflow="fold")
    for r in results:
        ok = bool(r.get("passed"))
        why = "" if ok else clip(str((r.get("reasons") or [""])[0]), 80)
        table.add_row(
            str(r.get("case_id")),
            str(r.get("split")),
            str(r.get("kind")),
            Text("PASS" if ok else "FAIL", style="bold green" if ok else "bold red"),
            str(r.get("interventions", 0)),
            Text(why, style="red"),
        )
    return Group(table, Text(), head)
