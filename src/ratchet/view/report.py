"""One static HTML page from Atlas data: the replay matrix, the lineage and the results table.

The page is a document, not an app: no script, no network, one file, so it can sit in a README, be attached
to a post or be opened from disk. Everything that came from the store is escaped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

from ratchet.view.admission import (
    RUNS,
    FreshView,
    ReplayRow,
    fresh_view,
    procedures_not_applicable,
    replay_rows,
)
from ratchet.view.data import VersionNode
from ratchet.view.matrix import GROUP_TITLE, Matrix, RowKind, layout

CSS = """
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;
--good:#3fb950;--bad:#f85149;--warn:#d29922;--cyan:#58a6ff;--mag:#bc8cff}
@media (prefers-color-scheme:light){:root{--bg:#fff;--panel:#f6f8fa;--line:#d0d7de;--fg:#1f2328;--dim:#59636e;
--good:#1a7f37;--bad:#cf222e;--warn:#9a6700;--cyan:#0969da;--mag:#8250df}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
main{max-width:1100px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:18px;margin:40px 0 12px;color:var(--cyan)}
.sub{color:var(--dim);margin:0 0 24px}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 8px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 16px}
.stat b{display:block;font-size:22px}.stat span{color:var(--dim);font-size:12px}
table{border-collapse:collapse;width:100%;background:var(--panel);border:1px solid var(--line)}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--dim);font-weight:600}
td.c,th.c{text-align:center}
tr.grp td{background:var(--bg);font-weight:700;letter-spacing:.04em}
.hit{color:var(--good);font-weight:700}.miss{color:var(--bad);font-weight:700}
.block{background:var(--bad);color:#fff;font-weight:700}.pass{color:var(--good)}.na,.dim{color:var(--dim)}
.err{color:var(--warn);font-weight:700}
.badge{display:inline-block;padding:1px 10px;border-radius:4px;font-weight:700;color:#fff}
.incomplete{background:var(--warn)}.admitted{background:var(--good)}.rejected{background:var(--bad)}.replay-ok{background:var(--cyan)}
.legend{margin:12px 0 0;padding:0;list-style:none}.legend li{padding:3px 0}
.foot{margin-top:10px;font-weight:700}
.scroll{overflow-x:auto}
.stable{color:var(--good);font-weight:700}
.broken{background:var(--bad);color:#fff;font-weight:700}
h3{font-size:15px;margin:24px 0 8px}
ul.tree,ul.tree ul{list-style:none;margin:0;padding-left:22px;border-left:1px solid var(--line)}
ul.tree{border-left:0;padding-left:0}ul.tree li{padding:4px 0 4px 8px;position:relative}
.v{color:var(--cyan);font-weight:700}.rcpt{color:var(--mag)}.why{color:var(--bad);display:block;padding-left:20px}
.ok{color:var(--good);font-weight:700}.no{color:var(--bad);font-weight:700}.seed{color:var(--cyan);font-weight:700}
"""

_VERDICT_CLASS = {
    "ADMITTED": "admitted",
    "REJECTED": "rejected",
    "CLEARED": "replay-ok",
    "INCOMPLETE": "incomplete",
}
_MARK = {"seed": ("◆", "seed"), "promoted": ("✓", "ok"), "admitted": ("✓", "ok"), "rejected": ("✗", "no")}


@dataclass(frozen=True)
class ReportData:
    title: str
    roots: Sequence[VersionNode]
    matrices: Sequence[Matrix]
    generated: str


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def matrix_html(m: Matrix) -> str:
    n = len(m.columns)
    head = "".join(f'<th class="c">{_e(c.letter)}</th>' for c in m.columns)
    rows = [f"<tr><th>recorded trace</th>{head}</tr>"]
    lines = layout(m, budget=10**6, full=True)
    for ln in lines:
        if ln.kind == "group":
            cells = "".join(f'<td class="c">{_e(_group_cell(m, ln.group, c))}</td>' for c in range(n))
            rows.append(
                f'<tr class="grp"><td>{GROUP_TITLE[ln.group]} ({len(m.rows_of(ln.group))})</td>{cells}</tr>'
            )
        elif ln.kind == "row":
            cells = "".join(_cell_html(m, ln.row, c) for c in range(n))
            rows.append(f"<tr><td>{_e(m.rows[ln.row].case_id)}</td>{cells}</tr>")
    verdicts = "".join(_verdict_html(m, c) for c in range(n))
    rows.append(f'<tr class="grp"><td>VERDICT</td>{verdicts}</tr>')
    legend = "".join(_legend_item(m, c) for c in range(n))
    rate = m.checks / m.elapsed_s if m.elapsed_s > 0 else float(m.checks)
    foot = (
        f"{len(m.rows)} traces, {m.elapsed_s:.2f} s, {m.model_calls} model calls "
        f'<span class="dim">&middot; {m.checks:,} guard checks ({rate:,.0f}/s) on recorded state</span>'
    )
    return (
        f'<div class="scroll"><table>{"".join(rows)}</table></div>'
        f'<ul class="legend">{legend}</ul><div class="foot">{foot}</div>'
    )


def _verdict_html(m: Matrix, c: int) -> str:
    label = m.verdict(c).label
    return f'<td class="c"><span class="badge {_VERDICT_CLASS[label]}">{_e(label)}</span></td>'


def _group_cell(m: Matrix, g: RowKind, c: int) -> str:
    if g == "trap_failure":
        hit, total = m.hits(c)
        return f"{hit}/{total} targets"
    return f"{m.count(g, c)} blocked"


def _cell_html(m: Matrix, r: int, c: int) -> str:
    cell, kind = m.cells[(r, c)], m.rows[r].kind
    at = f" @{cell.seq}" if cell.seq is not None else ""
    if cell.error:
        return '<td class="c err">! error</td>'
    if kind == "trap_failure":
        if cell.fired:
            return f'<td class="c hit">&#9679; BLOCK{at}</td>'
        if cell.late:
            return f'<td class="c err">&#9680; late{at}</td>'
        return '<td class="c miss">&#9675; missed</td>' if r in m.targets[c] else '<td class="c na">-</td>'
    if kind in ("past_success", "benign_twin"):
        if cell.fired:
            return f'<td class="c block">&#10007; BLOCK{at}</td>'
        return '<td class="c pass">&middot; pass</td>'
    return f'<td class="c hit">&#9679; BLOCK{at}</td>' if cell.fired else '<td class="c na">&middot;</td>'


def _legend_item(m: Matrix, c: int) -> str:
    col, v = m.columns[c], m.verdict(c)
    why = "; ".join(v.reasons) or "fires on its targets and on nothing that worked"
    rcpt = (
        f' <span class="rcpt">[{_e(col.decision.receipt)}]</span>'
        if col.decision and col.decision.receipt
        else ""
    )
    cls = "ok" if v.ok else "no"
    return (
        f'<li><span class="{cls}">{"&#10003;" if v.ok else "&#10007;"}</span> <b>{_e(col.letter)}</b> '
        f'<span class="v">{_e(col.guard.id)}</span> &mdash; {_e(col.guard.description)}<br>'
        f'<span class="{cls}">{_e(v.label)}</span>: {_e(why)}{rcpt} '
        f'<span class="dim">replay {_e(v.replay)}</span></li>'
    )


def lineage_html(roots: Sequence[VersionNode]) -> str:
    def node(n: VersionNode, parent: VersionNode | None) -> str:
        mark, cls = _MARK[n.status]
        d = n.decision
        what = "seed" if n.status == "seed" else (d.what if d else "")
        active = f' <span class="badge admitted">ACTIVE rev {n.active_revision}</span>' if n.active else ""
        out = (
            f'<span class="{cls}">{mark}</span> <span class="v">{_e(n.version)}</span>{active} '
            f"{_e(what)} {_score(n, parent)}"
        )
        if d and d.receipt:
            out += f' <span class="rcpt">{_e(d.receipt)}</span>'
        if n.status == "rejected" and d:
            out += f'<span class="why">{_e("; ".join(d.reasons()) or "rejected")}</span>'
        kids = sorted(n.children, key=lambda c: c.status == "rejected")
        inner = f"<ul>{''.join(node(k, n) for k in kids)}</ul>" if kids else ""
        return f"<li>{out}{inner}</li>"

    return f'<ul class="tree">{"".join(node(r, None) for r in roots)}</ul>'


def _score(n: VersionNode, parent: VersionNode | None) -> str:
    s = n.stats
    if s is None:
        return '<span class="dim">not evaluated</span>'
    held = f"{s.heldout_passed}/{s.heldout_total}"
    if parent is not None and parent.stats is not None:
        held = f"{parent.stats.heldout_passed} &rarr; {held}"
    return (
        f'<span class="dim">held-out</span> <b>{held}</b> <span class="dim">&middot; train</span> '
        f'{s.train_passed}/{s.train_total} <span class="dim">&middot; false interventions</span> '
        f'<b class="{"pass" if s.false_interventions == 0 else "miss"}">{s.false_interventions}</b> '
        f'<span class="rcpt">{_e(s.mode)}</span>'
    )


def results_html(roots: Sequence[VersionNode]) -> str:
    head = "".join(
        f"<th>{h}</th>"
        for h in (
            "version",
            "parent",
            "outcome",
            "change",
            "train",
            "held-out",
            "false int.",
            "evaluation",
            "receipt",
        )
    )
    rows = []
    for r in roots:
        for n in r.walk():
            s, d = n.stats, n.decision
            label = {"seed": "seed", "promoted": "promoted", "admitted": "admitted", "rejected": "rejected"}[
                n.status
            ]
            cls = "no" if n.status == "rejected" else ("seed" if n.status == "seed" else "ok")
            cells = [
                f'<td class="v">{_e(n.version)}</td>',
                f"<td>{_e(n.parent or '-')}</td>",
                f'<td class="{cls}">{label}</td>',
                f"<td>{_e(d.what if d else '-')}</td>",
                f"<td>{s.train_passed}/{s.train_total}</td>" if s else '<td class="dim">-</td>',
                f"<td>{s.heldout_passed}/{s.heldout_total}</td>" if s else '<td class="dim">-</td>',
                f"<td>{s.false_interventions}</td>" if s else '<td class="dim">-</td>',
                f"<td>{_e(s.mode)}</td>" if s else '<td class="dim">-</td>',
                f'<td class="rcpt">{_e(d.receipt) if d and d.receipt else "-"}</td>',
            ]
            rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table><tr>{head}</tr>{"".join(rows)}</table></div>'


def admission_html(roots: Sequence[VersionNode]) -> str:
    """For each decision: the replay gate per guard, and the four paired runs case by case."""
    out: list[str] = []
    for n in (n for r in roots for n in r.walk()):
        d = n.decision
        if d is None:
            continue
        mark = '<span class="ok">ADMITTED</span>' if d.admitted else '<span class="no">REJECTED</span>'
        body = f"<h3>{_e(d.parent)} &rarr; {_e(d.candidate)} {mark} {_e(d.what)}</h3>"
        for c in d.checks:
            ev = c.get("evidence") or {}
            if c["name"] == "replay" and (rows := replay_rows(ev)):
                body += _replay_table_html(rows, procedures_not_applicable(ev))
            if c["name"] == "fresh" and (v := fresh_view(ev)) is not None:
                body += _fresh_table_html(v)
        out.append(body)
    return "".join(out)


_ROLE_CLASS = {
    "broken": "broken",
    "stable": "stable",
    "invalid": "err",
    "protected": "dim",
    "unrecovered": "na",
}
_ROLE_WORDS = {
    "broken": "PROTECTED CASE BROKE",
    "stable": "STABLE RECOVERY",
    "invalid": "NO VALID EVIDENCE",
    "protected": "protected",
    "unrecovered": "still failing",
}


def _fresh_table_html(v: FreshView) -> str:
    head = "".join(f'<th class="c">{r}</th>' for r in RUNS)
    rows = []
    for row in v.rows:
        cells = "".join(
            f'<td class="c {"pass" if o == "pass" else ("miss" if o == "fail" else "err")}">{_e(o)}</td>'
            for o in row.outcomes
        )
        rows.append(
            f"<tr><td>{_e(row.case_id)}</td>{cells}"
            f'<td class="{_ROLE_CLASS[row.role]}">{_ROLE_WORDS[row.role]}</td></tr>'
        )
    pend = "; C2 (the confirming run) has not run yet" if v.confirm_pending else ""
    return (
        f'<div class="scroll"><table><tr><th>held-out case</th>{head}<th>role</th></tr>'
        f"{''.join(rows)}</table></div>"
        f'<div class="foot">stable recoveries {len(v.stable)} (need {v.needed}) &middot; protected '
        f"{len(v.protected)} &middot; broken {len(v.broken)} &middot; parent discordance D={v.discordance} "
        f'<span class="dim">(reported, not a threshold){pend}</span></div>'
    )


def _replay_table_html(rows: Sequence[ReplayRow], procedures: Sequence[str]) -> str:
    cells = []
    for r in rows:
        cls = {"CLEARED": "ok", "INCOMPLETE": "err"}.get(r.verdict, "no")
        hit = "-" if r.targets_total is None else f"{r.targets_hit}/{r.targets_total}"
        cells.append(
            f'<tr><td>{_e(r.guard_id)}</td><td>{_e(r.coverage)}</td><td class="{cls}">{_e(r.verdict)}</td>'
            f"<td>{hit}</td><td>{r.false_fires}</td><td>{r.benign_fires}</td><td>{r.errors}</td></tr>"
        )
    notes = "".join(f'<div class="dim">procedure {_e(p)}: replay NOT APPLICABLE</div>' for p in procedures)
    head = "".join(
        f"<th>{h}</th>"
        for h in ("guard", "coverage", "verdict", "targets hit", "false fires", "benign twin fires", "errors")
    )
    return f'<div class="scroll"><table><tr>{head}</tr>{"".join(cells)}</table></div>{notes}'


def _stats(roots: Sequence[VersionNode]) -> str:
    nodes = [n for r in roots for n in r.walk()]
    promoted = [n for n in nodes if n.status in ("promoted", "admitted")]
    rejected = [n for n in nodes if n.status == "rejected"]
    scored = [n for n in nodes if n.stats and n.status in ("seed", "promoted")]
    first = scored[0].stats if scored else None
    last = scored[-1].stats if scored else None
    items = [
        (str(len(nodes)), "versions"),
        (str(len(promoted)), "admitted"),
        (str(len(rejected)), "rejected"),
    ]
    if first and last:
        items.append(
            (f"{first.heldout_passed} &rarr; {last.heldout_passed} / {last.heldout_total}", "held-out passes")
        )
        items.append((str(last.false_interventions), "false interventions, latest"))
    return "".join(f'<div class="stat"><b>{v}</b><span>{_e(k)}</span></div>' for v, k in items)


def render_html(data: ReportData) -> str:
    matrix = "".join(
        f"<h2>Replay matrix: parent {_e(m.parent)}</h2>"
        '<p class="sub">Every recorded trace against every candidate guard. No model is called.</p>'
        f"{matrix_html(m)}"
        for m in data.matrices
    )
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(data.title)}</title><style>{CSS}</style></head><body><main>"
        f'<h1>{_e(data.title)}</h1><p class="sub">Generated from MongoDB Atlas, {_e(data.generated)}</p>'
        f'<div class="stats">{_stats(data.roots)}</div>'
        f"{matrix}<h2>Lineage</h2>{lineage_html(data.roots)}"
        f"<h2>Results</h2>{results_html(data.roots)}"
        f"<h2>Admission screens</h2>{admission_html(data.roots)}</main></body></html>\n"
    )
