"""Render a run's `results.json` as the Markdown summary table kept in `docs/results/agentdojo.md`."""

from __future__ import annotations

from typing import Any


def _cell(m: dict[str, Any]) -> str:
    rate = m.get("rate")
    pct = f"{100 * rate:.1f}%" if rate is not None else "n/a"
    err = f", {m['errors']} err" if m.get("errors") else ""
    return f"{pct} ({m['hits']}/{m['total']}{err})"


def _guard_lines(changes: dict[str, Any], parent_guards: set[str]) -> list[str]:
    out = []
    for g in changes.get("guards", []):
        if g["id"] not in parent_guards:
            out.append(f"  - guard `{g['id']}`: {g['description']}")
    return out


def render(results: dict[str, Any]) -> str:
    p = results["provenance"]
    lines = [
        f"### {p['suite']} ({p['benchmark_version']}), run `{results['run_id']}`",
        "",
        f"- Agent model `{p['model']}`, improver model `{p['improver_model']}`, endpoint `{p['endpoint']}`",
        f"- AgentDojo package {p['agentdojo_package_version']}"
        + (
            f", reference clone commit `{p['agentdojo_reference_clone_commit']}`"
            if p.get("agentdojo_reference_clone_commit")
            else ""
        ),
        f"- Ratchet commit `{p.get('ratchet_commit')}`, split seed {p['seed']}, attack `{p['attack']}`",
        f"- Held-out user tasks: {', '.join(p['split']['heldout_users'])}",
        f"- Held-out injection tasks: {', '.join(p['split']['heldout_injections'])}",
        f"- Cases: {p['cases']['train']} training, {p['cases']['heldout']} held-out "
        "(the improver reads training traces only)",
        "",
        "| Version | Evaluation | Role | Gen | Utility | Utility under attack | Targeted ASR "
        "| Ratchet pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = results["metrics"]
    for v in results["versions"]:
        mine = [r for r in rows if r["version"] == v and "heldout" in r]
        if not mine:
            lines.append(f"| `{v}` | not evaluated | | | | | | |")
        for r in mine:
            h = r["heldout"]
            lines.append(
                f"| `{v}` | {r['eval_id']} | {r.get('role') or ''} | {r.get('generation') or ''} "
                f"| {_cell(h['utility'])} | {_cell(h['utility_under_attack'])} "
                f"| {_cell(h['targeted_attack_success'])} | {_cell(h['ratchet_passed'])} |"
            )
    lines.append("")
    changes = results.get("admitted", {})
    seen: set[str] = set()
    admitted_lines: list[str] = []
    for v in results["versions"][1:]:
        c = changes.get(v)
        if c is None:
            continue
        admitted_lines.append(f"- `{v}` (parent `{c['parent']}`):")
        admitted_lines += _guard_lines(c, seen)
        seen |= {g["id"] for g in c.get("guards", [])}
        for proc in c.get("procedures", []):
            if proc["name"] not in seen:
                admitted_lines.append(f"  - procedure `{proc['name']}` when {proc['when']}")
                seen.add(proc["name"])
    if admitted_lines:
        lines += ["Admitted versions and their changes:", "", *admitted_lines, ""]
    else:
        lines += ["No candidate was admitted in this run.", ""]
    for note in results.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"
