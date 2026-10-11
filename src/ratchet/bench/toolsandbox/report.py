"""Markdown and CSV from every recorded benchmark run: the benchmark's scores per category, with denominators.

Everything is read from the run directories under `private/bench/toolsandbox/`; nothing is typed in by hand.
Every run found is listed, scenarios that were not run or lost included, so the report cannot be a selection.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ratchet.bench.toolsandbox.metrics import HEADLINE, LOST, NOT_RUN, RAN, Row, summarize, summarize_category
from ratchet.bench.toolsandbox.store import RunStore

PROTOCOL = """\
## Protocol

- Each scenario runs in ToolSandbox's own execution context: its databases (settings, contacts, messages,
  reminders), its Python tools, its execution environment and its user simulator (prompt, few-shot examples
  and `end_conversation` tool), at the pinned commit below. The score is ToolSandbox's, unchanged: its
  milestone and minefield evaluation of the whole conversation, a similarity between 0 and 1 per scenario.
  A scenario that reaches a minefield scores 0.
- The agent runs in Ratchet's worker loop under a Ratchet harness. The seed harness is ToolSandbox's own
  system message to the agent, every tool the agent can be shown, no procedures and no guards. A reply to
  the user passes through the worker as a `respond` tool, so the allowlist and every guard see it; the agent
  model itself sees a plain tool-calling transcript.
- The scenario set is ToolSandbox's, built under the seed its own runner uses: 1,032 scenarios, which are
  129 base scenarios in 8 forms each (no distractor tools, 3, 10 or all distractor tools, and four
  scramblings of the 3-distractor form).
- Category scores follow ToolSandbox's bookkeeping: a scenario counts in each category it carries, and the
  3-distractor scenarios that carry a scrambling tag are not counted under `THREE_DISTRACTION_TOOLS`.
  `ALL_CATEGORIES` is every scenario.
- A scenario NOT RUN (its necessary tools call a third-party API and the key is absent) or lost to the model
  endpoint after the predeclared retries has no similarity. It stays in `n`. Every category shows the mean
  over the scenarios that ran (`ran` of `n`) and the mean over all `n` with the others counted as 0, which
  is how ToolSandbox's own runner treats a scenario that produced no score.
- `passed` counts scenarios at similarity 0.9 or more, Ratchet's binary reading for the worker and admission
  gates; `exact` counts scenarios at 1.0. Neither replaces the similarity.
- Differences from ToolSandbox's own runner, each recorded so the numbers are not read as a leaderboard
  entry: the agent is called through an OpenAI-compatible endpoint with the harness's tool list and takes one
  tool call per turn (the first; the benchmark's parallel-call permutation check needs the calls together,
  and the worker runs and guards them one at a time); a call to a tool the scenario does not offer returns an
  error to the agent and changes nothing, where the benchmark's runner stops the scenario with an exception;
  rate-limited user-simulator calls are retried with backoff.
"""


@dataclass
class Run:
    path: str
    config: dict[str, Any]
    report: dict[str, Any]
    rows: list[Row]
    cases: dict[str, dict[str, Any]] = field(default_factory=dict)


def rows_for(store: RunStore, report: dict[str, Any]) -> list[Row]:
    """The report's results joined with the case list and the benchmark's recorded grades."""
    cases = store.read_cases()
    graded = store.graded_by_run()
    rows: list[Row] = []
    for r in report["results"]:
        case = cases[r["case_id"]]
        common = {
            "case_id": r["case_id"],
            "name": case["name"],
            "categories": tuple(case["categories"]),
            "split": r["split"],
        }
        reasons = r.get("reasons") or []
        if r.get("infra_failed") or r.get("error"):
            rows.append(Row(**common, status=LOST, detail=str(r.get("error") or "lost to the endpoint")))
        elif reasons and str(reasons[0]).startswith("NOT RUN"):
            rows.append(Row(**common, status=NOT_RUN, detail=str(reasons[0])))
        elif r.get("run_id") in graded:
            g = graded[r["run_id"]]["graded"]
            rows.append(
                Row(
                    **common,
                    status=RAN,
                    similarity=g["similarity"],
                    milestone_similarity=g["milestone_similarity"],
                    minefield_similarity=g["minefield_similarity"],
                    turn_count=g["turn_count"],
                )
            )
        else:
            rows.append(Row(**common, status=LOST, detail="no grade was recorded for this run"))
    return rows


def _shard_key(run: Run) -> tuple[Any, ...]:
    cfg = run.config
    return (
        run.report.get("harness_version"),
        cfg.get("split"),
        cfg.get("trial"),
        cfg.get("agent_model"),
        cfg.get("user_model"),
        cfg.get("limit"),
        cfg.get("shard", {}).get("count", 1),
    )


def merge_shards(runs: list[Run]) -> list[Run]:
    """One run per trial: the shards of a trial are put back together, in the order of the first shard found.

    A shard that was not recorded leaves its scenarios in the count as lost (never dropped from the
    denominator), and the run's path says which shards are present.
    """
    merged: list[Run] = []
    groups: dict[tuple[Any, ...], list[Run]] = {}
    for run in runs:
        count = run.config.get("shard", {}).get("count", 1)
        if count == 1:
            merged.append(run)
            continue
        key = _shard_key(run)
        if key not in groups:
            groups[key] = []
            merged.append(run)  # placeholder, replaced below
        groups[key].append(run)
    out: list[Run] = []
    for run in merged:
        count = run.config.get("shard", {}).get("count", 1)
        out.append(_merge(groups[_shard_key(run)]) if count > 1 else run)
    return out


def _merge(shards: list[Run]) -> Run:
    first = shards[0]
    count = first.config["shard"]["count"]
    present = sorted(r.config["shard"]["index"] for r in shards)
    rows = [row for r in shards for row in r.rows]
    have = {row.case_id for row in rows}
    for case_id, case in first.cases.items():
        if case_id not in have:
            rows.append(
                Row(
                    case_id,
                    case["name"],
                    tuple(case["categories"]),
                    case["split"],
                    LOST,
                    detail=f"its shard was not recorded (shards {present} of {count} are)",
                )
            )
    results = [res for r in shards for res in r.report["results"]]
    finished = max(str(r.report.get("finished", "")) for r in shards)
    report = {**first.report, "results": results, "finished": finished}
    config = {**first.config, "shard": {"index": present, "count": count}}
    folder = str(Path(first.path).parent / Path(first.path).name.split("-shard-")[0])
    return Run(
        f"{folder} (shards {','.join(map(str, present))} of {count})", config, report, rows, first.cases
    )


def collect(root: Path) -> list[Run]:
    """Every recorded run under `root`, in path order, with the shards of a trial merged."""
    runs: list[Run] = []
    for evals in sorted(root.rglob("evals.jsonl")):
        folder = evals.parent
        if not (folder / "cases.json").exists():
            continue
        store = RunStore(folder)
        config = (
            json.loads((folder / "config.json").read_text(encoding="utf-8"))
            if (folder / "config.json").exists()
            else {}
        )
        for rec in store.read("evals"):
            rep = rec["report"]
            if rep.get("type") == "eval" and "results" in rep:
                runs.append(
                    Run(str(folder.relative_to(root)), config, rep, rows_for(store, rep), store.read_cases())
                )
    return merge_shards(runs)


def _fmt(value: float | None, places: int = 3) -> str:
    return "" if value is None else f"{value:.{places}f}"


def _category_table(rows: list[Row], only: tuple[str, ...] | None = None) -> list[str]:
    lines = [
        "| Category | n | ran | not run | lost | similarity (ran) | similarity (all n) | milestone (ran) "
        "| milestone (all n) | minefield (ran) | exact | passed | turns (ran) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for category, s in summarize(rows).items():
        if only is not None and category not in only:
            continue
        lines.append(
            f"| `{category}` | {s['n']} | {s['ran']} | {s['not_run']} | {s['lost']} "
            f"| {_fmt(s['similarity_ran'])} | {_fmt(s['similarity_all'])} "
            f"| {_fmt(s['milestone_ran'])} | {_fmt(s['milestone_all'])} | {_fmt(s['minefield_ran'])} "
            f"| {s['exact']}/{s['ran']} | {s['passed']}/{s['ran']} | {_fmt(s['turns_ran'], 1)} |"
        )
    return lines


def _title(run: Run) -> str:
    cfg = run.config
    sample = f", seeded sample of {cfg['limit']}" if cfg.get("limit") else ""
    where = f"{cfg.get('split', '?')} (trial {cfg.get('trial', '?')}{sample})"
    return f"`{run.report['harness_version']}` on {where}"


def _run_section(run: Run) -> list[str]:
    cfg = run.config
    out = [f"### {_title(run)}", ""]
    out.append(
        f"- Run directory `{run.path}`; agent model `{cfg.get('agent_model', '?')}`, user simulator model "
        f"`{cfg.get('user_model', '?')}`, temperature {cfg.get('temperature', '?')}; "
        f"tool-sandbox `{str(cfg.get('tool_sandbox_commit', '?'))[:12]}`."
    )
    out.append(f"- Started {cfg.get('started', '?')}, finished {run.report.get('finished', '?')}.")
    summary = summarize_category(run.rows)
    out.append(
        f"- {summary['n']} scenarios: {summary['ran']} ran, {summary['not_run']} NOT RUN, "
        f"{summary['lost']} lost to the endpoint."
    )
    not_run = sorted({r.detail for r in run.rows if r.status == NOT_RUN})
    for detail in not_run:
        count = sum(1 for r in run.rows if r.status == NOT_RUN and r.detail == detail)
        out.append(f"- NOT RUN ({count} scenarios): `{detail}`")
    lost = [r for r in run.rows if r.status == LOST]
    for r in lost[:20]:
        out.append(f"- Lost `{r.name}`: {r.detail}")
    if len(lost) > 20:
        out.append(f"- ... and {len(lost) - 20} more lost scenarios, listed in the CSV.")
    out += ["", "**Headline categories**", "", *_category_table(run.rows, ("ALL_CATEGORIES", *HEADLINE)), ""]
    out += ["**Every category**", "", *_category_table(run.rows), ""]
    return out


def render(root: Path, csv_name: str = "toolsandbox-scenarios.csv") -> str:
    runs = collect(root)
    out = [
        "# ToolSandbox results",
        "",
        "ToolSandbox, run as a Ratchet world: the agent runs in Ratchet's worker loop under a harness, every",
        "tool call executes in ToolSandbox's own execution environment, the user is ToolSandbox's simulator,",
        "and every conversation is graded by ToolSandbox's own milestone and minefield evaluation,",
        "unchanged.",
        "Generated by `python -m ratchet.bench.toolsandbox report`; the raw traces, harnesses and the",
        "benchmark's grade of every scenario are kept as JSON lines in each run directory. Every scenario of",
        f"every run is listed in [{csv_name}]({csv_name}).",
        "",
        PROTOCOL,
        "## Overview (all scenarios of each run)",
        "",
    ]
    if not runs:
        return "\n".join([*out, "No run recorded yet.", ""])
    out += [
        "| Harness | Split | Agent model | n | ran | not run | lost | similarity (ran) | similarity (all n) "
        "| milestone (ran) | exact | passed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        s = summarize_category(run.rows)
        out.append(
            f"| `{run.report['harness_version']}` | {run.config.get('split', '?')} "
            f"| `{run.config.get('agent_model', '?')}` | {s['n']} | {s['ran']} | {s['not_run']} "
            f"| {s['lost']} | {_fmt(s['similarity_ran'])} | {_fmt(s['similarity_all'])} "
            f"| {_fmt(s['milestone_ran'])} "
            f"| {s['exact']}/{s['ran']} | {s['passed']}/{s['ran']} |"
        )
    out += ["", "## Runs", ""]
    for run in runs:
        out += _run_section(run)
    return "\n".join(out).rstrip() + "\n"


def render_csv(root: Path) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        ["run", "harness", "case_id", "scenario", "split", "status", "similarity", "milestone_similarity",
         "minefield_similarity", "turns", "categories", "detail"]
    )  # fmt: skip
    for run in collect(root):
        for r in run.rows:
            writer.writerow(
                [run.path, run.report["harness_version"], r.case_id, r.name, r.split, r.status,
                 "" if r.similarity is None else r.similarity,
                 "" if r.milestone_similarity is None else r.milestone_similarity,
                 "" if r.minefield_similarity is None else r.minefield_similarity,
                 "" if r.turn_count is None else r.turn_count, " ".join(r.categories), r.detail]
            )  # fmt: skip
    return buf.getvalue()
