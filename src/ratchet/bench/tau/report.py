"""Markdown summary of every recorded tau-bench run for one domain: counts, denominators, pass^k, decisions.

Everything is read from the run records under `private/bench/tau/`; nothing is typed in by hand. Every
evaluation is listed, failed and crashed runs included, so the summary cannot be a selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ratchet.bench.tau.metrics import by_task, summary
from ratchet.bench.tau.store import JsonlStore
from ratchet.bench.tau.tasks import DOMAINS

MAX_K = 4

PROTOCOL = """\
## Protocol

- Each task runs in tau-bench's own environment: its database, tools, task list, user simulator (prompt and
  code) and reward, at the pinned commit below. The reward is tau-bench's, unchanged: the final database must
  equal the one the task's gold actions produce and every required output must have been said to the user.
- The agent runs in Ratchet's worker loop under a Ratchet harness. The seed harness is tau-bench's own agent
  prompt (the domain policy), every tau-bench tool, no procedures and no guards. Replies to the customer pass
  through the worker as a `respond` tool, so the allowlist and every guard see them; the agent model itself
  sees tau-bench's plain tool-calling transcript and tools.
- As in tau-bench's tool-calling agent: one tool call per turn (the first), at most 30 agent actions, and a
  conversation that never ends scores 0.
- Differences from tau-bench's own runner: the agent request carries temperature 0 through an
  OpenAI-compatible endpoint; rate-limited user-simulator calls are retried with backoff (tau-bench retries
  none), so an endpoint limit is not scored as an agent failure.
- Held-out tasks are never shown to the improver; it reads train traces only. Every run is counted: a run
  that crashed is a failed run, and runs stopped early are listed with the reason.
"""


def collect(root: Path) -> list[dict[str, Any]]:
    """Every recorded evaluation report: trial reports and every evaluation an evolve run persisted."""
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("evals.jsonl")):
        domain = _domain_of(path, root)
        for rec in JsonlStore(path.parent).read("evals"):
            rep = rec["report"]
            if rep.get("type") == "eval" and "results" in rep:
                rows.append({**rep, "domain": domain, "source": str(path.parent.relative_to(root))})
    return rows


def _domain_of(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return next((p for p in parts if p in DOMAINS), "unknown")


def summarize(rows: list[dict[str, Any]], max_k: int) -> list[dict[str, Any]]:
    table: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for rep in rows:
        for r in rep["results"]:
            table.setdefault((rep["domain"], rep["harness_version"], r["split"]), []).append(r)
    out = []
    for (domain, version, split), results in sorted(table.items()):
        s = summary(by_task(results), range(1, max_k + 1))
        s["errors"] = sum(1 for r in results if r.get("error"))
        out.append({"domain": domain, "harness": version, "split": split, **s})
    return out


def _pct(value: float | None) -> str:
    return "" if value is None else f"{100 * value:.1f}%"


def _cell(row: dict[str, Any], k: int) -> str:
    """pass^k over every task, or over the tasks with at least k trials, with their count named."""
    if f"pass^{k}" in row:
        return _pct(row[f"pass^{k}"])
    sub = row.get(f"pass^{k}_subset")
    return f"{_pct(sub['value'])} ({sub['tasks']} of {row['tasks']} tasks)" if sub else ""


def _configs(root: Path, domain: str) -> list[dict[str, Any]]:
    out = []
    for path in sorted(root.rglob("config.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("domain", "retail") == domain:
            out.append({**cfg, "source": str(path.parent.relative_to(root))})
    return out


def _decisions(root: Path, domain: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "evolve" / domain).glob("*/decisions.jsonl")):
        for rec in JsonlStore(path.parent).read("decisions"):
            d = rec["decision"]
            failed = [c["name"] for c in d.get("checks", []) if not c.get("passed")]
            rows.append(
                {
                    "run": path.parent.name,
                    "id": rec["decision_id"],
                    "admitted": bool(d.get("admitted")),
                    "failed": failed,
                }
            )
    return rows


def _results(root: Path, domain: str) -> list[dict[str, Any]]:
    out = []
    for path in sorted((root / "evolve" / domain).glob("*/result.json")):
        res = json.loads(path.read_text(encoding="utf-8"))
        out.append({**res, "run": path.parent.name})
    return out


def unfinished(root: Path, domain: str, counted: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """Runs recorded in an evolve store whose evaluation never finished (the run was stopped first).

    `counted` holds (source, harness) pairs already covered by a finished evaluation. Each remaining group
    is reported as what it is: completed runs of an evaluation that did not complete.
    """
    out = []
    for path in sorted((root / "evolve" / domain).glob("*/traces.jsonl")):
        source = str(path.parent.relative_to(root))
        cfg_path = path.parent / "config.json"
        record = (
            json.loads(cfg_path.read_text(encoding="utf-8")).get("split_record", {})
            if cfg_path.exists()
            else {}
        )
        held = set(record.get("heldout", []))
        runs: dict[str, dict[str, Any]] = {}
        for t in JsonlStore(path.parent).read("traces"):
            runs[t["run_id"]] = t
        groups: dict[str, list[dict[str, Any]]] = {}
        for t in runs.values():
            if (source, t["harness_version"]) not in counted:
                groups.setdefault(t["harness_version"], []).append(t)
        for version, ts in sorted(groups.items()):
            h = [t for t in ts if t["case_id"] in held]
            tr = [t for t in ts if t["case_id"] not in held]
            out.append(
                {
                    "source": source,
                    "harness": version,
                    "heldout": (sum(t["outcome"]["passed"] for t in h), len(h)),
                    "train": (sum(t["outcome"]["passed"] for t in tr), len(tr)),
                }
            )
    return out


INTRO = (
    "tau-bench is a set of simulated customer-service tasks, here the {domain} domain. A simulated "
    "customer talks to the agent, and the final database state is graded. pass^k is the share of "
    "tasks the agent passes in all k repeated tries. This page holds Ratchet's measured results, "
    "starting from a baseline harness (the code around the model: prompt, tools and rules)."
)


def render(root: Path, domain: str) -> str:
    rows = [r for r in collect(root) if r["domain"] == domain]
    table = summarize(rows, MAX_K)
    configs = _configs(root, domain)
    lines = [f"# tau-bench {domain}: Ratchet results", "", INTRO.format(domain=domain), ""]
    lines += [
        "Generated by `python -m ratchet.bench.tau report --domain " + domain + "` from the run records.",
        "",
    ]
    notes = root / f"notes-{domain}.md"
    if notes.exists():
        lines += [notes.read_text(encoding="utf-8").rstrip(), ""]
    lines += [PROTOCOL, "## Setup", ""]
    keys = ("agent_model", "user_model", "improver_model", "temperature", "tau_bench_commit", "workers")
    counted = {r["source"] for r in rows} | {f"evolve/{domain}/{r['run']}" for r in _results(root, domain)}
    stopped = [c for c in configs if c["source"] not in counted]
    seen: set[str] = set()
    for cfg in configs:
        if cfg in stopped:
            continue
        rec = cfg.get("split_record", {})
        fields = ", ".join(f"{k} `{cfg[k]}`" for k in keys if k in cfg)
        n_train, n_held = len(rec.get("train", [])), len(rec.get("heldout", []))
        split = f"seed `{rec.get('seed')}`, {n_train} train / {n_held} held-out"
        line = f"- `{cfg['source']}`: {fields}; split {split}, digest `{str(rec.get('digest', ''))[:16]}`"
        if cfg.get("note"):
            line += f". Note: {cfg['note']}"
        if cfg.get("stopped"):
            line += f". Stopped: {cfg['stopped']}"
        if line not in seen:
            seen.add(line)
            lines.append(line)
    if stopped:
        lines += [
            "",
            "Runs stopped before any evaluation finished (traces kept, nothing counted from them):",
            "",
        ]
        lines += [f"- `{c['source']}`" + (f": {c['stopped']}" if c.get("stopped") else "") for c in stopped]
    lines += ["", "## pass^k by harness and split", ""]
    lines += [
        "A run that crashed is a failed run. pass^k is tau-bench's estimator, mean over tasks of "
        "C(successes, k) / C(trials, k). Where some tasks have fewer than k trials, the cell gives pass^k "
        "over the tasks that have k, with their count; those tasks are the seeded held-out subset an evolve "
        "run evaluated, not a selection by outcome.",
        "",
        "| harness | split | tasks | trials/task | runs | successes | crashed | "
        + " | ".join(f"pass^{k}" for k in range(1, MAX_K + 1))
        + " |",
        "|---|---|---|---|---|---|---|" + "---|" * MAX_K,
    ]
    for t in table:
        ks = " | ".join(_cell(t, k) for k in range(1, MAX_K + 1))
        lines.append(
            f"| `{t['harness']}` | {t['split']} | {t['tasks']} | {t['trials_per_task']} | {t['runs']} | "
            f"{t['successes']} | {t['errors']} | {ks} |"
        )
    lines += ["", "## Every evaluation", ""]
    lines += [
        "| source | role | gen | harness | held-out passed | train passed | crashed |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        held, train = r.get("heldout", {}), r.get("train", {})
        lines.append(
            f"| `{r['source']}` | {r.get('role', 'trial')} | {r.get('generation', '')} "
            f"| `{r['harness_version']}` | {held.get('passed', 0)}/{held.get('total', 0)} "
            f"| {train.get('passed', 0)}/{train.get('total', 0)} | {r.get('errors', 0)} |"
        )
    partial = unfinished(root, domain, {(r["source"], r["harness_version"]) for r in rows})
    if partial:
        lines += ["", "## Runs of evaluations that did not finish", ""]
        lines += [
            "Completed runs of a harness whose evaluation was stopped before it finished. They are not in "
            "the pass^k table; the counts are what finished, not a sample chosen afterwards.",
            "",
            "| source | harness | held-out passed / runs | train passed / runs |",
            "|---|---|---|---|",
        ]
        for u in partial:
            lines.append(
                f"| `{u['source']}` | `{u['harness']}` | {u['heldout'][0]}/{u['heldout'][1]} "
                f"| {u['train'][0]}/{u['train'][1]} |"
            )
    decisions = _decisions(root, domain)
    if decisions:
        lines += ["", "## Admission decisions", ""]
        lines += ["| evolve run | parent -> candidate | admitted | failed checks |", "|---|---|---|---|"]
        for d in decisions:
            lines.append(f"| {d['run']} | `{d['id']}` | {d['admitted']} | {', '.join(d['failed']) or '-'} |")
    for res in _results(root, domain):
        lines += ["", f"## Evolve run `{res['run']}`", ""]
        lines.append(f"- final harness `{res['final_harness']}`, active pointer {res.get('active')}")
        lines.append(f"- wall time {res.get('wall_s')} s, generations {len(res.get('generations', []))}")
        for g in res.get("generations", []):
            level1 = g.get("level1", {})
            lines.append(
                f"- generation {g.get('generation')}: parent `{g.get('parent')}`, "
                f"clusters {len(g.get('clusters', []))}, "
                f"candidates {level1.get('candidates')}, evaluated {level1.get('evaluated')}, "
                f"admitted {level1.get('admitted')}, promoted `{level1.get('promoted')}`"
                + (f"; note: {g['note']}" if g.get("note") else "")
            )
    return "\n".join(lines) + "\n"
