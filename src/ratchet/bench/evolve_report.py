"""Markdown for evolve runs, read from their stores: parent scores, every candidate, every decision.

An evolve store is a directory with a `config.json` that names its `generations`, plus the JSON Lines files
the improvement loop writes (`evals.jsonl`, `decisions.jsonl`, `traces.jsonl`) and, when the run finished,
`result.json`. A run stopped or failed on purpose carries `STOPPED.txt` or `FAILED.txt` with the reason
(or a `stopped` key in its config); that text is shown as written, as is a `NOTE.txt`. A conversation lost
to the model endpoint is counted and shown as lost, never as a pass and never dropped from a denominator.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

STOPPED = "STOPPED.txt"
FAILED = "FAILED.txt"
NOTE = "NOTE.txt"  # written by whoever reads the records, for what they show and the tables cannot say


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _config(folder: Path) -> dict[str, Any]:
    path = folder / "config.json"
    if not path.is_file():
        return {}
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def unique_stores(paths: list[Path]) -> list[Path]:
    """The run directories, each run once: a copy (same directory name, byte-identical `config.json`) of a
    directory already listed is dropped. The copy kept is the one holding more evaluation lines, so a
    partial copy never hides a fuller one."""
    best: dict[tuple[str, bytes], Path] = {}
    order: list[tuple[str, bytes]] = []
    for path in sorted(paths):
        cfg = path / "config.json"
        key = (path.name, cfg.read_bytes() if cfg.is_file() else str(path).encode())
        if key not in best:
            order.append(key)
            best[key] = path
        elif len(_jsonl(path / "evals.jsonl")) > len(_jsonl(best[key] / "evals.jsonl")):
            best[key] = path
    return [best[k] for k in order]


def _started(folder: Path) -> tuple[float, str]:
    """Sort key: the run's recorded start time, then its path (a run with no parseable start sorts last)."""
    try:
        when = datetime.strptime(str(_config(folder).get("started")), "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except ValueError:
        when = float("inf")
    return when, str(folder)


def evolve_stores(root: Path, domain: str | None = None) -> list[Path]:
    """Every evolve store under `root` (copies dropped) in the order the runs started, optionally only
    those of one domain."""
    found = []
    for cfg in sorted(root.rglob("config.json")):
        config = _config(cfg.parent)
        if "generations" not in config:
            continue
        if domain is not None and config.get("domain") != domain:
            continue
        found.append(cfg.parent)
    return sorted(unique_stores(found), key=_started)


def _reports(folder: Path) -> list[dict[str, Any]]:
    out = []
    for rec in _jsonl(folder / "evals.jsonl"):
        report = rec.get("report", rec)
        if isinstance(report, dict):
            out.append(report)
    return out


def _lost(report: dict[str, Any]) -> int:
    return sum(1 for r in report.get("results", []) if r.get("error") or r.get("infra_failed"))


def _count(report: dict[str, Any], split: str) -> str:
    c = report.get(split) or {}
    return f"{c.get('passed', 0)}/{c['total']}" if c.get("total") else "not run"


def status(folder: Path) -> str:
    """How the run ended, in the words its records give."""
    for name, label in ((FAILED, "Failed"), (STOPPED, "Stopped")):
        path = folder / name
        if path.is_file():
            return f"{label} (`{name}`): " + " ".join(path.read_text(encoding="utf-8").split())
    config = _config(folder)
    if config.get("stopped"):
        return f"Stopped: {config['stopped']}"
    if (folder / "result.json").is_file():
        planned = config.get("generations")
        done = len(json.loads((folder / "result.json").read_text(encoding="utf-8")).get("generations", []))
        return f"Finished: {done} of {planned} planned generations ran."
    return "Not finished, and no stop reason is recorded."


def _generations(folder: Path) -> list[dict[str, Any]]:
    """Per generation, its loop summary: from `result.json` when the run finished, else from the
    generation records in `evals.jsonl`."""
    result = folder / "result.json"
    if result.is_file():
        gens: list[dict[str, Any]] = json.loads(result.read_text(encoding="utf-8")).get("generations", [])
        return gens
    return [r for r in _reports(folder) if r.get("type") == "generation"]


def _split_of(folder: Path, reports: list[dict[str, Any]]) -> dict[str, str]:
    """Case id to split: from the evaluations, else the config's split record, else the store's cases."""
    split: dict[str, str] = {}
    cases = folder / "cases.json"
    if cases.is_file():
        for c in json.loads(cases.read_text(encoding="utf-8")):
            split[c["id"]] = c["split"]
    record = _config(folder).get("split_record")
    if isinstance(record, dict):
        for name in ("train", "heldout"):
            if isinstance(record.get(name), list):
                split.update({case: name for case in record[name]})
    for rep in reports:
        for r in rep.get("results", []):
            if "split" in r:
                split[r["case_id"]] = r["split"]
    return split


def _outside(folder: Path, evals: list[dict[str, Any]]) -> list[str]:
    """For a run that did not finish: the conversations no finished evaluation counts (those of the
    evaluation it was in when it stopped, plus any attempt that was retried), per harness, from the traces."""
    if (folder / "result.json").is_file():
        return []
    finished = {str(r.get("run_id")) for e in evals for r in e.get("results", []) if r.get("run_id")}
    runs: dict[str, dict[str, Any]] = {}
    for t in _jsonl(folder / "traces.jsonl"):
        if t["run_id"] not in finished:
            runs[t["run_id"]] = t
    parents = {str(e.get("harness_version")) for e in evals if str(e.get("role", "")).startswith("parent")}
    parents |= {h["version"] for h in _jsonl(folder / "harnesses.jsonl") if h.get("parent") is None}
    split = _split_of(folder, evals)
    groups: dict[str, list[dict[str, Any]]] = {}
    for t in runs.values():
        groups.setdefault(t["harness_version"], []).append(t)
    lines = []
    for version, ts in sorted(groups.items()):
        cells = []
        for name in ("heldout", "train"):
            sel = [t for t in ts if split.get(t["case_id"]) == name]
            cells.append(f"{sum(bool(t['outcome']['passed']) for t in sel)}/{len(sel)}")
        who = "Parent" if version in parents else "Candidate"
        lines.append(
            f"- {who} `{version}`: {len(ts)} recorded conversations that no finished evaluation counts "
            f"(held-out passed {cells[0]}, train passed {cells[1]}). No decision rests on them."
        )
    return lines


def render_run(folder: Path, label: str) -> list[str]:
    """One evolve run as Markdown lines."""
    config = _config(folder)
    reports = _reports(folder)
    evals = [r for r in reports if r.get("type") == "eval"]
    decisions = _jsonl(folder / "decisions.jsonl")
    models = ", ".join(
        f"{k.replace('_', ' ')} `{config[k]}`"
        for k in ("agent_model", "user_model", "improver_model")
        if config.get(k)
    )
    lines = [f"### Evolve run `{label}`", "", f"- {models}." if models else "- Models not recorded."]
    lines.append(f"- {status(folder)}")
    if (folder / NOTE).is_file():
        lines.append(f"- Note (`{NOTE}`): " + " ".join((folder / NOTE).read_text(encoding="utf-8").split()))
    for g in _generations(folder):
        level1 = g.get("level1", {})
        note = f" Note: {g['note']}." if g.get("note") else ""
        lines.append(
            f"- Generation {g.get('generation')}: {level1.get('candidates', 0)} candidates proposed, "
            f"{level1.get('evaluated', 0)} evaluated, {level1.get('admitted', 0)} admitted.{note}"
        )
    keys = sorted({str(d["decision"].get("keyid", ""))[:12] for d in decisions})
    admitted = sum(bool(d["decision"].get("admitted")) for d in decisions)
    if decisions:
        lines.append(
            f"- {len(decisions)} decision{'s' if len(decisions) != 1 else ''}, {admitted} admitted, "
            "signed with key " + ", ".join(f"`{k}`" for k in keys) + "."
        )
    else:
        lines.append("- No decision was made.")
    lines += [
        "",
        "Parent runs (the parent harness measured twice per generation; train is run in the first only):",
        "",
        "| gen | parent | run | held-out passed | train passed | lost to the endpoint |",
        "|---|---|---|---|---|---|",
    ]
    for r in evals:
        role = str(r.get("role", ""))
        if role.startswith("parent_run_"):
            run = role.removeprefix("parent_run_")
            lost = f"{_lost(r)} of {len(r.get('results', []))}"
            lines.append(
                f"| {r.get('generation', '')} | `{r['harness_version']}` | {run} "
                f"| {_count(r, 'heldout')} | {_count(r, 'train')} | {lost} |"
            )
    by_candidate = {str(d["decision"].get("candidate")): d for d in decisions}
    lines += [
        "",
        "Candidates (every evaluated candidate, with its decision):",
        "",
        "| gen | candidate | kind | held-out passed | train passed | lost to the endpoint | decision "
        "| failed checks |",
        "|---|---|---|---|---|---|---|---|",
    ]
    evaluated: set[str] = set()
    for r in evals:
        if r.get("role") != "candidate":
            continue
        version = r["harness_version"]
        evaluated.add(version)
        d = by_candidate.get(version)
        decision = d["decision"] if d else {}
        verdict = ("admitted" if decision.get("admitted") else "rejected") if d else "none recorded"
        failed = [c["name"] for c in decision.get("checks", []) if not c.get("passed")]
        kind = decision.get("context", {}).get("kind", "")
        lines.append(
            f"| {r.get('generation', '')} | `{version}` | {kind} | {_count(r, 'heldout')} "
            f"| {_count(r, 'train')} | {_lost(r)} of {len(r.get('results', []))} | {verdict} "
            f"| {', '.join(failed) or '-'} |"
        )
    for version, d in sorted(by_candidate.items()):
        if version not in evaluated:
            failed = [c["name"] for c in d["decision"].get("checks", []) if not c.get("passed")]
            verdict = "admitted" if d["decision"].get("admitted") else "rejected"
            lines.append(
                f"| | `{version}` | | not recorded | not recorded | | {verdict} "
                f"| {', '.join(failed) or '-'} |"
            )
    if not evaluated and not by_candidate:
        lines.append("| | none | | | | | | |")
    extra = _outside(folder, evals)
    if extra:
        lines += ["", *extra]
    return [*lines, ""]


def render_runs(root: Path, stores: list[Path]) -> list[str]:
    """Every store as a section, labelled by its path under `root`."""
    out: list[str] = []
    for folder in stores:
        out += render_run(folder, str(folder.relative_to(root)))
    return out
