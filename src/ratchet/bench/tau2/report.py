"""The results page and the leaderboard submission package, both generated from recorded runs.

`report` turns every run under `--out` into a Markdown page: setup, then per run the planned and recorded
conversations, successes, tau2-bench's pass^k, pass^k with endpoint-lost units counted as failures, and
the per-task success counts with exact task ids. No number on the page is typed by hand.

`package` builds a custom-type leaderboard submission from finished runs, one per domain: tau2-bench's
`Submission` model (validated by tau2-bench's own pydantic schema) with pass^k computed by tau2-bench's
`get_metrics`, the domain trajectory files in tau2-bench's single-file format, and methodology notes
generated from the frozen harness (version, digest, guards and procedures) and the admission receipts
given on the command line. It refuses runs that break the leaderboard's rules (a domain twice, a task
missing, fewer than four trials, differing model configurations) unless `--draft` is passed, and a draft
says so in its notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import ratchet.bench.tau2  # noqa: F401  # sets TAU2_DATA_DIR before tau2-bench is imported
from ratchet.bench.tau2.results import by_task, load_results, read_jsonl, summarize_run
from ratchet.bench.tau2.tasks import load_tasks

MIN_TRIALS = 4


def _all_runs(out: Path) -> list[Path]:
    return sorted(p.parent for p in Path(out).glob("runs/*/*/*/config.json"))


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def run_section(root: Path) -> list[str]:
    s = summarize_run(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    rows = read_jsonl(root / "runs.jsonl")
    lines = [
        f"### {s['domain']}, harness `{s['harness']}`, run `{root.name}`",
        "",
        f"- Agent model `{s['agent_model']}`, user simulator `{s['user_model']}`, assertion judge "
        f"`{config['judge_model']}`, temperature {config['temperature']}, seed {config['seed']} "
        f"(trial seeds {config['trial_seeds']}).",
        f"- Tasks: {s['tasks']} of the domain's base set; trials: {s['trials']}; conversations planned "
        f"{s['units_planned']}, recorded {s['units_recorded']}, completed {s['simulations']}, lost to the "
        f"model endpoint after {config['infra_retries']} retries {s['lost_to_endpoint']}.",
        f"- Successes: {s['successes']} of {s['units_recorded']} recorded conversations.",
    ]
    tau2 = s.get("tau2")
    if tau2:
        ks = " | ".join(f"pass^{k} {_pct(v)}" for k, v in sorted(tau2["pass_hat_k"].items()))
        lines.append(f"- tau2-bench metrics over the {s['simulations']} completed conversations: {ks}.")
    strict = s.get("pass_hat_k_lost_as_fail")
    if strict:
        ks = " | ".join(f"pass^{k} {_pct(v)}" for k, v in sorted(strict.items()))
        lines.append(
            f"- pass^k over all {s['units_recorded']} recorded units, a lost unit counted as a failure: {ks}."
        )
    else:
        lines.append(
            "- pass^k over all units: not computed, the run is not complete for every task and trial."
        )
    grouped = by_task(rows)
    lines += ["", "| task id | split | successes / trials |", "|---|---|---|"]
    split = {r["task_id"]: r["split"] for r in rows}
    for task in config["task_ids"]:
        runs = grouped.get(task, [])
        lines.append(f"| `{task}` | {split.get(task, '')} | {sum(runs)} / {len(runs)} |")
    return [*lines, ""]


def render_report(out: Path) -> str:
    runs = _all_runs(out)
    lines = [
        "# tau2-bench results",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `python -m ratchet.bench.tau2 report` from "
        "the run records under the run directory; every number below is computed from those records.",
        "",
        "Each conversation runs tau2-bench's own orchestrator, environment, user simulator and evaluator; "
        "only the agent's seat is taken by the Ratchet worker loop under the named harness. pass^k is "
        "tau2-bench's: the mean over tasks of C(c, k) / C(n, k) for c successes in n trials.",
        "",
    ]
    if not runs:
        return "\n".join([*lines, "No runs recorded yet.", ""])
    first = json.loads((runs[0] / "config.json").read_text(encoding="utf-8"))
    lines += [
        f"tau2-bench commit `{first['tau2_commit_installed']}` (data digest per run in its `config.json`).",
        "",
        "## Runs",
        "",
    ]
    for r in runs:
        lines += run_section(r)
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> int:
    text = render_report(Path(args.out))
    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


# -- leaderboard package ----------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_runs(runs: list[Path]) -> list[str]:
    """Every leaderboard rule the runs break (empty when they qualify)."""
    problems: list[str] = []
    configs = [json.loads((r / "config.json").read_text(encoding="utf-8")) for r in runs]
    domains = [c["domain"] for c in configs]
    for d in sorted(set(domains)):
        if domains.count(d) > 1:
            problems.append(f"domain {d} appears {domains.count(d)} times")
    keys = ("agent_model", "user_model", "judge_model", "temperature", "user_temperature")
    shapes = {json.dumps({k: c.get(k) for k in keys}, sort_keys=True) for c in configs}
    if len(shapes) > 1:
        problems.append(f"model configurations differ across domains: {sorted(shapes)}")
    harnesses = {c["harness"]["version"] for c in configs}
    if len(harnesses) > 1:
        problems.append(f"harness versions differ across domains: {sorted(harnesses)}")
    for r, c in zip(runs, configs, strict=True):
        s = summarize_run(r)
        total = len(load_tasks(c["domain"]))
        if c["tasks"] != total:
            problems.append(f"{c['domain']}: {c['tasks']} of {total} base tasks")
        if c["trials"] < MIN_TRIALS:
            problems.append(f"{c['domain']}: {c['trials']} trials, the leaderboard asks for {MIN_TRIALS}")
        if s["simulations"] != s["units_planned"]:
            problems.append(
                f"{c['domain']}: {s['simulations']} of {s['units_planned']} conversations completed"
            )
    return problems


def methodology_notes(runs: list[Path], receipts: list[Path], draft: list[str]) -> str:
    """Notes generated from the frozen harness and the run records."""
    configs = [json.loads((r / "config.json").read_text(encoding="utf-8")) for r in runs]
    c = configs[0]
    harness = json.loads((runs[0] / "harness.json").read_text(encoding="utf-8"))
    parts = [
        f"Agent: the Ratchet harness {harness['version']} (sha256 {harness['digest']}, created by "
        f"{harness['created_by']}, parent {harness['parent']}) driving {c['agent_model']} at temperature "
        f"{c['temperature']}. The harness's system prompt starts with tau2-bench's own half-duplex agent "
        "prompt and domain policy; Ratchet adds its procedures index and runs every tool call through the "
        "harness's tool allowlist and guards in a sandboxed guard runtime before the environment sees it. "
        "A blocked call is returned to the model as an error and never reaches the environment.",
        f"Guards ({len(harness['guards'])}): "
        + ("; ".join(f"{g['id']}: {g['description']}" for g in harness["guards"]) or "none")
        + ".",
        f"Procedures ({len(harness['procedures'])}): "
        + ("; ".join(f"{p['id']} ({p['name']}): when {p['when']}" for p in harness["procedures"]) or "none")
        + ".",
        "Benchmark side unchanged: tau2-bench's orchestrator, environments, tools, user simulator "
        f"({c['user_model']}, temperature {c['user_temperature']}), step limit {c['max_steps']}, error "
        f"limit {c['max_errors']}, trial seeds from seed {c['seed']}, and evaluate_simulation (evaluation "
        "type all). The agent's seat is a puppet agent fed by the Ratchet worker loop; each agent tool call "
        "is its own assistant message. The natural-language assertion judge (retail) is "
        f"{c['judge_model']} instead of the default gpt-4.1, which the evaluation endpoint does not serve.",
        f"Infrastructure: a conversation lost to the model endpoint is re-run at most {c['infra_retries']} "
        "times (predeclared); completed conversations are never re-run. Every attempt is kept in the run "
        "records.",
        f"tau2-bench commit {c['tau2_commit_installed']}; task data digests: "
        + ", ".join(f"{k['domain']} {k['tau2_data_digest'][:16]}" for k in configs)
        + f"; Ratchet commit {c['ratchet_commit']}.",
    ]
    if receipts:
        parts.append(
            "Admission receipts (signed DSSE envelopes): "
            + ", ".join(f"{p.name} sha256 {_sha256(p)}" for p in receipts)
            + "."
        )
    if draft:
        parts.append("DRAFT, not eligible as submitted: " + "; ".join(draft) + ".")
    return " ".join(parts)


def build_package(
    runs: list[Path],
    dest: Path,
    *,
    organization: str,
    model_organization: str,
    email: str,
    references: list[tuple[str, str]],
    receipts: list[Path],
    draft: bool,
) -> dict[str, Any]:
    from tau2.scripts.leaderboard.prepare_submission import (  # type: ignore[import-untyped]  # no types
        get_metrics,
        validate_submission_traj_set,
    )
    from tau2.scripts.leaderboard.submission import (  # type: ignore[import-untyped]  # no type info
        ContactInfo,
        Methodology,
        Reference,
        Results,
        Submission,
        Verification,
    )

    problems = check_runs(runs)
    if problems and not draft:
        raise SystemExit("not a qualifying submission: " + "; ".join(problems))
    loaded = [load_results(r) for r in runs]
    ok, why = validate_submission_traj_set(loaded)
    if not ok:
        raise SystemExit(f"tau2-bench rejects the trajectory set: {why}")
    _, domain_results, model, user_sim = get_metrics(loaded)
    traj_dir = dest / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for res in loaded:
        domain = res.info.environment_info.domain_name
        version = res.info.agent_info.implementation.split(":")[-1]
        name = f"{model}_{domain}_ratchet-{version}_{res.info.num_trials}trials.json"
        res.save(traj_dir / name, format="json")
        files[domain] = name
    configs = [json.loads((r / "config.json").read_text(encoding="utf-8")) for r in runs]
    harness_changed = any(
        json.loads((r / "harness.json").read_text(encoding="utf-8"))["procedures"] for r in runs
    )
    submission = Submission(
        model_name=model,
        model_organization=model_organization,
        submitting_organization=organization,
        submission_date=time.strftime("%Y-%m-%d"),
        submission_type="custom",
        modality="text",
        contact_info=ContactInfo(email=email),
        results=Results(**domain_results),
        trajectories_available=True,
        trajectory_files=files,
        references=[Reference(title=t, url=u, type="github") for t, u in references] or None,
        methodology=Methodology(
            evaluation_date=time.strftime("%Y-%m-%d"),
            tau2_bench_version=f"1.0.1 (commit {configs[0]['tau2_commit_installed']})",
            user_simulator=user_sim,
            notes=methodology_notes(runs, receipts, problems),
            verification=Verification(
                modified_prompts=harness_changed,
                omitted_questions=any("base tasks" in p for p in problems),
                details="Agent prompt is tau2-bench's own prompt plus the harness's procedures index"
                if harness_changed
                else "Agent prompt is tau2-bench's own agent prompt and domain policy, unmodified",
            ),
        ),
    )
    (dest / "submission.json").write_text(
        submission.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
    )
    return {"dest": str(dest), "domains": sorted(files), "problems": problems}


def cmd_package(args: argparse.Namespace) -> int:
    refs = [tuple(r.split("=", 1)) for r in args.reference]
    out = build_package(
        [Path(r) for r in args.runs],
        Path(args.dest),
        organization=args.organization,
        model_organization=args.model_organization,
        email=args.email,
        references=[(t, u) for t, u in refs],
        receipts=[Path(p) for p in args.receipt],
        draft=args.draft,
    )
    print(json.dumps(out, indent=2))
    return 0


def add_report_parser(sub: Any) -> None:
    p = sub.add_parser("report")
    p.add_argument("--write", default="", help="write the page here instead of printing it")
    p.set_defaults(func=cmd_report)


def add_package_parser(sub: Any) -> None:
    p = sub.add_parser("package")
    p.add_argument("runs", nargs="+", help="run directories, one per domain")
    p.add_argument("--dest", required=True)
    p.add_argument("--organization", required=True)
    p.add_argument("--model-organization", required=True)
    p.add_argument("--email", required=True)
    p.add_argument("--reference", action="append", default=[], help="TITLE=URL, repeatable")
    p.add_argument("--receipt", action="append", default=[], help="admission receipt file, repeatable")
    p.add_argument("--draft", action="store_true", help="package runs that do not qualify, marked as a draft")
    p.set_defaults(func=cmd_package)
