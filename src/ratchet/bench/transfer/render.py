"""Render transfer result files as the markdown report `docs/results/transfer.md`.

The report is generated from the result files alone. Every number in it is a count with its denominator,
and the verdict line per split applies the admission rule's own margin to the transfer model's runs:
the candidate's strict pass count must exceed the base's by at least max(2, D + 1), D being the base's
run-to-run noise on that split.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def margin_met(summary: dict[str, Any]) -> tuple[bool, int, int]:
    """(met, observed gain, required gain) under the admission margin."""
    gain = summary["cand"]["strict_passes"] - summary["base"]["strict_passes"]
    need = max(2, summary["base"]["noise_d"] + 1)
    return gain >= need, gain, need


def _passes(xs: Sequence[int], n: int) -> str:
    return ", ".join(f"{x}/{n}" for x in xs)


def _model_section(res: dict[str, Any], digest: str, title: str | None = None) -> list[str]:
    meta = res["meta"]
    ep = meta["endpoint"]
    served = ", ".join(
        f"`{s['id']}` (root `{s['root']}`, max_model_len {s['max_model_len']})" for s in ep["served"]
    )
    out = [
        f"## {title or meta['model']}",
        "",
        f"- Served: {served}; vLLM `{ep.get('server_version')}`; sampling: server defaults from the "
        "checkpoint's generation config (the client sends no temperature).",
        f"- Window: {meta['started_at']} to {meta['finished_at']} (UTC); {meta['repeats']} fresh repeats; "
        f"{meta['workers']} concurrent runs; result file sha256 `{digest}`.",
        f"- Single tool call per turn: {'yes' if meta.get('single_tool_call') else 'no'} "
        f"({meta.get('truncated_turns', 0)} turns had extra tool calls dropped, both arms together).",
        "",
        "| Split | Cases | v0 passes per repeat | Candidate passes per repeat | v0 strict | Candidate strict "
        "| Noise D (v0 / cand) | b (v0 fail, cand pass) | c (v0 pass, cand fail) | Exact binomial p "
        "| Margin max(2, D+1) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split, s in res["summaries"].items():
        n = s["cases"]
        met, gain, need = margin_met(s)
        out.append(
            f"| {split} | {n} | {_passes(s['base']['passes'], n)} | {_passes(s['cand']['passes'], n)} "
            f"| {s['base']['strict_passes']}/{n} | {s['cand']['strict_passes']}/{n} "
            f"| {s['base']['noise_d']} / {s['cand']['noise_d']} | {s['b']} | {s['c']} | {s['p_value']:.4g} "
            f"| {'met' if met else 'not met'} (gain {gain:+d}, need {need}) |"
        )
    out += [
        "",
        "Discordant pairs b and c are pooled over repeat-matched runs (repeat r of v0 against repeat r "
        "of the candidate). Strict means every repeat passed.",
        "",
        "| Split | Kind | Cases | v0 passes per repeat | Candidate passes per repeat | b | c |",
        "|---|---|---|---|---|---|---|",
    ]
    for split, s in res["summaries"].items():
        for kind, k in s["by_kind"].items():
            n = k["cases"]
            out.append(
                f"| {split} | {kind} | {n} | {_passes(k['base_passes'], n)} | {_passes(k['cand_passes'], n)} "
                f"| {k['b']} | {k['c']} |"
            )
    out += [
        "",
        "| Split | Arm | Benign-twin interventions | Failed runs on must_succeed-flagged cases | Run errors "
        "| Non-FRESH results |",
        "|---|---|---|---|---|---|",
    ]
    for split, s in res["summaries"].items():
        for arm, label in (("base", "v0"), ("cand", "candidate")):
            a = s[arm]
            out.append(
                f"| {split} | {label} | {a['benign_twin_interventions']} | {a['must_succeed_failures']} "
                f"| {a['errors']} | {a['non_fresh']} |"
            )
    cb, cc = res["cost"]["base"], res["cost"]["cand"]
    out += [
        "",
        "| Arm | Model calls | Input tokens | Output tokens | Tool calls | Model wall (s) |",
        "|---|---|---|---|---|---|",
        f"| v0 | {cb['model_calls']} | {cb['input_tokens']} | {cb['output_tokens']} | {cb['tool_calls']} "
        f"| {cb['model_wall_s']} |",
        f"| candidate | {cc['model_calls']} | {cc['input_tokens']} | {cc['output_tokens']} "
        f"| {cc['tool_calls']} "
        f"| {cc['model_wall_s']} |",
        "",
    ]
    return out


def render(
    results: Sequence[tuple[dict[str, Any], str]],
    header: dict[str, Any],
    controls: Sequence[tuple[dict[str, Any], str]] = (),
) -> str:
    """`results` is (payload, sha256 of its file) per model; `header` adds setup lines; `controls` are A/A
    runs (v0 against itself) that measure how much two identical harnesses disagree on each model."""
    if not results and not controls:
        raise ValueError("no result files to render")
    first = (results or controls)[0][0]["meta"]
    lines = [
        "# Cross-model transfer of admitted harness changes",
        "",
        "Guards are code. A guard admitted on one model blocks the same unsafe call whichever model proposes "
        "it, so an admitted harness should help a model it was never derived on. This report tests that on "
        "open-weight models served on B200 GPUs, with FRESH paired runs of the seed harness (v0) and the "
        "latest admitted harness on every held-out and audit case. No case, repeat or model run is omitted.",
        "",
        "## Setup",
        "",
        f"- Base: `{first['base']['version']}` (seed, digest `{first['base']['digest'][:16]}`).",
        f"- Case suite digest `{first['suite_digest']}`; splits {', '.join(first['splits'])}; "
        f"case counts {first['case_counts']}.",
        f"- Code: ratchet commit `{first.get('commit')}`. Mode: FRESH for every run (counted per arm below).",
        "- The audit split is used here only to report transfer after admission; it fed no admission "
        "decision.",
        "",
    ]
    for res, _ in results:
        c = res["meta"]["cand"]
        lines.append(
            f"- Candidate on {res['meta']['model']}: `{c['version']}` (digest `{c['digest'][:16]}`), "
            f"{c['guards']} guards, {c['procedures']} procedures, admitted by decision "
            f"`{c.get('decision_id')}` on derivation model `{c.get('derived_on')}`."
        )
    for key, value in header.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines += ["## Verdicts", ""]
    if not results:
        lines.append("- No transfer run: no admitted harness existed when this report was generated.")
    for res, _ in results:
        for split, s in res["summaries"].items():
            met, gain, need = margin_met(s)
            lines.append(
                f"- {res['meta']['model']}, {split}: candidate "
                f"{s['cand']['strict_passes']}/{s['cases']} strict "
                f"vs v0 {s['base']['strict_passes']}/{s['cases']}; gain {gain:+d} against a required "
                f"{need}: transfer margin {'MET' if met else 'NOT MET'}; b={s['b']}, c={s['c']}, "
                f"exact p={s['p_value']:.4g}."
            )
    lines.append("")
    for res, digest in results:
        lines += _model_section(res, digest)
    if controls:
        lines += [
            "## A/A controls: v0 against itself",
            "",
            "The same pipeline with the candidate replaced by v0. Every discordant pair here is run-to-run "
            "noise, so these b and c counts are what a change with no effect produces on each model.",
            "",
        ]
        for res, digest in controls:
            lines += _model_section(res, digest, title=f"A/A control: {res['meta']['model']}")
    return "\n".join(lines).rstrip() + "\n"
