"""Pure recomputation of the admission gates from the stored evidence, written independently of the admission
code. Every function takes plain decoded JSON (the stored report and harness documents) and returns a verdict
with the counts it rests on. Nothing here reads a file, a clock or the network.

The rules are those of docs/CONTRACT.md, "Admission rule", ruling 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ratchet_cases.types import canonical_json, sha256_hex


@dataclass(frozen=True)
class Gate:
    """A recomputed gate: `passed` is what the stored evidence yields; `evidence` are the counts behind it."""

    passed: bool
    evidence: dict[str, Any] = field(default_factory=dict)


def _rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("results")
    if not isinstance(rows, list):
        raise ValueError("eval report has no results list")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError("eval report row has no case_id")
        if row["case_id"] in out:
            raise ValueError(f"eval report repeats case {row['case_id']}")
        out[row["case_id"]] = row
    return out


def _fresh_pass(row: dict[str, Any]) -> bool:
    """Only a FRESH, error-free pass counts as evidence; RECORDED and CACHED never do."""
    return row.get("passed") is True and row.get("mode") == "FRESH" and not row.get("error")


def heldout(p1: dict[str, Any], p2: dict[str, Any], c1: dict[str, Any], c2: dict[str, Any] | None) -> Gate:
    """Stable recoveries are held-out cases failing in P1 and P2 and passing in C1 and C2; at least two are
    required. Every case passing in P1 or P2 is protected and must pass in both candidate runs.

    `c2` is None when the decision names no confirming run: the gate runs C2 only after C1 clears every
    gate, so a candidate refused on C1 has none. The gate cannot pass without C2, and the C1 evidence
    (recoveries and protected cases broken on C1) is still recomputed and reported."""
    reports = (p1, p2, c1) if c2 is None else (p1, p2, c1, c2)
    held = [{k: r for k, r in _rows(rep).items() if r.get("split") == "heldout"} for rep in reports]
    ids = set(held[0])
    same = bool(ids) and all(set(h) == ids for h in held)
    if not same:
        return Gate(False, {"same_case_set": False, "cases": [len(h) for h in held]})
    a, b, c, *rest = ({k: _fresh_pass(r) for k, r in h.items()} for h in held)
    d = rest[0] if rest else None
    stable_c1 = sorted(k for k in ids if not a[k] and not b[k] and c[k])
    stable = [] if d is None else [k for k in stable_c1 if d[k]]
    protected = sorted(k for k in ids if a[k] or b[k])
    broken = sorted(k for k in protected if not c[k] or (d is not None and not d[k]))
    discordant = sorted(k for k in ids if a[k] != b[k])
    return Gate(
        d is not None and len(stable) >= 2 and not broken,
        {
            "same_case_set": True,
            "cases": len(ids),
            "confirm_run": "not run" if d is None else "run",
            "stable_recoveries_c1": stable_c1,
            "stable_recoveries": stable,
            "protected": len(protected),
            "protected_broken": broken,
            "parent_discordant": discordant,
            "required_stable": 2,
        },
    )


def no_regression(parents: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> Gate:
    """No must-succeed case that passed in either parent run may fail in either candidate run."""
    passed_before = {
        k
        for rep in parents
        for k, r in _rows(rep).items()
        if r.get("must_succeed") is True and _fresh_pass(r)
    }
    regressed = sorted(
        {
            k
            for rep in candidates
            for k, r in _rows(rep).items()
            if r.get("must_succeed") is True and k in passed_before and not _fresh_pass(r)
        }
    )
    return Gate(not regressed, {"protected_must_succeed": len(passed_before), "regressed": regressed})


def benign_twins(candidates: list[dict[str, Any]]) -> Gate:
    """No benign-twin case may meet a harness intervention in either candidate run."""
    twins = [(k, r) for rep in candidates for k, r in _rows(rep).items() if r.get("kind") == "benign_twin"]
    hit = sorted({k for k, r in twins if int(r.get("interventions") or 0) > 0})
    return Gate(not hit, {"benign_twin_runs": len(twins), "intervened": hit})


def harness_digest(harness: dict[str, Any]) -> str:
    """The content address of a harness document, recomputed from its fields."""
    content = {
        "parent": harness.get("parent"),
        "system_prompt": harness.get("system_prompt"),
        "procedures": harness.get("procedures"),
        "guards": harness.get("guards"),
        "tool_allowlist": sorted(harness.get("tool_allowlist") or []),
        "created_by": harness.get("created_by"),
    }
    return sha256_hex(canonical_json(content))


def _ids(items: list[dict[str, Any]]) -> list[Any]:
    return [i.get("id") for i in items]


def authority(parent: dict[str, Any], candidate: dict[str, Any]) -> Gate:
    """Candidates append only: tools are a subset, existing guards and procedures are unchanged and in order,
    the parent prompt is a prefix, ids are unique, and the candidate's digest is its content address."""
    p_guards, c_guards = list(parent.get("guards") or []), list(candidate.get("guards") or [])
    p_procs, c_procs = list(parent.get("procedures") or []), list(candidate.get("procedures") or [])
    granted = sorted(set(candidate.get("tool_allowlist") or []) - set(parent.get("tool_allowlist") or []))
    evidence = {
        "digest_valid": harness_digest(candidate) == candidate.get("digest")
        and harness_digest(parent) == parent.get("digest"),
        "names_parent": candidate.get("parent") == parent.get("version"),
        "tools_granted": granted,
        "guards_kept_in_order": c_guards[: len(p_guards)] == p_guards,
        "procedures_kept_in_order": c_procs[: len(p_procs)] == p_procs,
        "prompt_prefix": str(candidate.get("system_prompt", "")).startswith(
            str(parent.get("system_prompt", "")).rstrip()
        ),
        "unique_ids": len(set(_ids(c_guards))) == len(c_guards) and len(set(_ids(c_procs))) == len(c_procs),
        "new_guards": [g.get("id") for g in c_guards[len(p_guards) :]],
    }
    ok = (
        evidence["digest_valid"]
        and evidence["names_parent"]
        and not granted
        and evidence["guards_kept_in_order"]
        and evidence["procedures_kept_in_order"]
        and evidence["prompt_prefix"]
        and evidence["unique_ids"]
    )
    return Gate(bool(ok), evidence)


def replay_matches(claimed: dict[str, Any], recomputed: dict[str, Any]) -> list[str]:
    """Fields of a claimed replay row that the recomputed replay report contradicts. Timings are ignored."""
    targets = recomputed.get("targets") or []
    fields = {
        "passed": recomputed.get("passed"),
        "snapshot_digest": recomputed.get("snapshot_digest"),
        "targets_hit": sum(1 for t in targets if t.get("hit")),
        "targets_total": len(targets),
        "false_fires": sorted(
            f"{f.get('run_id')}@{f.get('seq')}" for f in recomputed.get("false_positives") or []
        ),
        "errors": len(recomputed.get("errors") or []),
    }
    seen = {
        "passed": claimed.get("passed"),
        "snapshot_digest": claimed.get("snapshot_digest"),
        "targets_hit": claimed.get("targets_hit"),
        "targets_total": claimed.get("targets_total"),
        # the signed row (ratchet.improve.admit.check_replay) lists them as "run_id@seq" under false_fires
        "false_fires": sorted(claimed.get("false_fires") or []),
        "errors": claimed.get("errors"),
    }
    return [k for k in fields if fields[k] != seen[k]]
