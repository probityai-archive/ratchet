"""Derive every id and number the demo and the posts quote, from Atlas and the run directories.

    uv run python docs/demo/derive_ids.py [--uri URI] [--db DB] [--level2 FILE] [--no-deep-verify]

The demo has two outcomes, chosen from what the store holds, and DEMO_OUTCOME names it:
  admitted  the store (--db, else RATCHET_DB, else the ratchet* database with the most admitted decisions,
            then the most decisions) holds an admitted decision whose receipt verifies
  refused   no store holds one; the admitted keys (ADMITTED_*, CLEARED_CAND, STABLE_*, PROTECTED_N,
            DISCORDANCE, HELDOUT_N) are then absent, and every other key is still required
Both outcomes show the refusal scene: REFUSED_* is the rejected decision closest to admission across the
live-model databases (every ratchet* database whose name does not start with `ratchet_demo`).

Writes `docs/demo/ids.env` (git-ignored, one `# source:` comment above each value) and the two short
summaries the demo shows (`docs/demo/out/results-summary.txt`, `docs/demo/out/level2-summary.txt`). Nothing
is typed by hand: each value is read from the store, from the stored result pages, or from the level-2
result file, and the file names where.

When any value does not exist yet, every missing one is printed with its reason, the command exits 1, and
the existing `ids.env` is deleted so a take can never be recorded against stale ids.

The store comes from `--uri/--db` or `RATCHET_MONGODB_URI/RATCHET_DB`. Choices, in order:
  ADMITTED_DECISION  newest admitted decision with a `fresh` gate whose receipt verifies (`ratchet verify`
                     must exit 0 on it unless --no-deep-verify)
  CLEARED_CAND       newest admitted guard candidate that has a rejected-on-replay guard beside it
  REJECTED_CAND      the newest guard candidate of the same parent that replay rejected, preferring one
                     that blocked a recorded success
  AUTHORITY_DECISION newest rejected decision whose failed gates include authority
  BAD_TRACE          newest failed refund trace of that parent with a timeout then a retried call
  RESUME_RUN         newest checkpoint still `running`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "out"
IDS = HERE / "ids.env"
RESULTS = ROOT / "docs" / "results"
LEVEL2_DEFAULTS = (RESULTS / "level2.json", ROOT / "private" / "improve" / "l2.json")
EVOLVE_LOG = "docs/demo/out/evolve.log"
GATE_REPLAY, GATE_FRESH = "replay", "fresh"
AUTHORITY_GATES = ("authority", "structure")
BAD_TRACE_FAMILY = "refund_lost_reply"
OUTCOME_ADMITTED, OUTCOME_REFUSED = "admitted", "refused"
ADMITTED_ONLY = (
    "ADMITTED_DECISION",
    "ADMITTED_CAND",
    "CLEARED_CAND",
    "STABLE_RECOVERIES",
    "STABLE_NEEDED",
    "PROTECTED_N",
    "DISCORDANCE",
    "HELDOUT_N",
)
DEMO_DB_PREFIX = "ratchet_demo"
LEVEL2_DB = "ratchet_l2"
LEVEL2_ID = re.compile(r"^(level2-[^:]+):([^:]+):")

CaseInfoFn = Callable[[str], Any]


def rel(path: Path) -> str:
    """The path relative to the repository when it is inside it, else as given."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass
class Derived:
    """Values with where each came from, and every reason a value could not be derived."""

    values: dict[str, tuple[str, str]] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)

    def set(self, key: str, value: object, source: str) -> None:
        self.values[key] = (str(value), source)

    def fail(self, key: str, reason: str) -> None:
        self.missing[key] = reason


# -- the store ------------------------------------------------------------------------------------------


def _failed(doc: Mapping[str, Any]) -> list[str]:
    return [str(c.get("name")) for c in doc.get("checks") or [] if not c.get("passed")]


def _check(doc: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    return next((c for c in doc.get("checks") or [] if c.get("name") == name), None)


def _newest(docs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(docs, key=lambda d: str(d.get("decided_at", "")), reverse=True)


def decision_docs(store: Any) -> list[Mapping[str, Any]]:
    return [d for d in store.decisions() if d.get("type", "decision") == "decision"]


def pick_admitted(
    docs: Sequence[Mapping[str, Any]], verifies: Callable[[str], str | None]
) -> tuple[Mapping[str, Any] | None, str]:
    """The newest admitted decision with a fresh gate and a receipt that verifies. `verifies(id)` returns None
    when it does, else the reason it does not."""
    reasons: list[str] = []
    for d in _newest([d for d in docs if d.get("admitted")]):
        if _check(d, GATE_FRESH) is None:
            reasons.append(f"{d.get('_id')}: no `fresh` gate recorded")
        elif not d.get("envelope"):
            reasons.append(f"{d.get('_id')}: no signed receipt")
        elif (why := verifies(str(d["_id"]))) is not None:
            reasons.append(f"{d.get('_id')}: receipt does not verify ({why})")
        else:
            return d, ""
    if not reasons:
        return None, "no admitted decision is stored yet"
    return None, "; ".join(reasons[:3])


def replay_rejected(doc: Mapping[str, Any]) -> bool:
    return not doc.get("admitted") and GATE_REPLAY in _failed(doc)


def blocked_a_success(doc: Mapping[str, Any]) -> bool:
    """True when the decision's replay gate records a false fire on a passing or benign-twin trace."""
    ev = (_check(doc, GATE_REPLAY) or {}).get("evidence") or {}
    reports = [*(ev.get("reports") or []), *([ev["composed"]] if ev.get("composed") else [])]
    for r in reports:
        fires = r.get("false_fires", r.get("false_positives", 0))
        n = len(fires) if isinstance(fires, list) else int(fires or 0)
        if n or int(r.get("benign_twin_fires", 0)):
            return True
    return False


def has_guards(store: Any, resolve: Callable[[Any, Sequence[str]], Sequence[Any]], candidate: str) -> bool:
    refs = resolve(store, [candidate])
    return bool(refs) and bool(refs[0].guards)


def pick_pair(
    store: Any,
    docs: Sequence[Mapping[str, Any]],
    resolve: Callable[[Any, Sequence[str]], Sequence[Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    """(cleared, rejected): an admitted guard candidate and a replay-rejected guard candidate of the same
    parent, so one matrix shows both."""
    admitted = [
        d for d in _newest(docs) if d.get("admitted") and has_guards(store, resolve, str(d["candidate"]))
    ]
    for a in admitted:
        rejected = [
            d
            for d in _newest(docs)
            if replay_rejected(d)
            and d.get("parent") == a.get("parent")
            and has_guards(store, resolve, str(d["candidate"]))
        ]
        if rejected:
            rejected.sort(key=lambda d: not blocked_a_success(d))  # stable: newest first within each group
            return a, rejected[0]
    return None


def timeout_then_retry(events: Sequence[Any]) -> bool:
    """A tool call whose result was a timeout, followed by a second call to the same tool."""
    calls: list[str] = []
    timed_out: set[str] = set()
    for e in events:
        p = e.payload
        if e.kind == "tool_call":
            name = str((p.get("call") or p).get("name"))
            if name in timed_out:
                return True
            calls.append(name)
        elif e.kind == "tool_result" and (p.get("result") or p).get("error") == "timeout" and calls:
            timed_out.add(calls[-1])
    return False


def pick_bad_trace(
    store: Any, parent: str, case_info: CaseInfoFn, family: str | None = None, need_retry: bool = True
) -> str | None:
    """The newest failed trap trace of `family` for `parent`; with `need_retry`, only one where a timed-out
    tool call was retried."""
    family = family or BAD_TRACE_FAMILY
    for t in store.traces({"harness_version": parent}, newest_first=True):
        if t.outcome.passed:
            continue
        info = case_info(t.case_id)
        if (
            info is not None
            and info.family == family
            and info.kind == "trap"
            and (not need_retry or timeout_then_retry(t.events))
        ):
            return f"{t.run_id}/{t.case_id}"
    return None


def pick_replay_rows(store: Any, parent: str, case_info: CaseInfoFn, pick_rows: Callable[..., Any]) -> int:
    """How many traces the replay matrix covers for this parent: the newest trace of each case."""
    picked, _rows = pick_rows(store.traces({"harness_version": parent}), case_info)
    return len(picked)


def running_checkpoint(store: Any) -> str | None:
    cursor = store.db["checkpoints"].find({"state.status": "running"}).sort("updated_at", -1).limit(1)
    for doc in cursor:
        return str(doc["_id"])
    return None


def derive_authority_and_resume(store: Any, docs: Sequence[Mapping[str, Any]], out: Derived) -> None:
    authority = [d for d in _newest(docs) if not d.get("admitted") and set(_failed(d)) & set(AUTHORITY_GATES)]
    if authority:
        out.set(
            "AUTHORITY_DECISION",
            authority[0]["candidate"],
            f"candidate of decision {authority[0]['_id']}, "
            "the newest rejected decision that failed an authority gate",
        )
    else:
        out.fail("AUTHORITY_DECISION", "no rejected decision failed an authority gate")
    run = running_checkpoint(store)
    if run:
        out.set("RESUME_RUN", run, "newest checkpoint with status running")
        out.set("RESUME_CMD", f"ratchet resume {run}", "built from RESUME_RUN")
    else:
        out.fail("RESUME_RUN", "no checkpoint is still `running` (kill a run mid-case, then derive again)")
        out.fail("RESUME_CMD", "needs RESUME_RUN")


def pick_rejected(
    store: Any,
    docs: Sequence[Mapping[str, Any]],
    resolve: Callable[[Any, Sequence[str]], Sequence[Any]],
    parent: str | None,
) -> Mapping[str, Any] | None:
    """The newest replay-rejected guard candidate of `parent`, preferring one that blocked a success."""
    rejected = [
        d
        for d in _newest(docs)
        if replay_rejected(d)
        and (parent is None or d.get("parent") == parent)
        and has_guards(store, resolve, str(d["candidate"]))
    ]
    rejected.sort(key=lambda d: not blocked_a_success(d))
    return rejected[0] if rejected else None


@dataclass(frozen=True)
class Closeness:
    """How near a rejected decision came to admission, read from its fresh-evaluation gate."""

    doc: Mapping[str, Any]
    recoveries: list[str]
    recovery_runs: str
    protected: list[str]
    kept: list[str]
    regressed: list[str]
    discordant: list[str]
    heldout_n: int
    confirm_invalid: bool

    def rank(self) -> tuple[int, bool, int, int]:
        return (
            len(self.recoveries),
            self.kept == self.protected,
            -len(self.regressed),
            -len(_failed(self.doc)),
        )


def closeness(doc: Mapping[str, Any]) -> Closeness | None:
    gate = _check(doc, "heldout") or _check(doc, GATE_FRESH)
    ev = (gate or {}).get("evidence") or {}
    outcomes, order = ev.get("outcomes") or {}, list(ev.get("order") or [])
    if doc.get("admitted") or not outcomes or "C1" not in order:
        return None
    c1 = order.index("C1")
    both, first = list(ev.get("stable_recoveries") or []), list(ev.get("stable_recoveries_c1") or [])
    recoveries, runs = (
        (both, "both fresh runs") if len(both) >= len(first) else (first, "its first fresh run")
    )
    protected = list(ev.get("protected") or [])
    kept = [c for c in protected if (outcomes.get(c) or [None] * (c1 + 1))[c1] == "pass"]
    confirm_invalid = "C2" in order and all(
        (v[order.index("C2")] if len(v) > order.index("C2") else "invalid") == "invalid"
        for v in outcomes.values()
    )
    regressed = list(((_check(doc, "no_regression") or {}).get("evidence") or {}).get("regressed") or [])
    return Closeness(
        doc,
        recoveries,
        runs,
        protected,
        kept,
        regressed,
        list(ev.get("discordant") or []),
        int(ev.get("manifest") or len(outcomes)),
        confirm_invalid,
    )


def pick_closest(dbs: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[str, Closeness] | None:
    """The rejected decision closest to admission across databases: most recoveries, then every protected
    case kept, then fewest regressed must-succeed cases, then fewest failed gates."""
    scored = [(db, c) for db, docs in dbs.items() for d in docs if (c := closeness(d)) is not None]
    if not scored:
        return None
    return max(scored, key=lambda dc: (dc[1].rank(), str(dc[1].doc.get("decided_at", ""))))


def derive_refusal(out: Derived, picked: tuple[str, Closeness] | None) -> None:
    keys = (
        "REFUSED_DB",
        "REFUSED_CAND",
        "REFUSED_DECISION",
        "REFUSED_RECOVERIES",
        "REFUSED_RECOVERY_RUNS",
        "REFUSED_PROTECTED_N",
        "REFUSED_KEPT",
        "REFUSED_WHY",
        "REFUSED_HELDOUT_N",
        "PARENT_DISCORDANT",
        "PARENT_DISCORDANT_CASES",
    )
    if picked is None:
        for key in keys:
            out.fail(key, "no rejected decision records per-case fresh outcomes")
        return
    db, c = picked
    src = f"{db}, decision {c.doc['_id']}, fresh-evaluation gate"
    out.set("REFUSED_DB", db, "the live-model database holding the closest rejected decision")
    out.set("REFUSED_CAND", c.doc["candidate"], f"{src}: closest to admission of every live-model rejection")
    out.set("REFUSED_DECISION", c.doc["_id"], src)
    out.set("REFUSED_RECOVERIES", len(c.recoveries), src)
    out.set("REFUSED_RECOVERY_RUNS", c.recovery_runs, src)
    out.set("REFUSED_PROTECTED_N", len(c.protected), src)
    out.set("REFUSED_KEPT", len(c.kept), src)
    out.set("REFUSED_HELDOUT_N", c.heldout_n, src)
    why = []
    if c.regressed:
        n = len(c.regressed)
        why.append(f"{n} must-succeed case{'s' if n > 1 else ''} failed ({', '.join(c.regressed)})")
    if c.confirm_invalid:
        why.append("its confirmation run was invalid")
    if not why:
        why.append("failed " + ", ".join(_failed(c.doc)))
    out.set("REFUSED_WHY", " and ".join(why), src)
    out.set("PARENT_DISCORDANT", len(c.discordant), f"{src}: cases the two parent runs disagree on")
    out.set("PARENT_DISCORDANT_CASES", ", ".join(c.discordant) or "none", src)


def derive_level2_from_store(
    out: Derived, docs: Sequence[Mapping[str, Any]], db: str, out_dir: Path = OUT
) -> None:
    """Level 2 from the stored decisions of the level-2 runs (`level2-<run>:<arm>:...` ids), newest run that
    has both a seed and a search arm."""
    runs: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    newest: dict[str, str] = {}
    for d in docs:
        m = LEVEL2_ID.match(str(d.get("_id", "")))
        if m:
            runs.setdefault(m.group(1), {}).setdefault(m.group(2), []).append(d)
            newest[m.group(1)] = max(newest.get(m.group(1), ""), str(d.get("decided_at", "")))
    full = [r for r in runs if {"seed", "search"} <= set(runs[r])]
    if not full:
        for key in ("LEVEL2_FILE", "LEVEL2_SEARCH_YIELD", "LEVEL2_SEED_YIELD"):
            out.fail(
                key, f"no level-2 result file, and {db} records no run with both a seed and a search arm"
            )
        return
    run = max(full, key=lambda r: newest[r])
    head = f"{'arm':<12}{'proposed':>10}{'replay_cleared':>16}{'admitted':>10}{'yield':>8}"
    lines = [f"Level 2: improver instructions, same gate (Atlas {db}, run {run}, live models)", head]
    yields: dict[str, str] = {}
    for name in sorted(runs[run]):
        arm_docs = runs[run][name]
        cleared = sum(1 for d in arm_docs if (_check(d, GATE_REPLAY) or {}).get("passed"))
        admitted = sum(1 for d in arm_docs if d.get("admitted"))
        yields[name] = f"{admitted}/{len(arm_docs)}"
        lines.append(f"{name:<12}{len(arm_docs):>10}{cleared:>16}{admitted:>10}{yields[name]:>8}")
    out_dir.mkdir(exist_ok=True)
    summary = out_dir / "level2-summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    src = f"Atlas {db}, decisions of run {run}"
    out.set("LEVEL2_FILE", rel(summary), f"built from {src}")
    out.set("LEVEL2_SEARCH_YIELD", yields["search"], f"{src}, arm search: admitted/proposed")
    out.set("LEVEL2_SEED_YIELD", yields["seed"], f"{src}, arm seed: admitted/proposed")


def derive_from_store(
    store: Any,
    out: Derived,
    *,
    case_info: CaseInfoFn,
    verifies: Callable[[str], str | None],
    resolve: Callable[[Any, Sequence[str]], Sequence[Any]],
    pick_rows: Callable[..., Any],
    fresh_view: Callable[[Mapping[str, Any]], Any],
    refused_parent: str | None = None,
) -> None:
    """Every store value. With no admitted decision, the replay keys come from `refused_parent` (the parent
    of the refusal scene's candidate) and the admitted keys are left absent."""
    docs = decision_docs(store)
    out.set("N_DECISIONS", len(docs), "count of documents in `decisions`")
    out.set("N_REJECTED", sum(1 for d in docs if not d.get("admitted")), "decisions with admitted=false")
    derive_authority_and_resume(store, docs, out)

    admitted, why = pick_admitted(docs, verifies)
    if admitted is None:
        out.set("DEMO_OUTCOME", OUTCOME_REFUSED, f"no admitted decision verifies: {why}")
    else:
        out.set("DEMO_OUTCOME", OUTCOME_ADMITTED, f"decision {admitted['_id']} is admitted and verifies")
        out.set("ADMITTED_CAND", admitted["candidate"], f"candidate of decision {admitted['_id']}")
        view = fresh_view((_check(admitted, GATE_FRESH) or {}).get("evidence") or {})
        src = f"decision {admitted['_id']}, `fresh` gate"
        out.set(
            "ADMITTED_DECISION", admitted["_id"], f"newest admitted decision with a verifying receipt ({src})"
        )
        if view is None:
            out.fail("STABLE_RECOVERIES", f"{admitted['_id']} records no per-case outcome vectors")
        elif view.confirm_pending or view.broken or view.invalid:
            out.fail("STABLE_RECOVERIES", f"{admitted['_id']}: fresh gate is pending, broken or invalid")
        else:
            out.set("STABLE_RECOVERIES", len(view.stable), src)
            out.set("STABLE_NEEDED", view.needed, src)
            out.set("PROTECTED_N", len(view.protected), src)
            out.set("DISCORDANCE", view.discordance, src)
            out.set("HELDOUT_N", len(view.rows), src)

    pair = pick_pair(store, docs, resolve) if admitted is not None else None
    if admitted is None:
        rejected_only = pick_rejected(store, docs, resolve, refused_parent)
        if rejected_only is None:
            reason = f"no replay-rejected guard candidate of parent {refused_parent}"
            for key in ("REJECTED_CAND", "REPLAY_TRACES", "BAD_TRACE"):
                out.fail(key, reason)
            return
        pair = (rejected_only, rejected_only)
    if pair is None:
        reason = "no admitted guard candidate has a replay-rejected guard candidate of the same parent"
        for key in ("CLEARED_CAND", "REJECTED_CAND", "REPLAY_TRACES", "BAD_TRACE"):
            out.fail(key, reason)
    else:
        cleared, rejected = pair
        if admitted is not None:
            out.set(
                "CLEARED_CAND", cleared["candidate"], f"admitted guard candidate, decision {cleared['_id']}"
            )
        out.set(
            "REJECTED_CAND",
            rejected["candidate"],
            f"replay-rejected guard candidate of the same parent, decision {rejected['_id']}"
            + (" (blocked a recorded success)" if blocked_a_success(rejected) else ""),
        )
        parent = str(cleared["parent"])
        count = pick_replay_rows(store, parent, case_info, pick_rows)
        if count:
            out.set("REPLAY_TRACES", count, f"newest trace of each case recorded for parent {parent}")
        else:
            out.fail("REPLAY_TRACES", f"no traces recorded for parent {parent}")
        bad = pick_bad_trace(store, parent, case_info, need_retry=admitted is not None)
        if bad:
            out.set("BAD_TRACE", bad, f"newest failed {BAD_TRACE_FAMILY} trap trace of parent {parent}")
        else:
            out.fail(
                "BAD_TRACE",
                f"no failed {BAD_TRACE_FAMILY} trace of parent {parent} with a timeout then a retry",
            )


# -- the stored result pages and the level-2 file -------------------------------------------------------


def table(text: str, must_have: Sequence[str]) -> tuple[list[str], list[list[str]]] | None:
    """The first markdown table whose header holds every name in `must_have`: (header, rows)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("|") or i + 1 >= len(lines) or not re.match(r"^\|[-| :]+\|$", lines[i + 1]):
            continue
        head = [c.strip() for c in line.strip().strip("|").split("|")]
        if not all(name in head for name in must_have):
            continue
        rows = []
        for row in lines[i + 2 :]:
            if not row.startswith("|"):
                break
            rows.append([c.strip().strip("`") for c in row.strip().strip("|").split("|")])
        return head, rows
    return None


def render_columns(head: Sequence[str], rows: Sequence[Sequence[str]], names: Sequence[str]) -> list[str]:
    idx = [head.index(n) for n in names]
    cells = [[r[i] if i < len(r) else "" for i in idx] for r in rows]
    widths = [max(len(n), *(len(c[k]) for c in cells)) for k, n in enumerate(names)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return [fmt.format(*names).rstrip(), *(fmt.format(*c).rstrip() for c in cells)]


def agentdojo_summary(text: str) -> tuple[list[str], int, int] | None:
    head_names = ["Suite", "Agent model", "Status", "v0 utility under attack", "v0 targeted ASR"]
    found = table(text, head_names)
    if found is None:
        return None
    lines = render_columns(found[0], found[1], head_names)
    held: dict[str, int] = {}
    for m in re.finditer(r"^### (\w+) \(v[\d.]+\).*?\n(.*?)(?=^### |\Z)", text, re.S | re.M):
        n = re.search(r"Cases: \d+ training, (\d+) held-out", m.group(2))
        if n:
            held[m.group(1)] = max(held.get(m.group(1), 0), int(n.group(1)))
    return lines, sum(held.values()), len(held)


def tau_summary(text: str) -> tuple[list[str], int] | None:
    names = ["harness", "split", "tasks", "runs", "successes", "crashed", "pass^1"]
    found = table(text, names)
    counts = [int(m.group(1)) for m in re.finditer(r"\d+ train / (\d+) held-out", text)]
    if found is None or not counts:
        return None
    return render_columns(found[0], found[1], names), max(counts)


def tau_self_runs(text: str) -> tuple[str, str, str] | None:
    """(harness, "31 and 26", "40"): the held-out scores of the repeated runs of one unchanged harness in one
    evolve source (rows whose role starts with `parent_run`), from the `Every evaluation` table."""
    found = table(text, ["source", "role", "harness", "held-out passed"])
    if found is None:
        return None
    head, rows = found
    groups: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        cell = dict(zip(head, r, strict=False))
        if cell.get("role", "").startswith("parent_run") and "/" in cell.get("held-out passed", ""):
            groups.setdefault((cell["source"], cell["harness"]), []).append(cell["held-out passed"])
    for (_src, harness), scores in groups.items():
        totals = {s.split("/")[1] for s in scores}
        if len(scores) >= 2 and len(totals) == 1:
            return harness, " and ".join(s.split("/")[0] for s in scores), totals.pop()
    return None


def derive_from_results(out: Derived, results: Path = RESULTS, out_dir: Path = OUT) -> None:
    summary: list[str] = []
    dojo_file, tau_file = results / "agentdojo.md", results / "tau-retail.md"
    dojo = agentdojo_summary(dojo_file.read_text(encoding="utf-8")) if dojo_file.exists() else None
    tau = tau_summary(tau_file.read_text(encoding="utf-8")) if tau_file.exists() else None
    if dojo is None:
        out.fail("AGENTDOJO_CASES", "docs/results/agentdojo.md is missing or has no overview table")
    else:
        lines, cases, suites = dojo
        summary += ["AgentDojo, held-out cases only (docs/results/agentdojo.md)", *lines, ""]
        out.set(
            "AGENTDOJO_CASES", cases, "sum over suites of the largest `held-out` case count in agentdojo.md"
        )
        out.set("AGENTDOJO_SUITES", suites, "distinct suites in agentdojo.md")
    if tau is None:
        out.fail("TAU_CASES", "docs/results/tau-retail.md is missing or has no pass^k table")
    else:
        lines, tasks = tau
        summary += ["tau-bench retail (docs/results/tau-retail.md)", *lines, ""]
        out.set("TAU_CASES", tasks, "largest held-out task count in tau-retail.md")
        self_runs = tau_self_runs(tau_file.read_text(encoding="utf-8"))
        if self_runs is None:
            out.fail("TAU_SELF_SCORES", "tau-retail.md records no repeated runs of one unchanged harness")
        else:
            src = f"tau-retail.md, `Every evaluation`, parent runs of harness {self_runs[0]}"
            out.set("TAU_SELF_HARNESS", self_runs[0], src)
            out.set("TAU_SELF_SCORES", self_runs[1], src)
            out.set("TAU_SELF_TOTAL", self_runs[2], src)
    if dojo is not None and tau is not None:
        out_dir.mkdir(exist_ok=True)
        path = out_dir / "results-summary.txt"
        path.write_text("\n".join(summary), encoding="utf-8")
        out.set("RESULTS_FILE", rel(path), "built from the two result pages")


def derive_from_level2(out: Derived, candidates: Sequence[Path], out_dir: Path = OUT) -> None:
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        names = ", ".join(rel(p) for p in candidates)
        for key in ("LEVEL2_FILE", "LEVEL2_SEARCH_YIELD", "LEVEL2_SEED_YIELD"):
            out.fail(key, f"no level-2 result file (looked for {names})")
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    arms = result.get("arms") or {}
    if not result.get("claim") or not {"seed", "search"} <= set(arms):
        why = result.get("reason") or "the comparison recorded no arms"
        for key in ("LEVEL2_FILE", "LEVEL2_SEARCH_YIELD", "LEVEL2_SEED_YIELD"):
            out.fail(key, f"{path.name} records no level-2 claim: {why}")
        return
    cols = ["proposed", "invalid", "duplicates", "replay_cleared", "evaluated_fresh", "admitted"]
    head = (
        f"{'arm':<11}{'instructions':<14}{'slots':>6}" + "".join(f"{c:>16}" for c in cols) + f"{'yield':>8}"
    )
    lines = [f"Level 2: improver instructions, same budget (source {path.name})", head]
    for name, arm in arms.items():
        row = f"{name:<11}{arm['instructions_version']:<14}{arm['slots']:>6}"
        lines.append(row + "".join(f"{arm[c]:>16}" for c in cols) + f"{arm['admission_yield']:>8}")
    out_dir.mkdir(exist_ok=True)
    summary = out_dir / "level2-summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.set("LEVEL2_FILE", rel(summary), f"built from {rel(path)}")
    out.set("LEVEL2_SEARCH_YIELD", arms["search"]["admission_yield"], f"{path.name}, arm search")
    out.set("LEVEL2_SEED_YIELD", arms["seed"]["admission_yield"], f"{path.name}, arm seed")


def derive_constants(out: Derived) -> None:
    out.set("EVOLVE_LOG", EVOLVE_LOG, "the file `demo.py evolve-bg` writes")
    done = subprocess.run(
        ["git", "-C", str(ROOT), "remote", "get-url", "origin"], capture_output=True, text=True, check=False
    )
    url = done.stdout.strip()
    if done.returncode == 0 and url:
        out.set("REPO_URL", re.sub(r"\.git$", "", url), "`git remote get-url origin`")
    else:
        out.fail("REPO_URL", "this checkout has no `origin` remote")


# -- output ---------------------------------------------------------------------------------------------


def render_env(out: Derived, now: datetime) -> str:
    lines = [f"# Generated by docs/demo/derive_ids.py at {now:%Y-%m-%d %H:%M:%S} UTC. Do not edit.", ""]
    for key in sorted(out.values):
        value, source = out.values[key]
        lines += [f"# source: {source}", f"{key}={value}"]
    return "\n".join(lines) + "\n"


def write_or_fail(out: Derived, ids: Path = IDS, now: datetime | None = None) -> int:
    if out.missing:
        ids.unlink(missing_ok=True)
        for key in sorted(out.missing):
            print(f"NOT DERIVED  {key}: {out.missing[key]}", file=sys.stderr)
        print(
            f"derive_ids: {len(out.missing)} value(s) do not exist yet; "
            f"{ids.name} was removed, nothing written",
            file=sys.stderr,
        )
        return 1
    ids.write_text(render_env(out, now or datetime.now(UTC)), encoding="utf-8")
    print(f"wrote {rel(ids)}: {len(out.values)} values, each with its source")
    return 0


def ratchet_verifies(decision_id: str) -> str | None:
    """None when `ratchet verify <id>` exits 0, else the last line it printed."""
    done = subprocess.run(
        [sys.executable, "-m", "ratchet.cli", "verify", decision_id],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if done.returncode == 0:
        return None
    lines = (done.stdout + done.stderr).strip().splitlines()
    return lines[-1] if lines else f"exit {done.returncode}"


def rank_databases(counts: Mapping[str, tuple[int, int]]) -> str | None:
    """The database to read the demo from: most admitted decisions, then most decisions. `counts` maps a
    database name to (admitted, decisions)."""
    if not counts:
        return None
    return max(sorted(counts), key=lambda n: counts[n])


def ratchet_databases(client: Any) -> list[str]:
    return [n for n in client.list_database_names() if n.startswith("ratchet")]


def stored_decisions(client: Any, db: str) -> list[Mapping[str, Any]]:
    if "decisions" not in client[db].list_collection_names():
        return []
    return [d for d in client[db]["decisions"].find({}) if d.get("type", "decision") == "decision"]


def resolve_cross_db(out: Derived, key: str, found: Mapping[str, str], extra: Mapping[str, str]) -> None:
    """When `key` is missing in the demo database, take it from another database (`found` maps db -> value)
    and name that database in `<key>_DB`."""
    if key not in out.missing or not found:
        return
    db = sorted(found)[0]
    del out.missing[key]
    out.set(key, found[db], f"database {db} (the demo database has none)")
    out.set(f"{key.split('_')[0]}_DB", db, f"the database holding {key}")
    for k, v in extra.items():
        out.missing.pop(k, None)
        out.set(k, v.format(found[db]), f"built from {key}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=os.environ.get("RATCHET_MONGODB_URI", ""))
    ap.add_argument(
        "--db",
        default=os.environ.get("RATCHET_DB", ""),
        help="demo database (default: the ratchet* database with the most admitted, then most, decisions)",
    )
    ap.add_argument("--level2", type=Path, help="level-2 result JSON (default: docs/results/level2.json)")
    ap.add_argument(
        "--no-deep-verify", action="store_true", help="check the signature only, not `ratchet verify`"
    )
    args = ap.parse_args(argv)
    if not args.uri:
        print("derive_ids: set RATCHET_MONGODB_URI or pass --uri", file=sys.stderr)
        return 2

    from pymongo import MongoClient

    from ratchet.cli import LiveBackend
    from ratchet.store import Store
    from ratchet.view.admission import fresh_view
    from ratchet.view.data import resolve_candidates
    from ratchet.view.matrix import pick_rows
    from ratchet.view.verify import trusted_keys, verify_decision

    client: Any = MongoClient(args.uri)
    all_docs = {n: stored_decisions(client, n) for n in ratchet_databases(client)}
    live = {n: d for n, d in all_docs.items() if not n.startswith(DEMO_DB_PREFIX)}
    out = Derived()
    refusal = pick_closest(live)
    derive_refusal(out, refusal)

    db = args.db or rank_databases(
        {n: (sum(1 for d in docs if d.get("admitted")), len(docs)) for n, docs in all_docs.items()}
    )
    if not db:
        print("derive_ids: no ratchet* database exists", file=sys.stderr)
        return 2
    os.environ["RATCHET_MONGODB_URI"] = args.uri
    trusted = trusted_keys(None, os.environ)

    def run_store(name: str, refused_parent: str | None) -> Derived:
        os.environ["RATCHET_DB"] = name
        got = Derived()
        with Store(args.uri, name) as store:
            by_id = {str(d["_id"]): d for d in decision_docs(store)}

            def verifies(decision_id: str) -> str | None:
                res = verify_decision(by_id[decision_id], trusted)
                if not res.ok:
                    return res.summary
                return None if args.no_deep_verify else ratchet_verifies(decision_id)

            derive_from_store(
                store,
                got,
                case_info=LiveBackend().case_info,
                verifies=verifies,
                resolve=resolve_candidates,
                pick_rows=pick_rows,
                fresh_view=fresh_view,
                refused_parent=refused_parent,
            )
        return got

    refused_parent = str(refusal[1].doc["parent"]) if refusal else None
    got = run_store(db, refused_parent)
    if got.values.get("DEMO_OUTCOME", ("",))[0] == OUTCOME_REFUSED and refusal and refusal[0] != db:
        db = refusal[0]  # the refused outcome reads the database of its refusal scene, so one store serves it
        got = run_store(db, refused_parent)
    out.values.update(got.values)
    out.missing.update(got.missing)
    out.set(
        "RATCHET_DB",
        db,
        "the demo database: "
        + (
            "it holds the admitted decision"
            if got.values["DEMO_OUTCOME"][0] == OUTCOME_ADMITTED
            else "no database holds an admitted decision, so the database of the refusal scene"
        ),
    )

    authority = {
        n: str(d["candidate"])
        for n, docs in all_docs.items()
        for d in _newest(docs)[:1000]
        if not d.get("admitted") and set(_failed(d)) & set(AUTHORITY_GATES)
    }
    resolve_cross_db(out, "AUTHORITY_DECISION", authority, {})
    resume: dict[str, str] = {}
    for n in all_docs:
        cur = client[n]["checkpoints"].find({"state.status": "running"}).sort("updated_at", -1).limit(1)
        for doc in cur:
            resume[n] = str(doc["_id"])
    resolve_cross_db(out, "RESUME_RUN", resume, {"RESUME_CMD": "ratchet resume {}"})
    for key, db_key in (("AUTHORITY_DECISION", "AUTHORITY_DB"), ("RESUME_RUN", "RESUME_DB")):
        if key in out.values and db_key not in out.values:
            out.set(db_key, db, f"the demo database holds {key}")

    derive_from_results(out)
    level2 = [args.level2] if args.level2 else list(LEVEL2_DEFAULTS)
    if any(p.exists() for p in level2) or LEVEL2_DB not in all_docs:
        derive_from_level2(out, level2)
    else:
        derive_level2_from_store(out, all_docs[LEVEL2_DB], LEVEL2_DB)
    derive_constants(out)
    return write_or_fail(out)


if __name__ == "__main__":
    sys.exit(main())
