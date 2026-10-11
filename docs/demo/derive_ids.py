"""Derive every id and number the demo and the posts quote, from Atlas and the run directories.

    uv run python docs/demo/derive_ids.py [--uri URI] [--db DB] [--level2 FILE] [--no-deep-verify]

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


def pick_bad_trace(store: Any, parent: str, case_info: CaseInfoFn, family: str | None = None) -> str | None:
    family = family or BAD_TRACE_FAMILY
    for t in store.traces({"harness_version": parent}, newest_first=True):
        if t.outcome.passed:
            continue
        info = case_info(t.case_id)
        if (
            info is not None
            and info.family == family
            and info.kind == "trap"
            and timeout_then_retry(t.events)
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


def derive_from_store(
    store: Any,
    out: Derived,
    *,
    case_info: CaseInfoFn,
    verifies: Callable[[str], str | None],
    resolve: Callable[[Any, Sequence[str]], Sequence[Any]],
    pick_rows: Callable[..., Any],
    fresh_view: Callable[[Mapping[str, Any]], Any],
) -> None:
    docs = decision_docs(store)
    out.set("N_DECISIONS", len(docs), "count of documents in `decisions`")
    out.set("N_REJECTED", sum(1 for d in docs if not d.get("admitted")), "decisions with admitted=false")

    admitted, why = pick_admitted(docs, verifies)
    if admitted is None:
        out.fail("ADMITTED_DECISION", why)
    else:
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

    pair = pick_pair(store, docs, resolve)
    if pair is None:
        reason = "no admitted guard candidate has a replay-rejected guard candidate of the same parent"
        for key in ("CLEARED_CAND", "REJECTED_CAND", "REPLAY_TRACES", "BAD_TRACE"):
            out.fail(key, reason)
    else:
        cleared, rejected = pair
        out.set("CLEARED_CAND", cleared["candidate"], f"admitted guard candidate, decision {cleared['_id']}")
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
        bad = pick_bad_trace(store, parent, case_info)
        if bad:
            out.set("BAD_TRACE", bad, f"newest failed {BAD_TRACE_FAMILY} trap trace of parent {parent}")
        else:
            out.fail(
                "BAD_TRACE",
                f"no failed {BAD_TRACE_FAMILY} trace of parent {parent} with a timeout then a retry",
            )

    authority = [d for d in _newest(docs) if not d.get("admitted") and set(_failed(d)) & set(AUTHORITY_GATES)]
    if authority:
        out.set(
            "AUTHORITY_DECISION",
            authority[0]["_id"],
            "newest rejected decision that failed an authority gate",
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


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=os.environ.get("RATCHET_MONGODB_URI", ""))
    ap.add_argument("--db", default=os.environ.get("RATCHET_DB", "ratchet"))
    ap.add_argument("--level2", type=Path, help="level-2 result JSON (default: docs/results/level2.json)")
    ap.add_argument(
        "--no-deep-verify", action="store_true", help="check the signature only, not `ratchet verify`"
    )
    args = ap.parse_args(argv)
    if not args.uri:
        print("derive_ids: set RATCHET_MONGODB_URI or pass --uri", file=sys.stderr)
        return 2

    from ratchet.cli import LiveBackend
    from ratchet.store import Store
    from ratchet.view.admission import fresh_view
    from ratchet.view.data import resolve_candidates
    from ratchet.view.matrix import pick_rows
    from ratchet.view.verify import trusted_keys, verify_decision

    os.environ["RATCHET_MONGODB_URI"], os.environ["RATCHET_DB"] = args.uri, args.db
    trusted = trusted_keys(None, os.environ)
    out = Derived()
    with Store(args.uri, args.db) as store:
        by_id = {str(d["_id"]): d for d in decision_docs(store)}

        def verifies(decision_id: str) -> str | None:
            res = verify_decision(by_id[decision_id], trusted)
            if not res.ok:
                return res.summary
            return None if args.no_deep_verify else ratchet_verifies(decision_id)

        derive_from_store(
            store,
            out,
            case_info=LiveBackend().case_info,
            verifies=verifies,
            resolve=resolve_candidates,
            pick_rows=pick_rows,
            fresh_view=fresh_view,
        )
    derive_from_results(out)
    derive_from_level2(out, [args.level2] if args.level2 else list(LEVEL2_DEFAULTS))
    derive_constants(out)
    return write_or_fail(out)


if __name__ == "__main__":
    sys.exit(main())
