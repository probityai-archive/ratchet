"""Recall: show the proposer what worked and what failed on similar failures before it writes a candidate.

Two phases, and the lessons of failures are kept beside the successes, with the full results of each prior
attempt:

1. Similarity. The cluster's mechanism summary is looked up in the store's failure memory with hybrid
   search: Atlas Search keywords and Atlas Vector Search fused by the server's `$rankFusion`. (`$scoreFusion`
   needs MongoDB 8.3 and the sandbox runs 8.0.34, so it is not used.)
2. Utility. For each retrieved cluster, every candidate that was proposed for it and what happened to it, from
   the generation records and signed decisions already in the store. Ranked admitted with no false fires
   first, then admitted, then rejected with the reason; ties by similarity.

Each guard and procedure of the current harness also gets counters: helpful is
the number of held-out cases that stably recovered only after it was added, harmful the number of
must-succeed cases it broke plus benign-twin cases it intervened on.

Secrecy. Recall runs under the improver's database identity, which cannot read the held-out database, and it
adds three more layers. Failure memory is written from training clusters only. Every trace id that reaches a
`PriorFix` belongs to a training case, checked against the set of cases the improver was allowed to see; an id
that cannot be shown to be a training id is withheld and only counted. Everything protected (held-out
recoveries, regressions, referee false fires on held-out traces) is reduced to counts, which is the frozen
sanitized report shape of ruling 2.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from ratchet.improve.diagnose import FailureCluster
from ratchet.improve.llm import Chat, Turn
from ratchet.store import NewFailure, SimilarFailure
from ratchet.types import Harness

FixKind = Literal["guard", "procedure", "prompt_edit"]
FixOutcome = Literal["admitted_clean", "admitted", "rejected", "active"]

DEFAULT_BUDGET_CHARS = 6000
DEFAULT_MAX_SOURCE_CHARS = 1500
MIN_SOURCE_CHARS = 200
_TIER = {"admitted_clean": 0, "admitted": 1, "rejected": 2}
_ID_PREFIX = "cluster-"


class RecallStore(Protocol):
    """The slice of `ratchet.store.Store` recall uses. Reads, plus one idempotent write of failure memory."""

    db: Any

    def put_failures(self, failures: Sequence[NewFailure]) -> list[str]: ...

    def wait_indexed(self, timeout_s: float = 60.0, poll_s: float = 0.25) -> int: ...

    def similar_failures_hybrid(
        self, text: str, k: int = 5, *, rerank: bool = False
    ) -> list[SimilarFailure]: ...

    def evals(self, filter: Mapping[str, Any] | None = None) -> list[dict[str, Any]]: ...

    def decisions(self, filter: Mapping[str, Any] | None = None) -> list[dict[str, Any]]: ...

    def cases(self, split: str = "train") -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class PriorFix:
    """One earlier candidate (or one active item of the current harness) with what happened to it.

    Every number is a count. The only trace ids are training ids (`false_fire_ids`); protected evidence is
    never named, only counted.
    """

    kind: FixKind
    candidate_id: str
    cluster_id: str
    cluster_summary: str
    outcome: FixOutcome
    description: str = ""
    source: str | None = None
    name: str = ""
    when: str = ""
    steps: tuple[str, ...] = ()
    text: str = ""
    similarity: float = 0.0
    generation: int | None = None
    gates_failed: tuple[str, ...] = ()
    status: str = ""
    targets_hit: int = 0
    targets_total: int = 0
    false_fires: int = 0
    composed_false_fires: int = 0
    benign_twin_interventions: int = 0
    false_fire_ids: tuple[str, ...] = ()
    false_fire_reasons: tuple[str, ...] = ()
    false_fires_withheld: int = 0
    stable_recoveries: int = 0
    protected_broken: int = 0
    helpful: int = 0
    harmful: int = 0
    decisions: int = 0
    attempts: int = 1
    rank: int = 0
    of_total: int = 0
    source_chars_omitted: int = 0

    @property
    def tier(self) -> int:
        return _TIER.get(self.outcome, 3)


@dataclass(frozen=True)
class Recalled:
    """What recall found for one cluster, with the bounds it applied stated."""

    fixes: list[PriorFix]
    considered: int
    omitted: int
    source_chars_omitted: int
    retrieved_clusters: list[str] = field(default_factory=list)

    @property
    def statement(self) -> str:
        return (
            f"{len(self.fixes)} of {self.considered} prior fixes shown, {self.omitted} omitted for the "
            f"prompt budget, {self.source_chars_omitted} source characters cut"
        )


# -- failure memory ---------------------------------------------------------------------------------


def failure_text(cluster: FailureCluster) -> str:
    """The text a cluster is stored and looked up by: its mechanism and the key facts behind it."""
    key = cluster.key
    return f"{cluster.summary} | reason: {key.get('reason', '')} | tool: {key.get('tool', '')}"


def failure_id(cluster: FailureCluster) -> str:
    digest = hashlib.sha256(failure_text(cluster).encode("utf-8")).hexdigest()[:8]
    return f"{_ID_PREFIX}{cluster.id}-{digest}"


def cluster_of(failure: str) -> str | None:
    """The cluster id a failure-memory id was written for, or None for a failure we did not write."""
    if not failure.startswith(_ID_PREFIX):
        return None
    head, _, tail = failure[len(_ID_PREFIX) :].rpartition("-")
    return head if head and tail else None


# -- reading the store's history --------------------------------------------------------------------


def _run_id_of(failure_ref: str) -> str:
    return failure_ref.split("@", 1)[0]


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _count(value: Any) -> int:
    """The size of a list-valued evidence field, or an integer count when the field is already one."""
    if isinstance(value, list | tuple | set):
        return len(value)
    return _int(value)


def _check(decision: Mapping[str, Any], name: str) -> dict[str, Any]:
    for c in decision.get("checks") or ():
        if c.get("name") == name:
            return dict(c.get("evidence") or {}) | {"_passed": bool(c.get("passed"))}
    return {}


def _referee_false_fires(decision: Mapping[str, Any]) -> tuple[int, int, list[str]]:
    """(individual false fires, composed false fires, their `run@seq` ids) from a decision's replay check."""
    ev = _check(decision, "replay")
    reports = [r for r in ev.get("reports") or () if isinstance(r, Mapping)]
    refs = [str(x) for r in reports for x in r.get("false_fires") or ()]
    composed = ev.get("composed")
    composed_n = _count(composed.get("false_fires")) if isinstance(composed, Mapping) else 0
    return len(refs), composed_n, refs


def _targets(decision: Mapping[str, Any]) -> tuple[int, int]:
    ev = _check(decision, "replay")
    reports = [r for r in ev.get("reports") or () if isinstance(r, Mapping)]
    return (
        sum(_int(r.get("targets_hit")) for r in reports),
        sum(_int(r.get("targets_total")) for r in reports),
    )


def _failed_gates(decision: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(c["name"]) for c in decision.get("checks") or () if not c.get("passed"))


def _counters(decision: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """(stable recoveries, protected broken, must-succeed regressions + benign-twin cases, twin hits)."""
    # the held-out comparison is the one check whose evidence lists the stable recoveries
    fresh = next(
        (
            dict(c.get("evidence") or {})
            for c in decision.get("checks") or ()
            if "stable_recoveries" in (c.get("evidence") or {})
        ),
        {},
    )
    stable = _count(fresh.get("stable_recoveries"))
    broken = _count(fresh.get("protected_broken"))
    regressed = _count(_check(decision, "no_regression").get("regressed"))
    twin = _check(decision, "benign_twins")
    twin_cases = _count(twin.get("intervened"))
    return stable, broken, regressed + twin_cases, _int(twin.get("interventions"))


class _Candidate:
    """Everything the store says about one candidate id, joined across generation records and decisions."""

    def __init__(self, cid: str) -> None:
        self.cid = cid
        self.kind: str = ""
        self.cluster_id: str = ""
        self.change: dict[str, Any] = {}
        self.generation: int | None = None
        self.status: str = ""
        self.replay: dict[str, Any] | None = None
        self.attempts = 0
        self.decisions: list[dict[str, Any]] = []

    def absorb(self, raw: Mapping[str, Any], generation: int | None) -> None:
        self.attempts += 1
        self.kind = str(raw.get("kind") or self.kind)
        self.cluster_id = str(raw.get("cluster_id") or self.cluster_id)
        self.change = dict(raw.get("change") or self.change)
        if generation is not None and (self.generation is None or generation >= self.generation):
            self.generation = generation
            self.status = str(raw.get("status") or self.status)
            if raw.get("replay"):
                self.replay = dict(raw["replay"])

    def best_decision(self) -> dict[str, Any] | None:
        if not self.decisions:
            return None
        return min(
            self.decisions,
            key=lambda d: (
                not d.get("admitted"),
                sum(_referee_false_fires(d)[:2]),
                -_int((d.get("context") or {}).get("generation")),
            ),
        )


def _join(
    generations: Iterable[Mapping[str, Any]], decisions: Iterable[Mapping[str, Any]]
) -> dict[str, _Candidate]:
    out: dict[str, _Candidate] = {}
    for rec in sorted(generations, key=lambda r: _int(r.get("generation"))):
        gen = _int(rec.get("generation"))
        for raw in rec.get("candidates") or ():
            cid = str(raw.get("id") or "")
            if cid:
                out.setdefault(cid, _Candidate(cid)).absorb(raw, gen)
    for d in decisions:
        ctx = d.get("context") or {}
        cid = str(ctx.get("id") or "")
        if not cid or ctx.get("composite_of"):
            continue
        cand = out.setdefault(cid, _Candidate(cid))
        if cand.attempts == 0:
            cand.absorb(ctx, _int(ctx.get("generation")) or None)
        cand.decisions.append(dict(d))
    return out


# -- the recall object ------------------------------------------------------------------------------


class Recall:
    """Failure memory plus prior-fix history for the proposer, read under the improver's identity.

    `train_case_ids` is the set of cases the improver is allowed to see. It is unioned with the training
    cases in the store and the `visible_case_ids` of earlier decisions; a trace id is shown only when its
    case is in that set.
    """

    def __init__(
        self,
        store: RecallStore,
        *,
        train_case_ids: Collection[str] = (),
        k: int = 5,
        budget_chars: int = DEFAULT_BUDGET_CHARS,
        max_source_chars: int = DEFAULT_MAX_SOURCE_CHARS,
        rerank: bool = False,
    ) -> None:
        self.store = store
        self.k = k
        self.budget_chars = budget_chars
        self.max_source_chars = max_source_chars
        self.rerank = rerank
        self._train: set[str] = set(train_case_ids)
        self._store_train_loaded = False

    # -- training-case whitelist --

    def train_ids(self) -> set[str]:
        if not self._store_train_loaded:
            self._train |= {str(c["_id"]) for c in self.store.cases("train")}
            self._store_train_loaded = True
        return self._train

    def _shown_ids(self, refs: Sequence[tuple[str, str | None]]) -> dict[str, bool]:
        """For each `run@seq` reference, whether its trace belongs to a training case. Unknown means no."""
        train = self.train_ids()
        need = sorted({_run_id_of(r) for r, case in refs if case is None})
        case_of: dict[str, str] = {}
        if need:
            for doc in self.store.db["traces"].find({"run_id": {"$in": need}}, {"run_id": 1, "case_id": 1}):
                case_of[str(doc["run_id"])] = str(doc["case_id"])
        out: dict[str, bool] = {}
        for ref, case in refs:
            known = case if case is not None else case_of.get(_run_id_of(ref))
            out[ref] = known is not None and known in train
        return out

    # -- writing failure memory --

    def remember(self, clusters: Sequence[FailureCluster]) -> list[str]:
        """Store each cluster's mechanism in failure memory. Only clusters with a training member are written.

        Idempotent: a cluster already stored under the same text is not embedded again.
        """
        train = self.train_ids()
        failures: list[NewFailure] = []
        for c in clusters:
            member = next((m for m in c.members if m.case_id in train), None)
            if member is None:
                continue
            failures.append(
                NewFailure(
                    failure_text(c),
                    case_family=c.key.get("tool", "") or "unknown",
                    trace_ref=f"{member.run_id}/{member.case_id}",
                    failure_id=failure_id(c),
                )
            )
        if not failures:
            return []
        ids = self.store.put_failures(failures)
        self.store.wait_indexed()  # a hybrid query right after a write could otherwise see one leg only
        return ids

    # -- reading it back --

    def recall(self, cluster: FailureCluster, harness: Harness | None = None) -> Recalled:
        """Prior fixes for clusters similar to `cluster`, ranked and bounded, plus counters for `harness`."""
        hits = self.store.similar_failures_hybrid(failure_text(cluster), self.k, rerank=self.rerank)
        score: dict[str, float] = {}
        summary: dict[str, str] = {}
        for h in hits:
            cid = cluster_of(h.id)
            if cid is not None and h.score > score.get(cid, -1.0):
                score[cid], summary[cid] = h.score, h.summary
        history = self._history(sorted(score))
        fixes = [self._to_fix(c, score, summary) for c in history.values()]
        fixes = self._privatise(fixes, history)
        fixes.sort(key=lambda f: (f.tier, -f.similarity, f.candidate_id))
        active = self._active(harness)
        shown, active, cut = self._fit(fixes, active)
        return Recalled(shown + active, len(fixes), len(fixes) - len(shown), cut, sorted(score))

    def _history(self, cluster_ids: list[str]) -> dict[str, _Candidate]:
        if not cluster_ids:
            return {}
        gens = self.store.evals({"type": "generation", "candidates.cluster_id": {"$in": cluster_ids}})
        decs = [
            d
            for d in self.store.decisions({"context.cluster_id": {"$in": cluster_ids}})
            if d.get("type", "decision") == "decision"
        ]
        for d in decs:
            self._train |= {str(x) for x in d.get("visible_case_ids") or ()}
        joined = _join(gens, decs)
        return {cid: c for cid, c in joined.items() if c.cluster_id in cluster_ids}

    def _to_fix(self, c: _Candidate, score: Mapping[str, float], summary: Mapping[str, str]) -> PriorFix:
        best = c.best_decision()
        change = c.change
        steps = tuple(str(s) for s in change.get("steps") or ())
        base: dict[str, Any] = {
            "kind": c.kind if c.kind in ("guard", "procedure", "prompt_edit") else "guard",
            "candidate_id": c.cid,
            "cluster_id": c.cluster_id,
            "cluster_summary": summary.get(c.cluster_id, ""),
            "description": str(change.get("description") or change.get("name") or "")[:300],
            "source": change.get("source"),
            "name": str(change.get("name") or ""),
            "when": str(change.get("when") or ""),
            "steps": steps,
            "text": str(change.get("append") or ""),
            "similarity": float(score.get(c.cluster_id, 0.0)),
            "generation": c.generation,
            "status": c.status,
            "attempts": max(c.attempts, 1),
            "decisions": len(c.decisions),
        }
        train_rp = c.replay or {}
        base["targets_hit"] = _int(train_rp.get("targets_hit"))
        base["targets_total"] = _int(train_rp.get("targets_total"))
        base["false_fires"] = len(train_rp.get("false_positives") or ())
        base["benign_twin_interventions"] = _int(train_rp.get("benign_twin_fires"))
        if best is not None:
            single, composed, _ = _referee_false_fires(best)
            hit, total = _targets(best)
            stable, broken, harmful, twins = _counters(best)
            base |= {
                "false_fires": single,
                "composed_false_fires": composed,
                "targets_hit": hit or base["targets_hit"],
                "targets_total": total or base["targets_total"],
                "stable_recoveries": stable,
                "protected_broken": broken,
                "helpful": stable,
                "harmful": harmful,
                "benign_twin_interventions": twins,
            }
            if best.get("admitted"):
                clean = single == 0 and composed == 0 and twins == 0
                base["outcome"] = "admitted_clean" if clean else "admitted"
            else:
                base["outcome"] = "rejected"
                base["gates_failed"] = _failed_gates(best)
        else:
            base["outcome"] = "rejected"
            base["gates_failed"] = (c.status,) if c.status else ()
        return PriorFix(**base)

    def _privatise(self, fixes: list[PriorFix], history: Mapping[str, _Candidate]) -> list[PriorFix]:
        """Attach the false-fire trace ids that are provably training ids; count the rest as withheld."""
        wanted: dict[str, list[tuple[str, str | None, str]]] = {}
        for f in fixes:
            c = history[f.candidate_id]
            rows: list[tuple[str, str | None, str]] = []
            for fp in (c.replay or {}).get("false_positives") or ():
                rows.append(
                    (f"{fp['run_id']}@{fp['seq']}", fp.get("case_id"), str(fp.get("reason", ""))[:200])
                )
            best = c.best_decision()
            if best is not None:
                rows += [(ref, None, "") for ref in _referee_false_fires(best)[2]]
            wanted[f.candidate_id] = rows
        verdict = self._shown_ids([(ref, case) for rows in wanted.values() for ref, case, _ in rows])
        out: list[PriorFix] = []
        for f in fixes:
            seen: dict[str, str] = {}
            for ref, _case, reason in wanted[f.candidate_id]:
                if verdict.get(ref) and (ref not in seen or (reason and not seen[ref])):
                    seen[ref] = reason
            total = max(f.false_fires, len(wanted[f.candidate_id]))
            out.append(
                replace(
                    f,
                    false_fire_ids=tuple(sorted(seen)),
                    false_fire_reasons=tuple(seen[r] for r in sorted(seen) if seen[r]),
                    false_fires_withheld=max(total - len(seen), 0),
                    false_fires=total,
                )
            )
        return out

    def _active(self, harness: Harness | None) -> list[PriorFix]:
        """Helpful and harmful counters for each guard and procedure of the current harness."""
        if harness is None:
            return []
        items: dict[str, tuple[FixKind, str]] = {g.id: ("guard", g.description) for g in harness.guards} | {
            p.id: ("procedure", p.name) for p in harness.procedures
        }
        if not items:
            return []
        decs = self.store.decisions({"context.change.id": {"$in": sorted(items)}})
        out: list[PriorFix] = []
        for item_id, (kind, label) in items.items():
            mine = [
                d
                for d in decs
                if d.get("type", "decision") == "decision"
                and not (d.get("context") or {}).get("composite_of")
                and ((d.get("context") or {}).get("change") or {}).get("id") == item_id
            ]
            counts = [_counters(d) for d in mine]
            out.append(
                PriorFix(
                    kind=kind,
                    candidate_id=item_id,
                    cluster_id=str(((mine[0].get("context") or {}).get("cluster_id")) if mine else ""),
                    cluster_summary="",
                    outcome="active",
                    description=label[:300],
                    helpful=sum(c[0] for c in counts),
                    harmful=sum(c[2] for c in counts),
                    stable_recoveries=sum(c[0] for c in counts),
                    protected_broken=sum(c[1] for c in counts),
                    decisions=len(mine),
                )
            )
        return out

    def _fit(
        self, fixes: list[PriorFix], active: list[PriorFix]
    ) -> tuple[list[PriorFix], list[PriorFix], int]:
        """The leading fixes whose rendering fits the budget, long sources cut first, nothing silently."""
        room = self.budget_chars - len(_header(len(fixes), 0, 0)) - sum(len(_render_fix(a)) for a in active)
        shown: list[PriorFix] = []
        cut_total = 0
        for f in fixes:
            f = _cap_source(f, self.max_source_chars)
            text_len = len(_render_fix(f))
            if text_len > room and f.source:
                over = text_len - room
                keep = len(f.source) - over - len(_CUT_MARK.format(n=len(f.source)))
                if keep >= MIN_SOURCE_CHARS:
                    f = _cap_source(f, keep)
                    text_len = len(_render_fix(f))
            if text_len > room:
                continue
            room -= text_len
            cut_total += f.source_chars_omitted
            shown.append(f)
        total = len(fixes)
        shown = [replace(f, rank=i, of_total=total) for i, f in enumerate(shown, start=1)]
        return shown, [replace(a, of_total=total) for a in active], cut_total


# -- rendering --------------------------------------------------------------------------------------

_CUT_MARK = "\n...(cut, {n} characters in the full source)"


def _cap_source(f: PriorFix, limit: int) -> PriorFix:
    if f.source is None or len(f.source) <= limit:
        return f
    return replace(
        f, source=f.source[:limit], source_chars_omitted=f.source_chars_omitted + len(f.source) - limit
    )


def _header(total: int, shown: int, cut: int) -> str:
    return (
        f"Prior fixes for similar failures ({shown} of {total} shown, ranked: admitted with no false fires, "
        f"then admitted, then rejected; {total - shown} omitted for the prompt budget, {cut} source "
        "characters cut). Build on what worked, never repeat a rejected approach unchanged, and fix the "
        "reason it was rejected.\n"
    )


def _outcome_line(f: PriorFix) -> str:
    label = {"admitted_clean": "ADMITTED, no false fires", "admitted": "ADMITTED", "rejected": "REJECTED"}[
        f.outcome
    ]
    bits = [label]
    if f.outcome == "rejected" and f.gates_failed:
        bits.append("failed " + ", ".join(f.gates_failed))
    if f.kind == "guard":
        bits.append(f"replay hit {f.targets_hit}/{f.targets_total} targets")
        bits.append(f"{f.false_fires} false fires")
        if f.composed_false_fires:
            bits.append(f"{f.composed_false_fires} when composed")
    if f.decisions:
        bits.append(f"held-out: {f.stable_recoveries} stable recoveries, {f.harmful} harmed")
    return "; ".join(bits)


def _render_fix(f: PriorFix) -> str:
    if f.outcome == "active":
        return (
            f"- {f.kind} {f.candidate_id} ({f.description}): helpful {f.helpful}, harmful {f.harmful} "
            f"over {f.decisions} decisions\n"
        )
    lines = [f"{f.rank or '-'}. {f.kind} {f.candidate_id}: {_outcome_line(f)}"]
    lines.append(f"   for a cluster of failures like: {f.cluster_summary or f.cluster_id}")
    if f.description:
        lines.append(f"   {f.description}")
    if f.kind == "guard" and f.source:
        suffix = _CUT_MARK.format(n=len(f.source) + f.source_chars_omitted) if f.source_chars_omitted else ""
        lines.append("   source:\n" + "\n".join("     " + ln for ln in (f.source + suffix).splitlines()))
    elif f.kind == "procedure":
        lines.append(f"   procedure {f.name} (when {f.when}): " + "; ".join(f.steps))
    elif f.kind == "prompt_edit":
        lines.append(f"   prompt text: {f.text}")
    if f.false_fire_ids:
        shown = ", ".join(f.false_fire_ids[:8])
        lines.append(f"   wrongly blocked training runs: {shown}")
        lines += [f"   block reason seen there: {r}" for r in f.false_fire_reasons[:3]]
    if f.false_fires_withheld:
        lines.append(
            f"   {f.false_fires_withheld} more false fires on runs that are not training runs (not shown)"
        )
    return "\n".join(lines) + "\n"


def render_prior_fixes(fixes: Sequence[PriorFix]) -> str:
    """The prompt section for `fixes`, with the bounds that were applied stated at its head."""
    history = [f for f in fixes if f.outcome != "active"]
    active = [f for f in fixes if f.outcome == "active"]
    if not history and not active:
        return ""
    total = (history or active)[0].of_total or len(history)
    parts = []
    if history:
        cut = sum(f.source_chars_omitted for f in history)
        parts.append(_header(total, len(history), cut))
        parts += [_render_fix(f) for f in history]
    if active:
        parts.append(
            "Track record of the current harness's own guards and procedures (helpful = held-out cases "
            "that recovered only after it was added; harmful = must-succeed or benign-twin cases it hurt):\n"
        )
        parts += [_render_fix(f) for f in active]
    return "\n".join(parts)


# -- a chat wrapper that adds the section to a proposal request ----------------------------------------

_CLUSTER_LINE = re.compile(
    r"^Failure cluster (\S+) \(\d+ failures\): (.*)\nMechanism key: (\{.*\})$", re.MULTILINE
)


def cluster_of_prompt(prompt: str) -> FailureCluster | None:
    """The cluster a proposal prompt is about (id, summary, key), read back from the prompt text."""
    m = _CLUSTER_LINE.search(prompt)
    if m is None:
        return None
    try:
        key = {str(k): str(v) for k, v in json.loads(m.group(3)).items()}
    except (ValueError, AttributeError):
        return None
    return FailureCluster(m.group(1), key, [], m.group(2))


class RecallChat:
    """Wraps the improver's chat client: a proposal request gains the prior-fixes section for its cluster.

    This is the prompt-level stand-in for `propose(..., prior_fixes=...)`: it appends exactly the text that
    `render_prior_fixes` produces, so with and without the wrapper is the retrieval ablation. Requests that
    are not proposals (diagnosis summaries, instruction revision, guard repair) pass through unchanged.
    """

    def __init__(self, inner: Chat, recall: Recall, harness: Harness | None = None) -> None:
        self.inner = inner
        self.recall = recall
        self.harness = harness
        self.sections: dict[str, Recalled] = {}

    def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> Turn:
        out = [dict(m) for m in messages]
        user = next((m for m in reversed(out) if m.get("role") == "user"), None)
        if user is not None and "\n\nPropose " in str(user.get("content")):
            cluster = cluster_of_prompt(str(user["content"]))
            if cluster is not None:
                found = self.recall.recall(cluster, self.harness)
                self.sections[cluster.id] = found
                section = render_prior_fixes(found.fixes)
                if section:
                    body = str(user["content"])
                    at = body.rfind("\n\nPropose ")
                    user["content"] = f"{body[:at]}\n\n{section}{body[at:]}"
        return self.inner.complete(out, tools)
