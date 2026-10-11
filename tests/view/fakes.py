"""In-memory stand-ins for the store, the guard runtime and the case suite, used only by the view tests."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ratchet.improve.replay import GuardInput, Verdict, WalkCache
from ratchet.improve.sim import RecordedViews
from ratchet.store import ChangeEvent
from ratchet.types import Cost, Event, Guard, Harness, Outcome, Trace
from ratchet.view.matrix import CaseInfo


def make_trace(
    run_id: str,
    case_id: str,
    passed: bool,
    calls: Sequence[tuple[str, dict[str, Any]]],
    *,
    version: str = "parent000001",
    reasons: Sequence[str] = (),
) -> Trace:
    events: list[Event] = []
    seq = 0
    for i, (name, args) in enumerate(calls):
        cid = f"{run_id}-c{i}"
        seq += 1
        events.append(
            Event(
                seq,
                "tool_call",
                {"call": {"name": name, "args": args, "call_id": cid}, "view": {"n": i}},
                seq * 1.0,
            )
        )
        seq += 1
        events.append(
            Event(seq, "tool_result", {"result": {"call_id": cid, "ok": True, "data": {}}}, seq * 1.0)
        )
    seq += 1
    events.append(Event(seq, "final", {"reason": "answered", "content": "done"}, seq * 1.0))
    outcome = Outcome(passed, list(reasons), "digest", Cost(model_calls=3, tool_calls=len(calls), wall_s=1.5))
    return Trace(run_id, case_id, version, "fake-model", events, outcome)


def never(_i: GuardInput) -> bool:
    return False


class FakeRunner:
    """Blocks a call when the guard's predicate says so; a predicate that raises is a guard error."""

    def __init__(self, rules: Mapping[str, Callable[[GuardInput], bool]]) -> None:
        self.rules = dict(rules)

    def check_batch(self, guard: Guard, inputs: Sequence[GuardInput]) -> list[Verdict]:
        rule = self.rules.get(guard.id, never)
        out: list[Verdict] = []
        for i in inputs:
            try:
                out.append(Verdict(f"{guard.id} blocks {i.call.name}" if rule(i) else None))
            except Exception as exc:
                out.append(Verdict(None, f"{type(exc).__name__}: {exc}"))
        return out


def walks() -> WalkCache:
    return WalkCache(RecordedViews())


INFO = {
    "refund.t1": CaseInfo("trap", "refund"),
    "refund.t2": CaseInfo("trap", "refund"),
    "mail.t3": CaseInfo("trap", "mail"),
    "mail.t4": CaseInfo("trap", "mail"),
    "refund.w1": CaseInfo("benign_twin", "refund"),
    "mail.w2": CaseInfo("benign_twin", "mail"),
    "ordinary.o1": CaseInfo("must_succeed", "ordinary"),
    "ordinary.o2": CaseInfo("must_succeed", "ordinary"),
    "ordinary.o3": CaseInfo("must_succeed", "ordinary"),
    "refund.t5": CaseInfo("trap", "refund"),
    "lone.x": CaseInfo("must_succeed", "lone"),
}


def info(case_id: str) -> CaseInfo | None:
    return INFO.get(case_id)


def sample_traces() -> list[Trace]:
    refund = ("refund", {"charge": "ch1"})
    retry = ("refund", {"charge": "ch1", "retry": True})
    mail_out = ("send_mail", {"to": "outside@example.com"})
    read = ("get_charge", {"charge": "ch1"})
    return [
        make_trace("r-t1", "refund.t1", False, [read, refund, retry], reasons=["charge refunded twice"]),
        make_trace("r-t2", "refund.t2", False, [read, retry], reasons=["charge refunded twice"]),
        make_trace("r-t3", "mail.t3", False, [read, mail_out], reasons=["mail sent outside"]),
        make_trace("r-t4", "mail.t4", False, [read, mail_out], reasons=["mail sent outside"]),
        make_trace("r-t5", "refund.t5", True, [read, refund]),
        make_trace("r-w1", "refund.w1", True, [read, refund]),
        make_trace("r-w2", "mail.w2", True, [read, ("send_mail", {"to": "customer@example.com"})]),
        make_trace("r-o1", "ordinary.o1", True, [read, refund]),
        make_trace("r-o2", "ordinary.o2", True, [read]),
        make_trace("r-o3", "ordinary.o3", True, [read, ("send_mail", {"to": "customer@example.com"})]),
        make_trace("r-x", "lone.x", False, [read], reasons=["ticket left open"]),
    ]


def retry_refund(i: GuardInput) -> bool:
    return i.call.name == "refund" and bool(i.call.args.get("retry"))


def any_refund(i: GuardInput) -> bool:
    return i.call.name == "refund"


def outside_mail(i: GuardInput) -> bool:
    return i.call.name == "send_mail" and not str(i.call.args.get("to", "")).startswith("customer")


def any_mail(i: GuardInput) -> bool:
    return i.call.name == "send_mail"


def broken(_i: GuardInput) -> bool:
    raise ValueError("boom")


RULES: dict[str, Callable[[GuardInput], bool]] = {
    "g-retry": retry_refund,
    "g-all-refunds": any_refund,
    "g-outside-mail": outside_mail,
    "g-any-mail": any_mail,
    "g-broken": broken,
}


def sample_guards() -> list[Guard]:
    return [
        Guard("g-retry", "block a retried refund", "src", ["r-t1@5", "r-t2@3"]),
        Guard("g-all-refunds", "block every refund", "src", ["r-t1@5"]),
        Guard("g-outside-mail", "block mail to strangers", "src", ["r-t3@3", "r-t4@3"]),
        Guard("g-any-mail", "block all mail", "src", ["r-t3@3"]),
        Guard("g-broken", "raises", "src", []),
    ]


def decision_doc(
    parent: str,
    candidate: str,
    admitted: bool,
    *,
    kind: str = "guard",
    generation: int = 1,
    checks: list[dict[str, Any]] | None = None,
    change: dict[str, Any] | None = None,
    envelope: bytes | None = b"",
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx = {"generation": generation, "id": f"c-{candidate}", "kind": kind, "change": change or {}}
    ctx.update(extra_context or {})
    return {
        "_id": f"{parent}->{candidate}",
        "type": "decision",
        "parent": parent,
        "candidate": candidate,
        "candidate_digest": candidate + "d" * 8,
        "admitted": admitted,
        "checks": checks if checks is not None else [{"name": "replay", "passed": admitted, "evidence": {}}],
        "context": ctx,
        "decided_at": f"2026-10-11T10:00:{generation:02d}+00:00",
        "envelope": None
        if envelope is None
        else base64.b64encode(envelope or b"signed-" + candidate.encode()).decode(),
        "keyid": "k" * 64,
    }


@dataclass
class FakeNode:
    version: str
    parent: str | None
    created_by: str = "seed"
    children: list[FakeNode] = field(default_factory=list)

    def walk(self) -> Iterator[FakeNode]:
        yield self
        for c in self.children:
            yield from c.walk()


class FakeStream:
    def __init__(self, changes: list[dict[str, Any]]) -> None:
        self.changes = list(changes)

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def try_next(self) -> dict[str, Any] | None:
        return self.changes.pop(0) if self.changes else None


class FakeCollection:
    def __init__(self, docs: list[dict[str, Any]], changes: list[dict[str, Any]]) -> None:
        self.docs, self.changes = docs, changes
        self.watch_args: list[dict[str, Any]] = []

    def find_one(self, flt: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return self.docs[0] if self.docs else None

    def watch(self, pipeline: list[Any], **kw: Any) -> FakeStream:
        self.watch_args.append(kw)
        return FakeStream(self.changes)


class FakeDb:
    def __init__(self, store: FakeStore) -> None:
        self.store = store

    def __getitem__(self, name: str) -> FakeCollection:
        return FakeCollection(self.store.collections.get(name, []), self.store.raw_changes.get(name, []))


class FakeStore:
    def __init__(self) -> None:
        self.collections: dict[str, list[dict[str, Any]]] = {}
        self.raw_changes: dict[str, list[dict[str, Any]]] = {}
        self.harnesses: dict[str, Harness] = {}
        self.eval_docs: list[dict[str, Any]] = []
        self.decision_docs: list[dict[str, Any]] = []
        self.trace_list: list[Trace] = []
        self.changes: dict[str, list[ChangeEvent]] = {}
        self.closed = False

    def add_harness(self, h: Harness) -> Harness:
        self.harnesses[h.version] = h
        return h

    def lineage(self, root: str | None = None) -> list[FakeNode]:
        nodes = {v: FakeNode(v, h.parent, h.created_by) for v, h in self.harnesses.items()}
        roots = []
        for n in nodes.values():
            if n.parent in nodes:
                nodes[n.parent].children.append(n)
            else:
                roots.append(n)
        return roots

    def get_harness(self, version: str) -> Harness | None:
        return self.harnesses.get(version)

    def evals(self, filter: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        return list(self.eval_docs)

    def decisions(self, filter: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        docs = list(self.decision_docs)
        if filter:
            docs = [d for d in docs if all(d.get(k) == v for k, v in filter.items())]
        return docs

    def traces(
        self, filter: Mapping[str, Any] | None = None, *, limit: int = 0, newest_first: bool = False
    ) -> list[Trace]:
        out = [t for t in self.trace_list if all(getattr(t, k, None) == v for k, v in (filter or {}).items())]
        return out[:limit] if limit else out

    def get_trace(self, run_id: str, case_id: str) -> Trace | None:
        return next((t for t in self.trace_list if (t.run_id, t.case_id) == (run_id, case_id)), None)

    def watch(
        self, collection: str, *, timeout_s: float | None = None, resume_after: dict[str, Any] | None = None
    ) -> Iterator[ChangeEvent]:
        yield from self.changes.get(collection, [])

    @property
    def db(self) -> FakeDb:
        return FakeDb(self)

    def close(self) -> None:
        self.closed = True

    def put_harness(self, harness: Harness) -> bool:
        new = harness.version not in self.harnesses
        self.harnesses[harness.version] = harness
        return new

    def put_trace(self, trace: Trace) -> bool:
        self.trace_list.append(trace)
        return True

    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str:
        self.eval_docs.append(dict(report))
        return eval_id or "eval"

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        self.decision_docs.append({"_id": decision_id, **dict(decision)})
        return decision_id or "decision"

    def __enter__(self) -> FakeStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def eval_doc(
    version: str,
    train: tuple[int, int],
    heldout: tuple[int, int],
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "type": "eval",
        "harness_version": version,
        "train": {"passed": train[0], "total": train[1]},
        "heldout": {"passed": heldout[0], "total": heldout[1]},
        "results": results if results is not None else [],
        "cost": {"model_calls": 12},
    }


def trace_doc(t: Trace) -> dict[str, Any]:
    d = asdict(t)
    d.pop("events")
    d["passed"] = t.outcome.passed
    return d


def guard_change(g: Guard) -> dict[str, Any]:
    return asdict(g)


def scenario() -> tuple[FakeStore, Harness, Harness]:
    """A seed, one promoted child, two rejected candidates (a guard and a procedure), traces and evals."""
    store = FakeStore()
    guards = {g.id: g for g in sample_guards()}
    seed = store.add_harness(
        Harness.build(
            parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=["a"], created_by="seed"
        )
    )
    good = store.add_harness(seed.child(guards=[guards["g-retry"]], created_by="improver"))
    store.trace_list = [
        Trace(t.run_id, t.case_id, seed.version, t.model, t.events, t.outcome) for t in sample_traces()
    ]
    store.decision_docs = [
        decision_doc(
            seed.version,
            good.version,
            True,
            change=guard_change(guards["g-retry"]),
            checks=[
                {"name": "replay", "passed": True, "evidence": {"reports": [replay_report("g-retry")]}},
                {"name": "fresh", "passed": True, "evidence": fresh_evidence(STABLE)},
            ],
        ),
        decision_doc(
            seed.version,
            "rej00000001",
            False,
            generation=1,
            change=guard_change(guards["g-all-refunds"]),
            extra_context={"id": "c-g-all-refunds"},
            checks=[
                {
                    "name": "replay",
                    "passed": False,
                    "evidence": {
                        "reports": [
                            replay_report(
                                "g-all-refunds",
                                "REJECTED",
                                false_fires=["a@1", "b@1", "c@1"],
                                benign_twin_fires=1,
                            )
                        ]
                    },
                },
                {"name": "fresh", "passed": False, "evidence": fresh_evidence(REJECTED_RUNS)},
            ],
        ),
        decision_doc(
            seed.version,
            "rej00000002",
            False,
            kind="procedure",
            generation=1,
            change={"name": "double check"},
            extra_context={"id": "c-p-1"},
            checks=[
                {"name": "benign_twins", "passed": False, "evidence": {"intervened": ["refund.w1"]}},
            ],
        ),
    ]
    store.eval_docs = [
        eval_doc(seed.version, (2, 5), (3, 6), [{"interventions": 0, "kind": "trap"}]),
        eval_doc(
            good.version,
            (4, 5),
            (5, 6),
            [{"interventions": 1, "kind": "trap"}, {"interventions": 2, "kind": "benign_twin"}],
        ),
    ]
    return store, seed, good


def fresh_evidence(outcomes: dict[str, list[str]], *, pending: bool = False) -> dict[str, Any]:
    """The `fresh` gate's evidence for outcome vectors over P1, P2, C1 (and C2 unless pending), by the
    contract's rule: stable recoveries fail P1 and P2 and pass C1 and C2; protected cases pass P1 or P2 and
    must pass C1 and C2."""
    protected = sorted(k for k, v in outcomes.items() if "pass" in v[:2])
    need = 3 if pending else 4
    broken = sorted(k for k in protected if any(x != "pass" for x in outcomes[k][2:need]))
    stable = sorted(
        k for k, v in outcomes.items()
        if v[:2] == ["fail", "fail"] and all(x == "pass" for x in v[2:need]) and not pending
    )  # fmt: skip
    c1 = sorted(k for k, v in outcomes.items() if v[:2] == ["fail", "fail"] and v[2] == "pass")
    discordant = sorted(k for k, v in outcomes.items() if v[0] != v[1])
    return {
        "manifest": len(outcomes), "same_manifest": True, "order": ["P1", "P2", "C1", "C2"],
        "outcomes": {k: v[:need] for k, v in outcomes.items()},
        "stable_recoveries": stable, "stable_recoveries_c1": c1, "protected": protected,
        "protected_broken": broken, "parent_discordance": len(discordant), "discordant": discordant,
        "invalid_evidence": [], "initial_ok": not broken and len(c1) >= 2, "confirm_pending": pending,
        "runs": {},
    }  # fmt: skip


def replay_report(
    guard_id: str, verdict: str = "CLEARED", coverage: str = "11/11 traces checked", **kw: Any
) -> dict[str, Any]:
    return {
        "guard_id": guard_id, "passed": verdict == "CLEARED", "verdict": verdict, "coverage": coverage,
        "targets_hit": 1, "targets_total": 1, "false_fires": [], "benign_twin_fires": 0, "errors": 0, **kw,
    }  # fmt: skip


REJECTED_RUNS = {
    "inj.v3": ["fail", "fail", "pass", "pass"],
    "inj.v4": ["fail", "fail", "fail", "fail"],
    "ord.v3": ["pass", "pass", "pass", "fail"],
    "ref.v4": ["pass", "fail", "pass", "pass"],
}

STABLE = {
    "inj.v3": ["fail", "fail", "pass", "pass"],  # a stable recovery
    "inj.v4": ["fail", "fail", "pass", "pass"],  # a stable recovery
    "ref.v3": ["fail", "fail", "fail", "fail"],  # still failing
    "ord.v3": ["pass", "pass", "pass", "pass"],  # protected, kept
    "ref.v4": ["pass", "fail", "pass", "pass"],  # protected (parent discordant), kept
}
