"""The held-out referee: the one process that runs, reads and grades held-out cases, under its own key.

The loop (the improver) holds training cases only. Everything that touches a held-out case or its trace runs
here: the parent's held-out runs, each candidate's held-out run and confirming run, the referee-side replay
over every passing and benign-twin trace (held-out included), the admission rule, the decision signature
(the referee's key, which the improver never holds) and activation. The improver sends training evidence
and gets back totals: held-out passes and counts, the admission flag, the names of failed gates and the
receipt digest. It never receives a held-out case id, outcome or trace.

`ProcessReferee` runs the referee as a separate Python process, built in that process by a factory named
`module:function` from a JSON config, and speaks newline-delimited JSON to it; the referee parses data only,
never pickles. With an Atlas store, the referee's records live in the held-out database, which the
improver's database role cannot read (tests/store/test_store_roles.py). `RefereeCore` is the logic itself,
used inside that process and directly in tests.

    python -m ratchet.improve.referee FACTORY      (started by ProcessReferee; the protocol is on stdio)
"""

from __future__ import annotations

import base64
import contextlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import IO, Any, Protocol

from ratchet_cases.types import Trace, sha256_hex

from ratchet.canonical import canonical_json
from ratchet.improve import wire
from ratchet.improve.activate import Activation, Activator
from ratchet.improve.admit import Decision, admit, awaiting_confirmation
from ratchet.improve.diagnose import FailureCluster
from ratchet.improve.evaluate import CaseLike, EvalReport, RunCase, evaluate_many
from ratchet.improve.evaluator import evaluator_identity, require_same
from ratchet.improve.overfit import Literals
from ratchet.improve.receipt import Signer, receipt
from ratchet.improve.replay import GuardRunner, ReplayReport, WalkCache, replay
from ratchet.improve.sim import Simulator
from ratchet.types import Harness


class RefereeError(RuntimeError):
    """The referee refused a request or failed; the message is the referee's own."""


class RefereeStore(Protocol):
    def put_trace(self, trace: Trace) -> bool: ...
    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str: ...
    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str: ...


@dataclass
class RefereeParts:
    """What the referee holds. Built inside the referee's process by its factory."""

    cases: Sequence[CaseLike]
    run_case: RunCase
    model: Any
    simulator: Simulator
    runner: GuardRunner
    signer: Signer
    store: RefereeStore
    activator: Activator
    model_config: dict[str, Any] = field(default_factory=dict)
    suite_digest: str | None = None
    workers: int = 8
    repeats: int = 1
    close: Callable[[], None] = lambda: None


@dataclass(frozen=True)
class Totals:
    """What the improver may know about one run: counts, never which cases."""

    harness: str
    train_passed: int
    train_total: int
    heldout_passed: int
    heldout_total: int
    heldout_invalid: int


@dataclass(frozen=True)
class Verdict:
    """The referee's answer for one candidate: totals, the admission flag and the receipt that records it."""

    decision_id: str
    receipt_digest: str
    candidate: str
    admitted: bool
    failed: tuple[str, ...]
    heldout_passed: int
    heldout_total: int
    stable_recoveries: int
    rates: dict[str, Any]


class Referee(Protocol):
    """The improver's view of the referee."""

    def start(self, train_identity: Mapping[str, Any]) -> dict[str, Any]: ...
    def bootstrap(self, version: str) -> tuple[str | None, int]: ...
    def parent_runs(self, generation: int, parent: Harness, train: EvalReport) -> tuple[Totals, Totals]: ...
    def evaluate(self, role: str, harness: Harness, train: EvalReport) -> Totals: ...
    def decide(
        self,
        generation: int,
        candidate: Harness,
        train: EvalReport,
        targets: Sequence[FailureCluster],
        guard_clusters: Mapping[str, str],
        literals: Literals,
        visible_case_ids: Sequence[str],
        context: Mapping[str, Any],
    ) -> Verdict: ...
    def activate(
        self,
        *,
        candidate: str,
        expected: str | None,
        expected_revision: int,
        decision_id: str,
        receipt_digest: str,
        idempotency_key: str,
    ) -> Activation: ...
    def active(self) -> tuple[str | None, int]: ...
    def public_key(self) -> str: ...
    def close(self) -> None: ...


def combine_identity(train: Mapping[str, Any], heldout: Mapping[str, Any]) -> dict[str, Any]:
    """The evaluator over both splits from the improver's identity of its training cases and the referee's of
    its held-out cases. Component digests are named by split; the suite digest must agree."""
    if train["components"].get("suite_file") != heldout["components"].get("suite_file"):
        raise RefereeError("the improver and the referee name different case suites")
    components = {f"train_{k}": v for k, v in train["components"].items() if k != "suite_file"}
    components |= {f"heldout_{k}": v for k, v in heldout["components"].items() if k != "suite_file"}
    components["suite_file"] = heldout["components"].get("suite_file")
    if dict(train["model_config"]) != dict(heldout["model_config"]):
        raise RefereeError("the improver and the referee run different model configurations")
    return {
        "version": sha256_hex(canonical_json(components))[:12],
        "components": components,
        "model_config": dict(heldout["model_config"]),
        "cases": int(train["cases"]) + int(heldout["cases"]),
    }


def _merge(train: EvalReport, heldout: EvalReport) -> EvalReport:
    if train.harness_version != heldout.harness_version:
        raise RefereeError("the training and held-out runs are of different harnesses")
    if any(r.split == "heldout" for r in train.results):
        raise RefereeError("the improver sent held-out results")
    return EvalReport(
        train.harness_version,
        [*train.results, *heldout.results],
        train.wall_s + heldout.wall_s,
        [*train.traces, *heldout.traces],
    )


def totals(r: EvalReport) -> Totals:
    held = [x for x in r.results if x.split == "heldout"]
    return Totals(
        r.harness_version,
        len(r.passed("train")),
        r.count("train"),
        len(r.passed("heldout")),
        r.count("heldout"),
        sum(1 for x in held if x.infra_failed or x.error),
    )


class RefereeCore:
    """The referee's logic. One generation at a time: `parent_runs`, then `decide` for each candidate."""

    def __init__(self, parts: RefereeParts) -> None:
        if not parts.cases or any(c.split != "heldout" for c in parts.cases):
            raise ValueError("the referee holds held-out cases, and only those")
        self.parts = parts
        self.walks = WalkCache(parts.simulator)
        self.evaluator: dict[str, Any] | None = None
        self._train_identity: dict[str, Any] | None = None
        self._generation: tuple[int, Harness, tuple[EvalReport, EvalReport]] | None = None

    def _heldout_identity(self) -> dict[str, Any]:
        return evaluator_identity(self.parts.cases, self.parts.model_config, self.parts.suite_digest)

    def start(self, train_identity: Mapping[str, Any]) -> dict[str, Any]:
        self._train_identity = dict(train_identity)
        self.evaluator = combine_identity(train_identity, self._heldout_identity())
        self.parts.store.put_eval(
            {"type": "evaluator", **self.evaluator}, eval_id=f"evaluator-{self.evaluator['version']}"
        )
        return self.evaluator

    def bootstrap(self, version: str) -> tuple[str | None, int]:
        return self.parts.activator.bootstrap(version)

    def _run(self, h: Harness) -> EvalReport:
        p = self.parts
        return evaluate_many([h], p.cases, p.run_case, p.model, workers=p.workers, repeats=p.repeats)[0]

    def _persist(self, report: EvalReport, context: dict[str, Any]) -> None:
        for t in report.traces:
            self.parts.store.put_trace(t)
        self.parts.store.put_eval({**report.to_payload(), **context})

    def parent_runs(self, generation: int, parent: Harness, train: EvalReport) -> tuple[Totals, Totals]:
        """The parent runs twice, fresh, on the held-out cases; the first run joins the improver's training
        run as P1, the second is P2. The noise floor is measured, never assumed."""
        if self.evaluator is None:
            raise RefereeError("start the referee first")
        p1 = _merge(train, self._run(parent))
        p2 = self._run(parent)
        self._persist(p1, {"generation": generation, "role": "parent_run_1"})
        self._persist(p2, {"generation": generation, "role": "parent_run_2"})
        self._generation = (generation, parent, (p1, p2))
        return totals(p1), totals(p2)

    def evaluate(self, role: str, harness: Harness, train: EvalReport) -> Totals:
        """One run of `harness` on the held-out cases, joined to the improver's training run and kept under
        `role` (a baseline, say). Outside any generation; the improver gets the counts."""
        report = _merge(train, self._run(harness))
        self._persist(report, {"role": role})
        return totals(report)

    def _sets(self) -> tuple[list[Trace], list[Trace]]:
        """The clearance traces: every passing parent trace and every benign-twin trace, held-out included."""
        assert self._generation is not None
        p1 = self._generation[2][0]
        benign = {r.case_id for r in p1.results if r.kind == "benign_twin"}
        return [t for t in p1.traces if t.outcome.passed], [t for t in p1.traces if t.case_id in benign]

    def _replays(
        self,
        generation: int,
        parent: Harness,
        candidate: Harness,
        targets: Sequence[FailureCluster],
        guard_clusters: Mapping[str, str],
    ) -> tuple[dict[str, ReplayReport], ReplayReport | None]:
        """Rule 1 on the referee's side: each added guard against the failures it was written for, then every
        guard of the candidate at once, over every passing and benign-twin parent trace."""
        passing, benign = self._sets()
        by_id = {c.id: c for c in targets}
        old = {g.id for g in parent.guards}
        added = [g for g in candidate.guards if g.id not in old]
        out: dict[str, ReplayReport] = {}
        for guard in added:
            cl = by_id.get(guard_clusters.get(guard.id, ""))
            if cl is None:
                raise RefereeError(f"guard {guard.id} names no target cluster")
            out[guard.id] = replay(
                guard, cl.members, cl.traces, passing, self.walks, self.parts.runner, benign
            )
            self._persist_replay(generation, parent, out[guard.id])
        composed = None
        if added and targets:
            members = [m for c in targets for m in c.members]
            failing = {k: v for c in targets for k, v in c.traces.items()}
            composed = replay(
                candidate.guards, members, failing, passing, self.walks, self.parts.runner, benign
            )
            self._persist_replay(generation, parent, composed)
        return out, composed

    def _persist_replay(self, generation: int, parent: Harness, report: ReplayReport) -> None:
        self.parts.store.put_eval(
            {
                "type": "replay",
                "role": "referee",
                "generation": generation,
                "parent": parent.version,
                **report.to_payload(),
            }
        )

    def decide(
        self,
        generation: int,
        candidate: Harness,
        train: EvalReport,
        targets: Sequence[FailureCluster],
        guard_clusters: Mapping[str, str],
        literals: Literals,
        visible_case_ids: Sequence[str],
        context: Mapping[str, Any],
    ) -> Verdict:
        """Run the candidate on the held-out cases, replay its guards, admit or reject, and sign. A candidate
        that clears every gate on its first run gets one more fresh held-out run to confirm."""
        if self._generation is None or self._generation[0] != generation:
            raise RefereeError(f"generation {generation} has no parent runs")
        if self.evaluator is None or self._train_identity is None:
            raise RefereeError("start the referee first")
        require_same(self.evaluator, combine_identity(self._train_identity, self._heldout_identity()))
        _, parent, parent_runs = self._generation
        if candidate.parent != parent.version:
            raise RefereeError(f"candidate {candidate.version} is not a child of {parent.version}")
        c1 = _merge(train, self._run(candidate))
        self._persist(c1, {"generation": generation, "role": "candidate", "harness": candidate.version})
        replays, composed = self._replays(generation, parent, candidate, targets, guard_clusters)
        kw: dict[str, Any] = {
            "parent_runs": parent_runs,
            "cand_eval": c1,
            "replays": replays,
            "composed": composed,
            "literals": literals,
            "visible_case_ids": visible_case_ids,
            "evaluator": self.evaluator,
            "context": dict(context),
        }
        d = admit(parent, candidate, confirm_eval=None, **kw)
        if awaiting_confirmation(d):
            confirm = self._run(candidate)
            self._persist(
                confirm, {"generation": generation, "role": "confirm", "harness": candidate.version}
            )
            d = admit(parent, candidate, confirm_eval=confirm, **kw)
        return self._sign(d, c1)

    def _sign(self, d: Decision, c1: EvalReport) -> Verdict:
        signer = self.parts.signer
        env = receipt(d, signer)
        decision_id = f"{d.parent}->{d.candidate}"
        self.parts.store.put_decision(
            {**d.to_payload(), "envelope": base64.b64encode(env).decode("ascii"), "keyid": signer.keyid},
            decision_id=decision_id,
        )
        held = totals(c1)
        return Verdict(
            decision_id,
            sha256_hex(env),
            d.candidate,
            d.admitted,
            tuple(c.name for c in d.checks if not c.passed),
            held.heldout_passed,
            held.heldout_total,
            len(d.check("heldout").evidence["stable_recoveries"]),
            dict(d.rates),
        )

    def activate(
        self,
        *,
        candidate: str,
        expected: str | None,
        expected_revision: int,
        decision_id: str,
        receipt_digest: str,
        idempotency_key: str,
    ) -> Activation:
        return self.parts.activator.activate(
            candidate=candidate,
            expected=expected,
            expected_revision=expected_revision,
            decision_id=decision_id,
            receipt_digest=receipt_digest,
            idempotency_key=idempotency_key,
        )

    def active(self) -> tuple[str | None, int]:
        return self.parts.activator.active()

    def public_key(self) -> str:
        """The decision key's raw public key, hex: what a run publishes so its receipts can be checked."""
        from ratchet.verify.dsse import raw_public

        return raw_public(self.parts.signer.public_key).hex()

    def close(self) -> None:
        self.parts.close()

    def __enter__(self) -> RefereeCore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------------------------------------
# The process boundary
# ---------------------------------------------------------------------------------------------------------

Factory = Callable[[Mapping[str, Any]], RefereeParts]


def load_factory(name: str) -> Factory:
    module, _, attr = name.partition(":")
    if not module or not attr:
        raise RefereeError(f"factory {name!r} is not module:function")
    fn = getattr(importlib.import_module(module), attr)
    if not callable(fn):
        raise RefereeError(f"factory {name!r} is not callable")
    return fn  # type: ignore[no-any-return]


def _rates(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dispatch(core: RefereeCore, request: Mapping[str, Any]) -> Any:
    method, a = request["method"], request.get("params", {})
    if method == "start":
        return core.start(a["train_identity"])
    if method == "bootstrap":
        return list(core.bootstrap(str(a["version"])))
    if method == "parent_runs":
        p1, p2 = core.parent_runs(int(a["generation"]), wire.harness(a["parent"]), wire.report(a["train"]))
        return [asdict(p1), asdict(p2)]
    if method == "evaluate":
        return asdict(core.evaluate(str(a["role"]), wire.harness(a["harness"]), wire.report(a["train"])))
    if method == "decide":
        v = core.decide(
            int(a["generation"]),
            wire.harness(a["candidate"]),
            wire.report(a["train"]),
            [wire.cluster(c) for c in a["targets"]],
            {str(k): str(v) for k, v in a["guard_clusters"].items()},
            wire.literals(a["literals"]),
            [str(x) for x in a["visible_case_ids"]],
            dict(a["context"]),
        )
        return asdict(v)
    if method == "active":
        return list(core.active())
    if method == "public_key":
        return core.public_key()
    if method == "activate":
        return asdict(
            core.activate(
                **{
                    k: a[k]
                    for k in (
                        "candidate",
                        "expected",
                        "expected_revision",
                        "decision_id",
                        "receipt_digest",
                        "idempotency_key",
                    )
                }
            )
        )
    raise RefereeError(f"unknown method {method!r}")


def serve(factory: Factory, read: IO[str], write: IO[str]) -> int:
    """The referee's side: the first line is the config; then one JSON request per line, one answer each."""
    try:
        core = RefereeCore(factory(json.loads(read.readline())))
    except Exception as exc:
        return _refuse_start(write, exc)
    write.write(json.dumps({"ok": "ready"}) + "\n")
    write.flush()
    try:
        for line in read:
            request = json.loads(line)
            if request.get("method") == "close":
                write.write(json.dumps({"ok": None}) + "\n")
                write.flush()
                return 0
            try:
                answer = {"ok": _dispatch(core, request)}
            except Exception as exc:
                answer = {"error": f"{type(exc).__name__}: {exc}"}
            write.write(json.dumps(answer, default=str) + "\n")
            write.flush()
    finally:
        core.close()
    return 0


def _refuse_start(write: IO[str], exc: BaseException) -> int:
    write.write(json.dumps({"error": f"the referee could not start: {type(exc).__name__}: {exc}"}) + "\n")
    write.flush()
    return 1


class ProcessReferee:
    """The referee in its own process. Requests and answers are JSON lines on the child's stdin and stdout;
    anything the referee's code prints goes to its stderr."""

    def __init__(
        self,
        factory: str,
        config: Mapping[str, Any],
        *,
        env: Mapping[str, str] | None = None,
        python: str = sys.executable,
    ) -> None:
        self._proc = subprocess.Popen(
            [python, "-m", "ratchet.improve.referee", factory],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=dict(env) if env is not None else None,
        )
        try:
            self._send({**dict(config)})
            if self._read() != "ready":
                raise RefereeError("the referee did not start")
        except BaseException:
            self._end()
            raise

    def _send(self, doc: Mapping[str, Any]) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(doc) + "\n")
        self._proc.stdin.flush()

    def _read(self) -> Any:
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            try:
                status = self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                status = None
            raise RefereeError(f"the referee closed its output (exit status {status})")
        answer = json.loads(line)
        if "error" in answer:
            raise RefereeError(str(answer["error"]))
        return answer["ok"]

    def _call(self, method: str, **params: Any) -> Any:
        self._send({"method": method, "params": params})
        return self._read()

    def start(self, train_identity: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self._call("start", train_identity=dict(train_identity))
        return result

    def bootstrap(self, version: str) -> tuple[str | None, int]:
        active, revision = self._call("bootstrap", version=version)
        return active, int(revision)

    def parent_runs(self, generation: int, parent: Harness, train: EvalReport) -> tuple[Totals, Totals]:
        p1, p2 = self._call(
            "parent_runs",
            generation=generation,
            parent=wire.harness_doc(parent),
            train=wire.report_doc(train),
        )
        return Totals(**p1), Totals(**p2)

    def evaluate(self, role: str, harness: Harness, train: EvalReport) -> Totals:
        t = self._call("evaluate", role=role, harness=wire.harness_doc(harness), train=wire.report_doc(train))
        return Totals(**t)

    def decide(
        self,
        generation: int,
        candidate: Harness,
        train: EvalReport,
        targets: Sequence[FailureCluster],
        guard_clusters: Mapping[str, str],
        literals: Literals,
        visible_case_ids: Sequence[str],
        context: Mapping[str, Any],
    ) -> Verdict:
        v = self._call(
            "decide",
            generation=generation,
            candidate=wire.harness_doc(candidate),
            train=wire.report_doc(train),
            targets=[wire.cluster_doc(c) for c in targets],
            guard_clusters=dict(guard_clusters),
            literals=wire.literals_doc(literals),
            visible_case_ids=list(visible_case_ids),
            context=dict(context),
        )
        return Verdict(**{**v, "failed": tuple(v["failed"])})

    def activate(
        self,
        *,
        candidate: str,
        expected: str | None,
        expected_revision: int,
        decision_id: str,
        receipt_digest: str,
        idempotency_key: str,
    ) -> Activation:
        a = self._call(
            "activate",
            candidate=candidate,
            expected=expected,
            expected_revision=expected_revision,
            decision_id=decision_id,
            receipt_digest=receipt_digest,
            idempotency_key=idempotency_key,
        )
        return Activation(**a)

    def active(self) -> tuple[str | None, int]:
        version, revision = self._call("active")
        return version, int(revision)

    def public_key(self) -> str:
        return str(self._call("public_key"))

    def close(self) -> None:
        if self._proc.poll() is None:
            with contextlib.suppress(RefereeError, BrokenPipeError):
                self._call("close")
        self._end()

    def _end(self) -> None:
        """Close the child's input, reap it (killed if it has not exited in a minute), close its output."""
        stdin = self._proc.stdin
        if stdin is not None and not stdin.closed:
            with contextlib.suppress(BrokenPipeError):
                stdin.close()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        if self._proc.stdout is not None:
            self._proc.stdout.close()

    def __enter__(self) -> ProcessReferee:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m ratchet.improve.referee module:function", file=sys.stderr)
        return 2
    # The protocol owns the real stdout; everything else the referee's code prints goes to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        factory = load_factory(args[0])
    except Exception as exc:
        return _refuse_start(protocol, exc)
    return serve(factory, sys.stdin, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
