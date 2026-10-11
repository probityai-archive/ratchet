"""The `ratchet` command: run a harness, evolve it, replay a candidate, read the record, verify a receipt.

Every command reads and writes through a `Backend`, which holds the wiring to the other lanes (store,
agent, cases, improver). `LiveBackend` imports them lazily; tests hand in a fake. The views live in
`ratchet.view` and take plain data.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from rich.console import Console
from rich.text import Text

from ratchet.improve.replay import GuardRunner, WalkCache
from ratchet.improve.sim import Simulator
from ratchet.types import Harness, Trace
from ratchet.view import resume as resume_view
from ratchet.view.data import (
    CandidateRef,
    RunStore,
    ViewStore,
    WriteStore,
    build_lineage,
    decision_info,
    find_decision_doc,
    resolve_candidates,
)
from ratchet.view.evidence import collect
from ratchet.view.lineage import show_lineage
from ratchet.view.live import DEFAULT_COLLECTIONS, follow
from ratchet.view.matrix import (
    MAX_COLUMNS,
    CaseInfo,
    Matrix,
    MatrixColumn,
    compute_matrix,
    letters,
    show_matrix,
)
from ratchet.view.report import ReportData, render_html
from ratchet.view.show import decision_view, run_view, trace_view
from ratchet.view.telemetry import Sink, TelemetryStore, usage_line
from ratchet.view.verify import render_verify, trusted_keys, verify_decision, verify_envelope

URI_ENV, DB_ENV = "RATCHET_MONGODB_URI", "RATCHET_DB"


class CliError(Exception):
    """A problem the user can fix; printed without a traceback."""


class EvolveSummary(Protocol):
    @property
    def harness(self) -> Harness: ...
    @property
    def generations(self) -> Sequence[Any]: ...


class Backend(Protocol):
    """The wiring to the other lanes."""

    def open_store(self) -> AbstractContextManager[WriteStore]: ...
    def model(self) -> Any: ...
    def seed(self) -> Harness: ...
    def cases(self, split: str, family: str | None, ids: Sequence[str]) -> list[Any]: ...
    def case_info(self, case_id: str) -> CaseInfo | None: ...
    def get_case(self, case_id: str) -> Any: ...
    def resume_case(
        self, harness: Harness, case: Any, model: Any, store: WriteStore, run_id: str
    ) -> Trace: ...
    def evaluate(
        self,
        harness: Harness,
        cases: Sequence[Any],
        model: Any,
        store: WriteStore | None,
        workers: int,
        repeats: int,
    ) -> Any: ...
    def evolve(
        self,
        store: RunStore,
        model: Any,
        generations: int,
        cases: Sequence[Any],
        workers: int,
        repeats: int,
        log: Callable[[str], None],
    ) -> EvolveSummary: ...
    def runner(self) -> AbstractContextManager[GuardRunner]: ...
    def simulator(self) -> Simulator: ...
    def telemetry(self) -> Sink: ...


class LiveBackend:
    """Imports each lane on first use, so `ratchet --help` and the read-only views stay light."""

    def __init__(self, env: dict[str, str] | None = None) -> None:
        self.env = dict(os.environ) if env is None else env
        self._store: Any = None

    def open_store(self) -> AbstractContextManager[WriteStore]:
        from ratchet.store import Store

        uri = self.env.get(URI_ENV)
        if not uri:
            raise CliError(
                f"{URI_ENV} is not set (local Atlas: docker compose -f docker/compose.yaml up -d --wait)"
            )
        store = Store(uri, self.env.get(DB_ENV, "ratchet"))
        self._store = store
        return store

    def model(self) -> Any:
        from ratchet.agent.model import ModelClient

        return ModelClient.from_env(self.env)

    def seed(self) -> Harness:
        from ratchet.agent.seed import seed_harness

        return seed_harness()

    def cases(self, split: str, family: str | None, ids: Sequence[str]) -> list[Any]:
        from ratchet.cases import all_cases

        wanted = ("train", "heldout") if split == "all" else (split,)
        found = [
            c
            for c in all_cases()
            if c.split in wanted and (family is None or c.family == family) and (not ids or c.id in ids)
        ]
        if not found:
            raise CliError("no cases match that selection")
        return found

    def get_case(self, case_id: str) -> Any:
        from ratchet.cases import get

        try:
            return get(case_id)
        except KeyError as exc:
            raise CliError(f"unknown case {case_id!r}") from exc

    def resume_case(self, harness: Harness, case: Any, model: Any, store: WriteStore, run_id: str) -> Trace:
        from ratchet.agent.worker import run_case

        return run_case(harness, case, model, store, run_id=run_id)

    def case_info(self, case_id: str) -> CaseInfo | None:
        from ratchet.cases import all_cases

        for c in all_cases():
            if c.id == case_id:
                return CaseInfo(c.kind, c.family, c.split)
        return None

    def evaluate(
        self,
        harness: Harness,
        cases: Sequence[Any],
        model: Any,
        store: WriteStore | None,
        workers: int,
        repeats: int,
    ) -> Any:
        from ratchet.agent.worker import run_case
        from ratchet.improve.evaluate import evaluate

        return evaluate(harness, cases, run_case, model, store=store, workers=workers, repeats=repeats)

    def evolve(
        self,
        store: RunStore,
        model: Any,
        generations: int,
        cases: Sequence[Any],
        workers: int,
        repeats: int,
        log: Callable[[str], None],
    ) -> EvolveSummary:
        from ratchet.agent.guards import GuardRuntime
        from ratchet.agent.worker import run_case
        from ratchet.cases import all_cases
        from ratchet.improve.activate import MongoActivator
        from ratchet.improve.loop import Lab, evolve
        from ratchet.improve.receipt import Signer
        from ratchet.improve.sandbox import SandboxRunner
        from ratchet.improve.worldsim import WorldSimulator

        if self._store is None:
            raise CliError("evolve needs the store open first (open_store)")
        activator = MongoActivator(self._store.db.client, self._store.db.name)
        with GuardRuntime() as runtime:
            lab = Lab(
                cases=cases,
                run_case=run_case,
                simulator=WorldSimulator(all_cases()),
                runner=SandboxRunner(runtime),
                signer=Signer.from_env(),
                activator=activator,
                workers=workers,
                repeats=repeats,
                log=log,
            )
            return evolve(store, model, generations, lab=lab, seed=self.seed())

    def runner(self) -> AbstractContextManager[GuardRunner]:
        from ratchet.agent.guards import GuardRuntime
        from ratchet.improve.sandbox import SandboxRunner

        return _Sandbox(GuardRuntime(), SandboxRunner)

    def simulator(self) -> Simulator:
        from ratchet.cases import all_cases
        from ratchet.improve.worldsim import WorldSimulator

        return WorldSimulator(all_cases())

    def telemetry(self) -> Sink:
        from ratchet.telemetry import RaindropSink, ledger_default_path, resolve_write_key

        key = resolve_write_key(use_keyring=True)
        if key is None:
            raise CliError(
                "--raindrop needs RAINDROP_WRITE_KEY or keyring service raindrop account write-key"
            )
        return RaindropSink(key, ledger_path=ledger_default_path())


class _Sandbox:
    """Keeps the guard runtime open for the length of a `with` block and closes its child processes after."""

    def __init__(self, runtime: Any, make: Callable[[Any], GuardRunner]) -> None:
        self.runtime, self.make = runtime, make

    def __enter__(self) -> GuardRunner:
        return self.make(self.runtime)

    def __exit__(self, *exc: object) -> None:
        self.runtime.close()


# -- commands ---------------------------------------------------------------------------------------------


def _pick_harness(backend: Backend, store: WriteStore, version: str) -> Harness:
    if version == "seed":
        return backend.seed()
    found = store.get_harness(version)
    if found is None:
        raise CliError(f"no harness version {version!r} in the store (use `ratchet show lineage`)")
    return found


@contextmanager
def _telemetry(args: argparse.Namespace, backend: Backend, console: Console) -> Iterator[Sink | None]:
    """The Raindrop sink when `--raindrop` is on. Closed on the way out, with the monthly count printed."""
    if not args.raindrop:
        yield None
        return
    sink = backend.telemetry()
    try:
        yield sink
    finally:
        try:
            sink.close()
            console.print(Text(usage_line(sink), style="dim"))
        except Exception as exc:  # telemetry never fails the command
            console.print(Text(f"raindrop: close failed: {type(exc).__name__}: {exc}", style="yellow"))


def _label(args: argparse.Namespace) -> str:
    return str(args.experiment or f"ratchet-{datetime.now(UTC):%Y%m%d-%H%M}")


def cmd_run(args: argparse.Namespace, backend: Backend, console: Console) -> int:
    cases = backend.cases(args.split, args.family, args.case or [])
    with _telemetry(args, backend, console) as sink, backend.open_store() as store:
        harness = _pick_harness(backend, store, args.harness)
        store.put_harness(harness)
        console.print(Text(f"running {harness.version} on {len(cases)} cases ({args.split})", style="dim"))
        report = backend.evaluate(harness, cases, backend.model(), store, args.workers, args.repeats)
        payload = {**report.to_payload(), "role": "run", "split_selected": args.split}
        for t in report.traces:
            store.put_trace(t)
        store.put_eval(payload)
        if sink is not None:
            _emit_run(sink, report, cases, args.cohort, _label(args), console)
    console.print(run_view(report.to_payload()))
    return 0


def _emit_run(
    sink: Sink, report: Any, cases: Sequence[Any], cohort: str, label: str, console: Console
) -> None:
    try:
        summary = sink.emit_evaluation(report, cases, cohort=cohort, experiment=label)
        if getattr(summary, "refused", None):
            console.print(Text(f"raindrop: not sent: {summary.refused}", style="yellow"))
    except Exception as exc:  # telemetry never fails the run
        console.print(Text(f"raindrop: {type(exc).__name__}: {exc}", style="yellow"))


def cmd_resume(args: argparse.Namespace, backend: Backend, console: Console) -> int:
    with backend.open_store() as store:
        saved = store.resume(args.run_id)
        if saved is None:
            raise CliError(f"no checkpoint for run {args.run_id!r}: it never ran, or it ran without a store")
        if saved.get("status") != "running":
            raise CliError(f"run {args.run_id!r} already finished ({saved.get('status')}); nothing to resume")
        version = str(saved["harness_version"])
        harness = store.get_harness(version) or backend.seed()
        if harness.version != version:
            raise CliError(f"the harness {version} of this run is not in the store")
        case = backend.get_case(str(saved["case_id"]))
        console.print(resume_view.before(resume_view.build_report(saved)))
        trace = backend.resume_case(harness, case, backend.model(), store, args.run_id)
        store.put_trace(trace)
        console.print()
        console.print(resume_view.after(resume_view.build_report(saved, trace)))
    return 0


def cmd_evolve(args: argparse.Namespace, backend: Backend, console: Console) -> int:
    cases = backend.cases("all", None, [])
    with _telemetry(args, backend, console) as sink, backend.open_store() as store:

        def log(msg: str) -> None:
            console.print(Text(msg, style="dim"))

        def warn(msg: str) -> None:
            console.print(Text(msg, style="yellow"))

        target: RunStore = store if sink is None else TelemetryStore(store, sink, cases, _label(args), warn)
        result = backend.evolve(
            target, backend.model(), args.generations, cases, args.workers, args.repeats, log
        )
        console.print()
        for g in result.generations:
            console.print(_generation_line(g))
        console.print()
        show_lineage(console, build_lineage(store))
        console.print(Text(f"head: {result.harness.version}", style="bold cyan"))
    return 0


def _generation_line(g: Any) -> Text:
    level1 = getattr(g, "level1", None) or {}
    before, after = level1.get("parent_score") or {}, level1.get("child_score") or {}
    t = Text.assemble((f"generation {g.generation}", "bold"), f"  parent {g.parent}")
    if g.promoted:
        t.append(f"  promoted {g.promoted}", style="bold green")
        held = f"{before.get('heldout_passed')} -> {after.get('heldout_passed')}/{after.get('heldout_total')}"
        t.append(f"  held-out {held}")
    else:
        t.append(f"  nothing admitted{(': ' + g.note) if g.note else ''}", style="yellow")
    return t


def _columns(refs: Sequence[CandidateRef], limit: int, console: Console) -> list[MatrixColumn]:
    chosen: list[tuple[CandidateRef, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        if not ref.guards:
            console.print(Text(f"{ref.candidate}: {ref.note or 'no new guard to replay'}", style="yellow"))
        for g in ref.guards:
            if g.id not in seen:
                seen.add(g.id)
                chosen.append((ref, g))
    if len(chosen) > limit:
        console.print(
            Text(f"showing the first {limit} of {len(chosen)} guards (--max-columns)", style="yellow")
        )
        chosen = chosen[:limit]
    return [
        MatrixColumn(g, ref.candidate, letter, ref.decision)
        for (ref, g), letter in zip(chosen, letters(len(chosen)), strict=True)
    ]


def build_matrices(
    backend: Backend, store: ViewStore, candidates: Sequence[str], limit: int, console: Console
) -> list[Matrix]:
    """One matrix per parent harness: its candidates are replayed over that parent's own traces."""
    refs = resolve_candidates(store, candidates)
    if not refs:
        raise CliError(
            "no candidates recorded yet (run `ratchet evolve`, or name one: `ratchet replay <version>`)"
        )
    orphan = next((r for r in refs if not r.parent), None)
    if orphan is not None:
        console.print(Text(f"{orphan.candidate}: {orphan.note or 'no known parent'}", style="yellow"))
    out: list[Matrix] = []
    parents = list(dict.fromkeys(r.parent for r in refs if r.parent))
    if not parents:
        raise CliError("cannot tell which harness these candidates descend from")
    for parent in parents:
        columns = _columns([r for r in refs if r.parent == parent], limit, console)
        if not columns:
            continue
        traces = store.traces({"harness_version": parent})
        if not traces:
            console.print(Text(f"no traces recorded for parent harness {parent}", style="yellow"))
            continue
        with backend.runner() as runner:
            walks = WalkCache(backend.simulator())
            out.append(compute_matrix(parent, columns, traces, backend.case_info, walks, runner))
    if not out:
        raise CliError("nothing to replay: no candidate adds a guard, or its parent has no recorded traces")
    return out


def cmd_replay(args: argparse.Namespace, backend: Backend, console: Console) -> int:
    with backend.open_store() as store:
        matrices = build_matrices(backend, store, args.candidate, args.max_columns, console)
    for matrix in matrices:
        show_matrix(console, matrix, delay=args.delay, full=args.full)
    return 0


def cmd_show(args: argparse.Namespace, backend: Backend, console: Console) -> int:
    with backend.open_store() as store:
        if args.what == "lineage":
            show_lineage(console, build_lineage(store))
        elif args.what == "trace":
            console.print(_load_trace(store, _need(args, "trace")))
        else:
            doc = find_decision_doc(store, _need(args, "decision"))
            if doc is None:
                raise CliError(f"no decision {args.id!r}")
            console.print(decision_view(decision_info(doc)))
            if args.envelope_out:
                _write_envelope(doc, Path(args.envelope_out), console)
    return 0


def _write_envelope(doc: dict[str, Any], target: Path, console: Console) -> None:
    raw = doc.get("envelope")
    if not raw:
        raise CliError("this decision carries no signed envelope")
    target.write_bytes(base64.b64decode(str(raw)))
    console.print(
        Text(f"wrote the signed envelope to {target}; check it: ratchet verify {target}", style="green")
    )


def _need(args: argparse.Namespace, what: str) -> str:
    if not args.id:
        raise CliError(f"`ratchet show {what}` needs an id")
    return str(args.id)


def _load_trace(store: ViewStore, ref: str) -> Any:
    if "/" in ref:
        run_id, _, case_id = ref.partition("/")
        found = store.get_trace(run_id, case_id)
    else:
        many = store.traces({"run_id": ref})
        found = many[0] if many else None
    if found is None:
        raise CliError(f"no trace {ref!r} (use run_id or run_id/case_id)")
    return trace_view(found)


def cmd_verify(args: argparse.Namespace, backend: Backend, console: Console) -> int:
    pem = Path(args.pubkey_file) if args.pubkey_file else None
    trusted = trusted_keys(args.pubkey, backend_env(backend), pem)
    envelope_file = args.envelope or (
        args.decision if args.decision and Path(args.decision).is_file() else None
    )
    if envelope_file:
        res = verify_envelope(Path(envelope_file).read_bytes(), trusted)
        name = str(envelope_file)
    else:
        with backend.open_store() as store:
            doc = find_decision_doc(store, args.decision)
            if doc is None:
                raise CliError(f"no decision {args.decision!r}")
            evidence = collect(store, doc, backend.runner, backend.simulator())
        res, name = verify_decision(doc, trusted, evidence), str(doc.get("_id"))
        for note in evidence.notes:
            console.print(Text(f"evidence: {note}", style="yellow"))
    console.print(render_verify(res, name))
    return 0 if res.ok else 1


def backend_env(backend: Backend) -> dict[str, str]:
    env = getattr(backend, "env", None)
    return dict(env) if isinstance(env, dict) else dict(os.environ)


def cmd_live(args: argparse.Namespace, backend: Backend, console: Console) -> int:
    with backend.open_store() as store:
        console.print(
            Text(f" RATCHET  live: {', '.join(args.collections)}", style="bold black on bright_cyan")
        )
        show_lineage(console, build_lineage(store))
        console.print(Text("waiting for changes (ctrl-c to stop)", style="dim"))
        follow(store, console, args.collections, timeout_s=args.timeout, max_events=args.max_events)
    return 0


def cmd_report(args: argparse.Namespace, backend: Backend, console: Console) -> int:
    with backend.open_store() as store:
        roots = build_lineage(store)
        try:
            matrices = build_matrices(backend, store, args.candidate, args.max_columns, console)
        except CliError as exc:
            console.print(Text(f"report has no replay matrix: {exc}", style="yellow"))
            matrices = []
    data = ReportData(args.title, roots, matrices, datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))
    Path(args.output).write_text(render_html(data), encoding="utf-8")
    console.print(Text(f"wrote {args.output}", style="bold green"))
    return 0


# -- parser -----------------------------------------------------------------------------------------------


def _raindrop_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--raindrop",
        action="store_true",
        help="send runs and decisions to Raindrop (key: RAINDROP_WRITE_KEY or keyring raindrop/write-key)",
    )
    p.add_argument("--experiment", help="label that keeps one comparison apart in Raindrop")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ratchet",
        description="An agent harness that turns each failure into a proven guard, with every version and "
        f"proof in MongoDB Atlas. Set {URI_ENV} (and {DB_ENV}) to point at the store.",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    run = sub.add_parser(
        "run",
        help="run one harness version over a case split (all = train + held-out; audit only when named)",
    )
    run.add_argument("--harness", default="seed", help="harness version, or 'seed' (default)")
    run.add_argument("--split", choices=["train", "heldout", "audit", "all"], default="all")
    run.add_argument("--family", help="only this case family")
    run.add_argument("--case", action="append", help="only this case id (repeatable)")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--repeats", type=int, default=1)
    _raindrop_flags(run)
    run.add_argument("--cohort", choices=["baseline", "candidate"], default="baseline")
    run.set_defaults(func=cmd_run)

    rs = sub.add_parser("resume", help="resume an interrupted case run from its Atlas checkpoint")
    rs.add_argument("run_id")
    rs.set_defaults(func=cmd_resume)

    ev = sub.add_parser("evolve", help="run the ratchet loop for N generations")
    ev.add_argument("--generations", "-n", type=int, default=3)
    ev.add_argument("--workers", type=int, default=8)
    ev.add_argument("--repeats", type=int, default=1)
    _raindrop_flags(ev)
    ev.set_defaults(func=cmd_evolve)

    rp = sub.add_parser("replay", help="replay candidate guards over every recorded trace (no model calls)")
    rp.add_argument("candidate", nargs="*", help="candidate versions or ids; default: the latest generation")
    rp.add_argument("--full", action="store_true", help="show every row instead of fitting the terminal")
    rp.add_argument(
        "--delay", type=float, default=0.04, help="seconds between revealed rows (0 = print at once)"
    )
    rp.add_argument("--max-columns", type=int, default=MAX_COLUMNS)
    rp.set_defaults(func=cmd_replay)

    sh = sub.add_parser("show", help="print the lineage, a trace, or a decision")
    sh.add_argument("what", choices=["lineage", "trace", "decision"])
    sh.add_argument("id", nargs="?", help="trace id (run_id or run_id/case_id) or decision id")
    sh.add_argument(
        "--envelope-out", metavar="FILE", help="decision only: write the signed DSSE envelope here"
    )
    sh.set_defaults(func=cmd_show)

    vf = sub.add_parser("verify", help="verify a decision's signed DSSE receipt")
    vf.add_argument(
        "decision", nargs="?", default="", help="decision id (parent->candidate) or candidate version"
    )
    vf.add_argument(
        "--pubkey", help="trusted Ed25519 public key, 32 bytes of hex (default: from RATCHET_SIGNING_KEY)"
    )
    vf.add_argument(
        "--pubkey-file", help="trusted Ed25519 public key as PEM (default: keys/ratchet-decisions.pub)"
    )
    vf.add_argument("--envelope", help="verify a DSSE envelope file instead of a stored decision")
    vf.set_defaults(func=cmd_verify)

    lv = sub.add_parser("live", help="follow the change streams and print what lands")
    lv.add_argument("--collections", nargs="+", default=list(DEFAULT_COLLECTIONS))
    lv.add_argument("--timeout", type=float, default=None, help="stop after this many seconds")
    lv.add_argument("--max-events", type=int, default=None, help="stop after this many lines")
    lv.set_defaults(func=cmd_live)

    rt = sub.add_parser("report", help="write one static HTML page: lineage, replay matrix, results")
    rt.add_argument("candidate", nargs="*")
    rt.add_argument("-o", "--output", default="ratchet-report.html")
    rt.add_argument("--title", default="Ratchet: every change proven before it is kept")
    rt.add_argument("--max-columns", type=int, default=MAX_COLUMNS)
    rt.set_defaults(func=cmd_report)
    return p


def main(
    argv: Sequence[str] | None = None, *, backend: Backend | None = None, console: Console | None = None
) -> int:
    out = console or Console()
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args, backend or LiveBackend(), out))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if os.environ.get("RATCHET_DEBUG"):
            traceback.print_exc()
        err = Console(stderr=True) if console is None else console
        text = str(exc) if isinstance(exc, CliError) else f"{type(exc).__name__}: {exc}"
        err.print(Text(f"ratchet: {text}", style="bold red"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
