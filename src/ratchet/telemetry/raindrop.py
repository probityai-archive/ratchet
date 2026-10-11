"""Raindrop sink: show the harness getting better inside Raindrop's own product.

What is sent, and how Raindrop reads it (docs captured under research/_raw/raindrop/):

- One event per case run, named `support_case`. Input is the customer message, output is the final reply,
  and the tool calls the agent made are attached as tool spans. A call that a guard blocked is an errored
  tool span carrying the block message.
- Signals on that event: `task_success` / `task_failure`, `guard_block` (one per run that was blocked),
  and `unwarranted_intervention` (a guard fired on work that should have gone through).
- One event per admission decision, named `ratchet_admission`, with an `admitted` or `rejected` signal and
  one boolean property per gate.

Cohorts. Raindrop's Experiments compare a Baseline and an Experiment cohort; each is defined by a date range
plus conditions on model, feature flag, property or tool. Both cohorts here are produced on the same day by
the same model, so the only way to tell them apart is a property. Every run event therefore carries:

    harness_version     the 12-hex harness version id (condition "property harness_version = X")
    harness_cohort      "baseline" or "candidate"     (condition "property harness_cohort = ...")
    harness_generation  depth in the lineage, 0 for the seed
    experiment          a label naming the comparison, so several comparisons never mix

Pick `task_failure` as the outcome signal in the experiment. Creating an experiment needs the Pro plan; on
the free Hobby plan the Events page, Signals and a filtered view per cohort still work.

Budget. The free plan ingests 1,000 events a month and then stops for the rest of the month. The sink keeps a
monthly ledger on disk, reserves each event before it is queued and refuses once `max_events` is reached, so a
re-run can never push the account past the cap. Signals are not events and are counted separately.

Safety. The write key is held only by the SDK client and is never logged or put in an event. The sink does
no network work on the caller's thread: events go through a bounded queue to one background thread, and a
full queue drops the event and counts it. With no key the sink is a no-op.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import queue
import subprocess
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ratchet.types import Event, Trace

log = logging.getLogger("ratchet.telemetry.raindrop")

RUN_EVENT = "support_case"
ADMISSION_EVENT = "ratchet_admission"
RUN_USER = "ratchet-customer"
IMPROVER_USER = "ratchet-improver"
MONTHLY_FREE_EVENTS = 1000
DEFAULT_MAX_EVENTS = 900
MAX_REASON_CHARS = 500

SIGNAL_SUCCESS = "task_success"
SIGNAL_FAILURE = "task_failure"
SIGNAL_GUARD = "guard_block"
SIGNAL_UNWARRANTED = "unwarranted_intervention"
SIGNAL_ADMITTED = "admitted"
SIGNAL_REJECTED = "rejected"

COHORT_BASELINE = "baseline"
COHORT_CANDIDATE = "candidate"

Props = dict[str, str | int | float | bool]
SignalSpec = tuple[str, str | None, dict[str, Any]]


class CaseLike(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def family(self) -> str: ...
    @property
    def split(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def must_succeed(self) -> bool: ...
    @property
    def customer_message(self) -> str: ...


class ReportLike(Protocol):
    @property
    def harness_version(self) -> str: ...
    @property
    def traces(self) -> Sequence[Trace]: ...
    @property
    def results(self) -> Sequence[Any]: ...


class DecisionLike(Protocol):
    @property
    def parent(self) -> str: ...
    @property
    def candidate(self) -> str: ...
    @property
    def admitted(self) -> bool: ...
    @property
    def checks(self) -> Sequence[Any]: ...
    @property
    def rates(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CaseInfo:
    """What the sink needs to know about a case, whatever the case model looks like."""

    id: str
    family: str = "unknown"
    split: str = "unknown"
    kind: str = "unknown"
    must_succeed: bool = False
    customer_message: str = ""

    @classmethod
    def of(cls, case: CaseLike) -> CaseInfo:
        return cls(case.id, case.family, case.split, case.kind, case.must_succeed, case.customer_message)


@dataclass
class EmitSummary:
    """What one emit call did. `refused` is set when a whole cohort was held back."""

    sent: int = 0
    signals: int = 0
    duplicates: int = 0
    over_budget: int = 0
    dropped: int = 0
    refused: str | None = None
    event_ids: list[str] = field(default_factory=list)

    def merge(self, other: EmitSummary) -> None:
        self.sent += other.sent
        self.signals += other.signals
        self.duplicates += other.duplicates
        self.over_budget += other.over_budget
        self.dropped += other.dropped
        self.event_ids += other.event_ids


def ledger_default_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ratchet" / "raindrop-ledger.json"


def resolve_write_key(*, use_keyring: bool = False) -> str | None:
    """The write key from RAINDROP_WRITE_KEY, else (opt-in) the system keyring. Never printed or logged."""
    key = os.environ.get("RAINDROP_WRITE_KEY", "").strip()
    if key or not use_keyring:
        return key or None
    try:
        out = subprocess.run(
            ["secret-tool", "lookup", "service", "raindrop", "account", "write-key"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


class _Ledger:
    """Events reserved this calendar month, persisted so separate runs share one budget."""

    def __init__(self, path: Path | None, now: Callable[[], datetime]) -> None:
        self._path = path
        self._now = now
        self._month = self._month_key()
        self._count = self._load()

    def _month_key(self) -> str:
        return self._now().strftime("%Y-%m")

    def _load(self) -> int:
        if self._path is None or not self._path.exists():
            return 0
        try:
            data = json.loads(self._path.read_text())
            return int(data["events"]) if data.get("month") == self._month else 0
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return 0

    @property
    def count(self) -> int:
        if self._month != self._month_key():
            self._month, self._count = self._month_key(), 0
        return self._count

    def add(self, n: int = 1) -> None:
        self._count = self.count + n
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"month": self._month, "events": self._count}))
            tmp.replace(self._path)
        except OSError:
            log.warning("raindrop ledger not written: %s", self._path)


class _Dispatcher:
    """One daemon thread behind a bounded queue, so the caller never waits on the network."""

    def __init__(self, maxsize: int) -> None:
        self._q: queue.Queue[Callable[[], None]] = queue.Queue(maxsize)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, job: Callable[[], None]) -> bool:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="raindrop-sink", daemon=True)
                self._thread.start()
        try:
            self._q.put_nowait(job)
        except queue.Full:
            return False
        return True

    def _run(self) -> None:
        while True:
            job = self._q.get()
            try:
                job()
            except Exception:
                log.exception("raindrop job failed")
            finally:
                self._q.task_done()

    def drain(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while self._q.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True


def _cap(text: str, limit: int = MAX_REASON_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def final_reply(trace: Trace) -> str:
    """The agent's last words: the `final` event's content, else the last model message that had any."""
    for kind in ("final", "model"):
        for e in reversed(trace.events):
            if e.kind == kind and e.payload.get("content"):
                return str(e.payload["content"])
    return ""


@dataclass(frozen=True)
class ToolStep:
    name: str
    args: dict[str, Any]
    output: Any
    error: str | None
    duration_ms: float


def _closing_events(trace: Trace) -> dict[str, Event]:
    """For each call id, the tool_result or guard_block event that closed the call."""
    closing: dict[str, Event] = {}
    for e in trace.events:
        if e.kind == "tool_result":
            call_id = (e.payload.get("result") or {}).get("call_id")
        elif e.kind == "guard_block":
            call_id = e.payload.get("call_id")
        else:
            continue
        if call_id:
            closing[str(call_id)] = e
    return closing


def _step_outcome(done: Event | None) -> tuple[Any, str | None]:
    if done is None:
        return None, None
    if done.kind == "guard_block":
        return None, f"guard blocked ({done.payload.get('source')}): {done.payload.get('message')}"
    res = done.payload.get("result") or {}
    return res.get("data"), None if res.get("ok", True) else str(res.get("error") or "error")


def tool_steps(trace: Trace) -> list[ToolStep]:
    """Pair each tool call with its result, or with the guard block that stopped it."""
    closing = _closing_events(trace)
    steps: list[ToolStep] = []
    for e in trace.events:
        if e.kind != "tool_call":
            continue
        call = e.payload.get("call") or {}
        done = closing.get(str(call.get("call_id")))
        output, error = _step_outcome(done)
        ms = max(0.0, (done.ts - e.ts) * 1000.0) if done is not None else 0.0
        steps.append(ToolStep(str(call.get("name", "?")), dict(call.get("args") or {}), output, error, ms))
    return steps


def guard_blocks(trace: Trace) -> list[Event]:
    return [e for e in trace.events if e.kind == "guard_block"]


def interventions(trace: Trace) -> int:
    """Guard blocks that came from a guard or the allowlist, not from malformed arguments."""
    return sum(1 for e in guard_blocks(trace) if e.payload.get("source") != "arguments")


def unwarranted(trace: Trace, case: CaseInfo) -> bool:
    """A guard stopped work that had to go through: a benign twin, or ordinary in-policy work."""
    return interventions(trace) > 0 and case.kind in ("benign_twin", "must_succeed")


def _flatten_numbers(prefix: str, value: Any, out: dict[str, float | int | bool], depth: int = 0) -> None:
    if isinstance(value, bool | int | float):
        out[prefix] = value
    elif isinstance(value, Mapping) and depth < 2:
        for k, v in value.items():
            _flatten_numbers(f"{prefix}_{k}", v, out, depth + 1)


class RaindropSink:
    """Send Ratchet traces, evaluations and admission decisions to Raindrop. See the module docstring."""

    def __init__(
        self,
        write_key: str | None = None,
        *,
        client: Any | None = None,
        project_id: str | None = None,
        max_events: int = DEFAULT_MAX_EVENTS,
        ledger_path: Path | None = None,
        extra_properties: Mapping[str, str | int | float | bool] | None = None,
        tracing: bool = True,
        queue_size: int = 2048,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._now = now or (lambda: datetime.now(UTC))
        self._max = max_events
        self._ledger = _Ledger(ledger_path, self._now)
        self._extra = dict(extra_properties or {})
        self._dispatch = _Dispatcher(queue_size)
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._sent = 0
        self._signals = 0
        self._dropped = 0
        self._client: Any | None = client
        if client is None and write_key:
            self._client = self._make_client(write_key, project_id, tracing)

    @staticmethod
    def _make_client(write_key: str, project_id: str | None, tracing: bool) -> Any | None:
        try:
            with warnings.catch_warnings():
                # raindrop-ai 0.0.75 imports traceloop-sdk, whose models still use pydantic's class-based
                # `Config`. That warning comes from their code, fires once at import, and is the only one
                # ignored here.
                warnings.filterwarnings("ignore", message="Support for class-based `config` is deprecated")
                module = importlib.import_module("raindrop")
                return module.Raindrop(
                    api_key=write_key, project_id=project_id, tracing_enabled=tracing, auto_instrument=False
                )
        except Exception:
            log.exception("raindrop client not created; telemetry is off")
            return None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def sent(self) -> int:
        """Events whose send finished without an error."""
        return self._sent

    @property
    def signals_sent(self) -> int:
        return self._signals

    @property
    def dropped(self) -> int:
        """Events lost to a full queue or a failed send."""
        return self._dropped

    @property
    def reserved_this_month(self) -> int:
        return self._ledger.count

    def remaining(self) -> int:
        return max(0, self._max - self._ledger.count)

    # -- reservation ------------------------------------------------------------------------------------

    def _reserve(self, event_id: str, summary: EmitSummary) -> bool:
        with self._lock:
            if event_id in self._seen:
                summary.duplicates += 1
                return False
            if self._ledger.count >= self._max:
                summary.over_budget += 1
                return False
            self._seen.add(event_id)
            self._ledger.add()
            return True

    def _enqueue(self, job: Callable[[], None], summary: EmitSummary, event_id: str) -> None:
        if self._dispatch.submit(job):
            summary.sent += 1
            summary.event_ids.append(event_id)
            return
        summary.dropped += 1
        with self._lock:
            self._dropped += 1

    def _record_success(self, signals: int) -> None:
        with self._lock:
            self._sent += 1
            self._signals += signals

    def _record_failure(self, what: str) -> None:
        log.exception("raindrop send failed for %s", what)
        with self._lock:
            self._dropped += 1

    # -- runs -------------------------------------------------------------------------------------------

    def run_properties(
        self,
        trace: Trace,
        case: CaseInfo,
        *,
        cohort: str | None = None,
        generation: int | None = None,
        experiment: str | None = None,
    ) -> Props:
        cost = trace.outcome.cost
        props: Props = {
            "harness_version": trace.harness_version,
            "case_id": case.id,
            "case_family": case.family,
            "split": case.split,
            "kind": case.kind,
            "must_succeed": case.must_succeed,
            "passed": trace.outcome.passed,
            "reasons": _cap("; ".join(trace.outcome.reasons)),
            "guard_blocks": len(guard_blocks(trace)),
            "interventions": interventions(trace),
            "model_calls": cost.model_calls,
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens,
            "tool_calls": cost.tool_calls,
            "wall_s": round(cost.wall_s, 3),
            "run_id": trace.run_id,
        }
        if cohort is not None:
            props["harness_cohort"] = cohort
        if generation is not None:
            props["harness_generation"] = generation
        if experiment is not None:
            props["experiment"] = experiment
        props.update(self._extra)
        return props

    def emit_trace(
        self,
        trace: Trace,
        case: CaseLike | CaseInfo,
        *,
        cohort: str | None = None,
        generation: int | None = None,
        experiment: str | None = None,
    ) -> EmitSummary:
        """Queue one event for a case run, with its signals. Returns at once; call `flush()` to wait."""
        summary = EmitSummary()
        info = case if isinstance(case, CaseInfo) else CaseInfo.of(case)
        if not self.enabled or not self._reserve(trace.run_id, summary):
            return summary
        props = self.run_properties(trace, info, cohort=cohort, generation=generation, experiment=experiment)
        signals = self._run_signals(trace, info)
        steps = tool_steps(trace)
        reply = final_reply(trace)
        summary.signals += len(signals)

        def job() -> None:
            self._send_run(trace, info, reply, props, steps, signals)

        self._enqueue(job, summary, trace.run_id)
        return summary

    @staticmethod
    def _run_signals(trace: Trace, case: CaseInfo) -> list[SignalSpec]:
        passed = trace.outcome.passed
        out: list[SignalSpec] = [
            (SIGNAL_SUCCESS if passed else SIGNAL_FAILURE, "POSITIVE" if passed else "NEGATIVE", {})
        ]
        blocks = guard_blocks(trace)
        if blocks:
            out.append(
                (SIGNAL_GUARD, None, {"source": str(blocks[0].payload.get("source")), "count": len(blocks)})
            )
        if unwarranted(trace, case):
            out.append((SIGNAL_UNWARRANTED, "NEGATIVE", {"kind": case.kind, "count": interventions(trace)}))
        return out

    def _send_run(
        self,
        trace: Trace,
        case: CaseInfo,
        reply: str,
        props: Props,
        steps: list[ToolStep],
        signals: list[SignalSpec],
    ) -> None:
        client = self._client
        if client is None:
            return
        try:
            interaction = client.begin(
                user_id=f"{RUN_USER}-{case.family}",
                event=RUN_EVENT,
                event_id=trace.run_id,
                properties=props,
                input=case.customer_message,
                convo_id=case.id,
                model=trace.model,
            )
            for step in steps:
                interaction.track_tool(
                    name=step.name,
                    input=step.args,
                    output=step.output,
                    duration_ms=step.duration_ms,
                    error=step.error,
                )
            interaction.finish(output=reply or "(no reply)")
            for name, sentiment, sprops in signals:
                client.track_signal(event_id=trace.run_id, name=name, sentiment=sentiment, properties=sprops)
        except Exception:
            self._record_failure(f"run event {trace.run_id}")
            return
        self._record_success(len(signals))

    # -- evaluations ------------------------------------------------------------------------------------

    def emit_evaluation(
        self,
        report: ReportLike,
        cases: Iterable[CaseLike] | Mapping[str, CaseLike] = (),
        *,
        cohort: str,
        generation: int | None = None,
        experiment: str | None = None,
    ) -> EmitSummary:
        """Queue every run of one evaluation under one cohort tag, all or nothing for the budget.

        `cohort` is "baseline" for the v0 evaluation and "candidate" for the vN evaluation, so Raindrop's
        Experiments can split them with a property condition. If the whole cohort does not fit in the
        remaining monthly budget nothing is sent and `refused` says why: half a cohort would bias the
        comparison.
        """
        summary = EmitSummary()
        if not self.enabled:
            summary.refused = "no write key"
            return summary
        infos = self._infos(report, cases)
        fresh = [t for t in report.traces if t.run_id not in self._seen]
        if len(fresh) > self.remaining():
            summary.refused = f"{len(fresh)} events do not fit in the {self.remaining()} left this month"
            return summary
        for trace in fresh:
            info = infos.get(trace.case_id) or CaseInfo(trace.case_id)
            summary.merge(
                self.emit_trace(trace, info, cohort=cohort, generation=generation, experiment=experiment)
            )
        return summary

    @staticmethod
    def _infos(report: ReportLike, cases: Iterable[CaseLike] | Mapping[str, CaseLike]) -> dict[str, CaseInfo]:
        infos: dict[str, CaseInfo] = {}
        for r in report.results:
            infos[r.case_id] = CaseInfo(r.case_id, split=r.split, kind=r.kind, must_succeed=r.must_succeed)
        pool = cases.values() if isinstance(cases, Mapping) else cases
        for c in pool:
            infos[c.id] = CaseInfo.of(c)
        return infos

    # -- admission --------------------------------------------------------------------------------------

    def emit_decision(self, decision: DecisionLike, *, experiment: str | None = None) -> EmitSummary:
        """Queue one event for an admission decision, with an `admitted` or `rejected` signal."""
        summary = EmitSummary()
        event_id = f"admission-{decision.parent}-{decision.candidate}"
        if not self.enabled or not self._reserve(event_id, summary):
            return summary
        failed = [str(c.name) for c in decision.checks if not c.passed]
        props: Props = {
            "parent_version": decision.parent,
            "candidate_version": decision.candidate,
            "harness_version": decision.candidate,
            "admitted": bool(decision.admitted),
            "failed_checks": ", ".join(failed) or "none",
        }
        for c in decision.checks:
            props[f"check_{c.name}"] = bool(c.passed)
        numbers: dict[str, float | int | bool] = {}
        _flatten_numbers("rate", dict(decision.rates), numbers)
        props.update(list(numbers.items())[:40])
        if experiment is not None:
            props["experiment"] = experiment
        props.update(self._extra)
        verdict = "ADMITTED" if decision.admitted else "REJECTED: " + (", ".join(failed) or "no reason")
        text = f"Candidate harness {decision.candidate} proposed over {decision.parent}."
        signal: SignalSpec = (
            SIGNAL_ADMITTED if decision.admitted else SIGNAL_REJECTED,
            "POSITIVE" if decision.admitted else "NEGATIVE",
            {"failed_checks": props["failed_checks"]},
        )
        summary.signals += 1

        def job() -> None:
            self._send_decision(event_id, text, verdict, props, signal)

        self._enqueue(job, summary, event_id)
        return summary

    def _send_decision(
        self, event_id: str, text: str, verdict: str, props: Props, signal: SignalSpec
    ) -> None:
        client = self._client
        if client is None:
            return
        name, sentiment, sprops = signal
        try:
            interaction = client.begin(
                user_id=IMPROVER_USER, event=ADMISSION_EVENT, event_id=event_id, properties=props, input=text
            )
            interaction.finish(output=verdict)
            client.track_signal(event_id=event_id, name=name, sentiment=sentiment, properties=sprops)
        except Exception:
            self._record_failure(f"admission event {event_id}")
            return
        self._record_success(1)

    # -- lifecycle --------------------------------------------------------------------------------------

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait for queued sends, then drain the SDK's own buffer. True when everything went out in time."""
        done = self._dispatch.drain(timeout)
        if self._client is None:
            return done
        try:
            self._client.flush()
        except Exception:
            log.exception("raindrop flush failed")
            return False
        return done

    def close(self, timeout: float = 30.0) -> bool:
        """Flush, then shut the SDK client down."""
        ok = self.flush(timeout)
        if self._client is None:
            return ok
        try:
            self._client.shutdown()
        except Exception:
            log.exception("raindrop shutdown failed")
            return False
        return ok
