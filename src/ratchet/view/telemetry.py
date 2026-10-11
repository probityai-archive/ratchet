"""Send each generation to Raindrop without letting Raindrop touch a decision.

`TelemetryStore` wraps the store the improvement loop writes to. Every write goes to the real store first
and returns its result untouched; only then is it observed, inside a guard that swallows and reports any
error. The loop never reads anything back from the wrapper, so a sink that raises, hangs behind its queue or
refuses for budget cannot change what is admitted. Events leave when the loop writes a generation record:
the parent's evaluation as the baseline cohort, the promoted (else best) candidate's as the candidate
cohort, then one event per admission decision.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, Protocol

from ratchet.types import Harness, Trace
from ratchet.view.data import RunStore


class Sink(Protocol):
    """The slice of `ratchet.telemetry.RaindropSink` the CLI uses."""

    @property
    def sent(self) -> int: ...
    @property
    def dropped(self) -> int: ...
    @property
    def reserved_this_month(self) -> int: ...
    def remaining(self) -> int: ...
    def emit_evaluation(
        self,
        report: Any,
        cases: Any = (),
        *,
        cohort: str,
        generation: int | None = None,
        experiment: str | None = None,
    ) -> Any: ...
    def emit_decision(self, decision: Any, *, experiment: str | None = None) -> Any: ...
    def close(self, timeout: float = 30.0) -> bool: ...


def usage_line(sink: Sink) -> str:
    """Events used this month against the monthly cap the sink enforces."""
    used = sink.reserved_this_month
    return (
        f"raindrop: {used} of {used + sink.remaining()} monthly events used "
        f"({sink.sent} sent this run, {sink.dropped} dropped)"
    )


def _report(payload: Mapping[str, Any], traces: Mapping[str, Trace]) -> Any:
    """An eval payload plus its traces, in the shape the sink reads (`ReportLike`)."""
    results = [SimpleNamespace(**r) for r in payload.get("results") or []]
    found = [traces[r.run_id] for r in results if getattr(r, "run_id", None) in traces]
    return SimpleNamespace(harness_version=payload["harness_version"], traces=found, results=results)


def _decision(payload: Mapping[str, Any]) -> Any:
    checks = [SimpleNamespace(name=c["name"], passed=bool(c["passed"])) for c in payload.get("checks") or []]
    return SimpleNamespace(
        parent=payload["parent"],
        candidate=payload["candidate"],
        admitted=bool(payload["admitted"]),
        checks=checks,
        rates=payload.get("rates") or {},
    )


def _heldout_key(payload: Mapping[str, Any]) -> tuple[int, int]:
    return (
        int((payload.get("heldout") or {}).get("passed", 0)),
        int((payload.get("train") or {}).get("passed", 0)),
    )


class TelemetryStore:
    """A `RunStore` that forwards every write and reports each finished generation to the sink."""

    def __init__(
        self,
        inner: RunStore,
        sink: Sink,
        cases: Sequence[Any],
        experiment: str,
        warn: Callable[[str], None],
    ) -> None:
        self.inner, self.sink, self.cases = inner, sink, list(cases)
        self.experiment, self.warn = experiment, warn
        self._traces: dict[str, Trace] = {}
        self._evals: dict[str, Mapping[str, Any]] = {}
        self._candidates: list[Mapping[str, Any]] = []
        self._decisions: list[Mapping[str, Any]] = []
        self._depth: dict[str, int] = {}
        self._baselines: set[str] = set()

    # -- the store the loop writes to: forward first, observe after ------------------------------------------

    def get_harness(self, version: str) -> Harness | None:
        return self.inner.get_harness(version)

    def put_harness(self, harness: Harness) -> bool:
        return self.inner.put_harness(harness)

    def put_trace(self, trace: Trace) -> bool:
        result = self.inner.put_trace(trace)
        self._traces[trace.run_id] = trace
        return result

    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str:
        result = self.inner.put_eval(report, eval_id=eval_id)
        self._observe(self._on_eval, report)
        return result

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        result = self.inner.put_decision(decision, decision_id=decision_id)
        self._observe(self._on_decision, decision)
        return result

    # -- observation -----------------------------------------------------------------------------------------

    def _guarded(self, step: Callable[[], None]) -> None:
        try:
            step()
        except Exception as exc:  # telemetry must never reach the loop
            self.warn(f"raindrop: {type(exc).__name__}: {exc} (admission is unaffected)")

    def _observe(self, handler: Callable[[Mapping[str, Any]], None], doc: Any) -> None:
        if isinstance(doc, Mapping):
            self._guarded(lambda: handler(doc))

    def _on_eval(self, doc: Mapping[str, Any]) -> None:
        kind = doc.get("type")
        if kind == "eval" and doc.get("results") is not None:
            self._evals[str(doc["harness_version"])] = doc
            if doc.get("role") in ("candidate", "composite"):
                self._candidates.append(doc)
        elif kind == "generation":
            self._emit_generation(doc)

    def _on_decision(self, doc: Mapping[str, Any]) -> None:
        self._decisions.append(doc)

    def _note(self, what: str, summary: Any) -> None:
        refused = getattr(summary, "refused", None)
        if refused:
            self.warn(f"raindrop: {what} not sent: {refused}")

    def _emit_generation(self, rec: Mapping[str, Any]) -> None:
        parent, promoted = str(rec["parent"]), (rec.get("level1") or {}).get("promoted")
        base = self._depth.setdefault(parent, 0)
        chosen = self._chosen(promoted)
        decisions = self._decisions
        self._candidates, self._decisions = [], []
        if promoted:
            self._depth[str(promoted)] = base + 1
        steps: list[Callable[[], None]] = [
            lambda: self._emit_cohort(self._evals.get(parent), "baseline", base),
            lambda: self._emit_cohort(chosen, "candidate", base + 1),
            lambda: self._emit_decisions(decisions),
        ]
        for step in steps:  # independent: one failing step must not hold back the others
            self._guarded(step)

    def _emit_cohort(self, payload: Mapping[str, Any] | None, cohort: str, depth: int) -> None:
        if payload is None:
            return
        version = str(payload["harness_version"])
        if cohort == "baseline":
            if version in self._baselines:
                return
            self._baselines.add(version)
        summary = self.sink.emit_evaluation(
            _report(payload, self._traces),
            self.cases,
            cohort=cohort,
            generation=depth,
            experiment=self.experiment,
        )
        self._note(f"{cohort} {version}", summary)

    def _emit_decisions(self, decisions: Sequence[Mapping[str, Any]]) -> None:
        for d in decisions:
            self.sink.emit_decision(_decision(d), experiment=self.experiment)

    def _chosen(self, promoted: Any) -> Mapping[str, Any] | None:
        if promoted and str(promoted) in self._evals:
            return self._evals[str(promoted)]
        return max(self._candidates, key=_heldout_key, default=None)
