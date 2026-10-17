"""One read view over the improver's store and the referee's, for the people who audit a run.

The improver writes training runs and harness versions to its store; the referee writes held-out runs,
decisions and activation to its own. `ratchet show`, `verify`, `report` and `live` read both through this
view. Writes go to the first store only, and the loop itself is never handed this view.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ratchet_cases.types import Trace

from ratchet.store.types import Doc, LineageNode
from ratchet.types import Harness


class ReadUnion:
    """Reads from every store in order; writes and checkpoints go to the first."""

    def __init__(self, main: Any, *others: Any) -> None:
        self.main = main
        self.stores: tuple[Any, ...] = (main, *others)

    # -- the write side: the improver's store ------------------------------------------------------------

    def put_harness(self, harness: Harness) -> bool:
        return bool(self.main.put_harness(harness))

    def put_trace(self, trace: Trace) -> bool:
        return bool(self.main.put_trace(trace))

    def put_eval(self, report: Any, *, eval_id: str | None = None) -> str:
        return str(self.main.put_eval(report, eval_id=eval_id))

    def put_decision(self, decision: Any, *, decision_id: str | None = None) -> str:
        return str(self.main.put_decision(decision, decision_id=decision_id))

    def checkpoint(self, run_id: str, state: Mapping[str, Any]) -> int:
        return int(self.main.checkpoint(run_id, state))

    def resume(self, run_id: str) -> Doc | None:
        found: Doc | None = self.main.resume(run_id)
        return found

    # -- reads: every store ------------------------------------------------------------------------------

    def lineage(self, root: str | None = None) -> Sequence[LineageNode]:
        return list(self.main.lineage(root))

    def get_harness(self, version: str) -> Harness | None:
        for s in self.stores:
            h = s.get_harness(version)
            if h is not None:
                return h  # type: ignore[no-any-return]
        return None

    def evals(self, filter: Mapping[str, Any] | None = None) -> list[Doc]:
        return [d for s in self.stores for d in s.evals(filter)]

    def decisions(self, filter: Mapping[str, Any] | None = None) -> list[Doc]:
        return [d for s in self.stores for d in s.decisions(filter)]

    def traces(
        self, filter: Mapping[str, Any] | None = None, *, limit: int = 0, newest_first: bool = False
    ) -> list[Trace]:
        out = [t for s in self.stores for t in s.traces(filter, newest_first=newest_first)]
        return out[:limit] if limit else out

    def get_trace(self, run_id: str, case_id: str) -> Trace | None:
        for s in self.stores:
            t = s.get_trace(run_id, case_id)
            if t is not None:
                return t  # type: ignore[no-any-return]
        return None

    def active_record(self) -> Doc | None:
        """The active pointer from the referee's store, which owns activation, else the improver's."""
        for s in reversed(self.stores):
            doc = s.active_record()
            if doc is not None:
                return doc  # type: ignore[no-any-return]
        return None

    def watch(
        self, collection: str, *, timeout_s: float | None = None, resume_after: Doc | None = None
    ) -> Iterator[Any]:
        """Decisions and activations are the referee's records; everything else is the improver's."""
        referee_owned = collection in ("decisions", "activations") and len(self.stores) > 1
        owner = self.stores[-1] if referee_owned else self.main
        return owner.watch(collection, timeout_s=timeout_s, resume_after=resume_after)  # type: ignore[no-any-return]

    def close(self) -> None:
        for s in self.stores:
            s.close()

    def __enter__(self) -> ReadUnion:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
