"""One thread at a time inside the benchmark's process-global execution context.

`tool_sandbox` keeps the world in a module-level variable (`get_current_context`), which its own source calls
not thread safe; its runner gets isolation from one process per scenario. Ratchet's pools run many worlds in
threads, so every touch of that variable goes through this gate: the thread takes the lock, points the global
at its own world's context, and works. The one long operation inside a benchmark turn is the user simulator's
model call, which touches no context, so the simulator gives the gate up for the duration of the call.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from tool_sandbox.common.execution_context import (  # type: ignore[import-untyped]  # no py.typed
    ExecutionContext,
    get_current_context,
    set_current_context,
)


class ContextGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None

    @contextmanager
    def held(self, context: ExecutionContext) -> Iterator[None]:
        """Own the global context, set to `context`, for the block."""
        with self._lock:
            set_current_context(context)
            self._owner = threading.get_ident()
            try:
                yield
            finally:
                self._owner = None

    @contextmanager
    def released(self) -> Iterator[None]:
        """Inside `held`: let other threads in for the block, then retake the gate and restore the context."""
        if self._owner != threading.get_ident():
            raise RuntimeError("the gate is not held by this thread")
        context = get_current_context()
        self._owner = None
        self._lock.release()
        try:
            yield
        finally:
            self._lock.acquire()
            set_current_context(context)
            self._owner = threading.get_ident()


GATE = ContextGate()
