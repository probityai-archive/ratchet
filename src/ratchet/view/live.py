"""Follow Atlas change streams and print each new trace, candidate and decision as it lands."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.text import Text

from ratchet.view.data import ViewStore, decision_info
from ratchet.view.show import clip

DEFAULT_COLLECTIONS = ("harness_versions", "traces", "decisions", "evals", "failures")
TAGS = {
    "traces": ("TRACE", "blue"),
    "harness_versions": ("VERSION", "cyan"),
    "decisions": ("DECISION", "bold magenta"),
    "evals": ("EVAL", "yellow"),
    "failures": ("FAILURE", "red"),
}


@dataclass(frozen=True)
class Reconnecting:
    attempt: int
    error: Exception


def _stamp(now: float) -> Text:
    return Text(time.strftime("%H:%M:%S", time.localtime(now)) + " ", style="dim")


def _line(collection: str, body: Text, now: float) -> Text:
    tag, style = TAGS.get(collection, (collection.upper(), ""))
    return Text.assemble(_stamp(now), (f"{tag:<9}", style), body)


def _trace(doc: dict[str, Any]) -> Text:
    ok = bool(doc.get("passed"))
    out = doc.get("outcome") or {}
    t = Text.assemble(
        (" PASS " if ok else " FAIL ", "bold white on green" if ok else "bold white on red"),
        f" {doc.get('case_id')}  ",
        (f"harness {doc.get('harness_version')}", "dim"),
    )
    if not ok and out.get("reasons"):
        t.append("  " + clip(str(out["reasons"][0]), 70), style="red")
    return t


def _version(doc: dict[str, Any]) -> Text:
    guards = doc.get("guards") or []
    t = Text.assemble((f"{doc['_id']}", "bold cyan"), (f"  parent {doc.get('parent') or '-'}", "dim"))
    procs = len(doc.get("procedures") or [])
    t.append(f"  by {doc.get('created_by')}  {len(guards)} guard(s), {procs} procedure(s)")
    if doc.get("created_by") == "improver" and guards:
        t.append(
            "  newest: " + clip(str(guards[-1].get("description") or guards[-1].get("id")), 50), style="white"
        )
    return t


def _decision(doc: dict[str, Any]) -> Text:
    d = decision_info(doc)
    badge = ("✓ ADMITTED ", "bold black on green") if d.admitted else ("✗ REJECTED ", "bold white on red")
    t = Text.assemble(badge, f" {d.parent} -> {d.candidate}  ", (clip(d.what, 40), "white"))
    if not d.admitted:
        t.append("  " + clip("; ".join(d.reasons()), 70), style="red")
    if d.receipt:
        t.append(f"  {d.receipt}", style="magenta")
    return t


def _eval(doc: dict[str, Any]) -> Text | None:
    kind = doc.get("type")
    if kind == "generation":
        promoted = doc.get("promoted")
        before, after = doc.get("parent_score") or {}, doc.get("child_score") or {}
        t = Text.assemble((f"generation {doc.get('generation')}", "bold"), f"  parent {doc.get('parent')}")
        if promoted:
            t.append(f" -> promoted {promoted}", style="bold green")
            t.append(
                f"  held-out {before.get('heldout_passed')} -> {after.get('heldout_passed')}"
                f"/{after.get('heldout_total')}",
                style="green",
            )
        else:
            t.append("  nothing admitted", style="yellow")
        return t
    if kind == "eval":
        h, tr = doc.get("heldout") or {}, doc.get("train") or {}
        return Text(
            f"harness {doc.get('harness_version')}  train {tr.get('passed')}/{tr.get('total')}"
            f"  held-out {h.get('passed')}/{h.get('total')}"
        )
    if kind == "improver_instructions":
        return Text(
            f"improver instructions {doc.get('version')} (parent {doc.get('parent') or '-'})", style="dim"
        )
    return None


def format_change(
    collection: str, operation: str, document: dict[str, Any] | None, now: float
) -> Text | None:
    """One printable line for a change, or None when it is not worth showing (an update, a delete)."""
    if operation != "insert" or document is None:
        return None
    body: Text | None
    if collection == "traces":
        body = _trace(document)
    elif collection == "harness_versions":
        body = _version(document)
    elif collection == "decisions":
        body = _decision(document)
    elif collection == "evals":
        body = _eval(document)
    elif collection == "failures":
        body = Text(clip(str(document.get("summary") or ""), 100))
    else:
        body = None
    return None if body is None else _line(collection, body, now)


def follow(
    store: ViewStore,
    console: Console,
    collections: Sequence[str] = DEFAULT_COLLECTIONS,
    *,
    timeout_s: float | None = None,
    max_events: int | None = None,
    clock: Callable[[], float] = time.time,
    retries: int = 5,
    backoff_s: float = 1.0,
) -> int:
    """Print changes from every collection until `timeout_s` passes or `max_events` lines were printed.

    One thread per change stream feeds one queue; only this thread prints. A stream that drops is reopened
    from its last resume token, so nothing is missed or printed twice; after `retries` failures in a row it
    is reported and given up. Returns the number of lines printed."""
    events: queue.Queue[tuple[str, Any] | None] = queue.Queue()
    stop = threading.Event()
    deadline = None if timeout_s is None else time.monotonic() + timeout_s

    def pump(name: str) -> None:
        token: dict[str, Any] | None = None
        failures = 0
        try:
            while not stop.is_set():
                left = None if deadline is None else max(0.0, deadline - time.monotonic())
                try:
                    for ev in store.watch(name, timeout_s=left, resume_after=token):
                        token, failures = ev.token, 0
                        events.put((name, ev))
                    return
                except Exception as exc:
                    failures += 1
                    if failures > retries:
                        events.put((name, exc))
                        return
                    events.put((name, Reconnecting(failures, exc)))
                    time.sleep(backoff_s * failures)
        finally:
            events.put(None)

    threads = [threading.Thread(target=pump, args=(c,), daemon=True) for c in collections]
    for th in threads:
        th.start()
    printed, finished = 0, 0
    try:
        while finished < len(threads) and (max_events is None or printed < max_events):
            item = events.get()
            if item is None:
                finished += 1
                continue
            name, ev = item
            if isinstance(ev, Reconnecting):
                console.print(
                    Text(
                        f"{name}: stream dropped ({ev.error}); reconnecting, try {ev.attempt}", style="yellow"
                    )
                )
            elif isinstance(ev, Exception):
                console.print(Text(f"{name}: change stream stopped: {ev}", style="bold red"))
            else:
                line = format_change(name, ev.operation, ev.document, clock())
                if line is not None:
                    console.print(line)
                    printed += 1
    except KeyboardInterrupt:
        console.print(Text("stopped", style="dim"))
    finally:
        stop.set()
    return printed
