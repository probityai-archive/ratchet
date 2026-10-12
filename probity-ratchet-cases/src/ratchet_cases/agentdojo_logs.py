"""Read AgentDojo run logs for the benign-twin suite and compute the false intervention rate.

AgentDojo writes one JSON file per episode, at
``<logdir>/<pipeline>/<suite>/<user_task>/<attack>/<injection>.json``.
A benign twin is a no-attack episode, so its file is ``.../none/none.json`` and carries ``utility``. Run the
twins with their own ``--logdir``: the twin suites share their names with AgentDojo's own suites, and
this reader does not separate the two when they are mixed. Standard library only; AgentDojo need not be
installed to read the logs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUITES = ("workspace", "travel", "banking", "slack")


class LogError(ValueError):
    """A log file the report cannot read as a twin episode."""


@dataclass(frozen=True)
class Episode:
    pipeline: str
    suite: str
    user_task: str
    utility: bool


def episodes(logdir: Path) -> list[Episode]:
    """Every no-attack twin episode under `logdir`, sorted by pipeline, suite and task."""
    found: list[Episode] = []
    for path in sorted(logdir.glob("*/*/user_task_*/none/none.json")):
        suite = path.parents[2].name
        if suite not in SUITES:
            continue
        try:
            doc: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            raise LogError(f"{path}: {err}") from err
        if not isinstance(doc, dict) or not isinstance(doc.get("utility"), bool):
            raise LogError(f"{path}: no boolean 'utility'")
        found.append(Episode(path.parents[3].name, suite, path.parents[1].name, doc["utility"]))
    if not found:
        raise FileNotFoundError(
            f"no no-attack episode logs (*/<suite>/user_task_*/none/none.json) under {logdir}"
        )
    return found


def report(logdir: Path) -> dict[str, Any]:
    """Per pipeline: twins run, twins whose requested action was carried out, and the false intervention
    rate (twins failed / twins run), overall and per suite."""
    rows: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes(logdir):
        rows[ep.pipeline].append(ep)
    out: dict[str, Any] = {}
    for pipeline, eps in sorted(rows.items()):
        per_suite = {}
        for suite in SUITES:
            mine = [e for e in eps if e.suite == suite]
            if mine:
                per_suite[suite] = _rate(mine)
        out[pipeline] = {**_rate(eps), "suites": per_suite}
    return out


def _rate(eps: list[Episode]) -> dict[str, Any]:
    done = sum(e.utility for e in eps)
    return {"twins": len(eps), "carried_out": done, "false_intervention_rate": 1 - done / len(eps)}
