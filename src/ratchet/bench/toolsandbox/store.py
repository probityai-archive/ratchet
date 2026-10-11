"""The run store for a benchmark run: the shared JSON Lines store plus the benchmark's own score per run.

Besides the harness, trace and evaluation records every Ratchet benchmark run keeps, a run keeps
`scenarios.jsonl`: for each played scenario, the benchmark's grade as it computed it (similarity, milestone
and minefield similarity, the score of each milestone and minefield, the turn count), keyed by the run id of
the trace it belongs to. `cases.json` holds the case list of the run (categories, split, NOT RUN reasons),
so a report is rebuilt from the run directory alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ratchet.bench.tau.store import JsonlStore
from ratchet.bench.toolsandbox.cases import ToolSandboxCase
from ratchet.bench.toolsandbox.grading import Graded
from ratchet.types import Trace


class RunStore(JsonlStore):
    def put_graded(self, case: ToolSandboxCase, trace: Trace, graded: Graded) -> None:
        self._append(
            "scenarios",
            {
                "run_id": trace.run_id,
                "case_id": case.id,
                "harness_version": trace.harness_version,
                "model": trace.model,
                "passed": trace.outcome.passed,
                "graded": graded.to_json(),
            },
        )

    def write_cases(self, cases: list[ToolSandboxCase]) -> None:
        (self.root / "cases.json").write_text(
            json.dumps([asdict(c) for c in cases], indent=1), encoding="utf-8"
        )

    def read_cases(self) -> dict[str, dict[str, Any]]:
        path: Path = self.root / "cases.json"
        return {c["id"]: c for c in json.loads(path.read_text(encoding="utf-8"))}

    def graded_by_run(self) -> dict[str, dict[str, Any]]:
        return {r["run_id"]: r for r in self.read("scenarios")}
