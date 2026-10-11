"""Synthetic screens for testing the recording pipeline. Test data only: never used in a submission.

    python3 docs/demo/video/pipeline_fixture.py matrix|decision

The sample traces and guards come from the view tests' in-memory fakes, loaded by file path so the
fixture does not depend on how the tests directory is packaged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from rich.console import Console

from ratchet.view.data import decision_info
from ratchet.view.matrix import MatrixColumn, compute_matrix, letters, show_matrix
from ratchet.view.show import decision_view

FAKES = Path(__file__).resolve().parents[3] / "tests" / "view" / "fakes.py"
KEYS = ["g-retry", "g-all-refunds", "g-outside-mail", "g-any-mail", "g-broken"]


def load_fakes() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ratchet_view_fakes", FAKES)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {FAKES}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def matrix_screen(console: Console) -> None:
    fk = load_fakes()
    guards = {g.id: g for g in fk.sample_guards()}
    columns = []
    for i, key in enumerate(KEYS):
        decision = None
        if key in ("g-retry", "g-all-refunds"):
            admitted = key == "g-retry"
            checks = [{"name": "replay", "passed": admitted, "evidence": {}}]
            decision = decision_info(fk.decision_doc("p", f"cand{i}", admitted, checks=checks))
        columns.append(MatrixColumn(guards[key], f"cand{i}", letters(5)[i], decision))
    matrix = compute_matrix(
        "parent000001", columns, fk.sample_traces(), fk.info, fk.walks(), fk.FakeRunner(fk.RULES)
    )
    show_matrix(console, matrix, delay=0.06)


def decision_screen(console: Console) -> None:
    fk = load_fakes()
    checks = [
        {"name": "replay", "passed": True, "evidence": {}},
        {"name": "authority", "passed": True, "evidence": {}},
    ]
    console.print(
        decision_view(decision_info(fk.decision_doc("parent000001", "cand0000001", True, checks=checks)))
    )


def main() -> int:
    console = Console()
    which = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    (matrix_screen if which == "matrix" else decision_screen)(console)
    return 0


if __name__ == "__main__":
    sys.exit(main())
