"""The submission steps resolve for both demo outcomes, from exactly the keys derive_ids sets for each."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
STEPS = ROOT / "docs" / "demo" / "video" / "steps.submission.toml"


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


drive = load("ratchet_demo_drive", ROOT / "docs" / "demo" / "video" / "drive.py")
derive = load("ratchet_demo_derive_ids_for_drive", ROOT / "docs" / "demo" / "derive_ids.py")

COMMON = [
    "RATCHET_DB",
    "BAD_TRACE",
    "REJECTED_CAND",
    "REPLAY_TRACES",
    "AUTHORITY_DECISION",
    "AUTHORITY_DB",
    "RESULTS_FILE",
    "AGENTDOJO_CASES",
    "TAU_CASES",
    "LEVEL2_FILE",
    "LEVEL2_SEARCH_YIELD",
    "LEVEL2_SEED_YIELD",
    "REFUSED_DB",
    "REFUSED_CAND",
    "REFUSED_RECOVERIES",
    "REFUSED_RECOVERY_RUNS",
    "REFUSED_KEPT",
    "REFUSED_PROTECTED_N",
    "REFUSED_WHY",
    "PARENT_DISCORDANT",
    "REFUSED_HELDOUT_N",
    "TAU_SELF_SCORES",
    "TAU_SELF_TOTAL",
]


def ids_for(outcome: str) -> dict[str, str]:
    ids = {key: f"v-{key.lower()}" for key in COMMON}
    ids["DEMO_OUTCOME"] = outcome
    if outcome == "admitted":
        ids.update({key: f"v-{key.lower()}" for key in derive.ADMITTED_ONLY})
    return ids


@pytest.mark.parametrize("variant", [None, "45"])
def test_admitted_outcome_shows_the_admission_then_the_refusal(variant: str | None) -> None:
    steps, missing = drive.load_steps(STEPS, ids_for("admitted"), variant)
    assert missing == []
    names = [s["id"] for s in steps]
    assert names.index("admitted") < names.index("refusal")
    assert {"verify", "failure", "matrix"} <= set(names)
    assert not {"failure_live", "matrix_live"} & set(names)
    assert all(
        "Demo scenario" in s["captions"][0]["text"] for s in steps if s["id"] in ("failure", "admitted")
    )
    refusal = next(s for s in steps if s["id"] == "refusal")
    assert refusal["db"] == "v-refused_db"
    assert refusal["captions"][0]["text"].startswith("Live models")


def test_refused_outcome_needs_no_admitted_key_and_implies_no_admission() -> None:
    steps, missing = drive.load_steps(STEPS, ids_for("refused"), None)
    assert missing == []
    names = [s["id"] for s in steps]
    assert not {"admitted", "verify", "failure", "matrix"} & set(names)
    assert names[:3] == ["failure_live", "matrix_live", "refusal"]
    text = " ".join(c["text"] for s in steps for c in s["captions"]).lower()
    assert " admitted:" not in text
    assert "refused" in text
    assert all(s["db"] for s in steps)  # every step names the database it reads


def test_a_missing_outcome_or_key_is_reported() -> None:
    ids = ids_for("refused")
    del ids["DEMO_OUTCOME"]
    _steps, missing = drive.load_steps(STEPS, ids, None)
    assert "DEMO_OUTCOME" in missing
    ids = ids_for("refused")
    del ids["REFUSED_WHY"]
    assert drive.load_steps(STEPS, ids, None)[1] == ["REFUSED_WHY"]


def test_a_step_runs_with_its_own_database() -> None:
    assert drive.step_env({"db": "other"}, {"RATCHET_DB": "main", "X": "1"}) == {
        "RATCHET_DB": "other",
        "X": "1",
    }
    assert drive.step_env({"db": ""}, {"RATCHET_DB": "main"}) == {"RATCHET_DB": "main"}
