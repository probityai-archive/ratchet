"""Terminal tasks as Ratchet cases: the task directory format, the two-phase split, and the seeded sides.

A task directory holds `task.toml` (metadata, limits, the container image), `instruction.md` (what the
agent is asked to do), `tests/` (the task's own verifier, entry point `tests/test.sh`) and `solution/` (the
reference solution, never shown to the agent). The world runs the agent in the task's prebuilt image and
the grader runs `tests/test.sh` unchanged.

The two phases. Phase 1 is every task annotated `hard` with a 900-second agent timeout (12 tasks); phase 2 is
every task annotated `hard` with an 1800-second agent timeout (10 tasks). The split is derived from the task
metadata and checked against the committed lists `PHASE_1` and `PHASE_2`, so a change in the suite stops the
run instead of silently changing the experiment.

The sides. Inside each phase a fixed seed draws the held-out tasks; `split.json` beside this module records
the assignment and its digest, and was committed before any run.

- Phase 1 evolution: train = phase-1 train, held-out = phase-1 held-out.
- Phase 2 evolution: train = phase-1 train and phase-2 train, held-out = phase-2 held-out, audit = phase-1
  held-out. The audit cohort never reaches a decision; it is evaluated after phase 2 to test that phase-1
  gains did not erode. The phase-1 train tasks stay in the decision set, so the admission rule's protected
  cases cover them during phase 2.
"""

from __future__ import annotations

import hashlib
import json
import random
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ratchet.types import Split

Phase = Literal[1, 2]
DEFAULT_SEED = 20261011
HELDOUT_PER_PHASE: dict[int, int] = {1: 4, 2: 4}
SPLIT_FILE = Path(__file__).with_name("split.json")
# The second split version: 6 of 12 phase-1 tasks and 5 of 10 phase-2 tasks held out. The admission rule
# counts stable recoveries on held-out cases only and needs two; in the first phase-1 generation on the
# 4-task held-out set, one held-out task failed both parent runs, so no candidate could reach two.
WIDE_HELDOUT: dict[int, int] = {1: 6, 2: 5}
WIDE_SPLIT_FILE = Path(__file__).with_name("split-heldout-6-5.json")

PHASE_1: tuple[str, ...] = (
    "cancel-async-tasks",
    "configure-git-webserver",
    "fix-code-vulnerability",
    "gpt2-codegolf",
    "make-doom-for-mips",
    "model-extraction-relu-logits",
    "password-recovery",
    "polyglot-rust-c",
    "sparql-university",
    "torch-pipeline-parallelism",
    "torch-tensor-parallelism",
    "write-compressor",
)
PHASE_2: tuple[str, ...] = (
    "dna-assembly",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "llm-inference-batching-scheduler",
    "make-mips-interpreter",
    "mcmc-sampling-stan",
    "path-tracing",
    "path-tracing-reverse",
    "protein-assembly",
)
PHASE_TIMEOUT_S: dict[int, float] = {1: 900.0, 2: 1800.0}


class SuiteError(ValueError):
    """The task directory does not hold the suite this experiment was frozen against."""


@dataclass(frozen=True)
class TaskSpec:
    """One task as read from its directory. Every field is plain JSON, so the evaluator's frozen identity
    covers the task; `tests_digest` binds the exact verifier files the grader will run."""

    name: str
    root: str
    instruction: str
    image: str
    difficulty: str
    category: str
    agent_timeout_s: float
    verifier_timeout_s: float
    cpus: float = 1.0
    memory_mb: int = 2048
    verifier_env: dict[str, str] = field(default_factory=dict)
    tests_digest: str = ""

    @property
    def path(self) -> Path:
        return Path(self.root)

    @property
    def tests_dir(self) -> Path:
        return self.path / "tests"


def tree_digest(root: Path) -> str:
    """SHA-256 over every file under `root`: relative path and content, in path order."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


@dataclass(frozen=True)
class TerminalCase:
    """One task as a Ratchet case. Every task asks for work the agent can do, so every case must succeed."""

    id: str
    split: Split
    task: TaskSpec
    phase: int
    must_succeed: bool = True
    kind: str = "must_succeed"
    family: str = "terminal"


def load_task(root: Path) -> TaskSpec:
    meta = tomllib.loads((root / "task.toml").read_text(encoding="utf-8"))
    env = meta.get("environment", {})
    image = env.get("docker_image")
    if not image:
        raise SuiteError(f"{root.name}: task.toml names no prebuilt docker_image")
    if not (root / "tests" / "test.sh").is_file():
        raise SuiteError(f"{root.name}: tests/test.sh is missing")
    return TaskSpec(
        name=root.name,
        root=str(root),
        instruction=(root / "instruction.md").read_text(encoding="utf-8"),
        image=str(image),
        difficulty=str(meta.get("metadata", {}).get("difficulty", "")),
        category=str(meta.get("metadata", {}).get("category", "")),
        agent_timeout_s=float(meta["agent"]["timeout_sec"]),
        verifier_timeout_s=float(meta["verifier"]["timeout_sec"]),
        cpus=float(env.get("cpus", 1)),
        memory_mb=int(env.get("memory_mb", 2048)),
        verifier_env={str(k): str(v) for k, v in meta.get("verifier", {}).get("env", {}).items()},
        tests_digest=tree_digest(root / "tests"),
    )


def load_suite(root: Path) -> dict[str, TaskSpec]:
    """Every task under `root` (one directory per task, each with a `task.toml`), by name."""
    tasks = {p.parent.name: load_task(p.parent) for p in sorted(Path(root).glob("*/task.toml"))}
    if not tasks:
        raise SuiteError(f"no task.toml found under {root}")
    return tasks


def phase_of(task: TaskSpec) -> int | None:
    if task.difficulty != "hard":
        return None
    return next((p for p, t in PHASE_TIMEOUT_S.items() if task.agent_timeout_s == t), None)


def phases(tasks: Mapping[str, TaskSpec]) -> dict[int, list[str]]:
    """The two phases derived from metadata, refused unless they equal the committed lists."""
    derived: dict[int, list[str]] = {1: [], 2: []}
    for name, task in sorted(tasks.items()):
        p = phase_of(task)
        if p is not None:
            derived[p].append(name)
    for p, frozen in ((1, PHASE_1), (2, PHASE_2)):
        if tuple(derived[p]) != frozen:
            raise SuiteError(
                f"phase {p} derived from the suite is {derived[p]}, the frozen list is {list(frozen)}"
            )
    return derived


def heldout_of(names: Sequence[str], n_heldout: int, seed: int) -> list[str]:
    if not 0 <= n_heldout <= len(names):
        raise ValueError(f"n_heldout must be between 0 and {len(names)}")
    return sorted(random.Random(seed).sample(sorted(names), n_heldout))


def assignment(
    seed: int = DEFAULT_SEED, heldout: Mapping[int, int] = HELDOUT_PER_PHASE
) -> dict[str, dict[str, list[str]]]:
    """Per phase, the train and held-out task names drawn with `seed` (phase 2 uses `seed + 1`)."""
    out: dict[str, dict[str, list[str]]] = {}
    for p, names in ((1, PHASE_1), (2, PHASE_2)):
        held = heldout_of(names, heldout[p], seed + p - 1)
        out[str(p)] = {"train": [n for n in names if n not in held], "heldout": held}
    return out


def split_record(seed: int = DEFAULT_SEED, heldout: Mapping[int, int] = HELDOUT_PER_PHASE) -> dict[str, Any]:
    sides = assignment(seed, heldout)
    digest = hashlib.sha256(json.dumps(sides, sort_keys=True).encode("utf-8")).hexdigest()
    record: dict[str, Any] = {"seed": seed, "phases": sides, "digest": digest}
    if dict(heldout) != HELDOUT_PER_PHASE:
        record["heldout_per_phase"] = {str(p): n for p, n in sorted(heldout.items())}
    return record


def committed_record(path: Path = SPLIT_FILE) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return record


def _sides(phase: int, seed: int, heldout: Mapping[int, int]) -> list[tuple[str, Split]]:
    sides = assignment(seed, heldout)
    p1, p2 = sides["1"], sides["2"]
    if phase == 1:
        return [(n, "train") for n in p1["train"]] + [(n, "heldout") for n in p1["heldout"]]
    if phase == 2:
        return (
            [(n, "train") for n in p1["train"] + p2["train"]]
            + [(n, "heldout") for n in p2["heldout"]]
            + [(n, "audit") for n in p1["heldout"]]
        )
    raise ValueError(f"phase must be 1 or 2, got {phase}")


def phase_cases(
    tasks: Mapping[str, TaskSpec],
    phase: int,
    seed: int = DEFAULT_SEED,
    heldout: Mapping[int, int] = HELDOUT_PER_PHASE,
) -> list[TerminalCase]:
    """The cases of one evolution phase, with their sides (see the module docstring)."""
    found = phases(tasks)
    home = {n: p for p, names in found.items() for n in names}
    return [TerminalCase(case_id(n), split, tasks[n], home[n]) for n, split in _sides(phase, seed, heldout)]


def task_cases(
    tasks: Mapping[str, TaskSpec], names: Iterable[str], split: Split = "heldout"
) -> list[TerminalCase]:
    """Plain evaluation cases for named tasks (smoke runs, transfer and audit evaluations)."""
    out = []
    for n in names:
        if n not in tasks:
            raise SuiteError(f"no task named {n!r} in the suite")
        out.append(TerminalCase(case_id(n), split, tasks[n], phase_of(tasks[n]) or 0))
    return out


def case_id(name: str) -> str:
    return f"term-{name}"
