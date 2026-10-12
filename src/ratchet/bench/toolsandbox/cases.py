"""The benchmark's scenarios as Ratchet cases, with a recorded train / held-out split.

The scenario set is the benchmark's own: `named_scenarios` builds every base scenario and its augmentations
(extra distractor tools, scrambled tool names, descriptions, argument names, descriptions and types), 1,032
in all at the pinned commit. The benchmark shuffles with Python's global random generator, so the set is built
under the seed its own runner uses (42), which makes the scenario names, tool lists and tool order the same
in every process and on every machine; the generator state of the caller is restored afterwards.

A scenario whose necessary tools call a third-party API (the benchmark's `rapid_api_search_tools` need a
RapidAPI key) cannot run without that key. It is kept in the set, marked `not_run` with the exact error the
tool raises, and counted in every denominator; it is never scored as a pass or a fail.

Split. The nine variants of one base scenario share their task, so they are never split apart: base
scenarios are shuffled with a seed and the first `n_train` are train, the rest held-out. A run over both
sides is a run over the whole benchmark, which is how the full-set scores are reported.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

from ratchet_cases.types import Split
from tool_sandbox.common.execution_context import (
    ExecutionContext,
    ScenarioCategories,
)
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_discovery import (
    ToolBackend,
    get_all_tools,
)
from tool_sandbox.scenarios import named_scenarios
from tool_sandbox.tools.rapid_api_search_tools import (
    rapid_api_get_request,
)

from ratchet.bench.toolsandbox.gate import GATE

BENCHMARK_SEED = 42  # the seed the benchmark's own runner sets before building its scenarios
DEFAULT_SEED = 20261011
DEFAULT_TRAIN_BASES = 26
RAPID_MODULE = "tool_sandbox.tools.rapid_api_search_tools"
RAPID_KEY_SERVICES = ("rapidapi", "rapid-api", "rapid_api")
END_TOOL = "end_conversation"
_VARIANTS = (
    "",
    "_3_distraction_tools",
    "_10_distraction_tools",
    "_all_tools",
    "_3_distraction_tools_tool_description_scrambled",
    "_3_distraction_tools_arg_type_scrambled",
    "_3_distraction_tools_arg_description_scrambled",
    "_3_distraction_tools_tool_name_scrambled",
)

_LOCK = threading.Lock()
_SCENARIOS: dict[str, Scenario] | None = None


@dataclass(frozen=True)
class ToolSandboxCase:
    """One scenario. `name` is the benchmark's scenario name; `base` is the scenario its variant extends.

    The family is the base scenario: its eight forms share one task, so they are never split apart.
    """

    id: str
    split: Split
    name: str
    base: str
    categories: tuple[str, ...]
    family: str
    not_run: str | None = None
    must_succeed: bool = True
    kind: str = "must_succeed"


def scenarios() -> dict[str, Scenario]:
    """The benchmark's 1,032 scenarios by name, built once per process under the benchmark's seed."""
    global _SCENARIOS
    with _LOCK:
        if _SCENARIOS is None:
            state = random.getstate()
            random.seed(BENCHMARK_SEED)
            try:
                _SCENARIOS = dict(named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT))
            finally:
                random.setstate(state)
        return _SCENARIOS


def category_names(scenario: Scenario) -> tuple[str, ...]:
    return tuple(sorted(str(c) for c in scenario.categories))


def base_of(name: str, bases: set[str]) -> str:
    """The base scenario a variant name extends."""
    for variant in _VARIANTS:
        if variant and name.endswith(variant) and name.removesuffix(variant) in bases:
            return name.removesuffix(variant)
    if name in bases:
        return name
    raise KeyError(f"{name} is not a base scenario or a known variant of one")


def base_names(all_scenarios: dict[str, Scenario]) -> list[str]:
    """Base scenarios: the ones the benchmark tags as having no distractor tools."""
    tag = ScenarioCategories.NO_DISTRACTION_TOOLS
    return sorted(n for n, s in all_scenarios.items() if tag in s.categories)


def rapid_tool_names() -> set[str]:
    """Names of the benchmark's tools that call a third-party API."""
    return {n for n, fn in get_all_tools(ToolBackend.DEFAULT).items() if fn.__module__ == RAPID_MODULE}


def necessary_tools(scenario: Scenario) -> set[str]:
    """The tools a base scenario's task needs: its allow list without the user's `end_conversation`."""
    return set(scenario.starting_context.tool_allow_list or []) - {END_TOOL}


def keyring_key() -> str | None:
    """The RapidAPI key from the environment, else from the keyring (service names are tried in turn)."""
    if os.environ.get("RAPID_API_KEY"):
        return os.environ["RAPID_API_KEY"]
    for service in RAPID_KEY_SERVICES:
        try:
            out = subprocess.run(
                ["secret-tool", "lookup", "service", service, "account", "api-key"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return None


def missing_key_error() -> str | None:
    """The exact error a RapidAPI tool raises with no key, measured by calling the benchmark's request
    function in a fresh context; None when a key is set and the tools can run."""
    if os.environ.get("RAPID_API_KEY"):
        return None
    with GATE.held(ExecutionContext()):
        try:
            rapid_api_get_request("https://example.invalid", {}, {})
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
    return None


def build_cases(
    *, seed: int = DEFAULT_SEED, n_train: int = DEFAULT_TRAIN_BASES, error: str | None = None
) -> list[ToolSandboxCase]:
    """Every scenario as a case. `error` (default: measured now) marks scenarios that need a missing key."""
    everything = scenarios()
    bases = base_names(everything)
    if not 0 <= n_train <= len(bases):
        raise ValueError(f"n_train must be between 0 and {len(bases)}")
    train = set(random.Random(seed).sample(bases, n_train))
    rapid = rapid_tool_names()
    missing = missing_key_error() if error is None else error
    needs = {b: bool(necessary_tools(everything[b]) & rapid) for b in bases}
    base_set = set(bases)
    cases: list[ToolSandboxCase] = []
    for name in sorted(everything):
        base = base_of(name, base_set)
        blocked = missing if needs[base] else None
        cases.append(
            ToolSandboxCase(
                id=f"toolsandbox-{name}",
                split="train" if base in train else "heldout",
                name=name,
                base=base,
                categories=category_names(everything[name]),
                family=base,
                not_run=blocked,
                must_succeed=blocked is None,
                kind="not_run" if blocked else "must_succeed",
            )
        )
    return cases


def split_record(cases: list[ToolSandboxCase], seed: int) -> dict[str, Any]:
    """The split as data: seed, ids per side, and a digest of the whole assignment."""
    rows = [{"id": c.id, "split": c.split, "base": c.base, "not_run": c.not_run} for c in cases]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "seed": seed,
        "train": [c.id for c in cases if c.split == "train"],
        "heldout": [c.id for c in cases if c.split == "heldout"],
        "not_run": [c.id for c in cases if c.not_run],
        "digest": digest,
    }
