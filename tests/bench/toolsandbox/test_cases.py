from __future__ import annotations

import random
import subprocess
from typing import Any

import pytest
from conftest import MISSING_KEY

from ratchet.bench.toolsandbox import cases as cases_module
from ratchet.bench.toolsandbox.cases import (
    BENCHMARK_SEED,
    ToolSandboxCase,
    base_names,
    base_of,
    build_cases,
    category_names,
    keyring_key,
    missing_key_error,
    necessary_tools,
    rapid_tool_names,
    scenarios,
    split_record,
)


def test_the_scenario_set_is_the_benchmarks_1032() -> None:
    everything = scenarios()
    assert len(everything) == 1032
    assert len(base_names(everything)) == 129
    assert scenarios() is everything


def test_the_set_is_built_under_the_benchmarks_seed_and_the_callers_random_state_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drawn: list[float] = []

    def fake_named(preferred_tool_backend: Any) -> dict[str, Any]:
        drawn.append(random.random())
        return {"only": object()}

    monkeypatch.setattr(cases_module, "named_scenarios", fake_named)
    monkeypatch.setattr(cases_module, "_SCENARIOS", None)
    random.seed(7)
    before = random.getstate()
    built = scenarios()
    assert random.getstate() == before
    assert drawn == [random.Random(BENCHMARK_SEED).random()]
    assert scenarios() is built
    assert list(built) == ["only"]


def test_every_base_has_eight_forms(all_cases: list[ToolSandboxCase]) -> None:
    per_base: dict[str, int] = {}
    for c in all_cases:
        per_base[c.base] = per_base.get(c.base, 0) + 1
    assert set(per_base.values()) == {8}
    assert len(all_cases) == 1032
    assert len({c.id for c in all_cases}) == 1032


def test_a_variant_is_never_split_from_its_base(all_cases: list[ToolSandboxCase]) -> None:
    side: dict[str, set[str]] = {}
    for c in all_cases:
        side.setdefault(c.base, set()).add(c.split)
    assert all(len(v) == 1 for v in side.values())
    assert {c.split for c in all_cases} == {"train", "heldout"}
    assert len({c.base for c in all_cases if c.split == "train"}) == 26
    assert all(c.family == c.base for c in all_cases)


def test_the_split_follows_the_seed() -> None:
    a = [c.split for c in build_cases(seed=1, n_train=10, error=MISSING_KEY)]
    b = [c.split for c in build_cases(seed=1, n_train=10, error=MISSING_KEY)]
    c = [c.split for c in build_cases(seed=2, n_train=10, error=MISSING_KEY)]
    assert a == b
    assert a != c


@pytest.mark.parametrize("n", [-1, 130])
def test_n_train_outside_the_base_count_is_refused(n: int) -> None:
    with pytest.raises(ValueError, match="n_train"):
        build_cases(n_train=n, error=MISSING_KEY)


def test_rapid_tools_are_the_five_that_call_a_third_party() -> None:
    assert rapid_tool_names() == {
        "convert_currency",
        "search_lat_lon",
        "search_location_around_lat_lon",
        "search_stock",
        "search_weather_around_lat_lon",
    }


def test_scenarios_that_need_the_missing_key_are_not_run_and_stay_counted(
    all_cases: list[ToolSandboxCase],
) -> None:
    rapid = rapid_tool_names()
    everything = scenarios()
    for c in all_cases:
        needs = bool(necessary_tools(everything[c.base]) & rapid)
        assert (c.not_run is not None) == needs
        assert c.must_succeed == (not needs)
        assert c.kind == ("not_run" if needs else "must_succeed")
    blocked = [c for c in all_cases if c.not_run]
    assert len(blocked) % 8 == 0 and blocked
    assert {c.not_run for c in blocked} == {MISSING_KEY}


def test_with_a_key_nothing_is_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPID_API_KEY", "k")
    assert missing_key_error() is None
    assert all(c.not_run is None for c in build_cases())


def test_the_missing_key_error_is_the_one_the_tool_raises() -> None:
    message = missing_key_error()
    assert message is not None
    assert message.startswith("PermissionError: Please provide 'RAPID_API_KEY'")


def test_base_of_recognises_each_form_and_refuses_strangers() -> None:
    bases = {"cellular_off"}
    for suffix in ("", "_3_distraction_tools", "_10_distraction_tools", "_all_tools"):
        assert base_of("cellular_off" + suffix, bases) == "cellular_off"
    assert base_of("cellular_off_3_distraction_tools_tool_name_scrambled", bases) == "cellular_off"
    with pytest.raises(KeyError, match="not a base scenario"):
        base_of("wifi_off_all_tools", bases)


def test_category_names_are_sorted_strings() -> None:
    names = category_names(scenarios()["cellular_off"])
    assert names == tuple(sorted(names))
    assert "NO_DISTRACTION_TOOLS" in names


def test_split_record_names_every_case_and_digests_the_assignment(
    all_cases: list[ToolSandboxCase],
) -> None:
    rec = split_record(all_cases, 5)
    assert len(rec["train"]) + len(rec["heldout"]) == 1032
    assert set(rec["not_run"]) <= {c.id for c in all_cases}
    other = [ToolSandboxCase(**{**c.__dict__, "split": "heldout"}) for c in all_cases]
    assert split_record(other, 5)["digest"] != rec["digest"]
    assert split_record(all_cases, 5)["digest"] == rec["digest"]


class _Run:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode, self.stdout = returncode, stdout


def test_keyring_key_prefers_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPID_API_KEY", "from-env")
    assert keyring_key() == "from-env"


def test_keyring_key_tries_each_service_name(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def fake_run(cmd: list[str], **_kw: Any) -> _Run:
        asked.append(cmd[3])
        return _Run(0, "secret\n") if cmd[3] == "rapid-api" else _Run(1, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert keyring_key() == "secret"
    assert asked == ["rapidapi", "rapid-api"]


def test_keyring_key_is_none_when_no_name_holds_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Run(0, "\n"))
    assert keyring_key() is None


def test_keyring_key_is_none_without_the_keyring_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_a: Any, **_k: Any) -> _Run:
        raise FileNotFoundError("secret-tool")

    monkeypatch.setattr(subprocess, "run", missing)
    assert keyring_key() is None
