from __future__ import annotations

import pytest
from ratchet_cases.types import Cost, ToolCall
from tau_testkit import ScriptedUser

from ratchet.bench.tau.runner import make_env, seed_harness, tau_tool_names
from ratchet.bench.tau.tasks import DEFAULT_SEED, VIEW_TABLES, default_cases, split_single, tau_tasks
from ratchet.bench.tau.world import RESPOND, TauWorld, outcome_of


def test_airline_split_divides_the_fifty_test_tasks() -> None:
    cases = default_cases("airline")
    assert len(cases) == 50 and len(tau_tasks("test", "airline")) == 50
    assert sum(c.split == "train" for c in cases) == 20
    assert all(c.domain == "airline" and c.tau_split == "test" for c in cases)
    assert cases[0].id == "tau-airline-test-000"
    assert cases == split_single(50, "airline", seed=DEFAULT_SEED, n_train=20)
    with pytest.raises(ValueError):
        tau_tasks("train", "airline")
    with pytest.raises(ValueError):
        split_single(5, "airline", n_train=6)


def test_airline_seed_harness_and_tools() -> None:
    h = seed_harness("airline")
    assert h.system_prompt.startswith("# Airline Agent Policy")
    assert "book_reservation" in h.tool_allowlist and RESPOND in h.tool_allowlist
    assert h.tool_allowlist == sorted([*tau_tool_names("airline"), RESPOND])
    assert h.version != seed_harness("retail").version


def test_airline_world_views_users_and_reservations(monkeypatch: pytest.MonkeyPatch) -> None:
    from tau_bench.envs.airline import MockAirlineDomainEnv  # type: ignore[import-untyped]  # no type info

    env = MockAirlineDomainEnv(user_strategy="human", task_split="test", task_index=0)
    env.user = ScriptedUser()
    env.reset(task_index=0)
    world = TauWorld(env, "hi", VIEW_TABLES["airline"])
    world.call(ToolCall("get_reservation_details", {"reservation_id": "4WQ150"}, "c1"))
    view = world.view().to_json()
    assert list(view["reservations"]) == ["4WQ150"]
    assert list(view["users"]) == ["chen_jackson_3290"]
    world.call(ToolCall(RESPOND, {"content": "bye"}, "c2"))
    assert world.done
    assert not outcome_of(world, Cost()).passed


def test_make_env_builds_the_domain_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import tau_bench.envs.base as base  # type: ignore[import-untyped]  # no type info

    monkeypatch.setattr(base, "load_user", lambda **_kw: ScriptedUser(opening="hello from the user"))
    case = default_cases("airline")[0]
    env, opening = make_env(case, "user-model")
    assert opening == "hello from the user"
    assert env.task_index == case.index and "reservations" in env.data
