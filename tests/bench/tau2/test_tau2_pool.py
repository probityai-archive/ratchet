from __future__ import annotations

import threading
from typing import Any

import pytest
from tau2_testkit import STOP, case, config, fresh_factory

import ratchet.bench.tau2.__main__ as cli
from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.bench.tau2.runner import EndpointPool, Tau2Runner, seed_harness


def test_the_pool_sends_each_conversation_to_the_least_busy_endpoint() -> None:
    users: list[Any] = []

    def factory(cfg: Any, t: Any, seed: int | None, sim_id: str) -> Any:
        orch = fresh_factory([STOP])(cfg, t, seed, sim_id)
        orch.user.llm_args = {}
        users.append(orch.user)
        return orch

    runners = [
        Tau2Runner(config("airline"), factory=factory, user_api_base=u) for u in ("http://a", "http://b")
    ]
    models = [ScriptedModel([scripted_turn("Hi")], name="a"), ScriptedModel([scripted_turn("Hi")], name="b")]
    pool = EndpointPool(runners, list(models))
    pool.inflight[0] = 5  # endpoint a looks busy
    trace = pool(seed_harness("airline"), case("airline", "0"), None, None, run_id="p1")
    assert trace.model == "b" and pool.inflight == [5, 0]
    assert len(models[1].requests) == 1 and not models[0].requests
    sim = pool.pop_simulation("p1")
    assert sim is not None and pool.pop_simulation("p1") is None
    assert [u.llm_args for u in users] == [{"api_base": "http://b"}]


def test_the_user_simulator_is_pointed_at_the_runner_endpoint() -> None:
    captured: dict[str, Any] = {}

    def factory(cfg: Any, t: Any, seed: int | None, sim_id: str) -> Any:
        orch = fresh_factory([STOP])(cfg, t, seed, sim_id)
        orch.user.llm_args = {"temperature": 0.0}
        captured["user"] = orch.user
        return orch

    runner = Tau2Runner(config("airline"), factory=factory, user_api_base="http://x/v1")
    runner(
        seed_harness("airline"), case("airline", "0"), ScriptedModel([scripted_turn("Hi")]), None, run_id="u1"
    )
    assert captured["user"].llm_args == {"temperature": 0.0, "api_base": "http://x/v1"}


def test_pool_runtime_reaches_every_runner_and_shapes_are_checked() -> None:
    runners = [Tau2Runner(config("airline")), Tau2Runner(config("airline"))]
    pool = EndpointPool(runners, [object(), object()])
    marker: Any = threading.Lock()
    pool.runtime = marker
    assert pool.runtime is marker and all(r.runtime is marker for r in runners)
    with pytest.raises(ValueError):
        EndpointPool(runners, [object()])
    with pytest.raises(ValueError):
        EndpointPool([], [])


def test_endpoints_builds_a_pool_for_several_urls() -> None:
    cfg = config("airline")
    one, models = cli.endpoints(cfg, "http://a/v1", "m", "k", 0.0)
    assert isinstance(one, Tau2Runner) and len(models) == 1
    pooled, models = cli.endpoints(cfg, "http://a/v1, http://b/v1", "m", "k", None)
    assert isinstance(pooled, EndpointPool) and [r.user_api_base for r in pooled.runners] == [
        "http://a/v1",
        "http://b/v1",
    ]
    for m in models:
        m.close()
    own, models = cli.endpoints(cfg, "http://a/v1,http://b/v1", "m", "k", None, own_user_endpoint=True)
    assert isinstance(own, EndpointPool) and [r.user_api_base for r in own.runners] == [None, None]
    for m in models:
        m.close()


def test_a_runner_without_a_user_endpoint_leaves_the_user_configuration_alone() -> None:
    captured: dict[str, Any] = {}

    def factory(cfg: Any, t: Any, seed: int | None, sim_id: str) -> Any:
        orch = fresh_factory([STOP])(cfg, t, seed, sim_id)
        orch.user.llm_args = {"temperature": 0.0}
        captured["user"] = orch.user
        return orch

    runner = Tau2Runner(config("airline"), factory=factory)
    runner(
        seed_harness("airline"), case("airline", "0"), ScriptedModel([scripted_turn("Hi")]), None, run_id="n1"
    )
    assert captured["user"].llm_args == {"temperature": 0.0}
