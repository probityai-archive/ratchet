from __future__ import annotations

from dataclasses import replace
from typing import Any

from ratchet_cases.types import Event, Trace

from ratchet.improve.evaluate import evaluate, evaluate_many, interventions, model_calls
from ratchet.types import Guard, Harness

from .fakes import BLOCK_ALL_REFUNDS, NO_DUP_GUARD, FakeRunCase, fake_cases

SEED = Harness.build(
    parent=None,
    system_prompt="p",
    procedures=[],
    guards=[],
    tool_allowlist=["payments_refund"],
    created_by="seed",
)
GUARDED = SEED.child(guards=[Guard("g", "d", NO_DUP_GUARD)], created_by="improver")


def test_evaluate_counts_by_split_and_keeps_traces() -> None:
    rep = evaluate(SEED, fake_cases(), FakeRunCase(), model=None, workers=3)
    assert rep.passed("train") == {"plain-3"} and rep.passed("heldout") == {"plain-6"}
    assert rep.count() == 6 and len(rep.traces) == 6
    assert {t.case_id for t in rep.train_traces()} == {"timeout-1", "timeout-2", "plain-3"}
    assert len({r.run_id for r in rep.results}) == 6  # every run gets its own id
    payload = rep.to_payload()
    assert payload["train"] == {"passed": 1, "total": 3} and payload["cost"]["tool_calls"] > 0
    assert payload["fresh_calls"] == 6 and payload["cached_calls"] == 0
    assert payload["results"][2]["kind"] == "benign_twin"


def test_evaluate_many_shares_the_pool_and_records_errors() -> None:
    run = FakeRunCase(explode="plain-6")
    a, b = evaluate_many([SEED, GUARDED], fake_cases(), run, model=None, workers=4)
    assert run.calls == 16  # plain-6 is retried twice per harness under the predeclared limit
    assert b.passed("heldout") == {"timeout-4", "timeout-5"}
    assert b.by_case()["timeout-4"].interventions == 1 and b.by_case()["plain-3"].interventions == 0
    err = b.by_case()["plain-6"]
    assert not err.passed and err.error and err.reasons[0].startswith("run error")
    assert err.infra_failed and err.attempts == 3
    assert b.to_payload()["errors"] == 1 and a.harness_version == SEED.version


def test_interventions_count_harness_blocks_only() -> None:
    blocker = SEED.child(guards=[Guard("b", "d", BLOCK_ALL_REFUNDS)], created_by="improver")
    rep = evaluate(blocker, fake_cases(), FakeRunCase(), None)
    assert rep.by_case()["plain-6"].interventions == 1
    t = rep.traces[0]
    arg_block = Event(99, "guard_block", {"call_id": "x", "source": "arguments"}, 0.0)
    assert interventions(replace(t, events=[*t.events, arg_block])) == interventions(t)


def test_cached_model_calls_are_counted_apart() -> None:
    t = evaluate(SEED, fake_cases()[:1], FakeRunCase(), None).traces[0]
    cached = Event(98, "model", {"content": "x", "cached": True}, 0.0)
    assert model_calls(replace(t, events=[*t.events, cached])) == (1, 1)


def test_repeats_count_a_case_only_when_every_repeat_passes() -> None:
    class Flaky(FakeRunCase):
        def __call__(self, harness: Any, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
            t = super().__call__(harness, case, model, store, run_id=run_id)
            if case.id == "plain-3" and self.calls % 2 == 0:
                return replace(t, outcome=replace(t.outcome, passed=False, reasons=["flaked"]))
            return t

    rep = evaluate(SEED, fake_cases()[2:3], Flaky(), model=None, workers=1, repeats=2)
    assert rep.passed() == set() and rep.by_case()["plain-3"].reasons == ["flaked"]


def test_runs_lost_to_the_endpoint_are_retried_then_marked_invalid() -> None:
    from ratchet.improve.evaluate import INFRA_REASON

    class Flaky(FakeRunCase):
        def __call__(self, harness: Any, case: Any, model: Any, store: Any, *, run_id: str) -> Trace:
            t = super().__call__(harness, case, model, store, run_id=run_id)
            if self.calls < 3 or case.id == "timeout-1":
                reasons = [*t.outcome.reasons, f"{INFRA_REASON}: HTTP 503"]
                return replace(t, outcome=replace(t.outcome, passed=False, reasons=reasons))
            return t

    rep = evaluate(SEED, fake_cases()[:2], Flaky(), None, workers=1)
    r1, r2 = rep.results
    assert r1.infra_failed and r1.attempts == 3
    assert not r2.infra_failed and r2.attempts == 1
    assert rep.to_payload()["infra_failed"] == 1 and rep.to_payload()["infra_retries"] == 2
