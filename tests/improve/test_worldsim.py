"""The production simulator against a real case, run by the real worker with a scripted model."""

from __future__ import annotations

from dataclasses import replace

from ratchet.agent.model import ScriptedModel, scripted_turn
from ratchet.agent.seed import seed_harness
from ratchet.agent.worker import run_case
from ratchet.cases import all_cases
from ratchet.improve.diagnose import diagnose
from ratchet.improve.replay import WalkCache, replay
from ratchet.improve.worldsim import WorldSimulator
from ratchet.types import Guard, Trace

from .fakes import LocalRunner

CASES = {c.id: c for c in all_cases()}
ARGS = {"charge_id": "ch_3MoK8", "amount_cents": 3200, "reason": "cracked kettle"}

GUARD = """
def check(call, view, history):
    if call.name != "payments_refund":
        return None
    charge = call.args.get("charge_id")
    done = [r for r in view.get("payments", {}).get("refunds", {}).values()
            if r.get("charge_id") == charge and r.get("status") == "succeeded"]
    if done:
        return "a refund for this charge already succeeded at the provider; read it instead of retrying"
    return None
"""


def _run(case_id: str, keys: list[str], run_id: str) -> Trace:
    case = CASES[case_id]
    turns = [scripted_turn(None, [("tickets_get", {"ticket_id": case.ticket_id})], id_prefix="a")]
    turns += [scripted_turn(None, [("payments_refund", {**ARGS, "idempotency_key": k})], id_prefix=f"k{i}")
              for i, k in enumerate(keys)]  # fmt: skip
    turns.append(scripted_turn("done"))
    return run_case(seed_harness(), case, ScriptedModel(turns), None, run_id=run_id)


def test_walk_rebuilds_the_world_and_locates_the_duplicate_refund() -> None:
    bad = _run("refund_lost_reply.v1", ["k1", "k2"], "bad")
    sim = WorldSimulator(CASES.values())
    walk = sim.walk(bad)
    assert walk.faithful, walk.note
    assert [s.call.name for s in walk.steps] == ["tickets_get", "payments_refund", "payments_refund"]
    cluster = next(c for c in diagnose([bad], sim) if "twice" in c.key["reason"])
    assert cluster.members[0].seq == walk.steps[2].seq and cluster.key["mode"] == "commission"
    assert cluster.key["prior_error"] == "timeout"


def test_a_guard_over_real_views_clears_and_a_tampered_record_is_flagged() -> None:
    bad = _run("refund_lost_reply.v1", ["k1", "k2"], "bad")
    good = _run("refund_lost_reply.v1", ["k1"], "good")
    sim = WorldSimulator(CASES)
    cluster = next(c for c in diagnose([bad], sim) if "twice" in c.key["reason"])
    rep = replay(
        Guard("g", "d", GUARD), cluster.members, cluster.traces, [good], WalkCache(sim), LocalRunner()
    )
    assert rep.targets[0]["hit"] and not rep.false_positives
    events = [replace(e, payload={**e.payload, "view": {"forged": True}}) if e.kind == "tool_call" else e
              for e in good.events]  # fmt: skip
    walk = sim.walk(replace(good, events=events))
    assert not walk.faithful and "differ" in walk.note
