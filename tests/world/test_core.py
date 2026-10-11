from __future__ import annotations

import pytest

from ratchet.types import ToolCall
from ratchet.world import Fault, StateView, World
from tests.world.helpers import Caller, company, world


def test_tools_are_openai_function_schemas() -> None:
    specs = World(company().build()).tools()
    names = [s["function"]["name"] for s in specs]
    assert names == sorted(set(names)) and len(names) >= 30
    for spec in specs:
        assert spec["type"] == "function"
        fn = spec["function"]
        assert fn["description"] and fn["parameters"]["type"] == "object"
        assert set(fn["parameters"]["required"]) <= set(fn["parameters"]["properties"])
    assert World(company().build()).tool_names() == names


def test_unknown_tool_and_bad_arguments_are_refused_without_effect() -> None:
    w = world()
    call = Caller(w)
    before = w.digest()
    assert call("nope").error == "unknown tool: nope"
    assert call("payments_get_charge").error == "invalid_arguments: args.charge_id is required"
    assert (
        call("payments_get_charge", charge_id=5).error == "invalid_arguments: args.charge_id must be string"
    )
    assert (
        call("payments_get_charge", charge_id="ch_a", x=1).error
        == "invalid_arguments: args.x is not a known field"
    )
    bad_int = call("payments_refund", charge_id="ch_a", amount_cents=True, idempotency_key="k", reason="r")
    assert bad_int.error == "invalid_arguments: args.amount_cents must be integer"
    assert call("tickets_set_status", ticket_id="T-1", status="closed").error.startswith(  # type: ignore[union-attr]
        "invalid_arguments: args.status must be one of"
    )
    nested = call("orders_modify_items", order_id="ord_a", changes=[{"line_id": "L1"}])
    assert nested.error == "invalid_arguments: args.changes[0].new_sku is required"
    assert call("orders_modify_items", order_id="ord_a", changes=[{"line_id": "L1", "new_sku": 1}]).error == (
        "invalid_arguments: args.changes[0].new_sku must be string"
    )
    assert w.step == 8
    assert w.view()["payments"] == StateView(company().build())["payments"]
    assert w.digest() != before  # the clock moved


def test_clock_advances_one_minute_per_agent_call_and_not_for_others() -> None:
    w = world()
    call = Caller(w)
    start = w.now()
    call("tickets_get", ticket_id="T-1")
    call("tickets_get", actor="worker:dana", ticket_id="T-1")
    assert w.step == 1 and w.now() > start
    assert w.view().step == 1 and w.view().now == w.now()


def test_digest_is_deterministic_and_tracks_state() -> None:
    a, b = world(), world()
    assert a.digest() == b.digest()
    Caller(a).ok("tickets_add_note", ticket_id="T-1", body="x")
    Caller(b).ok("tickets_add_note", ticket_id="T-1", body="y")
    assert a.digest() != b.digest()


def test_view_is_read_only_and_round_trips() -> None:
    v = world().view()
    with pytest.raises(AttributeError):
        v.step = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        v["payments"]["charges"]["ch_a"]["amount_cents"] = 1
    assert "payments" in v and {"audit", "step", "now"} <= set(v.keys())
    data = v.to_json()
    data["payments"]["charges"]["ch_a"]["amount_cents"] = 1
    assert v["payments"]["charges"]["ch_a"]["amount_cents"] == 10000
    assert isinstance(v["tickets"]["tickets"]["T-1"]["notes"], tuple)


def test_unknown_system_in_initial_state_is_rejected() -> None:
    state = company().build()
    state["warehouse"] = {}
    with pytest.raises(ValueError, match="warehouse"):
        World(state)


def test_timeout_after_commit_changes_state_and_reports_timeout() -> None:
    w = world([Fault("f", "timeout_after_commit", tool="payments_refund", nth=1)])
    call = Caller(w)
    result = call("payments_refund", charge_id="ch_a", amount_cents=100, idempotency_key="k", reason="r")
    assert not result.ok and result.error == "timeout"
    assert call.ok("payments_list_refunds", charge_id="ch_a")["refunded_cents"] == 100
    assert w.view()["audit"]["world"][0]["result"] == {"tool": "payments_refund", "committed": True}


def test_timeout_before_commit_changes_nothing() -> None:
    w = world([Fault("f", "timeout_before_commit", tool="payments_refund", nth=2)])
    call = Caller(w)
    call.ok("payments_refund", charge_id="ch_a", amount_cents=100, idempotency_key="k1", reason="r")
    assert call(
        "payments_refund", charge_id="ch_a", amount_cents=100, idempotency_key="k2", reason="r"
    ).error == ("timeout")
    assert call.ok("payments_list_refunds", charge_id="ch_a")["refunded_cents"] == 100


def test_mutate_after_step_and_after_tool_with_delay() -> None:
    faults = [
        Fault(
            "rev",
            "mutate",
            after_step=1,
            ops=[{"op": "approval_revoke", "approval_id": "apr_1", "by": "sup", "reason": "no"}],
        ),
        Fault(
            "msg",
            "mutate",
            tool="orders_get",
            nth=1,
            delay=1,
            ops=[{"op": "customer_message", "ticket_id": "T-1", "body": "more", "requests": [{"type": "x"}]}],
        ),
        Fault(
            "dana",
            "mutate",
            tool="orders_get",
            nth=1,
            ops=[
                {
                    "op": "call",
                    "actor": "worker:dana",
                    "tool": "tickets_add_note",
                    "args": {"ticket_id": "T-1", "body": "mine"},
                }
            ],
        ),
        Fault(
            "pub",
            "mutate",
            after_step=2,
            ops=[
                {
                    "op": "policy_publish",
                    "doc_id": "refund-policy",
                    "version": {"version": 9, "body": "b", "rules": {}},
                }
            ],
        ),
    ]
    w = world(faults)
    call = Caller(w)
    assert call.ok("approvals_get", approval_id="apr_1")["approval"]["status"] == "granted"
    assert call.ok("approvals_get", approval_id="apr_1")["approval"]["status"] == "revoked"
    call.ok("orders_get", order_id="ord_a")
    ticket = call.ok("tickets_get", ticket_id="T-1")["ticket"]
    assert [n["author"] for n in ticket["notes"]] == ["worker:dana"] and len(ticket["messages"]) == 1
    ticket = call.ok("tickets_get", ticket_id="T-1")["ticket"]
    assert ticket["messages"][-1]["requests"] == [{"type": "x"}]
    assert call.ok("policy_get", doc_id="refund-policy")["policy"]["citation"] == "refund-policy@v9"
    assert w.view()["faults_fired"] == ("rev", "pub", "dana", "msg")


def test_unknown_fault_op_is_an_error() -> None:
    w = world([Fault("bad", "mutate", after_step=1, ops=[{"op": "explode"}])])
    with pytest.raises(ValueError, match="explode"):
        w.call(ToolCall("tickets_get", {"ticket_id": "T-1"}, "c1"))


def test_from_case_uses_initial_state_and_faults() -> None:
    from ratchet.cases import get

    case = get("refund_lost_reply.v1")
    w = World.from_case(case)
    assert w.view()["tickets"]["tickets"][case.ticket_id]["status"] == "open"
