"""`retail_faults` on stock tau2-bench: the task set, and three scripted agents graded on the live databases.

Run by the `tau2` CI job against tau2-bench installed from its repository at a pinned commit, with
`TAU2_DATA_DIR` pointing at that commit's `data/`. No model, no user simulator, no key.
"""

from __future__ import annotations

import pytest

from ratchet_cases import tau2
from ratchet_cases.tau2 import FaultKind, Policy

TASKS = tau2.faulted_tasks()
AFTER = [ft for ft in TASKS if ft.fault.kind is FaultKind.TIMEOUT_AFTER_COMMIT]
BEFORE = [ft for ft in TASKS if ft.fault.kind is FaultKind.TIMEOUT_BEFORE_COMMIT]


def _id(ft: tau2.FaultedTask) -> str:
    return ft.task_id


def test_the_fault_targets_are_every_retail_write_tool() -> None:
    assert tau2.write_tools() == [
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    ]


def test_each_write_tool_gets_both_timeouts_on_one_base_task() -> None:
    base = {
        "cancel_pending_order": "30",
        "exchange_delivered_order_items": "0",
        "modify_pending_order_address": "17",
        "modify_pending_order_items": "15",
        "modify_pending_order_payment": "40",
        "modify_user_address": "33",
        "return_delivered_order_items": "5",
    }
    expected = [
        {
            "task_id": f"retail_faults/{tool}/{kind.value}/{task_id}",
            "base_task_id": task_id,
            "tool_name": tool,
            "kind": kind.value,
        }
        for tool, task_id in base.items()
        for kind in FaultKind
    ]
    assert tau2.manifest() == expected
    assert len(TASKS) == 14


@pytest.mark.parametrize("ft", TASKS, ids=_id)
def test_an_agent_that_reads_before_retrying_passes(ft: tau2.FaultedTask) -> None:
    assert tau2.run(ft, Policy.RECONCILE).passed


@pytest.mark.parametrize("ft", BEFORE, ids=_id)
def test_an_agent_that_gives_up_leaves_the_change_undone(ft: tau2.FaultedTask) -> None:
    grade = tau2.run(ft, Policy.GIVE_UP)
    assert not grade.agent_db_match
    assert not grade.passed


@pytest.mark.parametrize("ft", AFTER, ids=_id)
def test_giving_up_after_a_commit_still_leaves_the_gold_state(ft: tau2.FaultedTask) -> None:
    assert tau2.run(ft, Policy.GIVE_UP).passed


@pytest.mark.parametrize("ft", TASKS, ids=_id)
def test_a_blind_retry_is_refused_by_the_tool_and_leaves_the_gold_state(ft: tau2.FaultedTask) -> None:
    # Measured at the pinned commit: every retail write tool refuses a second application to the same
    # record (a cancelled order is no longer pending, a modified order is no longer modifiable), so a blind
    # retry after a commit changes nothing. If a tool stops refusing, this test is where it shows.
    assert tau2.run(ft, Policy.BLIND_RETRY).passed


def test_the_fault_fires_once_and_the_retry_reaches_the_tool() -> None:
    ft = next(f for f in BEFORE if f.fault.tool_name == "cancel_pending_order")
    injector = tau2.FaultInjector(tau2.get_environment(), ft.fault)
    log = tau2.play(ft.task, injector.get_response, Policy.RECONCILE)
    writes = [
        (call.id, reply.error, reply.content) for call, reply in log if call.name == "cancel_pending_order"
    ]
    assert writes[0][1:] == (True, tau2.TIMEOUT)
    assert len(writes) == 2
    assert writes[1][1] is False
    assert injector.calls["cancel_pending_order"] == 2


def test_register_refuses_a_tau2_that_cannot_carry_faults() -> None:
    # Stock tau2-bench's Task drops unknown fields, so its orchestrator would run the set unfaulted.
    registered: list[object] = []

    class Registry:
        def register_tasks(self, *args: object, **kwargs: object) -> None:
            registered.append(args)

    with pytest.raises(RuntimeError, match="no `faults` field"):
        tau2.register(Registry())
    assert registered == []
