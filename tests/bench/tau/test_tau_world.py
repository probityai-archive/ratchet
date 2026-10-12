from __future__ import annotations

from ratchet_cases.types import Cost, ToolCall
from tau_testkit import ScriptedUser, gold_actions, make_world

from ratchet.bench.tau.world import RESPOND, STOP, outcome_of, record_diffs


def _do(world, name, args, cid="c"):  # type: ignore[no-untyped-def]
    return world.call(ToolCall(name, args, cid))


def test_tools_are_tau_bench_tools_plus_respond() -> None:
    world = make_world(0)
    names = world.tool_names()
    assert RESPOND in names
    assert "exchange_delivered_order_items" in names
    assert len(world.tools()) == len(names) == 17


def test_respond_returns_the_customer_reply_and_stop_ends_the_conversation() -> None:
    user = ScriptedUser(["My email is a@b.c", STOP])
    world = make_world(0, user)
    first = _do(world, RESPOND, {"content": "Hello, your email?"})
    assert first.ok and first.data == {"observation": "My email is a@b.c", "done": False}
    assert not world.done
    second = _do(world, RESPOND, {"content": "Anything else?"})
    assert second.data is not None and second.data["done"] is True
    assert world.done and world.ending is not None and world.ending.by == RESPOND
    assert user.heard == ["Hello, your email?", "Anything else?"]
    after = _do(world, "get_order_details", {"order_id": "#W2378156"})
    assert not after.ok and after.error == "the conversation has ended"


def test_respond_needs_string_content() -> None:
    world = make_world(0)
    result = _do(world, RESPOND, {"content": 3})
    assert not result.ok and "string" in (result.error or "")


def test_unknown_tool_and_tool_errors_are_reported_like_tau_bench() -> None:
    world = make_world(0)
    unknown = _do(world, "launch_rocket", {})
    assert not unknown.ok and unknown.error == "Unknown action launch_rocket"
    missing = _do(world, "get_order_details", {"order_id": "#W0000000"})
    assert not missing.ok and (missing.error or "").startswith("Error")
    assert missing.data is not None and missing.data["done"] is False


def test_view_holds_only_the_records_the_conversation_touched() -> None:
    world = make_world(0)
    assert world.view().to_json() == {"step": 0, "done": False, "users": {}, "orders": {}}
    _do(world, "get_order_details", {"order_id": "#W2378156"})
    view = world.view().to_json()
    assert list(view["orders"]) == ["#W2378156"]
    assert list(view["users"]) == ["yusuf_rossi_9620"]
    assert view["step"] == 1
    view["orders"].clear()
    assert world.view().to_json()["orders"], "the view is a copy"


def test_transfer_ends_the_conversation() -> None:
    world = make_world(10)
    result = _do(world, "transfer_to_human_agents", {"summary": "needs a human"})
    assert result.ok and result.data is not None and result.data["done"] is True
    assert world.ending is not None and world.ending.by == "transfer_to_human_agents"


def test_gold_actions_earn_reward_one() -> None:
    world = make_world(0)
    for i, (name, args) in enumerate(gold_actions(0)):
        assert _do(world, name, args, f"g{i}").ok
    digest_before = world.digest()
    _do(world, RESPOND, {"content": "Done."})
    outcome = outcome_of(world, Cost(model_calls=3))
    assert outcome.passed and outcome.reasons == []
    assert outcome.cost.model_calls == 3
    assert outcome.final_state_digest == digest_before == world.digest()


def test_wrong_change_fails_with_record_level_reasons() -> None:
    world = make_world(0)
    _do(world, "cancel_pending_order", {"order_id": "#W2378156", "reason": "no longer needed"})
    _do(world, "modify_user_address", {
        "user_id": "yusuf_rossi_9620", "address1": "1 Main St", "address2": "", "city": "X",
        "state": "PA", "country": "USA", "zip": "19122",
    })  # fmt: skip
    _do(world, RESPOND, {"content": "Done."})
    outcome = outcome_of(world, Cost())
    assert not outcome.passed
    assert any("users['yusuf_rossi_9620'].address" in r for r in outcome.reasons)
    assert any("orders['#W2378156']" in r for r in outcome.reasons)


def test_missing_output_is_a_reason() -> None:
    world = make_world(2)
    for i, (name, args) in enumerate(gold_actions(2)):
        _do(world, name, args, f"g{i}")
    _do(world, RESPOND, {"content": "Your return is set up."})
    outcome = outcome_of(world, Cost())
    assert not outcome.passed
    assert outcome.reasons == ["the agent never told the user '10'"]


def test_output_said_and_wrong_database_lists_both() -> None:
    world = make_world(2)
    _do(world, RESPOND, {"content": "There are 10 types."})
    _do(world, RESPOND, {"content": "Bye."})
    outcome = outcome_of(world, Cost())
    assert not outcome.passed
    assert outcome.reasons and all("never told" not in r for r in outcome.reasons)


def test_unfinished_conversation_scores_zero_and_says_what_was_left() -> None:
    world = make_world(0)
    final_before = world.env.data
    outcome = outcome_of(world, Cost())
    assert not outcome.passed
    assert outcome.reasons[0] == "the conversation did not end within the agent's action budget"
    assert any("orders['#W2378156']" in r for r in outcome.reasons[1:])
    assert world.env.data is final_before


def test_record_diffs_reports_missing_extra_and_truncates() -> None:
    final = {"t": {"a": {"x": 1}, "b": {"x": 1}}}
    gold = {"t": {"a": {"x": 2}, "c": {"x": 1}}}
    diffs = record_diffs(final, gold)
    assert diffs == [
        "t['a'].x is 1, the task needs 2",
        "t['b'] should not exist in the final database",
        "t['c'] is missing in the final database",
    ]
    many = record_diffs({"t": {str(i): {"x": i} for i in range(5)}}, {"t": {}}, limit=2)
    assert len(many) == 3 and many[-1] == "... and 3 more differences"
    long = record_diffs({"t": {"a": {"x": "y" * 400}}}, {"t": {"a": {"x": ""}}})
    assert "..." in long[0] and len(long[0]) < 260
