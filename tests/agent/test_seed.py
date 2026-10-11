from __future__ import annotations

from ratchet.agent.seed import SEED_PROMPT, all_tool_names, seed_harness


def test_seed_is_v0_with_full_allowlist_and_nothing_else() -> None:
    harness = seed_harness()
    assert harness.parent is None and harness.created_by == "seed" and harness.verify_digest()
    assert harness.procedures == [] and harness.guards == []
    assert harness.tool_allowlist == sorted(all_tool_names())
    assert {"payments_refund", "tickets_get", "mail_send"} <= set(harness.tool_allowlist)
    assert harness.system_prompt == SEED_PROMPT


def test_seed_is_content_addressed_and_accepts_a_tool_list() -> None:
    assert seed_harness().version == seed_harness().version
    narrow = seed_harness(["tickets_get"])
    assert narrow.tool_allowlist == ["tickets_get"] and narrow.version != seed_harness().version
