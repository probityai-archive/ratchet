from __future__ import annotations

from ratchet.improve.diagnose import diagnose
from ratchet.improve.propose import (
    SEED,
    Instructions,
    check_guard_source,
    make_candidate,
    passing_shapes,
    propose,
    revise_guard,
)
from ratchet.improve.replay import WalkCache, replay
from ratchet.types import Harness

from .fakes import (
    BLOCK_ALL_REFUNDS,
    NO_DUP_GUARD,
    FakeSimulator,
    LocalRunner,
    ScriptedChat,
    bad_timeout,
    good_plain,
    good_timeout,
)

PARENT = Harness.build(
    parent=None,
    system_prompt="You handle refunds.",
    procedures=[],
    guards=[],
    tool_allowlist=["payments_refund", "payments_list_refunds", "payments_get_charge"],
    created_by="seed",
)

REPLY = {
    "candidates": [
        {"kind": "guard", "description": "no duplicate refunds", "source": NO_DUP_GUARD, "rationale": "r"},
        {"kind": "guard", "description": "same again", "source": NO_DUP_GUARD},
        {"kind": "guard", "description": "bad", "source": "def nope():\n    pass\n"},
        {"kind": "guard", "description": "syntax", "source": "def check(call, view, history):\n  return (\n"},
        {"kind": "procedure", "name": "status before retry", "when": "a refund times out",
         "steps": ["call list_refunds", "retry only if no succeeded refund"]},
        {"kind": "procedure", "name": "", "when": "x", "steps": []},
        {"kind": "prompt_edit", "text": "A timeout does not mean the refund failed."},
        {"kind": "prompt_edit", "text": " "},
        {"kind": "teleport"},
        "not an object",
    ]
}  # fmt: skip


def _cluster():  # type: ignore[no-untyped-def]
    return diagnose([bad_timeout(1), bad_timeout(2)], FakeSimulator())[0]


def test_propose_builds_candidates_of_each_kind_and_records_rejections() -> None:
    chat = ScriptedChat([REPLY])
    passing = [good_timeout(5), good_plain(6), good_plain(7)]
    walks = WalkCache(FakeSimulator())
    prop = propose(_cluster(), PARENT, chat, passing=passing, walks=walks, n=5)
    kinds = [c.kind for c in prop.candidates]
    assert kinds == ["guard", "procedure", "prompt_edit"]  # duplicate guard collapsed
    assert len(prop.rejected) == 6
    guard_c, proc_c, edit_c = prop.candidates
    assert guard_c.harness.parent == PARENT.version and guard_c.harness.created_by == "improver"
    assert [g.id for g in guard_c.new_guards(PARENT)] == [guard_c.change["id"]]
    assert guard_c.harness.guards[0].origin_failure_ids == ["bad-1@5", "bad-2@5"]
    assert proc_c.harness.procedures[0].name == "status before retry"
    assert edit_c.harness.system_prompt.endswith("does not mean the refund failed.")
    assert guard_c.to_payload()["targets"] == ["bad-1@5", "bad-2@5"]
    assert guard_c.instructions_version == SEED.version
    prompt = chat.prompts()
    # The model sees the failing step, the real state view, and what passing runs looked like.
    assert "<== FAILING STEP" in prompt and '"refunds": [{"charge_id": "ch_1"' in prompt
    assert "payments_refund -> payments_list_refunds" in prompt and "<== same tool, passing" in prompt
    assert SEED.text in prompt


def test_model_failure_becomes_a_rejection() -> None:
    prop = propose(_cluster(), PARENT, ScriptedChat(["no json", "still none"]))
    assert not prop.candidates and "model call failed" in prop.rejected[0]["reason"]


def test_prompt_without_walks_or_passing_traces() -> None:
    chat = ScriptedChat([{"candidates": []}])
    propose(_cluster(), PARENT, chat)
    assert "(not available)" in chat.prompts() and "(no passing trace calls this tool)" in chat.prompts()


def test_check_guard_source() -> None:
    assert check_guard_source(NO_DUP_GUARD) is None
    assert "does not define" in str(check_guard_source("x = 1"))
    assert "compile" in str(check_guard_source("def check(call, view, history):\n  return (\n"))


def test_instructions_are_content_addressed() -> None:
    a = Instructions.build("x")
    b = Instructions.build("x", parent=a.version)
    assert a.version == b.version and b.parent == a.version and Instructions.build("y").version != a.version


def test_passing_shapes_counts_sequences() -> None:
    shapes = passing_shapes([good_plain(1), good_plain(2), good_timeout(3)])
    assert shapes[0].startswith("2x  payments_get_charge -> payments_refund")


def test_revise_guard_feeds_replay_failures_back() -> None:
    cluster = _cluster()
    walks = WalkCache(FakeSimulator())
    passing = [good_plain(6)]
    broad = make_candidate({"kind": "guard", "source": BLOCK_ALL_REFUNDS}, cluster, PARENT, SEED)
    assert not isinstance(broad, str)
    guard = broad.new_guards(PARENT)[0]
    rep = replay(guard, cluster.members, cluster.traces, passing, walks, LocalRunner())
    assert not rep.passed
    chat = ScriptedChat([{"description": "narrow", "source": NO_DUP_GUARD}])
    fixed = revise_guard(broad, rep, cluster, PARENT, chat, walks=walks)
    assert not isinstance(fixed, str)
    prompt = chat.prompts()
    assert (
        "FALSE POSITIVE in passing run plain-6" in prompt
        and "Call blocked wrongly: payments_refund" in prompt
    )
    rep2 = replay(fixed.new_guards(PARENT)[0], cluster.members, cluster.traces, passing, walks, LocalRunner())
    assert rep2.passed


def test_revise_guard_reports_misses_errors_and_model_failure() -> None:
    cluster = _cluster()
    walks = WalkCache(FakeSimulator())
    silent = make_candidate(
        {"kind": "guard", "source": "def check(call, view, history):\n    return view['x']\n"},
        cluster,
        PARENT,
        SEED,
    )
    assert not isinstance(silent, str)
    rep = replay(silent.new_guards(PARENT)[0], cluster.members, cluster.traces, [], walks, LocalRunner())
    chat = ScriptedChat(["no", "no"])
    assert "revision failed" in str(revise_guard(silent, rep, cluster, PARENT, chat, walks=walks))
    prompt = chat.prompts()
    assert "MISSED bad-1@5" in prompt and "ERROR in run" in prompt
