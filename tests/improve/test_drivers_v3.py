from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from ratchet.improve.run_det import normalized
from ratchet.improve.run_v3 import rebuild
from ratchet.types import Guard, Harness, Procedure

from .fakes import NO_DUP_GUARD, bad_timeout

SEED = Harness.build(parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=["t"],
                     created_by="seed")  # fmt: skip


def test_rebuild_reproduces_the_judged_candidate_by_digest() -> None:
    g = Guard("g1", "d", NO_DUP_GUARD, ["r@1"])
    cand = SEED.child(guards=[g], created_by="improver")
    assert (
        rebuild(SEED, {"candidate": cand.version, "context": {"kind": "guard", "change": asdict(g)}}) == cand
    )
    proc = Procedure("p1", "n", "w", ["s"], [])
    pc = SEED.child(procedures=[proc], created_by="improver")
    assert (
        rebuild(SEED, {"candidate": pc.version, "context": {"kind": "procedure", "change": asdict(proc)}})
        == pc
    )
    edit = SEED.child(system_prompt="p\n\nMore.", created_by="improver")
    ctx = {"kind": "prompt_edit", "change": {"append": "More."}}
    assert rebuild(SEED, {"candidate": edit.version, "context": ctx}) == edit
    with pytest.raises(SystemExit, match="rebuilt"):
        rebuild(SEED, {"candidate": "0" * 12, "context": ctx})


def test_normalized_traces_ignore_times_and_ids_but_not_behaviour() -> None:
    a = bad_timeout(1, run="r1")
    b = bad_timeout(1, run="r2")  # different run and call ids
    later = replace(b, events=[replace(e, ts=e.ts + 99) for e in b.events])
    assert normalized(a) == normalized(later)
    other = bad_timeout(2, run="r1")
    assert normalized(a) != normalized(other)
