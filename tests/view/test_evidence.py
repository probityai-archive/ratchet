from __future__ import annotations

from typing import Any

from ratchet.types import Guard, Harness, Procedure
from ratchet.view.evidence import collect, rebuild_candidate, stored_reports

from .fakes import FakeStore, eval_doc, scenario


def parent() -> Harness:
    return Harness.build(
        parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=["a"], created_by="seed"
    )


def test_rebuild_guard_procedure_and_prompt_edit_by_digest() -> None:
    p = parent()
    g = Guard("g1", "d", "src", ["r@1"])
    proc = Procedure("p1", "n", "w", ["s"], [])
    guard_change = {"id": "g1", "description": "d", "source": "src", "origin_failure_ids": ["r@1"]}
    proc_change = {"id": "p1", "name": "n", "when": "w", "steps": ["s"], "origin_failure_ids": []}
    cases: list[tuple[dict[str, Any], Harness]] = [
        ({"kind": "guard", "change": guard_change}, p.child(guards=[g], created_by="improver")),
        ({"kind": "procedure", "change": proc_change}, p.child(procedures=[proc], created_by="improver")),
        (
            {"kind": "prompt_edit", "change": {"append": "more"}},
            p.child(system_prompt="p\n\nmore", created_by="improver"),
        ),
    ]
    for ctx, expected in cases:
        built = rebuild_candidate(p, ctx, expected.digest)
        assert built is not None
        assert built.digest == expected.digest
        assert rebuild_candidate(p, ctx, "0" * 64) is None


def test_rebuild_refuses_what_it_cannot_describe() -> None:
    p = parent()
    assert rebuild_candidate(p, {"kind": "guard", "change": {}}, "x") is None
    assert rebuild_candidate(p, {"kind": "composite"}, "x") is None
    assert rebuild_candidate(p, {}, "x") is None


def test_stored_reports_are_found_by_the_digest_of_the_report_itself() -> None:
    from ratchet.verify import report_digest

    store = FakeStore()
    payload = eval_doc("v", (1, 2), (1, 2), [])
    store.eval_docs = [{**payload, "role": "candidate", "generation": 1, "_id": "x", "digest": "d"}]
    store.eval_docs.append(
        {"type": "replay", "role": "referee", "generation": 1, "parent": "p", "guard_id": "g"}
    )
    found = stored_reports(store)
    assert report_digest(payload) in found
    assert report_digest({"guard_id": "g"}) in found
    store.eval_docs.append({"type": "generation"})
    assert len(stored_reports(store)) == 2


def test_collect_without_a_parent_or_candidate_says_what_is_missing() -> None:
    store = FakeStore()
    ev = collect(store, {"parent": "nope", "candidate": "c", "report_digests": {}})
    assert ev.notes == ["the parent harness is not in the store"]
    assert ev.harnesses == {}
    s, seed, _ = scenario()
    ev = collect(
        s, {"parent": seed.version, "candidate": "ghost", "candidate_digest": "0" * 64, "context": {}}
    )
    assert "neither stored nor rebuildable" in ev.notes[0]
    assert list(ev.harnesses) == ["parent"]


def test_collect_supplies_harnesses_and_reports_but_not_replay_without_a_recorded_run() -> None:
    s, seed, good = scenario()
    doc = {
        "parent": seed.version,
        "candidate": good.version,
        "candidate_digest": good.digest,
        "report_digests": {"x": "0" * 64},
        "context": {"generation": 1},
    }
    from contextlib import nullcontext

    from .fakes import RULES, FakeRunner

    ev = collect(s, doc, lambda: nullcontext(FakeRunner(RULES)), None)
    assert set(ev.harnesses) == {"parent", "candidate"}
    assert ev.reports == {}
    assert ev.recomputed == {}
    from ratchet.improve.sim import RecordedViews

    ev = collect(s, doc, lambda: nullcontext(FakeRunner(RULES)), RecordedViews())
    assert any("not in the store" in n for n in ev.notes)


def test_the_store_supplies_the_traces_a_report_commits_and_the_suite_cases_it_names() -> None:
    from dataclasses import asdict

    from ratchet_cases.cases import get

    from ratchet.verify.runs import case_document, run_record
    from ratchet.view.evidence import report_traces, suite_cases

    from .fakes import make_trace

    store = FakeStore()
    kept, lost = make_trace("r1", "refund.t1", True, []), make_trace("r2", "refund.t2", False, [])
    store.trace_list = [kept]
    report = {
        "runs": [run_record(asdict(kept)), run_record(asdict(lost))],
        "results": [
            {"case_id": "refund_lost_reply.v3", "split": "heldout"},
            {"case_id": "not-in-the-suite", "split": "heldout"},
            {"case_id": "refund_lost_reply.v1", "split": "train"},
        ],
    }
    assert report_traces(store, {"p1": report, "legacy": {"results": []}}) == {"p1": [asdict(kept)]}
    assert suite_cases({"p1": report}) == {"refund_lost_reply.v3": case_document(get("refund_lost_reply.v3"))}
