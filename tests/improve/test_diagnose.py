from __future__ import annotations

from dataclasses import replace

from ratchet.improve.diagnose import diagnose, excerpt, locate, normalize, render_event
from ratchet.improve.llm import Meter
from ratchet.improve.sim import RecordedViews
from ratchet.types import Event

from .fakes import FakeSimulator, ScriptedChat, bad_timeout, good_plain, lazy, without_views


def test_normalize_strips_ids_and_quotes() -> None:
    assert normalize("Duplicate refund for charge ch_12 ('x')") == normalize(
        "duplicate refund for charge ch_9 ('y')"
    )


def test_clusters_by_mechanism_and_names_the_failing_step() -> None:
    traces = [bad_timeout(1), bad_timeout(2), lazy(3), good_plain(4)]
    clusters = diagnose(traces, FakeSimulator())
    assert [len(c.members) for c in clusters] == [2, 1]
    dup, omit = clusters
    assert dup.key["mode"] == "commission" and dup.key["tool"] == "payments_refund"
    assert dup.key["prior_error"] == "timeout" and dup.key["repeat"] == "same_args"
    assert {m.seq for m in dup.members} == {5}
    assert {m.id for m in dup.members} == {"bad-1@5", "bad-2@5"}
    assert set(dup.traces) == {"bad-1", "bad-2"}
    assert omit.key["mode"] == "omission" and omit.members[0].seq == lazy(3).events[-1].seq
    assert omit.key["tool"] == "<end>"
    payload = dup.to_payload()
    assert payload["members"][0]["id"].endswith("@5") and "timeout" in dup.summary


def test_model_writes_the_summary_but_not_the_grouping() -> None:
    chat = ScriptedChat(
        [{"mechanism": "retries a refund after a timeout without checking status"}, "garbage", "x"]
    )
    meter = Meter()
    clusters = diagnose([bad_timeout(1), lazy(2)], FakeSimulator(), chat, meter)
    by_mode = {c.key["mode"]: c for c in clusters}
    summaries = {c.summary for c in clusters}
    assert any(x.startswith("retries a refund") for x in summaries)
    assert len(by_mode) == 2 and any("model summary unavailable" in x for x in summaries)
    assert meter.model_calls == 3


def test_without_intermediate_grading_falls_back_to_mentions() -> None:
    t = bad_timeout(7)
    seq, mode = locate(t, FakeSimulator(grade=False).walk(t), t.outcome.reasons[0])
    assert (seq, mode) == (5, "commission")
    seq, mode = locate(t, None, "something vague")
    assert mode == "unlocated" and seq == t.events[-1].seq


def test_reason_that_clears_by_the_end_is_an_omission_at_the_end() -> None:
    t = bad_timeout(1)
    walk = FakeSimulator().walk(t)
    # A reason the intermediate grader never reports after the last call is located at the end.
    assert locate(t, walk, "customer not refunded for charge ch_1") == (t.events[-1].seq, "omission")


def test_walk_errors_are_reported_and_do_not_sink_diagnosis() -> None:
    class Broken:
        def walk(self, trace: object) -> object:
            raise ValueError("cannot rebuild")

    seen: list[str] = []
    clusters = diagnose([bad_timeout(1)], Broken(), on_walk_error=lambda t, e: seen.append(str(e)))  # type: ignore[arg-type]
    assert seen == ["cannot rebuild"] and clusters[0].key["mode"] == "commission"


def test_failed_trace_without_reasons_still_clusters() -> None:
    t = bad_timeout(1)
    t = replace(t, outcome=replace(t.outcome, reasons=[]))
    assert diagnose([t])[0].members[0].reason == "failed without a reason"


def test_excerpt_marks_the_failing_step_and_truncates() -> None:
    t = bad_timeout(1)
    text = excerpt(t, 5)
    assert "<== FAILING STEP" in text and text.count("\n") >= 4
    long = Event(0, "model", {"content": "x" * 1000}, 0.0)
    assert render_event(long).endswith("...(truncated)")


def test_recorded_views_reads_snapshots() -> None:
    t = without_views(bad_timeout(1))
    w = RecordedViews().walk(t)
    assert not w.faithful and w.initial_reasons is None and len(w.steps) == 3
    events = [
        replace(e, payload=dict(e.payload, view={"k": 1})) if e.kind == "tool_call" else e for e in t.events
    ]
    w2 = RecordedViews().walk(replace(t, events=events))
    assert w2.faithful and w2.steps[0].view == {"k": 1} and w2.steps[0].executed


def test_record_ids_emails_and_slugs_do_not_split_one_mechanism() -> None:
    a = "mail m_1 sent to customer-record, not the customer of record aiko.tanaka@docomo.example"
    b = "mail m_2 sent to customer@record.invalid, not the customer of record omar.farouk@yahoo.example"
    assert normalize(a) == normalize(b)


def test_endpoint_failures_are_not_diagnosed() -> None:
    from ratchet.improve.evaluate import INFRA_REASON

    t = bad_timeout(1)
    lost = replace(t, outcome=replace(t.outcome, reasons=[f"{INFRA_REASON}: timeout"]))
    assert diagnose([lost]) == []
