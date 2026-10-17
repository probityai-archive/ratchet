"""The archive: clade statistics, parent selection and rebasing, on built trees and a real Atlas lineage."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pymongo import MongoClient
from ratchet_cases.types import Cost

from ratchet.improve import archive as arc
from ratchet.improve.admit import Check, Decision, report_digest
from ratchet.improve.archive import (
    Archive,
    ArchiveSelector,
    RebaseError,
    _verdict,
    rebase,
    select_parent,
)
from ratchet.improve.evaluate import CaseResult, EvalReport
from ratchet.store.atlas import Store
from ratchet.types import Guard, Harness, Procedure

# -- builders: the records are produced by the real report and decision types ------------------------------


def report(version: str, heldout: Sequence[bool], train: Sequence[bool] = (), **bad: Any) -> EvalReport:
    """An evaluation of `version`: one result per outcome. `bad` edits the first result."""
    results = [
        CaseResult(f"h{i}", "heldout", True, "must_succeed", ok, [], f"run-{i}", Cost())
        for i, ok in enumerate(heldout)
    ] + [
        CaseResult(f"t{i}", "train", True, "must_succeed", ok, [], f"run-t{i}", Cost())
        for i, ok in enumerate(train)
    ]
    if bad and results:
        results[0] = CaseResult(**{**results[0].__dict__, **bad})
    return EvalReport(version, results, 1.0)


def eval_doc(
    r: EvalReport, role: str = "candidate", generation: int = 1, eid: str | None = None
) -> dict[str, Any]:
    return {
        "_id": eid or f"eval-{uuid.uuid4().hex[:8]}",
        **r.to_payload(),
        "role": role,
        "generation": generation,
    }


def decision_doc(
    parent: str,
    cand: str,
    admitted: bool,
    *,
    source: str | None = None,
    reports: dict[str, EvalReport] | None = None,
    at: str = "2026-10-11T10:00:00+00:00",
) -> dict[str, Any]:
    fresh = Check(
        "fresh", admitted, {"stable_recoveries": ["a", "b"] if admitted else [], "parent_discordance": 1}
    )
    d = Decision(
        parent=parent,
        candidate=cand,
        candidate_digest=f"digest-{cand}",
        admitted=admitted,
        checks=[fresh, Check("replay", True, {})],
        report_digests={k: report_digest(v.to_payload()) for k, v in (reports or {}).items()},
        context={"archive": {"source": source}} if source else {},
        decided_at=at,
    )
    return {"_id": f"{parent}->{cand}", **d.to_payload()}


def hv(version: str, parent: str | None, by: str = "improver") -> dict[str, Any]:
    return {"version": version, "parent": parent, "created_by": by}


def bools(passes: int, failures: int) -> list[bool]:
    return [True] * passes + [False] * failures


def tree() -> Archive:
    """S (2 of 4) with an admitted child A (4 of 4) and a rejected child R (1 of 4)."""
    return Archive.from_records(
        [hv("S", None, "seed"), hv("A", "S")],
        [
            eval_doc(report("S", bools(2, 2)), "parent_run_1"),
            eval_doc(report("A", bools(4, 0))),
            eval_doc(report("R", bools(1, 3))),
        ],
        [decision_doc("S", "A", True), decision_doc("S", "R", False)],
    )


# -- clade statistics ----------------------------------------------------------------------------------------


def test_leaf_clade_is_its_own_pass_rate_with_no_descendants() -> None:
    a = Archive.from_records([hv("S", None, "seed")], [eval_doc(report("S", bools(3, 1)))], [])
    c = a.clade("S")
    assert (c.descendants, c.passes, c.failures, c.rate) == (0, 3, 1, 0.75)
    assert c.admission_yield is None and c.gain_mean is None
    assert c.is_thin(1) and a.nodes["S"].status == "seed" and a.gain("S") is None


def test_one_admitted_and_one_rejected_child() -> None:
    a = tree()
    c = a.clade("S")
    assert (c.descendants, c.admitted_descendants, c.rejected_descendants) == (2, 1, 1)
    assert (c.passes, c.failures) == (2 + 4 + 1, 2 + 0 + 3)
    assert c.rate == pytest.approx(7 / 12)
    assert c.admission_yield == 0.5
    # the rejected child counts as an attempt and adds its outcomes to the pool, but earns no gain
    assert c.gain_n == 1 and c.gain_sum == pytest.approx(0.5) and c.gain_mean == pytest.approx(0.5)
    assert a.gain("A") == pytest.approx(0.5)
    assert a.nodes["R"].status == "rejected" and not a.nodes["R"].stored and not a.nodes["R"].eligible


def test_deep_chain_credits_every_ancestor() -> None:
    a = Archive.from_records(
        [hv("S", None, "seed"), hv("A", "S"), hv("B", "A"), hv("C", "B")],
        [eval_doc(report(v, bools(n, 4 - n))) for v, n in (("S", 1), ("A", 2), ("B", 3), ("C", 4))],
        [decision_doc("S", "A", True), decision_doc("A", "B", True), decision_doc("B", "C", True)],
    )
    assert [a.clade(v).descendants for v in "SABC"] == [3, 2, 1, 0]
    assert a.clade("S").rate == pytest.approx(10 / 16)
    assert a.clade("B").rate == pytest.approx(7 / 8)
    assert a.clade("S").gain_mean == pytest.approx(0.25)  # each step gains one case of four
    assert a.tip == "C"


def test_only_fresh_error_free_outcomes_on_the_split_count() -> None:
    r = report("S", bools(2, 1), train=bools(1, 1), mode="CACHED")
    s = report("S", bools(1, 1), error="run error")
    a = Archive.from_records([hv("S", None, "seed")], [eval_doc(r), eval_doc(s)], [])
    e = a.nodes["S"].evidence
    assert (e.passes, e.failures, e.invalid) == (1, 2, 2)
    train = Archive.from_records([hv("S", None, "seed")], [eval_doc(r)], [], split="train")
    assert (train.nodes["S"].evidence.passes, train.nodes["S"].evidence.failures) == (1, 1)


def test_duplicate_and_foreign_eval_documents_are_ignored() -> None:
    doc = eval_doc(report("S", bools(1, 1)), eid="same")
    other = {"_id": "x", "type": "replay", "harness_version": "S"}
    a = Archive.from_records([hv("S", None, "seed")], [doc, doc, other, {"type": "eval"}], [])
    assert a.nodes["S"].evidence.n == 2


def test_decision_names_the_evaluations_it_used() -> None:
    p1, c1 = report("S", bools(2, 2)), report("A", bools(3, 1))
    a = Archive.from_records(
        [hv("S", None, "seed"), hv("A", "S")],
        [eval_doc(p1, "parent_run_1"), eval_doc(c1), eval_doc(report("A", bools(0, 4)), "stray")],
        [decision_doc("S", "A", True, reports={"parent_run_1": p1, "candidate_run": c1})],
    )
    assert [e.named_by for e in a.nodes["A"].evidence.evals] == [("S->A:candidate_run",), ()]
    assert a.nodes["S"].evidence.evals[0].named_by == ("S->A:parent_run_1",)
    v = a.nodes["A"].verdict
    assert v is not None and v.admitted and v.stable_recoveries == 2 and v.parent_discordance == 1
    assert v.gates == {"fresh": True, "replay": True}


def test_latest_decision_wins_and_is_counted() -> None:
    a = Archive.from_records(
        [hv("S", None, "seed")],
        [],
        [
            decision_doc("S", "A", False, at="2026-10-11T10:00:00+00:00"),
            decision_doc("S", "A", True, at="2026-10-11T11:00:00+00:00"),
            decision_doc("S", "A", False, at="2026-10-11T09:00:00+00:00"),
        ],
    )
    v = a.nodes["A"].verdict
    assert v is not None and v.admitted and v.decisions == 3 and a.nodes["A"].status == "admitted"


def test_unknown_parent_and_unjudged_nodes_are_kept_but_never_parents() -> None:
    a = Archive.from_records([hv("X", "GONE"), hv("S", None, "seed")], [], [])
    assert a.nodes["GONE"].status == "unknown" and a.nodes["X"].status == "unjudged"
    assert not a.nodes["GONE"].eligible and not a.nodes["X"].eligible
    assert a.tip == "S"
    assert a.clade("GONE").descendants == 1


# -- source attribution --------------------------------------------------------------------------------------


def branching() -> Archive:
    """S -> A -> T is the chain; X was inspired by A but evaluated against T."""
    return Archive.from_records(
        [hv("S", None, "seed"), hv("A", "S"), hv("T", "A")],
        [eval_doc(report(v, bools(n, 4 - n))) for v, n in (("S", 1), ("A", 2), ("T", 2), ("X", 4))],
        [
            decision_doc("S", "A", True),
            decision_doc("A", "T", True),
            decision_doc("T", "X", True, source="A"),
        ],
    )


def test_a_rebased_child_is_credited_to_its_source_ancestor() -> None:
    a = branching()
    x = a.nodes["X"]
    assert (x.parent, x.source, a.attribution_parent("X")) == ("T", "A", "A")
    assert a.clade("T").descendants == 0
    assert a.clade("A").descendants == 2  # T and X
    assert a.clade("S").descendants == 3
    assert a.gain("X") == pytest.approx(0.5)  # measured against the comparison parent T, not the source


def test_a_source_that_is_not_an_ancestor_is_ignored_with_a_note() -> None:
    a = Archive.from_records(
        [hv("S", None, "seed"), hv("A", "S"), hv("B", "S")],
        [],
        [
            decision_doc("S", "A", True),
            decision_doc("S", "B", True, source="A"),
            decision_doc("A", "C", True, source="A"),
        ],
    )
    assert a.nodes["B"].source is None and a.attribution_parent("B") == "S"
    assert a.nodes["C"].source is None  # source equal to the parent is plain parentage
    assert len(a.notes) == 1 and "B: recorded source A" in a.notes[0]


# -- selection -----------------------------------------------------------------------------------------------


def two_clades() -> Archive:
    """Seed S; admitted children G (strong clade) and W (weak clade), each with two attempted children."""
    heads = [hv("S", None, "seed"), hv("G", "S"), hv("W", "S")]
    evals = [
        eval_doc(report(v, bools(p, 20 - p)))
        for v, p in (("S", 10), ("G", 18), ("W", 4), ("g1", 19), ("g2", 17), ("w1", 3), ("w2", 5))
    ]
    decisions = [decision_doc("S", "G", True), decision_doc("S", "W", True)]
    decisions += [decision_doc("G", c, True) for c in ("g1", "g2")] + [
        decision_doc("W", c, False) for c in ("w1", "w2")
    ]
    return Archive.from_records([*heads, hv("g1", "G"), hv("g2", "G")], evals, decisions)


def test_selector_falls_back_to_the_tip_under_the_threshold() -> None:
    c = select_parent(tree(), 3, tip="A", seed=1)
    assert (c.version, c.tip, c.rebase_required) == ("A", "A", False)
    assert c.reason["kind"] == "tip" and c.reason["min_descendants"] == 3 and c.reason["candidates"] == []


def test_selector_picks_a_qualifying_admitted_clade_and_requires_a_rebase() -> None:
    a = two_clades()
    greedy = select_parent(a, 2, tip="g2", explore=False)
    assert (greedy.version, greedy.tip, greedy.rebase_required) == ("G", "g2", True)
    assert greedy.reason["kind"] == "greedy"
    rows = {r["version"]: r for r in greedy.reason["candidates"]}
    assert set(rows) == {"S", "G", "W"}  # g1 and g2 are leaves; the seed, G and W have two or more attempts
    assert all(r["descendants"] >= 2 for r in rows.values())
    assert rows["G"]["clade_rate"] > rows["S"]["clade_rate"] > rows["W"]["clade_rate"]
    assert rows["S"]["descendants"] == 6 and rows["G"]["descendants"] == 2
    assert rows["W"]["admitted_descendants"] == 0


def test_thompson_sampling_is_reproducible_and_follows_the_evidence() -> None:
    a = two_clades()
    one = select_parent(a, 2, tip="g2", seed=7)
    again = select_parent(a, 2, tip="g2", seed=7)
    assert one == again and one.reason["kind"] == "thompson"
    picks = {select_parent(a, 2, tip="g2", seed=s).version for s in range(40)}
    assert "G" in picks and "W" not in picks  # the weak clade's posterior never wins here
    assert all("sample" in r and r["sample"] is not None for r in one.reason["candidates"])


def test_selection_payload_and_decision_context_round_trip_through_a_decision() -> None:
    choice = select_parent(two_clades(), 2, tip="g2", seed=3)
    payload = json.loads(json.dumps(choice.to_payload()))
    assert payload["comparison_parent"] == "g2" and payload["source"] == choice.version
    ctx = choice.decision_context()["archive"]
    assert ctx["source"] == choice.version and ctx["comparison_parent"] == "g2"
    doc = decision_doc("g2", "N", True, source=choice.version if choice.version != "g2" else None)
    again = Archive.from_records(
        [hv("S", None, "seed"), hv("G", "S"), hv("g1", "G"), hv("g2", "G")], [], [doc]
    )
    assert again.nodes["N"].parent == "g2"


def test_selector_input_errors() -> None:
    a = tree()
    with pytest.raises(ValueError, match="at least 1"):
        select_parent(a, 0)
    with pytest.raises(ValueError, match="tau"):
        select_parent(a, 1, tau=0)
    with pytest.raises(ValueError, match="no active tip"):
        select_parent(Archive([]), 1)
    with pytest.raises(ValueError, match="no active tip"):
        select_parent(a, 1, tip="missing")


def test_tau_sharpens_the_draw() -> None:
    a = two_clades()
    flat = [select_parent(a, 2, tip="g2", seed=s, tau=1.0).version for s in range(60)]
    sharp = [select_parent(a, 2, tip="g2", seed=s, tau=50.0).version for s in range(60)]
    best = a.clade("G").rate or 0.0
    assert best > (a.clade("S").rate or 1.0)
    assert sharp.count("G") >= flat.count("G")


class FakeStore:
    """Only what `ArchiveStore` names, over fixed documents."""

    def __init__(
        self, archive_docs: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        self.h, self.e, self.d = archive_docs

    def lineage(self, root: str | None = None) -> list[Any]:
        from ratchet.store import LineageNode

        nodes = {h["version"]: LineageNode(h["version"], h["parent"], h["created_by"], "d") for h in self.h}
        roots = []
        for n in nodes.values():
            if n.parent is None:
                roots.append(n)
            else:
                nodes[n.parent].children.append(n)
        return roots

    def evals(self, filter: Any = None) -> list[dict[str, Any]]:
        return self.e

    def decisions(self, filter: Any = None) -> list[dict[str, Any]]:
        return self.d


def test_archive_selector_reloads_and_varies_by_generation() -> None:
    a = two_clades()
    heads = [hv(v, n.parent, n.created_by) for v, n in a.nodes.items() if n.stored]
    evals = [
        e
        for n in a.nodes.values()
        for e in [eval_doc(report(n.version, bools(n.evidence.passes, n.evidence.failures)))]
    ]
    decisions = [
        decision_doc(n.parent or "", v, n.status == "admitted") for v, n in a.nodes.items() if n.verdict
    ]
    sel = ArchiveSelector(FakeStore((heads, evals, decisions)), 2, seed="run-1")
    assert sel.last is None
    first = sel("g2", 1)
    assert sel.last is first and first.reason["seed"] == "run-1:1"
    assert sel("g2", 1) == first
    assert sel("g2", 2).reason["seed"] == "run-1:2"


# -- rebase --------------------------------------------------------------------------------------------------


def g(gid: str, src: str = "def check(call, view, history):\n    return None\n") -> Guard:
    return Guard(gid, gid, src, [])


def harness(guards: Sequence[Guard] = (), prompt: str = "p", allow: Sequence[str] = ("a", "b", "c"),
            procs: Sequence[Procedure] = (), parent: str | None = None) -> Harness:  # fmt: skip
    return Harness.build(parent=parent, system_prompt=prompt, procedures=list(procs), guards=list(guards),
                         tool_allowlist=list(allow), created_by="seed")  # fmt: skip


def test_rebase_carries_the_change_onto_the_active_harness() -> None:
    old = harness([g("g1")])
    proc = Procedure("p1", "check", "when x", ["step"], [])
    child = old.child(
        guards=[*old.guards, g("g2")], procedures=[proc], system_prompt="p\n\nmore", tool_allowlist=["a", "b"]
    )
    active = harness([g("g1"), g("g9")], prompt="p\n\nlater", allow=["a", "b", "c", "d"])
    out = rebase(child, old, active)
    assert [x.id for x in out.guards] == ["g1", "g9", "g2"]
    assert [x.id for x in out.procedures] == ["p1"]
    assert out.system_prompt == "p\n\nlater\n\nmore"
    assert out.tool_allowlist == [
        "a",
        "b",
        "d",
    ]  # the narrowing is applied; the active harness keeps its own d
    assert out.parent == active.version and out.created_by == "improver" and out.verify_digest()


def test_rebase_onto_the_source_itself_reproduces_the_child_content() -> None:
    old = harness([g("g1")])
    child = old.child(guards=[*old.guards, g("g2")])
    again = old.child(guards=[*old.guards, g("g2")], created_by="improver")
    assert rebase(child, old, old).digest == again.digest


def test_rebase_skips_items_the_active_harness_already_has_and_refuses_clashes() -> None:
    old = harness([g("g1")])
    child = old.child(guards=[*old.guards, g("g2")], system_prompt="p\n\nmore")
    active = harness([g("g1"), g("g2")], prompt="p")
    assert [x.id for x in rebase(child, old, active).guards] == ["g1", "g2"]
    clash = harness([g("g1"), g("g2", "def check(call, view, history):\n    return 'x'\n")])
    with pytest.raises(RebaseError, match="different guard with id g2"):
        rebase(child, old, clash)
    already = harness([g("g1"), g("g2")], prompt="p\n\nmore")
    with pytest.raises(RebaseError, match="already present"):
        rebase(child, old, already)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"guards": [Guard("zz", "d", "s", [])]}, "keep its parent's guards"),
        ({"system_prompt": "changed"}, "does not start"),
        ({"tool_allowlist": ["a", "b", "c", "z"]}, "widens"),
    ],
)
def test_rebase_refuses_a_child_that_breaks_the_append_only_rule(
    change: dict[str, Any], message: str
) -> None:
    old = harness([g("g1")])
    with pytest.raises(RebaseError, match=message):
        rebase(old.child(**change), old, old)


def test_rebase_refuses_changed_procedures() -> None:
    p = Procedure("p1", "n", "w", [], [])
    old = harness(procs=[p])
    child = old.child(procedures=[Procedure("p1", "n2", "w", [], [])])
    with pytest.raises(RebaseError, match="procedures unchanged"):
        rebase(child, old, old)


# -- run directories, the command line and Atlas ------------------------------------------------------


def write_run(root: Path) -> Path:
    run = root / "run-1"
    run.mkdir()
    rows = {
        "harnesses.jsonl": [hv("S", None, "seed"), hv("A", "S")],
        "evals.jsonl": [
            {"eval_id": "e1", **report("S", bools(2, 2)).to_payload(), "role": "baseline"},
            {"eval_id": "e2", **report("A", bools(4, 0)).to_payload(), "role": "candidate", "generation": 1},
            {"eval_id": "e3", "type": "evaluator", "version": "v"},
        ],
        "decisions.jsonl": [{"decision_id": "S->A", **decision_doc("S", "A", True)}],
    }
    for name, recs in rows.items():
        (run / name).write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return run


def test_run_directory_loads_and_tolerates_missing_files(tmp_path: Path) -> None:
    a = Archive.from_run_dir(write_run(tmp_path))
    assert a.nodes["A"].status == "admitted" and a.clade("S").rate == pytest.approx(6 / 8)
    (tmp_path / "empty").mkdir()
    assert Archive.from_run_dir(tmp_path / "empty").nodes == {}


def test_payload_is_json_with_the_metric_label(tmp_path: Path) -> None:
    payload = json.loads(json.dumps(Archive.from_run_dir(write_run(tmp_path)).to_payload()))
    assert payload["metric"] == "descendant admission yield" and payload["formula"] == arc.FORMULA
    by = {n["version"]: n for n in payload["nodes"]}
    assert by["S"]["clade"]["descendants"] == 1 and by["S"]["clade"]["admission_yield"] == 1.0
    assert (
        by["A"]["gain"] == pytest.approx(0.5) and by["A"]["eligible"] is True
    )  # not stored? it is: see below
    assert by["A"]["verdict"]["gates"]["fresh"] is True


def test_command_line_writes_the_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = write_run(tmp_path)
    out = tmp_path / "x" / "archive.json"
    assert arc.main([str(run), "--out", str(out), "--min-descendants", "1", "--seed", "4"]) == 0
    data = json.loads(out.read_text())
    assert data["runs"][0]["name"] == "/".join(run.parts[-2:]) and data["runs"][0]["selection"]["source"] in {
        "S",
        "A",
    }
    assert arc.main([str(run), "--split", "train"]) == 0
    assert json.loads(capsys.readouterr().out)["runs"][0]["archive"]["split"] == "train"
    (tmp_path / "empty").mkdir()
    assert arc.main([str(tmp_path / "empty")]) == 0
    assert json.loads(capsys.readouterr().out)["runs"][0]["selection"] is None


@pytest.mark.parametrize("argv", [[], ["--uri", "mongodb://x"], ["--db", "d", "somedir"]])
def test_command_line_refuses_ambiguous_sources(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        arc.main(argv)


LOCAL_URI = "mongodb://ratchet:ratchet-local-dev@127.0.0.1:27018/?directConnection=true"


@pytest.fixture(scope="module")
def atlas_uri() -> str:
    """The local Atlas container (docker/compose.yaml). Fails at once when it is down; never starts it."""
    import os

    uri = os.environ.get("RATCHET_TEST_MONGODB_URI", LOCAL_URI)
    client: MongoClient[dict[str, Any]] = MongoClient(uri, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except Exception as exc:
        pytest.fail(
            f"local Atlas is not reachable ({type(exc).__name__}); "
            "run: docker compose -f docker/compose.yaml up -d --wait"
        )
    finally:
        client.close()
    return uri


@pytest.fixture
def atlas(atlas_uri: str) -> Iterator[Store]:
    name = f"ratchet_test_{uuid.uuid4().hex[:10]}"
    s = Store(atlas_uri, name)
    yield s
    s._client.drop_database(name)
    s.close()


def real_harnesses() -> tuple[Harness, Harness, Harness]:
    seed = harness([g("g1")])
    a = seed.child(guards=[*seed.guards, g("g2")], created_by="improver")
    b = a.child(guards=[*a.guards, g("g3")], created_by="improver")
    return seed, a, b


def test_atlas_lineage_becomes_an_archive(atlas: Store) -> None:
    seed, a, b = real_harnesses()
    rejected = a.child(
        guards=[*a.guards, g("g4")], created_by="improver"
    )  # never stored: only its decision is
    for h in (seed, a, b):
        atlas.put_harness(h)
    p1, ca = report(seed.version, bools(2, 2)), report(a.version, bools(4, 0))
    ctx: dict[str, Any] = {"generation": 1}
    atlas.put_eval({**p1.to_payload(), **ctx, "role": "parent_run_1"}, eval_id="e-p1")
    atlas.put_eval({**ca.to_payload(), **ctx, "role": "candidate"}, eval_id="e-ca")
    atlas.put_eval(
        {**report(a.version, bools(3, 1)).to_payload(), "generation": 2, "role": "parent_run_1"},
        eval_id="e-a2",
    )
    atlas.put_eval(
        {**report(b.version, bools(4, 0)).to_payload(), "generation": 2, "role": "candidate"}, eval_id="e-cb"
    )
    atlas.put_eval(
        {**report(rejected.version, bools(0, 4)).to_payload(), "role": "candidate"}, eval_id="e-cr"
    )
    atlas.put_eval({"type": "evaluator", "version": "x"}, eval_id="evaluator-x")
    docs = [
        decision_doc(seed.version, a.version, True, reports={"parent_run_1": p1, "candidate_run": ca}),
        decision_doc(a.version, b.version, True, source=seed.version),
        decision_doc(a.version, rejected.version, False),
    ]
    for d in docs:
        atlas.put_decision({k: v for k, v in d.items() if k != "_id"}, decision_id=d["_id"])

    arch = Archive.load(atlas)

    assert set(arch.nodes) == {seed.version, a.version, b.version, rejected.version}
    assert [n.status for n in arch.nodes.values() if n.stored] == ["seed", "admitted", "admitted"]
    assert arch.nodes[rejected.version].status == "rejected" and not arch.nodes[rejected.version].stored
    assert arch.tip == b.version
    # B was drawn from the seed but compared against A: it is credited to the seed's clade
    assert arch.nodes[b.version].source == seed.version and arch.nodes[b.version].parent == a.version
    clade = arch.clade(seed.version)
    assert (clade.descendants, clade.admitted_descendants, clade.rejected_descendants) == (3, 2, 1)
    assert (clade.passes, clade.failures) == (2 + 4 + 3 + 4 + 0, 2 + 0 + 1 + 0 + 4)
    named = [e.named_by for e in arch.nodes[a.version].evidence.evals]
    assert (f"{seed.version}->{a.version}:candidate_run",) in named
    choice = ArchiveSelector(atlas, 3, seed=1)(b.version, 1)
    assert choice.version == seed.version and choice.rebase_required and choice.tip == b.version
    json.dumps(arch.to_payload())


def test_atlas_command_line_reads_the_same_archive(atlas: Store, atlas_uri: str, tmp_path: Path) -> None:
    seed, a, _ = real_harnesses()
    atlas.put_harness(seed)
    atlas.put_harness(a)
    atlas.put_eval({**report(seed.version, bools(1, 1)).to_payload(), "role": "baseline"}, eval_id="e1")
    out = tmp_path / "atlas.json"
    assert (
        arc.main(["--uri", atlas_uri, "--db", atlas.db.name, "--out", str(out), "--min-descendants", "1"])
        == 0
    )
    run = json.loads(out.read_text())["runs"][0]
    assert {n["version"] for n in run["archive"]["nodes"]} == {seed.version, a.version}
    assert run["selection"]["comparison_parent"] == seed.version  # A has no decision: not a parent


@pytest.mark.parametrize("name", ["heldout", "fresh"])
def test_the_verdict_reads_the_held_out_gate_under_either_name(name: str) -> None:
    doc = {
        "parent": "S",
        "candidate": "A",
        "admitted": True,
        "checks": [
            {
                "name": name,
                "passed": True,
                "evidence": {"stable_recoveries": ["a", "b"], "parent_discordance": 0},
            }
        ],
    }
    v = _verdict(doc, 1)
    assert v is not None and v.stable_recoveries == 2 and v.parent_discordance == 0
