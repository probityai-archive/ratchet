from __future__ import annotations

import io

from rich.console import Console

from ratchet.view.data import VersionNode, VersionStats, build_lineage
from ratchet.view.lineage import lineage_tree, score, show_lineage, summary

from .fakes import FakeStore, scenario


def render(obj: object) -> str:
    c = Console(width=140, record=True, file=io.StringIO())
    c.print(obj)
    return c.export_text()


def test_tree_shows_scores_receipts_and_rejections() -> None:
    store, seed, good = scenario()
    out = render(lineage_tree(build_lineage(store)))
    assert seed.version in out
    assert "held-out 3/6" in out
    assert "held-out 3->5/6 (+2)" in out
    assert "false interventions 2" in out
    assert "rcpt:" in out
    assert "block a retried refund" in out
    assert "rej00000001" in out
    assert "held-out fresh passes 3, needs 5" in out
    assert "intervenes on 1 held-out benign twin" in out
    assert out.index(good.version) < out.index("rej00000001")  # the winner is listed before the losers


def test_summary_counts() -> None:
    store, _, _ = scenario()
    assert "4 versions" in str(summary(build_lineage(store)))
    assert "2 rejected" in str(summary(build_lineage(store)))


def test_score_without_stats_and_with_regression() -> None:
    assert str(score(None)) == "not evaluated"
    worse = score(
        VersionStats(heldout_passed=1, heldout_total=6), VersionStats(heldout_passed=3, heldout_total=6)
    )
    assert "(-2)" in str(worse)
    flat = score(
        VersionStats(heldout_passed=3, heldout_total=6), VersionStats(heldout_passed=3, heldout_total=6)
    )
    assert "(+0)" in str(flat)


def test_show_lineage_handles_empty_and_populated() -> None:
    buf = io.StringIO()
    c = Console(file=buf, width=120)
    show_lineage(c, build_lineage(FakeStore()))
    assert "no harness versions" in buf.getvalue()
    store, _, _ = scenario()
    show_lineage(c, build_lineage(store))
    assert "harness lineage" in buf.getvalue()


def test_rejected_node_without_reasons_still_prints() -> None:
    from ratchet.view.data import decision_info

    from .fakes import decision_doc

    d = decision_info(decision_doc("p", "c", False, checks=[]))
    node = VersionNode("c", "p", "improver", "rejected", stored=False, decision=d)
    assert "rejected" in render(lineage_tree([node]))
