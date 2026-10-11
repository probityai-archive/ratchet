from __future__ import annotations

from html.parser import HTMLParser

from ratchet.types import Guard
from ratchet.view.data import build_lineage, decision_info
from ratchet.view.matrix import MatrixColumn, compute_matrix, letters
from ratchet.view.report import ReportData, lineage_html, matrix_html, render_html, results_html

from .fakes import RULES, FakeRunner, decision_doc, info, sample_guards, sample_traces, scenario, walks


class Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)


def matrix(guards: list[Guard] | None = None, decided: bool = True):  # type: ignore[no-untyped-def]
    store, _, _ = scenario()
    gs = guards or sample_guards()
    cols = [
        MatrixColumn(
            g,
            f"c{i}",
            letters(len(gs))[i],
            decision_info(store.decision_docs[min(i, 2)]) if decided else None,
        )
        for i, g in enumerate(gs)
    ]
    return compute_matrix("parent000001", cols, sample_traces(), info, walks(), FakeRunner(RULES))


def test_page_has_every_section_and_is_static() -> None:
    store, seed, good = scenario()
    html = render_html(ReportData("Ratchet demo", build_lineage(store), [matrix()], "2026-10-11 12:00 UTC"))
    assert html.startswith("<!doctype html>")
    p = Tags()
    p.feed(html)
    assert "script" not in p.tags
    assert "link" not in p.tags
    assert "<title>Ratchet demo</title>" in html
    for part in ("Replay matrix: parent parent000001", "Lineage", "Results", seed.version, good.version):
        assert part in html
    assert "0 model calls" in html
    assert "ADMITTED" in html
    assert "REJECTED" in html
    assert "11 traces," in html
    assert "2026-10-11 12:00 UTC" in html


def test_page_without_matrix_or_versions_still_renders() -> None:
    html = render_html(ReportData("t", [], [], "now"))
    assert "Replay matrix" not in html
    assert "<h2>Lineage</h2>" in html


def test_everything_from_the_store_is_escaped() -> None:
    evil = Guard("g-<script>alert(1)</script>", "<img src=x onerror=alert(2)>", "src", ["r-t1@5"])
    store, seed, _ = scenario()
    store.decision_docs.append(
        decision_doc(seed.version, "evil0000001", False, change={"description": "<b>bold</b>"})
    )
    m = matrix([evil], decided=False)
    html = render_html(ReportData("<i>t</i>", build_lineage(store), [m], "<now>"))
    for raw in ("<script>alert", "<img src=x", "<b>bold</b>", "<i>t</i>", "<now>"):
        assert raw not in html
    assert "&lt;script&gt;alert(1)" in html
    assert "&lt;b&gt;bold&lt;/b&gt;" in html


def test_matrix_cells_cover_hit_block_pass_missed_late_and_error() -> None:
    miss = Guard("g-miss", "d", "s", ["r-t1@5"])
    late = Guard("g-retry", "d", "s", ["r-t1@3"])
    bad = Guard("g-broken", "d", "s", [])
    html = matrix_html(matrix([sample_guards()[1], miss, late, bad], decided=False))
    for cls in ("hit", "block", "pass", "miss", "err", "na"):
        assert f'class="c {cls}"' in html or f' {cls}"' in html
    assert "late @5" in html
    assert "REPLAY OK" in html or "REJECTED" in html


def test_lineage_and_results_tables() -> None:
    store, _, _ = scenario()
    roots = build_lineage(store)
    tree = lineage_html(roots)
    assert tree.count("<li>") == 4
    assert "held-out" in tree
    assert "rcpt:" in tree
    table = results_html(roots)
    assert table.count("<tr>") == 5
    assert "rejected" in table
    assert "promoted" in table
    assert "seed" in table
