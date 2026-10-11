"""The version tree: every harness version, how it scored, and why each child was kept or thrown away."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from rich.console import Console, RenderableType
from rich.text import Text
from rich.tree import Tree

from ratchet.view.data import VersionNode, VersionStats

STATUS_MARK = {
    "seed": ("◆", "bold cyan"),
    "promoted": ("✓", "bold green"),
    "admitted": ("✓", "green"),
    "rejected": ("✗", "bold red"),
}


def score(stats: VersionStats | None, parent: VersionStats | None = None) -> Text:
    """Held-out pass count (with the change from the parent), training count, false interventions."""
    if stats is None:
        return Text("not evaluated", style="dim")
    held = f"{stats.heldout_passed}/{stats.heldout_total}"
    if parent is not None:
        d = stats.heldout_passed - parent.heldout_passed
        held = f"{parent.heldout_passed}->{stats.heldout_passed}/{stats.heldout_total}"
        delta = Text(f" ({d:+d})", style="bold green" if d > 0 else ("bold red" if d < 0 else "dim"))
    else:
        delta = Text()
    fi_style = "green" if stats.false_interventions == 0 else "bold red"
    return Text.assemble(
        ("held-out ", "dim"), (held, "bold"), delta,
        ("  train ", "dim"), (f"{stats.train_passed}/{stats.train_total}", ""),
        ("  false interventions ", "dim"), (str(stats.false_interventions), fi_style),
        ("  ", ""), (stats.mode, "magenta"),
    )  # fmt: skip


def node_label(n: VersionNode, parent: VersionNode | None = None) -> Text:
    mark, style = STATUS_MARK[n.status]
    label = Text.assemble((f"{mark} ", style), (n.version, "bold cyan" if n.status != "rejected" else "red"))
    if n.active:
        label.append(f" ACTIVE rev {n.active_revision} ", style="bold black on bright_green")
    d = n.decision
    if n.status == "seed":
        label.append("  seed", style="dim")
    elif d is not None:
        label.append(f"  {d.what[:48]}", style="white")
    label.append("  ")
    label.append_text(score(n.stats, parent.stats if parent else None))
    if d is not None and d.receipt:
        label.append(f"  {d.receipt}", style="magenta")
    if n.status == "rejected" and d is not None:
        label.append("\n    " + ("; ".join(d.reasons()) or "rejected"), style="red")
    return label


def lineage_tree(roots: Sequence[VersionNode]) -> RenderableType:
    """A rich tree, one branch per root. Rejected children sit beside the promoted one that beat them."""
    tree = Tree(Text("harness lineage", style="bold bright_white"), guide_style="dim")

    def add(branch: Tree, n: VersionNode, parent: VersionNode | None) -> None:
        node = branch.add(node_label(n, parent))
        ordered = sorted(n.children, key=lambda c: c.status == "rejected")
        for c in ordered:
            add(node, c, n)

    for r in roots:
        add(tree, r, None)
    return tree


def summary(roots: Sequence[VersionNode]) -> Text:
    counts = Counter(n.status for r in roots for n in r.walk())
    total = sum(counts.values())
    return Text.assemble(
        (f"{total} versions", "bold"),
        (f"  {counts['promoted']} promoted", "green"),
        (f"  {counts['admitted']} admitted", "green"),
        (f"  {counts['rejected']} rejected", "red"),
    )


def show_lineage(console: Console, roots: Sequence[VersionNode]) -> None:
    if not roots:
        console.print("[yellow]no harness versions recorded yet[/]")
        return
    console.print(lineage_tree(roots))
    console.print(summary(roots))
