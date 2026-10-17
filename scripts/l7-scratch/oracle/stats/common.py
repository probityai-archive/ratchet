"""Shared loaders for the STATS review recomputations. Reads only the captured _raw copies."""

import collections
import json
import re
from pathlib import Path

# The dated research directory, two levels above this file (scripts/stats/), so the loaders work in any
# checkout or copy of the directory, not only in one path on one machine.
ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "_raw"
REC = RAW / "ratchet-runrecords"
RES = RAW / "ratchet-results"
N = 40
NULL_ARMS = ["parent", "identity_1", "identity_2"]
CANDS = ["candidate:ad50dac8194c", "candidate:b972467a3b9a"]


def v3_rows():
    return json.loads((REC / "v3_exec.json").read_text())


def v3_counts():
    """counts[(arm, case)] = passes of 40; panel_counts[(arm, case, panel)] = passes of 20."""
    rows = v3_rows()
    s = collections.defaultdict(int)
    sp = collections.defaultdict(int)
    for r in rows:
        s[(r["arm"], r["case_id"])] += int(r["passed"])
        sp[(r["arm"], r["case_id"], r["panel"])] += int(r["passed"])
    cases = sorted({r["case_id"] for r in rows})
    return s, sp, cases


def pooled_null(s, cases):
    return {c: sum(s[(a, c)] for a in NULL_ARMS) for c in cases}


def tau2_tables():
    text = (RES / "tau2.md").read_text()
    out = {}
    for block in re.split(r"\n### ", text)[1:]:
        head = block.splitlines()[0]
        m = re.match(r"(\w+), harness `\w+`, run `([\w-]+)`", head)
        if not m or m.group(2) != "oss-full-4x":
            continue
        rows = re.findall(
            r"^\| `(.+?)` \| (train|heldout) \| (\d+) / (\d+) \|$", block, re.MULTILINE
        )
        out[m.group(1)] = [(t, s, int(c), int(n)) for t, s, c, n in rows]
    return out


def f_two_run(p):
    """P(case survives the strict two-run no-regression check | no-op, iid Bernoulli(p))."""
    return 1.0 - (1.0 - (1.0 - p) ** 2) * (1.0 - p**2)
