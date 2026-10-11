"""Measurements behind docs/design/sequential-v3.md, from the 3,800 executions of ratchet_v3_full.

Run: python docs/design/sequential-v3/analysis.py > docs/design/sequential-v3/analysis.txt
Input: v3_full_executions.csv beside this file (exported from ratchet_v3_full.evals, type v3_execution).
Everything is exact binomial arithmetic or a replay of recorded outcomes; nothing is simulated.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

N = 40
LOOKS = (10, 20, 30, 40)
FIXED_LIMIT = 8
AGGREGATE_LIMIT_PP = 5.0
ALPHA_PER_CONTROL = 0.05
SPEND_RHO = 2.0
NULL_ARMS = ("parent", "identity_1", "identity_2")
COMPARED = ("identity_1", "identity_2", "candidate:ad50dac8194c", "candidate:b972467a3b9a")

Outcomes = dict[tuple[str, str], dict[int, int]]


def load() -> tuple[Outcomes, dict[tuple[str, int], int], list[str]]:
    outcomes: Outcomes = defaultdict(dict)
    panel: dict[tuple[str, int], int] = {}
    with (Path(__file__).parent / "v3_full_executions.csv").open() as f:
        for r in csv.DictReader(f):
            outcomes[(r["arm"], r["case_id"])][int(r["rep"])] = int(r["passed"])
            panel[(r["case_id"], int(r["rep"]))] = int(r["panel"])
    cases = sorted({c for _, c in outcomes})
    return outcomes, panel, cases


def pmf(n: int, p: float) -> list[float]:
    return [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]


def dist_diff(n: int, p: float, q: float) -> dict[int, float]:
    """Distribution of Bin(n, p) - Bin(n, q), independent arms."""
    a, b = pmf(n, p), pmf(n, q)
    out: dict[int, float] = defaultdict(float)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i - j] += x * y
    return out


def p_decline_over(limit: int, p: float, q: float | None = None) -> float:
    return sum(v for d, v in dist_diff(N, p, p if q is None else q).items() if d > limit)


# --- 1. identity-control spread and the decline limit -------------------------------------------


def spread(o: Outcomes, cases: list[str]) -> dict[str, int]:
    s = {k: sum(v.values()) for k, v in o.items()}
    print("1. IDENTITY-CONTROL SPREAD (three runs of the same harness, 40 executions each)")
    print(f"{'case':24s}  P  I1  I2   pooled   P-I1 P-I2 I1-I2")
    zs: list[float] = []
    for c in cases:
        k = [s[(a, c)] for a in NULL_ARMS]
        p = sum(k) / (3 * N)
        sd = math.sqrt(2 * N * p * (1 - p))
        ds = [k[0] - k[1], k[0] - k[2], k[1] - k[2]]
        zs += [d / sd for d in ds] if sd else []
        print(f"{c:24s} {k[0]:2d} {k[1]:3d} {k[2]:3d}   {p:.3f}   {ds[0]:+3d}  {ds[1]:+3d}  {ds[2]:+3d}")
    print(f"standardized null differences: n={len(zs)}, variance {sum(z * z for z in zs) / len(zs):.3f}")
    print("  (1.0 means binomial spread; the identity runs are neither over- nor under-dispersed)")
    corr = []
    for c in cases:
        for a, b in (("parent", "identity_1"), ("parent", "identity_2"), ("identity_1", "identity_2")):
            x, y = [o[(a, c)][i] for i in range(N)], [o[(b, c)][i] for i in range(N)]
            mx, my = sum(x) / N, sum(y) / N
            if 0 < mx < 1 and 0 < my < 1:
                cov = sum((u - mx) * (v - my) for u, v in zip(x, y, strict=True)) / N
                corr.append(cov / math.sqrt(mx * (1 - mx) * my * (1 - my)))
    print(f"same-seed correlation between identical arms: mean {sum(corr) / len(corr):+.3f}, n={len(corr)}")
    pooled = {c: sum(s[(a, c)] for a in NULL_ARMS) / (3 * N) for c in cases}
    per_case = [p_decline_over(FIXED_LIMIT, pooled[c]) for c in cases]
    fw = 1 - math.prod(1 - v for v in per_case)
    print(f"\nfixed limit {FIXED_LIMIT}: P(an unchanged identity arm declines > {FIXED_LIMIT} on some case)")
    print(f"  one control {fw:.3f}; either of two controls {1 - (1 - fw) ** 2:.3f}")
    smooth = {c: (sum(s[(a, c)] for a in NULL_ARMS) + 0.5) / (3 * N + 1) for c in cases}
    limits: dict[str, int] = {}
    for c in cases:
        lim = 0
        while p_decline_over(lim, smooth[c]) > ALPHA_PER_CONTROL / len(cases):
            lim += 1
        limits[c] = lim
    fw2 = 1 - math.prod(1 - p_decline_over(limits[c], smooth[c]) for c in cases)
    print(f"\ncalibrated limit: least L with P(decline > L | unchanged) <= {ALPHA_PER_CONTROL}/{len(cases)}")
    print("  (Jeffreys-smoothed pooled rate (k + 0.5) / 121, so 0/120 and 120/120 still get a spread)")
    for c in cases:
        worst = max(s[("parent", c)] - s[(a, c)] for a in NULL_ARMS[1:])
        print(f"  {c:24s} rate {smooth[c]:.3f}  limit {limits[c]:2d}  largest identity decline {worst:+d}")
    print(f"  one control {fw2:.3f}; either of two controls {1 - (1 - fw2) ** 2:.3f}")
    for a in COMPARED:
        over8 = [c for c in cases if s[("parent", c)] - s[(a, c)] > FIXED_LIMIT]
        over = [c for c in cases if s[("parent", c)] - s[(a, c)] > limits[c]]
        print(f"  {a:24s} over 8: {over8 or '-'}   over calibrated: {over or '-'}")
    return limits


# --- 2. sequential boundaries with the error budget spent across looks -------------------------


def bad_region(limit: int) -> list[tuple[float, float]]:
    """True rates where the candidate is worse than the tolerance: p - q >= limit / 40."""
    d = limit / N
    out = []
    for i in range(0, 101):
        p = i / 100
        j = 0
        while (q := round(p - d - j * 0.025, 4)) >= 0:
            out.append((p, q))
            j += 1
    return out


def sequential(p: float, q: float, bounds: list[int], limit: int) -> tuple[float, float, float]:
    """Exact P(case passes), expected executions per arm, and E[stop-time decline rate].
    Pass early at look k when D_k <= bounds[k]; fail early when D_k - (40 - n_k) > limit (certain)."""
    inc = dist_diff(LOOKS[0], p, q)
    alive = {0: 1.0}
    passed = ess = mean = 0.0
    prev = 0
    for k, n in enumerate(LOOKS):
        new: dict[int, float] = defaultdict(float)
        for d, w in alive.items():
            for e, v in inc.items():
                new[d + e] += w * v
        ess += (n - prev) * sum(alive.values())
        prev = n
        alive = {}
        last = k == len(LOOKS) - 1
        for d, w in new.items():
            if d <= bounds[k]:
                passed += w
                mean += w * d / n
            elif last or d - (N - n) > limit:
                mean += w * d / n
            else:
                alive[d] = w
    return passed, ess, mean


def design(limit: int) -> list[int]:
    """Largest pass boundaries such that, at every point of the bad region, the cumulative probability of
    passing by look k is at most (n_k / 40) ** rho times the fixed rule's probability of passing there."""
    bad = bad_region(limit)
    fixed = {pq: 1 - p_decline_over(limit, *pq) for pq in bad}
    bounds: list[int] = []
    for k, n in enumerate(LOOKS):
        lo = -n - 1
        best = lo
        top = limit if k == len(LOOKS) - 1 else limit - 1
        budget = (n / N) ** SPEND_RHO
        for b in range(top, lo, -1):  # pass probability grows with b, so the first feasible b is the largest
            trial = [*bounds, b] + [lo] * (len(LOOKS) - k - 1)
            if all(sequential(p, q, trial, limit)[0] <= budget * fixed[(p, q)] + 1e-12 for p, q in bad):
                best = b
                break
        bounds.append(best)
    return bounds


def check_design(limit: int, bounds: list[int]) -> tuple[float, float]:
    """Largest excess of sequential over fixed pass probability on a fine bad-region grid (must be <= 0),
    and the worst favourable bias of the stop-time decline rate anywhere."""
    excess = 0.0
    for i in range(0, 101):
        p = i / 100
        j = 0
        while (q := p - limit / N - j / 100) >= -1e-9:
            q = max(q, 0.0)
            excess = max(excess, sequential(p, q, bounds, limit)[0] - (1 - p_decline_over(limit, p, q)))
            j += 1
    bias = 0.0
    for i in range(0, 101, 2):
        p = i / 100
        for j in range(0, 41):
            q = p - j / 100
            if q < 0:
                break
            bias = min(bias, sequential(p, q, bounds, limit)[2] - (p - q))
    return excess, bias


# --- 3. replay of the recorded cohort --------------------------------------------------------------


def replay(
    o: Outcomes,
    panel: dict[tuple[str, int], int],
    cases: list[str],
    limit_of: Callable[[str], int],
    bounds_of: Callable[[int], list[int]],
    bias_pp: float,
) -> None:
    def first(arm: str, case: str, n: int) -> int:
        return sum(o[(arm, case)][i] for i in range(n))

    stops: dict[tuple[str, str], tuple[int, str]] = {}
    for a in COMPARED:
        for c in cases:
            lim, b = limit_of(c), bounds_of(limit_of(c))
            for k, n in enumerate(LOOKS):
                sp, sa = first("parent", c, n), first(a, c, n)
                d = sp - sa
                recovery_open = sp <= 10 and n - sa <= 10
                if d - (N - n) > lim:
                    stops[(a, c)] = (n, "fail")
                    break
                if n == N:
                    stops[(a, c)] = (n, "pass" if d <= b[-1] else "fail")
                    break
                if d <= b[k] and not recovery_open:
                    stops[(a, c)] = (n, "pass")
                    break
    fixed = {
        (a, c): "pass" if first("parent", c, N) - first(a, c, N) <= limit_of(c) else "fail"
        for a in COMPARED
        for c in cases
    }
    parent_used = sum(max(stops[(a, c)][0] for a in COMPARED) for c in cases)
    used = {a: sum(stops[(a, c)][0] for c in cases) for a in COMPARED}
    total = parent_used + sum(used.values())
    print(f"  executions: parent {parent_used}/760, " + ", ".join(f"{a} {u}/760" for a, u in used.items()))
    saved = 3800 - total
    print(f"  total {total} of 3800 ({100 * total / 3800:.1f}%), saving {saved} ({100 * saved / 3800:.1f}%)")
    for a in COMPARED:
        looks = Counter(stops[(a, c)][0] for c in cases)
        fails = sorted(c for c in cases if stops[(a, c)][1] == "fail")
        print(f"  {a:24s} settled at looks {dict(sorted(looks.items()))}  failing: {fails or '-'}")
    disagree = [k for k in stops if stops[k][1] != fixed[k]]
    print(f"  per-case verdicts that differ from the fixed rule with the same limits: {disagree or 'none'}")
    first_fail = {
        a: min((stops[(a, c)][0] for c in cases if stops[(a, c)][1] == "fail"), default=None)
        for a in COMPARED
    }
    print(f"  look at which each arm first has a certain case failure: {first_fail}")
    limit_pp = AGGREGATE_LIMIT_PP + bias_pp
    print(
        f"  aggregate, stop-time rates, limit {AGGREGATE_LIMIT_PP:.2f} - {-bias_pp:.2f} = {limit_pp:.2f} pp:"
    )
    for a in COMPARED:
        parts = []
        ok_seq = ok_fix = True
        for name, keep in (("pooled", None), ("panel 0", 0), ("panel 1", 1)):
            ds, df = [], []
            for c in cases:
                n = stops[(a, c)][0]
                run = [i for i in range(n) if keep is None or panel[(c, i)] == keep]
                full = [i for i in range(N) if keep is None or panel[(c, i)] == keep]
                if run:
                    ds.append(sum(o[("parent", c)][i] - o[(a, c)][i] for i in run) / len(run))
                df.append(sum(o[("parent", c)][i] - o[(a, c)][i] for i in full) / len(full))
            seq_pp, fix_pp = 100 * sum(ds) / len(ds), 100 * sum(df) / len(df)
            ok_seq &= seq_pp <= limit_pp
            ok_fix &= fix_pp <= AGGREGATE_LIMIT_PP
            parts.append(f"{name} {seq_pp:+.2f} (fixed {fix_pp:+.2f})")
        print(f"    {a:24s} " + "  ".join(parts) + f"  pass: sequential {ok_seq}, fixed {ok_fix}")


def main() -> None:
    o, panel, cases = load()
    limits = spread(o, cases)
    print("\n2. SEQUENTIAL DECLINE TEST, looks after 10/20/30/40 executions per arm, spending (n/40)^2")
    cache: dict[int, list[int]] = {}
    bias: dict[int, float] = {}
    for lim in sorted({FIXED_LIMIT, *limits.values()}):
        cache[lim] = design(lim)
        excess, bias[lim] = check_design(lim, cache[lim])
        eq = {p: round(sequential(p, p, cache[lim], lim)[1], 1) for p in (0.0, 0.05, 0.3, 0.5, 0.7, 1.0)}
        print(
            f"  limit {lim:2d}: pass boundaries {cache[lim]}; max excess P(pass) over fixed, bad region "
            f"{excess:+.1e}; worst stop-time bias {100 * bias[lim]:+.2f} pp"
        )
        print(f"            expected executions per arm when the arms are equal, by pass rate: {eq}")
    # the aggregate limit is reduced by the worst stop-time bias of any limit in use
    print("\n3. REPLAY OF ratchet_v3_full, fixed limit 8 (the adopted v3 tolerances)")
    replay(o, panel, cases, lambda c: FIXED_LIMIT, cache.__getitem__, 100 * bias[FIXED_LIMIT])
    print("\n4. REPLAY OF ratchet_v3_full, calibrated per-case limits")
    used = min(bias[limits[c]] for c in cases)
    replay(o, panel, cases, limits.__getitem__, cache.__getitem__, 100 * used)


if __name__ == "__main__":
    main()
