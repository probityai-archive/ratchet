"""Exact null failure rates of the v3 aggregate check by cohort size (docs/CONTRACT.md, minimum cohort size).

An unchanged arm and the parent each run 20 executions per case per panel. With every case at pass rate p,
a panel's aggregate decline exceeds 5 points when the summed count difference exceeds n; the pooled one
when the two panels' sums exceed 2n. Prints, per n, the probability that one panel fails and that at least
one of the three aggregate checks fails. Exact convolution; no simulation.
"""

from __future__ import annotations

from math import comb


def _difference(p: float, m: int = 20) -> dict[int, float]:
    b = [comb(m, k) * p**k * (1 - p) ** (m - k) for k in range(m + 1)]
    out: dict[int, float] = {}
    for x in range(m + 1):
        for y in range(m + 1):
            out[x - y] = out.get(x - y, 0.0) + b[x] * b[y]
    return out


def _sum(n: int, p: float) -> dict[int, float]:
    one, total = _difference(p), {0: 1.0}
    for _ in range(n):
        nxt: dict[int, float] = {}
        for i, u in total.items():
            for j, v in one.items():
                nxt[i + j] = nxt.get(i + j, 0.0) + u * v
        total = nxt
    return total


def rates(n: int, p: float = 0.5) -> tuple[float, float]:
    s = _sum(n, p)
    panel = sum(u for a, u in s.items() if a > n)
    ok = sum(u * v for a, u in s.items() for b, v in s.items() if a <= n and b <= n and a + b <= 2 * n)
    return panel, 1.0 - ok


if __name__ == "__main__":
    for n in (1, 2, 4, 8, 19, 40):
        panel, any_check = rates(n)
        print(f"n={n:2d}  one panel fails {panel:.4f}  any aggregate check fails {any_check:.4f}")
