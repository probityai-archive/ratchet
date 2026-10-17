"""The distribution functions the k-session test needs, in pure Python, so a bundle checks offline anywhere.

Normal tail from `math.erfc`; chi-square tail from the regularized upper incomplete gamma function; Student
t from the regularized incomplete beta function, both by the series and continued fractions of Numerical
Recipes (Press et al., 3rd ed., sections 6.2 and 6.4), evaluated to about 1e-14 relative. Quantiles are found
by bisection on the distribution function, which is monotone, to full double precision.
"""

from __future__ import annotations

import math
from collections.abc import Callable

_EPS = 1e-16
_TINY = 1e-300
_ITMAX = 10_000


def norm_sf(z: float) -> float:
    """P(Z > z) for a standard normal Z."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_isf(p: float) -> float:
    """The z with P(Z > z) = p, for 0 < p < 1."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return _bisect(lambda z: -norm_sf(z), -p, -40.0, 40.0)


def _gamma_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by its series, for x < a + 1."""
    term = total = 1.0 / a
    ap = a
    for _ in range(_ITMAX):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) by its continued fraction (modified Lentz), x >= a + 1."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = _TINY if abs(d) < _TINY else d
        c = b + an / c
        c = _TINY if abs(c) < _TINY else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, df: int) -> float:
    """P(X > x) for X chi-square with `df` degrees of freedom."""
    if df < 1:
        raise ValueError("df must be at least 1")
    if x <= 0.0:
        return 1.0
    a, y = df / 2.0, x / 2.0
    return 1.0 - _gamma_series(a, y) if y < a + 1.0 else _gamma_cf(a, y)


def _beta_cf(a: float, b: float, x: float) -> float:
    """The continued fraction for the regularized incomplete beta function (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = _TINY if abs(d) < _TINY else d
    d = 1.0 / d
    h = d
    for m in range(1, _ITMAX):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = _TINY if abs(d) < _TINY else d
        c = 1.0 + aa / c
        c = _TINY if abs(c) < _TINY else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = _TINY if abs(d) < _TINY else d
        c = 1.0 + aa / c
        c = _TINY if abs(c) < _TINY else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def t_sf(t: float, df: int) -> float:
    """P(T > t) for Student's t with `df` degrees of freedom."""
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def t_isf(p: float, df: int) -> float:
    """The t with P(T > t) = p, for 0 < p < 1."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return _bisect(lambda t: -t_sf(t, df), -p, -1e6, 1e6)


def _bisect(f: Callable[[float], float], target: float, lo: float, hi: float) -> float:
    """The x in [lo, hi] where the increasing function f reaches target, to the last representable bit."""
    for _ in range(2_000):
        mid = (lo + hi) / 2.0
        if mid in (lo, hi):
            break
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
