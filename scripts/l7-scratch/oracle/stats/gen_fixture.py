"""Build the D37 oracle fixture from r8_stats.py's own test functions (run on the box, numpy and scipy)."""
import json, math, sys
import numpy as np
from scipy.stats import t as tdist
sys.path.insert(0, __import__("os").path.dirname(__file__))
import r8_stats as r8

K, N, C, ENDPOINTS = 8, 5, 6, 5
rng = np.random.default_rng(20261017)
base = np.array([0.2, 0.35, 0.5, 0.65, 0.8, 0.5])
cohorts = []
for idx in range(24):
    effect = [0.0, 0.0, 0.25, 0.4][idx % 4]
    shape = (1, K, C, N)
    xs = []
    for a in range(4):
        p = np.clip(base + (effect if a == 3 else 0.0), 0, 1)
        xs.append(rng.random(shape) < p[None, None, :, None])
    # distinct dispatch times per session; the candidate runs later in odd cohorts to move T
    times = []
    for a in range(4):
        t = rng.random(shape) + (0.15 if (a == 3 and idx % 3 == 1) else 0.0)
        times.append(np.round(t, 9))
    eps = []
    for a in range(4):
        if a == 3 and idx % 5 == 4:
            eps.append(np.where(rng.random(shape) < 0.6, 0, rng.integers(0, ENDPOINTS, shape)))
        else:
            eps.append(rng.integers(0, ENDPOINTS, shape))
    counts = np.stack([x.sum(-1) for x in xs])
    p_e = float(r8.check_e(eps, 1)[0])
    p_t = float(r8.check_t(times, 1, K)[0])
    p_tr = float(r8.check_trend(times, xs, N)[0])
    z_p = float(r8.check_p(xs, counts, N)[0])
    par, i1, i2, cand = counts
    harm = bool(r8.harm_ok(cand.sum(1), (par + i1 + i2).sum(1), K * N, 3 * K * N)[0])
    crit = tdist.ppf(0.95, K - 1)
    ts, degenerate = [], False
    for ref in (par, i1, i2):
        d = ((cand - ref) / N).sum(-1)[0]
        sd = d.std(ddof=1)
        degenerate |= bool(sd == 0)
        ts.append(float(d.mean() / sd * math.sqrt(K)) if sd > 0 else None)
    ok_logs = (p_e >= r8.A_E) and ((p_t >= r8.A_T) or (p_tr >= r8.A_TR)) and (z_p < r8.Z_P)
    finite = [t for t in ts if t is not None]
    iut = (not degenerate) and min(finite) >= crit and harm
    cohorts.append({
        "passed": [x[0].astype(int).tolist() for x in xs],
        "times": [t[0].tolist() for t in times],
        "endpoints": [e[0].astype(int).tolist() for e in eps],
        "expect": {"p_endpoint": p_e, "p_time": p_t, "p_trend": p_tr, "z_pairing": z_p, "harm_ok": harm,
                   "t": ts, "degenerate": degenerate, "admitted": bool(iut and ok_logs), "logs_ok": bool(ok_logs)},
    })
out = {"source": "r8_stats.py sha256 16421bfdf4316ce946b5e49c221cb421fdbbea18976e628219aebe2ac572d1d0",
       "numpy": np.__version__, "scipy": __import__("scipy").__version__, "k": K, "n": N, "cases": C,
       "t_critical": float(crit), "z_pairing": float(r8.Z_P), "cohorts": cohorts}
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
print("cohorts", len(cohorts), "admitted", sum(c["expect"]["admitted"] for c in cohorts),
      "logs_fire", sum(not c["expect"]["logs_ok"] for c in cohorts), "harm_fail", sum(not c["expect"]["harm_ok"] for c in cohorts))
