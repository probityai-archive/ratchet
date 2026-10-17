"""Round-8 residuals of the interleaved design (ledger D34) measured, and the gaps closed where first principles
allow (ledger D37).

Round 8 found that a no-effect candidate passes D34 while the unchanged arms agree, whenever the candidate's own
bytes change how it is served or how it varies: candidate-only dispersion, a candidate-only session effect, a
candidate whose executions run later under a serving trend, a candidate routed to one endpoint, a candidate that
reacts more strongly than the copies to shared serving, and persistent arm-level states; plus three unchanged-arm
rows D34 did not print, shared seeds on deterministic serving, and suites too small for the agreement check.
This script reproduces every one of them on the model and rates of `r7_stats.py`, then measures four changes,
each derived from what can and cannot carry the information:

  log checks  Computable from the execution log the gate already writes (endpoint and completion time of every
              execution, rep index per arm). Each makes the cohort inconclusive; constants below.
              E  endpoint mix: Pearson chi-square test of homogeneity on the 2 x E table (the candidate's
                 executions per endpoint; the three unchanged arms' executions per endpoint, summed), every
                 session summed into one table, endpoints with no execution dropped, df = E' - 1;
                 inconclusive when p < A_E.
              T  completion time: in each session, the two-sided Mann-Whitney z of the candidate's completion
                 times against the three unchanged arms' (normal approximation, tie-corrected, no continuity
                 correction); the session z combined as sum / sqrt(sessions); inconclusive when the two-sided
                 p < A_T.
              P  pairing: for each pair of unchanged arms, the excess of (case, rep) slots where both pass over
                 the hypergeometric expectation given each arm's per-case count, summed over cases and sessions,
                 over the square root of the summed hypergeometric variance; inconclusive when the largest of
                 the three reaches Z_P = z(1 - A_P / 3).
              T* completion time, conditional: T's veto applies only when the unchanged arms' outcomes also
                 trend with completion time: in each (session, case) stratum, sum of (t - mean t)(y - mean y)
                 over the three unchanged arms' executions, summed over strata, over the square root of the
                 summed permutation variance (sum (t - mean t)^2 * sum (y - mean y)^2 / (m - 1), m executions
                 in the stratum); veto when T's p < A_T AND this two-sided p < A_TR. A candidate that runs
                 longer samples later conditions, which biases the comparison only if serving drifts within
                 the session, and a drift shows in the unchanged arms. The trend is computed on dispatch
                 times where the log records them (in the simulation the unchanged arms' dispatch and
                 completion times coincide); on completion times it also picks up any link between an
                 execution's outcome and its duration, which only makes T* stricter.
              Under random interleaving the arm labels are exchangeable for endpoint, time and rep, so each
              check's false-inconclusive rate is its alpha by construction (measured below).
  IUT         Intersection-union gain test: the candidate must show a gain against EACH unchanged arm on its
              own, not against their pool. Its null is "the candidate is no better than at least one unchanged
              arm", so its level is at most 0.05 whenever one unchanged arm is right, however the other two are
              broken, with no threshold to calibrate (it replaces the agreement check). Single session (U1):
              each z is `z_ref` of the candidate against one arm, variance scaled by max(1, phi_hat), phi_hat
              from the three unchanged arms as in D34; admitted when the smallest z >= z(0.95).
  replication k independent sessions of n runs per case per arm, every arm fresh in every session (D37). Per
              session s and unchanged arm j, D_sj = sum over cases of (candidate passes - arm j passes) / n;
              t_j = mean_s(D_sj) / (sd_s(D_sj) / sqrt(k)); admitted when the smallest t_j >= t(0.95, k - 1)
              and the harm screen passes; a zero sample SD makes the cohort inconclusive. Any arm-level effect
              that is drawn afresh in each session (candidate-only dispersion or session effect, a different
              response to shared serving, persistent arm states, routing whose endpoint effects change between
              sessions) enters the session-to-session variance that t_j uses, so it costs power, not level.
              Compared on the same draws: "2 of 3" (the second-largest t_j must pass: protects against one
              broken unchanged arm, not two) and "pooled" (one t-test of the candidate against the mean of the
              three unchanged arms: no protection against a broken unchanged arm). D37 = IUT + E + T* + P.
  protocol    Identity arms draw seeds from their own streams (no seed shared with the parent at a (case, rep));
              the gate assigns every execution's endpoint uniformly at random and records it, so routing cannot
              follow the candidate (check E then only verifies that the assignment was honoured).

Harm screen everywhere: per case, one-sided exact test of the candidate's pooled passes against the pooled
passes of the three unchanged arms at 0.05 (D34's screen; with k sessions, kn against 3kn).

Model as `r7_stats.py` (v3 pooled null rates of the three unchanged arms, 19 cases, 14 intermediate). One change
to the interleaved model, to match the gate: every execution's dispatch time is uniform over the session (random
interleaving), not 5 per block. Shared serving: 8 time blocks and 5 endpoints per session, logit effects
N(0, block SD) per block plus a linear trend per block, N(0, endpoint SD) per endpoint, drawn per session (or once
for every session where a row says persistent); an execution's effect is its arm's response (1 for unchanged arms)
times (block effect of the block it completes in + effect of its endpoint). A lagging candidate completes `lag`
blocks later and is served in that block (FABLE's model; a block is one eighth of a session). Dispersion as r7:
a per-arm, per-case Beta rate with intra-class correlation (phi - 1) / 39, phi defined at 40 runs. A session
effect: one N(0, SD) logit shift per arm per session on every intermediate case; "mean-preserving" centres the
candidate's per-case intercept so its expected rate over sessions equals the unchanged arms'. Persistent arm
state (ASTRA O48): each fresh arm draws one fair coin and every outcome of the arm equals it.

Column `mean diff`: the simulated mean of (candidate minus the unchanged arms' mean) summed over cases, in points
per intermediate case: the size of whatever expected difference a "no effect" row carries.

Run: python3 -I scripts/stats/r8_stats.py > scripts/stats/r8_stats.out   (numpy 2.4.4, scipy 1.16.1; reads
`_raw/ratchet-runrecords/v3_exec.json`; R8_PROCS sets the number of worker processes and never the result.)
"""

import hashlib
import itertools
import json
import math
import os
import sys
from functools import cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import chi2, norm, rankdata, t as tdist

HERE = Path(__file__).resolve().parent
V3 = HERE.parent.parent / "_raw" / "ratchet-runrecords" / "v3_exec.json"
NULL_ARMS = ("parent", "identity_1", "identity_2")
CANDS = ("candidate:ad50dac8194c", "candidate:b972467a3b9a")

SIMS = 50_000
CHUNK = 2_500
N = 40
BLOCKS, ENDPOINTS = 8, 5
ZN = norm.ppf(0.95)
A_E = 0.01
A_T = 0.01
A_P = 0.01
A_TR = 0.05
Z_P = norm.ppf(1 - A_P / 3)
UNIF = np.full(ENDPOINTS, 1.0 / ENDPOINTS)

_raw_bytes = V3.read_bytes()
V3_SHA = hashlib.sha256(_raw_bytes).hexdigest()
ROWS = json.loads(_raw_bytes)
CASES = sorted({r["case_id"] for r in ROWS})
C = len(CASES)
IDX = {c: i for i, c in enumerate(CASES)}
_k = dict.fromkeys(CASES, 0)
for _r in ROWS:
    if _r["arm"] in NULL_ARMS:
        _k[_r["case_id"]] += int(_r["passed"])
P = np.array([_k[c] / 120 for c in CASES])
MID = (P > 0) & (P < 1)
POL = IDX["policy_changed.v6"]
Z0 = [i for i in range(C) if P[i] == 0][:2]


# ---------------------------------------------------------------------------------------------------------------
# Tests (r7_stats.py's, generalised to n runs per arm)
# ---------------------------------------------------------------------------------------------------------------


@cache
def low_table(n1, n2):
    """P(first count <= a | totals), exact conditional (hypergeometric) distribution, n1 against n2 runs."""
    t = np.zeros((n1 + 1, n2 + 1))
    for a in range(n1 + 1):
        for b in range(n2 + 1):
            tt = a + b
            lo = max(0, tt - n2)
            t[a, b] = sum(
                math.comb(n1, x) * math.comb(n2, tt - x) for x in range(lo, a + 1)
            ) / math.comb(n1 + n2, tt)
    return t


def phi_three(par, i1, i2, n):
    xbar = (par + i1 + i2) / 3.0
    sq = ((par - xbar) ** 2 + (i1 - xbar) ** 2 + (i2 - xbar) ** 2).sum(-1)
    pi = (par + i1 + i2) / (3.0 * n)
    vs = (2 * n * pi * (1 - pi)).sum(-1)
    return np.maximum(1.0, np.where(vs > 0, sq / np.where(vs > 0, vs, 1), 1.0))


def z_ref(cand, ref, nc, nref, scale):
    d = (cand / nc - ref / nref).sum(-1)
    pi = (cand + ref) / (nc + nref)
    var = (pi * (1 - pi) * (1 / nc + 1 / nref)).sum(-1) * scale
    return np.where(var > 0, d / np.sqrt(np.where(var > 0, var, 1)), 0.0)


def z_pair(a, b, n):
    d = (b - a).sum(-1)
    pi = (a + b) / (2.0 * n)
    var = (2 * n * pi * (1 - pi)).sum(-1)
    return np.where(var > 0, d / np.sqrt(np.where(var > 0, var, 1)), 0.0)


def agree_stat(par, i1, i2, n, scaled=True):
    """Largest pairwise aggregate |z| among the three unchanged arms, over sqrt(phi_hat) when scaled (D34)."""
    zs = [np.abs(z_pair(a, b, n)) for a, b in ((par, i1), (par, i2), (i1, i2))]
    m = np.maximum(np.maximum(zs[0], zs[1]), zs[2])
    return m / np.sqrt(phi_three(par, i1, i2, n)) if scaled else m


def harm_ok(cand, ref_pool, nc, nref):
    return ~(low_table(nc, nref)[cand, ref_pool] <= 0.05).any(-1)


def q_frozen():
    """D34's agreement constant, recomputed exactly as `r7_stats.py` does (seed 20261040)."""
    rng = np.random.default_rng(20261040)
    a = []
    for _ in range(3):
        pp = np.broadcast_to(P, (SIMS, C)).copy()
        a.append(rng.binomial(20, pp) + rng.binomial(20, pp))
    return float(np.quantile(agree_stat(a[0], a[1], a[2], N), 0.95))


def calibrate_suite(p, seed):
    """95th percentiles of the scaled and unscaled agreement statistics, 3 binomial arms at the suite's rates."""
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(3):
        pp = np.broadcast_to(p, (SIMS, len(p))).copy()
        a.append(rng.binomial(20, pp) + rng.binomial(20, pp))
    return (
        float(np.quantile(agree_stat(a[0], a[1], a[2], N), 0.95)),
        float(np.quantile(agree_stat(a[0], a[1], a[2], N, scaled=False), 0.95)),
    )


Q_FROZEN = q_frozen()


# ---------------------------------------------------------------------------------------------------------------
# Mean-preserving centring (Gauss-Hermite, probabilists' weights)
# ---------------------------------------------------------------------------------------------------------------

_GX, _GW = np.polynomial.hermite_e.hermegauss(96)
_GW = _GW / math.sqrt(2 * math.pi)


def ln_mean(mu, s):
    return float((_GW * expit(mu + s * _GX)).sum())


def centre(target, s):
    return brentq(lambda m: ln_mean(m, s) - target, -40.0, 40.0, xtol=1e-14)


def centred_logits(base, s_ref, s_cand):
    """Per-case candidate intercepts whose mean rate under logit noise SD s_cand equals the unchanged arms' mean
    rate under SD s_ref (intermediate cases only; others keep their rate)."""
    out = logit(np.clip(base, 1e-12, 1 - 1e-12))
    for j in np.where(MID & (base > 0) & (base < 1))[0]:
        out[j] = centre(ln_mean(out[j], s_ref), s_cand)
    return out


# ---------------------------------------------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------------------------------------------


def shift(d, base=None):
    b = P if base is None else base
    q = b.copy()
    q[MID] = np.clip(b[MID] + d, 0.0, 1.0)
    return q


def with_fix(base):
    q = base.copy()
    q[Z0] = 1.0
    return q


def with_harm(base, drop):
    q = base.copy()
    q[POL] = max(0.0, q[POL] - drop)
    return q


def spec(**kw):
    s = {
        "p": P,
        "rates": (P, P, P, P),
        "phi": (1.0, 1.0, 1.0, 1.0),
        "sess": (0.0, 0.0, 0.0, 0.0),
        "centre_sess": False,
        "bsd": 0.0,
        "esd": 0.0,
        "trend": 0.0,
        "beta": (1.0, 1.0, 1.0, 1.0),
        "centre_beta": False,
        "lag": (0.0, 0.0, 0.0, 0.0),
        "ep": (UNIF, UNIF, UNIF, UNIF),
        "ep_persist": False,
        "astra": False,
        "rho": 0.0,
    }
    s.update(kw)
    return s


def arm_rates(sp, rng, a, ch, k):
    """Per-session, per-case rate of arm `a` after its session effect and dispersion; and, for a mean-preserving
    response to shared serving, the candidate's per-case logit intercepts."""
    p = sp["rates"][a]
    cc = len(p)
    m = (p > 0) & (p < 1)
    base = np.broadcast_to(p, (ch, k, cc)).copy()
    lp_fixed = None
    if a == 3 and sp["centre_beta"]:
        s_ref = math.hypot(sp["bsd"], sp["esd"])
        lp_fixed = centred_logits(p, s_ref, sp["beta"][3] * s_ref)
    if sp["sess"][a] > 0:
        eps = rng.normal(0.0, sp["sess"][a], size=(ch, k, 1))
        centred = sp["centre_sess"] and a == 3
        mu = centred_logits(p, 0.0, sp["sess"][a])[m] if centred else logit(p[m])
        base[..., m] = expit(mu[None, None, :] + eps)
    ph = np.broadcast_to(np.asarray(sp["phi"][a], float), (cc,))
    for j in np.where((ph > 1.0) & m)[0]:
        r = (ph[j] - 1) / (N - 1)
        ab = (1 - r) / r
        base[..., j] = rng.beta(base[..., j] * ab, (1 - base[..., j]) * ab)
    return base, lp_fixed


def arm_draw(sp, rng, a, shape, shared, parent):
    """Executions of arm `a`: completion time (in blocks), endpoint and outcome of every (session, case, run)."""
    ch, k, cc, _ = shape
    base, lp_fixed = arm_rates(sp, rng, a, ch, k)
    tau = rng.random(shape) * BLOCKS + sp["lag"][a]
    if np.array_equal(sp["ep"][a], UNIF):
        ep = rng.integers(0, ENDPOINTS, size=shape, dtype=np.int8)
    else:
        cum = np.cumsum(sp["ep"][a])[:-1]
        ep = np.searchsorted(
            cum, rng.random(shape, dtype=np.float32), side="right"
        ).astype(np.int8)
    if sp["astra"]:
        return np.broadcast_to(rng.random((ch, k, 1, 1)) < 0.5, shape), tau, ep
    pp = base[..., None].astype(np.float32)
    if shared is not None:
        b, e, rowoff, eoff = shared
        eff = b[rowoff + np.minimum(BLOCKS - 1, tau.astype(np.int64))] + e[eoff + ep]
        lp = (
            np.broadcast_to(lp_fixed, (ch, k, cc))
            if lp_fixed is not None
            else logit(np.clip(base, 1e-12, 1 - 1e-12))
        )
        mid_g = (sp["p"] > 0) & (sp["p"] < 1)
        pp = np.where(
            mid_g[None, None, :, None],
            expit(lp[..., None] + sp["beta"][a] * eff),
            base[..., None],
        )
        pp = pp.astype(np.float32)
    x = rng.random(shape, dtype=np.float32) < pp
    if a in (1, 2) and sp["rho"] > 0:
        x = np.where(rng.random(shape, dtype=np.float32) < sp["rho"], parent, x)
    return x, tau, ep


def check_e(eps, ch):
    """Endpoint check: chi-square p of the candidate's endpoint counts against the unchanged arms' (summed)."""
    cnt = []
    for ep in eps:
        flat = (
            ep.reshape(ch, -1).astype(np.int64) + (np.arange(ch) * ENDPOINTS)[:, None]
        )
        cnt.append(
            np.bincount(flat.ravel(), minlength=ch * ENDPOINTS).reshape(ch, ENDPOINTS)
        )
    tab = np.stack([cnt[3], cnt[0] + cnt[1] + cnt[2]], 1).astype(float)
    col = tab.sum(1)
    exp = tab.sum(2)[:, :, None] * col[:, None, :] / tab.sum((1, 2))[:, None, None]
    x2 = np.where(
        col[:, None, :] > 0, (tab - exp) ** 2 / np.where(exp > 0, exp, 1), 0.0
    ).sum((1, 2))
    return chi2.sf(x2, np.maximum((col > 0).sum(1) - 1, 1))


def check_t(times, ch, k):
    """Completion-time check: per session Mann-Whitney z of the candidate against the unchanged arms, combined
    over sessions; ranks from one integer sort with the candidate flag in the lowest bit (a tie needs two equal
    53-bit draws)."""
    n1 = times[3][0, 0].size
    n2 = 3 * n1
    enc = [(t.reshape(ch, k, -1) * 2.0**36).astype(np.int64) << 1 for t in times]
    enc[3] = enc[3] | 1
    srt = np.sort(np.concatenate(enc, -1), -1)
    u = ((srt & 1) * np.arange(1, n1 + n2 + 1)).sum(-1) - n1 * (n1 + 1) / 2
    zs = (u - n1 * n2 / 2) / math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return 2 * norm.sf(np.abs(zs.sum(-1) / math.sqrt(k)))


def check_trend(times, xs, n):
    """Outcome against completion time among the unchanged arms, stratified by session and case (permutation z)."""
    tt = np.concatenate(times[:3], -1)
    yy = np.concatenate(xs[:3], -1).astype(np.float64)
    tc = tt - tt.mean(-1, keepdims=True)
    yc = yy - yy.mean(-1, keepdims=True)
    num = (tc * yc).sum((-1, -2, -3))
    den = ((tc**2).sum(-1) * (yc**2).sum(-1) / (3 * n - 1)).sum((-1, -2))
    z = np.where(den > 0, num / np.sqrt(np.where(den > 0, den, 1)), 0.0)
    return 2 * norm.sf(np.abs(z))


def check_p(xs, counts, n):
    """Pairing check: largest standardised excess of same-(case, rep) both-pass slots over the three pairs."""
    zp = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        both = (xs[a] & xs[b]).sum(-1)
        ca, cb = counts[a], counts[b]
        var = ca * cb * (n - ca) * (n - cb) / (n * n * (n - 1.0))
        num = (both - ca * cb / n).sum((1, 2))
        den = var.sum((1, 2))
        zp.append(np.where(den > 0, num / np.sqrt(np.where(den > 0, den, 1)), 0.0))
    return np.maximum(np.maximum(zp[0], zp[1]), zp[2])


def draw_chunk(sp, rng, ch, k, n):
    """One chunk of `ch` comparisons, k sessions of n runs per case per arm. Returns counts (4, ch, k, C) and the
    log-check statistics: endpoint-check p, completion-time-check p, unchanged-arm trend p, pairing statistic."""
    shape = (ch, k, len(sp["p"]), n)
    shared = None
    if sp["bsd"] > 0 or sp["esd"] > 0 or sp["trend"] != 0:
        b = rng.normal(0.0, sp["bsd"], size=(ch, k, BLOCKS)) + sp["trend"] * (
            np.arange(BLOCKS) - (BLOCKS - 1) / 2
        )
        e = rng.normal(
            0.0, sp["esd"], size=(ch, 1 if sp["ep_persist"] else k, ENDPOINTS)
        )
        shared = (
            b.ravel(),
            np.ascontiguousarray(np.broadcast_to(e, (ch, k, ENDPOINTS))).ravel(),
            (np.arange(ch * k) * BLOCKS).reshape(ch, k, 1, 1),
            (np.arange(ch * k) * ENDPOINTS).reshape(ch, k, 1, 1),
        )
    xs, times, eps = [], [], []
    for a in range(4):
        x, tau, ep = arm_draw(sp, rng, a, shape, shared, xs[0] if xs else None)
        xs.append(x)
        times.append(tau)
        eps.append(ep)
    counts = np.stack([x.sum(-1) for x in xs])
    return (
        counts,
        check_e(eps, ch),
        check_t(times, ch, k),
        check_trend(times, xs, n),
        check_p(xs, counts, n),
    )


def run_job(job):
    """job = (seed, spec, k, n, extra). Returns a dict of event counts over SIMS comparisons."""
    seed, sp, k, n, extra = job
    rng = np.random.default_rng(seed)
    acc = {"diff": 0.0}

    def add(name, arr):
        acc[name] = acc.get(name, 0) + int(np.count_nonzero(arr))

    nmid = int(((sp["p"] > 0) & (sp["p"] < 1)).sum())
    for _ in range(SIMS // CHUNK):
        X, p_e, p_t, p_tr, p_stat = draw_chunk(sp, rng, CHUNK, k, n)
        okE, okP = p_e >= A_E, p_stat < Z_P
        okT = p_t >= A_T
        okTs = okT | (p_tr >= A_TR)
        ok = okE & okT & okP
        oks = okE & okTs & okP
        add("failE", ~okE)
        add("failT", ~okT)
        add("failTs", ~okTs)
        add("failP", ~okP)
        par, i1, i2, cand = X
        acc["diff"] += float(
            ((cand - (par + i1 + i2) / 3.0) / n).sum(-1).mean(-1).sum()
        )
        harm = harm_ok(cand.sum(1), (par + i1 + i2).sum(1), k * n, 3 * k * n)
        if k == 1:
            par, i1, i2, cand = par[:, 0], i1[:, 0], i2[:, 0], cand[:, 0]
            p3 = phi_three(par, i1, i2, n)
            gain = z_ref(cand, par + i1 + i2, n, 3 * n, p3) >= ZN
            add("gain", gain)
            stats = agree_stat(par, i1, i2, n)
            ustat = agree_stat(par, i1, i2, n, scaled=False)
            for qname, q, st in extra:
                agree = (stats if st == "scaled" else ustat) < q
                add(f"d34_{qname}", gain & harm & agree)
                add(f"d34inc_{qname}", ~agree)
                add(f"d34s_{qname}", gain & harm & agree & oks)
                add(f"d34sinc_{qname}", ~(agree & oks))
            zu = np.minimum(
                np.minimum(z_ref(cand, par, n, n, p3), z_ref(cand, i1, n, n, p3)),
                z_ref(cand, i2, n, n, p3),
            )
            u1 = (zu >= ZN) & harm
            add("u1", u1)
            add("u1c", u1 & ok)
            add("u1cinc", ~ok)
            add("u1s", u1 & oks)
            add("u1sinc", ~oks)
        else:
            crit = tdist.ppf(0.95, k - 1)
            ts = []
            degenerate = np.zeros(CHUNK, bool)
            for ref in (par, i1, i2):
                d = ((cand - ref) / n).sum(-1)
                sd = d.std(-1, ddof=1)
                degenerate |= sd == 0
                ts.append(d.mean(-1) / np.where(sd > 0, sd, 1) * math.sqrt(k))
            ts = np.sort(np.stack(ts), 0)
            iut = (ts[0] >= crit) & ~degenerate & harm
            two = (ts[1] >= crit) & ~degenerate & harm
            add("r", iut)
            add("rc", iut & ok)
            add("rcinc", ~ok | degenerate)
            add("rs", iut & oks)
            add("rsinc", ~oks | degenerate)
            add("r2s", two & oks)
            d = ((cand - (par + i1 + i2) / 3.0) / n).sum(-1)
            sd = d.std(-1, ddof=1)
            tp = d.mean(-1) / np.where(sd > 0, sd, 1) * math.sqrt(k)
            add("rpools", (tp >= crit) & (sd > 0) & harm & oks)
    acc["diff"] = acc["diff"] / SIMS / max(nmid, 1) * 100
    return acc


# ---------------------------------------------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------------------------------------------

E25 = np.array([0.25] + [0.1875] * 4)
E40 = np.array([0.40] + [0.15] * 4)
E60 = np.array([0.60] + [0.10] * 4)
C3 = (0.0, 0.0, 0.0, 1.0)

SCEN = [
    # label, kind, spec, figure reported in round 8 (and by whom)
    ("binomial, no effect", "null", spec(), "0.039 (r7 `is`)"),
    ("common dispersion 2.0 on every arm", "null", spec(phi=(2.0,) * 4), "0.036 (r7)"),
    ("candidate-only dispersion 1.6", "null", spec(phi=(1, 1, 1, 1.6)), "0.061 (r7)"),
    ("candidate-only dispersion 2.0", "null", spec(phi=(1, 1, 1, 2.0)), "0.067 (r7)"),
    (
        "candidate-only session effect, logit SD 0.2",
        "null",
        spec(sess=(0, 0, 0, 0.2)),
        "0.171 (r7)",
    ),
    (
        "candidate-only session effect, logit SD 0.2, mean-preserving",
        "null",
        spec(sess=(0, 0, 0, 0.2), centre_sess=True),
        "-",
    ),
    (
        "candidate-only session effect, logit SD 0.4, mean-preserving",
        "null",
        spec(sess=(0, 0, 0, 0.4), centre_sess=True),
        "-",
    ),
    (
        "session effect on every arm, logit SD 0.2",
        "null",
        spec(sess=(0.2,) * 4),
        "0.097 (r7)",
    ),
    (
        "persistent arm state: every outcome of an arm equals one fair coin",
        "null",
        spec(astra=True),
        ">= 0.0625 (ASTRA)",
    ),
    (
        "both identity arms down 3 points",
        "null",
        spec(rates=(P, shift(-0.03), shift(-0.03), P)),
        "0.173 (FABLE)",
    ),
    (
        "one identity arm down 3 points",
        "null",
        spec(rates=(P, shift(-0.03), P, P)),
        "0.078 (FABLE)",
    ),
    (
        "one identity arm down 5 points",
        "null",
        spec(rates=(P, shift(-0.05), P, P)),
        "0.076 (REVIEWER-2)",
    ),
    (
        "parent alone down 5 points",
        "null",
        spec(rates=(shift(-0.05), P, P, P)),
        "0.077 (REVIEWER-2)",
    ),
    (
        "parent and one identity arm down 5 points",
        "null",
        spec(rates=(shift(-0.05), shift(-0.05), P, P)),
        "0.208 (FABLE)",
    ),
    (
        "both identity arms down 5 points",
        "null",
        spec(rates=(P, shift(-0.05), shift(-0.05), P)),
        "0.204 (r7)",
    ),
    (
        "identity arm 1 down 10 points",
        "null",
        spec(rates=(P, shift(-0.10), P, P)),
        "0.010 (r7)",
    ),
    (
        "every arm but the parent drifted +5 points",
        "null",
        spec(rates=(P, shift(0.05), shift(0.05), shift(0.05))),
        "0.075 (r7)",
    ),
    (
        "interleaved, block SD 0.5, endpoint SD 0.5",
        "null",
        spec(bsd=0.5, esd=0.5),
        "0.038 (r7)",
    ),
    (
        "interleaved 0.2 / 0.2, candidate responds x2.0",
        "null",
        spec(bsd=0.2, esd=0.2, beta=(1, 1, 1, 2.0)),
        "0.095 (STATS)",
    ),
    (
        "interleaved 0.5 / 0.5, candidate responds x0.5",
        "null",
        spec(bsd=0.5, esd=0.5, beta=(1, 1, 1, 0.5)),
        "0.101 (STATS)",
    ),
    (
        "interleaved 0.5 / 0.5, candidate responds x1.5",
        "null",
        spec(bsd=0.5, esd=0.5, beta=(1, 1, 1, 1.5)),
        "0.114 (STATS)",
    ),
    (
        "interleaved 0.5 / 0.5, candidate responds x2.0",
        "null",
        spec(bsd=0.5, esd=0.5, beta=(1, 1, 1, 2.0)),
        "0.226 (STATS)",
    ),
    (
        "interleaved 0.5 / 0.5, candidate responds x0.5, mean-preserving",
        "null",
        spec(bsd=0.5, esd=0.5, beta=(1, 1, 1, 0.5), centre_beta=True),
        "-",
    ),
    (
        "interleaved 0.5 / 0.5, candidate responds x2.0, mean-preserving",
        "null",
        spec(bsd=0.5, esd=0.5, beta=(1, 1, 1, 2.0), centre_beta=True),
        "-",
    ),
    (
        "trend +0.05 logit per block, candidate lags 1 block",
        "null",
        spec(bsd=0.2, esd=0.2, trend=0.05, lag=(0, 0, 0, 1.0)),
        "0.086 (FABLE)",
    ),
    (
        "trend +0.10 logit per block, candidate lags 1 block",
        "null",
        spec(bsd=0.2, esd=0.2, trend=0.10, lag=(0, 0, 0, 1.0)),
        "0.150 (FABLE)",
    ),
    (
        "trend +0.10 logit per block, candidate lags 2 blocks",
        "null",
        spec(bsd=0.2, esd=0.2, trend=0.10, lag=(0, 0, 0, 2.0)),
        "0.329 (FABLE)",
    ),
    (
        "trend +0.10 logit per block, candidate lags 0.25 block",
        "null",
        spec(bsd=0.2, esd=0.2, trend=0.10, lag=(0, 0, 0, 0.25)),
        "-",
    ),
    (
        "trend +0.10 logit per block, candidate lags 0.5 block",
        "null",
        spec(bsd=0.2, esd=0.2, trend=0.10, lag=(0, 0, 0, 0.5)),
        "-",
    ),
    (
        "trend +0.20 logit per block, candidate lags 0.25 block",
        "null",
        spec(bsd=0.2, esd=0.2, trend=0.20, lag=(0, 0, 0, 0.25)),
        "-",
    ),
    (
        "trend +0.20 logit per block, candidate lags 0.5 block",
        "null",
        spec(bsd=0.2, esd=0.2, trend=0.20, lag=(0, 0, 0, 0.5)),
        "-",
    ),
    (
        "trend +0.10 logit per block shared by every arm, no lag",
        "null",
        spec(bsd=0.2, esd=0.2, trend=0.10),
        "0.038 (FABLE, at +0.05)",
    ),
    (
        "no trend, candidate lags 0.5 block",
        "null",
        spec(bsd=0.2, esd=0.2, lag=(0, 0, 0, 0.5)),
        "-",
    ),
    (
        "no trend, candidate lags 1 block",
        "null",
        spec(bsd=0.2, esd=0.2, lag=(0, 0, 0, 1.0)),
        "0.044 (FABLE)",
    ),
    (
        "candidate routed 25% to endpoint 1 (others 18.75%), endpoint SD 0.5",
        "null",
        spec(bsd=0.2, esd=0.5, ep=(UNIF, UNIF, UNIF, E25)),
        "-",
    ),
    (
        "candidate routed 40% to endpoint 1 (others 15%), endpoint SD 0.5",
        "null",
        spec(bsd=0.2, esd=0.5, ep=(UNIF, UNIF, UNIF, E40)),
        "0.089 (FABLE)",
    ),
    (
        "candidate routed 60% to endpoint 1 (others 10%), endpoint SD 0.5",
        "null",
        spec(bsd=0.2, esd=0.5, ep=(UNIF, UNIF, UNIF, E60)),
        "0.184 (FABLE)",
    ),
    (
        "candidate routed 40% to endpoint 1, endpoint effects persistent across sessions",
        "null",
        spec(bsd=0.2, esd=0.5, ep=(UNIF, UNIF, UNIF, E40), ep_persist=True),
        "-",
    ),
    (
        "identity arms reuse the parent's (case, rep) outcome, rho 0.1 (shared seed)",
        "null",
        spec(rho=0.1),
        "-",
    ),
    ("the same, rho 0.5", "null", spec(rho=0.5), "0.058 (STATS)"),
    ("the same, rho 1.0", "null", spec(rho=1.0), "0.074 (STATS)"),
    (
        "candidate +3 points from its own bytes on this serving (an effect of the bytes)",
        "effect",
        spec(rates=(P, P, P, shift(0.03))),
        "0.368 (REVIEWER-2)",
    ),
    (
        "every intermediate case +10 points",
        "power",
        spec(rates=(P, P, P, shift(0.10))),
        "0.934 (r7)",
    ),
    (
        "every intermediate case +5 points",
        "power",
        spec(rates=(P, P, P, shift(0.05))),
        "-",
    ),
    (
        "every intermediate case +10 points, dispersion 1.6 on every arm",
        "power",
        spec(rates=(P, P, P, shift(0.10)), phi=(1.6,) * 4),
        "0.846 (r7)",
    ),
    (
        "two zero-rate cases to 1.0",
        "power",
        spec(rates=(P, P, P, with_fix(P))),
        "0.646 (r7)",
    ),
    (
        "every intermediate case +10 points, candidate lags 0.5 block, no trend",
        "power",
        spec(rates=(P, P, P, shift(0.10)), bsd=0.2, esd=0.2, lag=(0, 0, 0, 0.5)),
        "-",
    ),
    (
        "every intermediate case +10 points, candidate lags 0.5 block, trend +0.10",
        "power",
        spec(
            rates=(P, P, P, shift(0.10)),
            bsd=0.2,
            esd=0.2,
            trend=0.10,
            lag=(0, 0, 0, 0.5),
        ),
        "-",
    ),
    (
        "every intermediate case +10 points, interleaved 0.5 / 0.5",
        "power",
        spec(rates=(P, P, P, shift(0.10)), bsd=0.5, esd=0.5),
        "0.936 (r7)",
    ),
    (
        "two zero-rate cases fixed, policy_changed.v6 down 20 points",
        "harm",
        spec(rates=(P, P, P, with_harm(with_fix(P), 0.20))),
        "0.226 (r7)",
    ),
]

CONFIGS = [
    (2, 20),
    (4, 10),
    (8, 5),
    (10, 4),
    (4, 20),
    (8, 10),
    (16, 5),
    (8, 15),
    (12, 10),
]
SEED_A = 20261300
SEED_B = 20261400
SEED_C = 20261700
SUITES = (2, 3, 4, 6, 14)


def suite_rates(kk):
    mids = np.where(MID)[0]
    order = mids[np.argsort(-(P[mids] * (1 - P[mids])), kind="stable")]
    ends = np.where(~MID)[0]
    return np.sort(np.concatenate([order[:kk], ends]))


def fmt(n):
    return (
        f"0 of {SIMS}"
        if n == 0
        else (f"{n / SIMS:.3f}" if n >= 25 else f"{n} of {SIMS}")
    )


def jobs():
    out = []
    extra = (("frozen", Q_FROZEN, "scaled"),)
    for i, (_, _, sp, _) in enumerate(SCEN):
        out.append(("A", i, (SEED_A + i, sp, 1, N, extra)))
    for ci, (k, n) in enumerate(CONFIGS):
        for i, (_, _, sp, _) in enumerate(SCEN):
            out.append(("B", (ci, i), (SEED_B + 100 * ci + i, sp, k, n, ())))
    for si, kk in enumerate(SUITES):
        sel = suite_rates(kk)
        p = P[sel]
        mid = (p > 0) & (p < 1)
        qs, qu = calibrate_suite(p, SEED_C + 90 + si)

        def sh(d, p=p, mid=mid):
            q = p.copy()
            q[mid] = np.clip(p[mid] + d, 0.0, 1.0)
            return q

        rows = (
            spec(p=p, rates=(p, p, p, p)),
            spec(p=p, rates=(p, sh(-0.20), p, p)),
            spec(p=p, rates=(p, sh(-0.40), p, p)),
            spec(p=p, rates=(p, p, p, sh(0.10))),
        )
        extra = (
            ("frozen", Q_FROZEN, "scaled"),
            ("cal", qs, "scaled"),
            ("calu", qu, "unscaled"),
        )
        for ri, sp in enumerate(rows):
            out.append(
                ("C", (si, ri, qs, qu), (SEED_C + 10 * si + ri, sp, 1, N, extra))
            )
            out.append(("C8", (si, ri), (SEED_C + 50 + 10 * si + ri, sp, 8, 5, ())))
    return out


# ---------------------------------------------------------------------------------------------------------------
# The v3 run's own executions under the log checks and U1
# ---------------------------------------------------------------------------------------------------------------


def v3_e_t(ce, re, ct, rt):
    """E (chi-square, df) and T (tie-corrected Mann-Whitney z) for one arm against a reference group."""
    tab = np.stack([ce, re])
    col = tab.sum(0)
    keep = col > 0
    ex = tab.sum(1)[:, None] * col[None, :] / tab.sum()
    x2 = float(((tab[:, keep] - ex[:, keep]) ** 2 / ex[:, keep]).sum())
    rk = rankdata(np.concatenate([rt, ct]))
    n1, n2 = len(ct), len(rt)
    u = rk[n2:].sum() - n1 * (n1 + 1) / 2
    _, tie = np.unique(np.concatenate([rt, ct]), return_counts=True)
    nn = n1 + n2
    var = n1 * n2 / 12 * ((nn + 1) - (tie**3 - tie).sum() / (nn * (nn - 1)))
    return x2, int(keep.sum()) - 1, (u - n1 * n2 / 2) / math.sqrt(var)


def v3_table():
    """The v3 executions as arrays: arm name, case index, rep, outcome, completion time (s), endpoint index."""
    from datetime import datetime

    t = [datetime.fromisoformat(r["created_at"]) for r in ROWS]
    eps = sorted({r["endpoint"] for r in ROWS})
    return (
        np.array([r["arm"] for r in ROWS]),
        np.array([IDX[r["case_id"]] for r in ROWS]),
        np.array([r["rep"] for r in ROWS]),
        np.array([bool(r["passed"]) for r in ROWS]),
        np.array([(x - min(t)).total_seconds() for x in t]),
        np.array([eps.index(r["endpoint"]) for r in ROWS]),
    )


def v3_trend_z(arm, case, t, y):
    """T*'s trend statistic on the unchanged arms of one session, stratified by case."""
    num = den = 0.0
    for c in range(C):
        s = np.isin(arm, NULL_ARMS) & (case == c)
        tc, yc = t[s] - t[s].mean(), y[s] - y[s].mean()
        num += float((tc * yc).sum())
        den += float((tc**2).sum() * (yc**2).sum() / (s.sum() - 1))
    return num / math.sqrt(den)


def v3_pairing(arm, case, rep, y):
    out = []
    grids = {}
    for a in NULL_ARMS:
        g = np.zeros((C, N), bool)
        s = arm == a
        g[case[s], rep[s]] = y[s]
        grids[a] = g
    for a, b in itertools.combinations(NULL_ARMS, 2):
        both = (grids[a] & grids[b]).sum(1)
        ca, cb = grids[a].sum(1), grids[b].sum(1)
        ex = ca * cb / N
        var = ca * cb * (N - ca) * (N - cb) / (N * N * (N - 1.0))
        z = (both - ex).sum() / math.sqrt(var.sum())
        out.append(
            f"- {a} and {b}: excess both-pass slots {float((both - ex).sum()):+.2f}, z {z:+.2f}"
        )
    return out


def v3_section():
    arm, case, rep, y, t, ep = v3_table()

    def counts(a):
        return np.bincount(case[(arm == a) & y], minlength=C)

    def epmix(names):
        return np.bincount(ep[np.isin(arm, names)], minlength=ENDPOINTS).astype(float)

    par, i1, i2 = (counts(a) for a in NULL_ARMS)
    p3 = phi_three(par, i1, i2, N)
    z_tr = v3_trend_z(arm, case, t, y.astype(float))
    lines = [
        (
            f"Trend of outcome against completion time among the three unchanged arms, stratified by case: "
            f"z {z_tr:+.2f}, two-sided p {2 * norm.sf(abs(z_tr)):.4f} (T* vetoes a time difference only "
            f"when this p < {A_TR})."
        ),
        "",
        (
            "| arm | executions per endpoint (8801 to 8805) | E: chi-square, df, p | mean completion (s "
            "after the first record) | T: Mann-Whitney z, two-sided p | U1: z against parent / identity 1 "
            "/ identity 2 |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for a in (*NULL_ARMS, *CANDS):
        others = [o for o in NULL_ARMS if o != a]
        ce = epmix([a])
        x2, df, z = v3_e_t(ce, epmix(others), t[arm == a], t[np.isin(arm, others)])
        zu = "- (unchanged arm; E and T against the other two)"
        if a in CANDS:
            zu = " / ".join(
                f"{float(z_ref(counts(a), r, N, N, p3)):+.2f}" for r in (par, i1, i2)
            )
        lines.append(
            f"| {a} | {' / '.join(str(int(v)) for v in ce)} | {x2:.2f}, {df}, {chi2.sf(x2, df):.4f} | "
            f"{t[arm == a].mean():.1f} | {z:+.2f}, {2 * norm.sf(abs(z)):.4f} | {zu} |"
        )
    lines += [
        "",
        (
            "Pairing of the unchanged arms by (case, rep) (the v3 run shared seed 1000 + rep across "
            f"arms): Z_P = {float(Z_P):.4f}"
        ),
        "",
        *v3_pairing(arm, case, rep, y),
        f"- phi_hat from the three unchanged arms: {float(p3):.4f}",
    ]
    return lines


# ---------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------


def main():
    procs = int(os.environ.get("R8_PROCS", os.cpu_count() or 1))
    js = jobs()
    for k, n in {(1, N), *CONFIGS, (8, 5)}:
        low_table(k * n, 3 * k * n)
    with get_context("fork").Pool(procs) as pool:
        res = pool.map(run_job, [j[2] for j in js], chunksize=1)
    R = {(sec, key): r for (sec, key, _), r in zip(js, res, strict=True)}
    null_rows = [i for i, s in enumerate(SCEN) if s[1] == "null"]
    pw = [i for i, s in enumerate(SCEN) if s[1] == "power"]
    hm = next(i for i, s in enumerate(SCEN) if s[1] == "harm")
    pr = print
    pr(
        "# Round 8 residuals of the interleaved design measured, and the design that closes them (ledger D37)"
    )
    pr()
    pr(
        f"Input `_raw/ratchet-runrecords/v3_exec.json`, SHA-256 {V3_SHA}; {len(ROWS)} executions, {C} cases,"
    )
    pr(
        f"{int(MID.sum())} intermediate at the pooled null rates of the three unchanged arms."
    )
    pr()
    pr("Constants:")
    pr(
        f"- D34 agreement threshold, recomputed as `r7_stats.py` (seed 20261040): Q_FROZEN = {Q_FROZEN!r}"
    )
    pr(f"- z(0.95) = {float(ZN)!r}; Z_P = z(1 - A_P / 3) = {float(Z_P)!r}")
    pr(
        f"- log-check alphas: A_E = {A_E}, A_T = {A_T}, A_P = {A_P}; trend alpha of T*: A_TR = {A_TR}"
    )
    pr(
        "- t(0.95, k - 1): "
        + ", ".join(
            f"k = {k}: {float(tdist.ppf(0.95, k - 1))!r}"
            for k in sorted({c[0] for c in CONFIGS})
        )
    )
    pr(
        f"- {SIMS} comparisons per cell, in chunks of {CHUNK}; Monte Carlo SE {math.sqrt(0.05 * 0.95 / SIMS):.4f} "
        f"near 0.05, {math.sqrt(0.01 * 0.99 / SIMS):.4f} near 0.01, {math.sqrt(0.25 / SIMS):.4f} at most"
    )
    pr(
        f"- seeds: Section 1 row i uses {SEED_A} + i; Section 2 configuration c (in the order listed) row i uses "
        f"{SEED_B} + 100 c + i; Section 3 suite s row r uses {SEED_C} + 10 s + r (one session) and "
        f"{SEED_C} + 50 + 10 s + r (8 sessions), its calibration {SEED_C} + 90 + s"
    )
    pr()
    pr(
        "T* = the completion-time check made conditional: it vetoes only when the candidate's completion times differ"
    )
    pr(
        "(T's p < A_T) AND the unchanged arms' outcomes trend with completion time (stratified by session and case,"
    )
    pr(
        "the permutation z of sum (t - mean t)(y - mean y), two-sided p < A_TR). A lag can bias the comparison only"
    )
    pr(
        "if serving drifts within the session, and the unchanged arms are where a drift shows."
    )
    pr()
    pr(
        "Row kinds: `null` = no effect of the candidate's bytes on its pass rate under the conditions every arm"
    )
    pr(
        "shares (a design must hold 0.05); `effect` = the candidate's bytes change its pass rate on this serving"
    )
    pr(
        "(any reference reads that as an effect); `power` = a real gain; `harm` = a gain with a hidden regression."
    )

    pr()
    pr(
        "## 1. One session of 40 runs per case per arm (3,040 executions): D34, its log-checked form, and U1"
    )
    pr()
    pr(
        "| # | scenario | kind | round 8 | mean diff | D34 gain only | D34: admitted / inconclusive | "
        "D34 + E, T*, P: admitted / inconclusive | U1: admitted | U1 + E, T, P: admitted / inconclusive | "
        "U1 + E, T*, P: admitted / inconclusive |"
    )
    pr("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (label, kind, _, r8) in enumerate(SCEN):
        a = R[("A", i)]
        pr(
            f"| {i} | {label} | {kind} | {r8} | {a['diff']:+.2f} | {fmt(a['gain'])} | "
            f"{fmt(a['d34_frozen'])} / {fmt(a['d34inc_frozen'])} | "
            f"{fmt(a['d34s_frozen'])} / {fmt(a['d34sinc_frozen'])} | {fmt(a['u1'])} | "
            f"{fmt(a['u1c'])} / {fmt(a['u1cinc'])} | {fmt(a['u1s'])} / {fmt(a['u1sinc'])} |"
        )
    a0 = R[("A", 0)]
    pr()
    pr(
        f"Each log check alone on row 0 (no effect): E fails {fmt(a0['failE'])}, T {fmt(a0['failT'])}, "
        f"T* {fmt(a0['failTs'])}, P {fmt(a0['failP'])}."
    )
    pr(
        "Largest no-effect admission over the `null` rows: D34 "
        + ", ".join(
            f"{name} {fmt(max(R[('A', i)][key] for i in null_rows))} (row "
            f"{max(null_rows, key=lambda i: R[('A', i)][key])})"
            for name, key in (
                ("as frozen", "d34_frozen"),
                ("D34 + E, T*, P", "d34s_frozen"),
                ("U1", "u1"),
                ("U1 + E, T, P", "u1c"),
                ("U1 + E, T*, P", "u1s"),
            )
        )
        + "."
    )

    pr()
    pr(
        "## 2. k independent sessions of n runs per case per arm: the replicated intersection-union t-test (D37)"
    )
    pr()
    pr(
        "Columns: the IUT t-test alone; with E, T and P; with E, T* and P (D37); the same test requiring 2 of the 3"
    )
    pr(
        "pairwise t (with E, T*, P); a t-test against the pooled unchanged arms (with E, T*, P). Executions = 4 arms"
    )
    pr(
        "x 19 cases x k x n. Inconclusive counts a failed log check or a zero sample SD."
    )
    summary = []
    for ci, (k, n) in enumerate(CONFIGS):
        pr()
        pr(
            f"### k = {k} sessions, n = {n} runs ({4 * C * k * n:,} executions, {k * n} runs per case per arm)"
        )
        pr()
        pr(
            "| # | scenario | kind | mean diff | IUT t | IUT + E, T, P: admitted / inconclusive | "
            "D37 = IUT + E, T*, P: admitted / inconclusive | 2 of 3 + E, T*, P | pooled + E, T*, P |"
        )
        pr("|---|---|---|---|---|---|---|---|---|")
        for i, (label, kind, _, _) in enumerate(SCEN):
            b = R[("B", (ci, i))]
            pr(
                f"| {i} | {label} | {kind} | {b['diff']:+.2f} | {fmt(b['r'])} | {fmt(b['rc'])} / {fmt(b['rcinc'])} | "
                f"{fmt(b['rs'])} / {fmt(b['rsinc'])} | {fmt(b['r2s'])} | {fmt(b['rpools'])} |"
            )
        worst = {
            key: max(null_rows, key=lambda i: R[("B", (ci, i))][key])
            for key in ("rs", "r2s", "rpools")
        }
        summary.append((ci, k, n, worst))
    pr()
    pr("### Summary of Section 2 (D37 = IUT + E, T*, P unless named)")
    pr()
    pr(
        "| k | n | runs per case per arm | executions | D37: largest no-effect admission (row) | 2 of 3: largest "
        "(row) | pooled: largest (row) | D37 inconclusive, row 0 | D37 power: "
        + " / ".join(f"row {i}" for i in pw)
        + f" | D37, harm row {hm} |"
    )
    pr("|---|---|---|---|---|---|---|---|---|---|")
    for ci, k, n, worst in summary:

        def g(i, key, ci=ci):
            return R[("B", (ci, i))][key]

        pr(
            f"| {k} | {n} | {k * n} | {4 * C * k * n:,} | {fmt(g(worst['rs'], 'rs'))} ({worst['rs']}) | "
            f"{fmt(g(worst['r2s'], 'r2s'))} ({worst['r2s']}) | {fmt(g(worst['rpools'], 'rpools'))} "
            f"({worst['rpools']}) | {fmt(g(0, 'rsinc'))} | "
            + " / ".join(fmt(g(i, "rs")) for i in pw)
            + f" | {fmt(g(hm, 'rs'))} |"
        )
    wd = max(null_rows, key=lambda i: R[("A", i)]["d34_frozen"])
    pr(
        f"| D34, one session | 40 | 40 | {4 * C * N:,} | D34: {fmt(R[('A', wd)]['d34_frozen'])} ({wd}) | - | - | "
        f"{fmt(a0['d34inc_frozen'])} | "
        + " / ".join(fmt(R[("A", i)]["d34_frozen"]) for i in pw)
        + f" | {fmt(R[('A', hm)]['d34_frozen'])} |"
    )
    held = [
        (k, n)
        for ci, k, n, w in summary
        if R[("B", (ci, w["rs"]))]["rs"] <= 0.05 * SIMS
    ]
    pr()
    pr(
        "Configurations in which D37 holds every `null` row at or below 0.05: "
        + (", ".join(f"k = {k}, n = {n}" for k, n in held) if held else "none")
        + "."
    )

    pr()
    pr(
        "## 3. Small suites: the k highest-variance intermediate cases of v3 plus its 5 zero- or one-rate cases"
    )
    pr()
    pr(
        "Agreement threshold for D34: frozen (Q_FROZEN), or calibrated per suite (95th percentile over 50,000"
    )
    pr(
        "binomial cohorts of three unchanged arms at the suite's rates), scaled by sqrt(phi_hat) as D34 or unscaled."
    )
    pr("U1 and D37 have no threshold to calibrate.")
    pr()
    pr(
        "| intermediate cases | Q per suite, scaled / unscaled | row | D34 frozen: admitted / inconclusive | "
        "D34 calibrated: admitted / inconclusive | D34 calibrated, unscaled: admitted / inconclusive | U1 | "
        "D37 at k = 8, n = 5: admitted / inconclusive |"
    )
    pr("|---|---|---|---|---|---|---|---|")
    names = (
        "no effect",
        "identity 1 down 20",
        "identity 1 down 40",
        "every intermediate case +10",
    )
    for si, kk in enumerate(SUITES):
        for ri in range(4):
            key = next(k for (s, k) in R if s == "C" and k[0] == si and k[1] == ri)
            c = R[("C", key)]
            c8 = R[("C8", (si, ri))]
            pr(
                f"| {kk} | {key[2]!r} / {key[3]!r} | {names[ri]} | "
                f"{fmt(c['d34_frozen'])} / {fmt(c['d34inc_frozen'])} | {fmt(c['d34_cal'])} / {fmt(c['d34inc_cal'])} | "
                f"{fmt(c['d34_calu'])} / {fmt(c['d34inc_calu'])} | {fmt(c['u1'])} | "
                f"{fmt(c8['rs'])} / {fmt(c8['rsinc'])} |"
            )

    pr()
    pr(
        "## 4. The log checks and U1 on the v3 run's own executions (one session; D37 needs k sessions)"
    )
    pr()
    for line in v3_section():
        pr(line)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SIMS = int(sys.argv[1])
    main()
