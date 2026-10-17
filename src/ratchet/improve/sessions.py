"""The k-session admission rule (ledger D37, P1 Appendix F): protocol, single-host runner, bundle and test.

Protocol. k independent sessions; in each, the parent, two identity arms and the candidate are fresh arms
that run n executions of every case, all randomly interleaved. Every execution's endpoint is drawn
uniformly at random from the session's pool and recorded, so routing cannot follow the candidate. Each arm
draws its seeds from its own stream, so no seed is shared between arms at a (case, rep). A lost execution
is rerun once, in a randomly placed later slot of its session's schedule, before any outcome is read.

Test. For session s and unchanged arm j, D_sj = sum over cases of (candidate passes - arm j passes) / n and
t_j = mean_s(D_sj) / (sd_s(D_sj) / sqrt(k)). The candidate is admitted only if the smallest t_j is at least
t(0.95, k - 1), no sample SD is zero, the harm screen passes (per case, a one-sided exact test of the
candidate's kn runs against the unchanged arms' 3kn at 0.05), every execution has a valid outcome, and none
of the log checks fires:

  E   endpoint mix: Pearson chi-square homogeneity of the candidate's executions per endpoint against the
      unchanged arms', every session summed into one table, empty endpoints dropped; fires at p < 0.01.
  T*  completion time: per session, the two-sided Mann-Whitney z of the candidate's completion times against
      the unchanged arms' (normal approximation, tie-corrected, no continuity correction), combined as
      sum / sqrt(k); it fires when that p < 0.01 AND the unchanged arms' outcomes trend with dispatch time
      (per session and case, sum (t - mean t)(y - mean y) over the square root of its permutation variance,
      summed over strata) at two-sided p < 0.05.
  P   pairing: per pair of unchanged arms, the excess of same-(case, rep) both-pass slots over the
      hypergeometric expectation, summed over cases and sessions, over the root of the summed variance;
      fires when the largest of the three reaches z(1 - 0.01 / 3).

Any failed log check, invalid execution or zero SD makes the comparison inconclusive, never a rejection.
There is no padding controller. Everything in the verdict is recomputed from the session bundle by
`evaluate_bundle`, offline, with no package beyond this one.

    python -m ratchet.improve.sessions verify BUNDLE_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ratchet_cases.types import sha256_hex

from ratchet.canonical import canonical_json
from ratchet.improve import _dist

SCHEMA = "ratchet-k-session-bundle/v1"
UNCHANGED = ("parent", "identity_1", "identity_2")
ARMS = (*UNCHANGED, "candidate")
# The frozen D37 constants, as pre-registered (scipy's quantiles). `check_constants` holds every bundle's
# declared values to these and to this module's own quantiles.
D37_T_CRITICAL_K8 = 1.894578605061305
D37_Z_PAIRING = 2.7130518884727213
ALPHA_GAIN = 0.05
ALPHA_HARM = 0.05
ALPHA_ENDPOINT = 0.01
ALPHA_TIME = 0.01
ALPHA_TREND = 0.05
ALPHA_PAIRING = 0.01
CONSTANT_TOLERANCE = 1e-9


class BundleError(ValueError):
    """The bundle is malformed or inconsistent with its own manifest; no verdict is read from it."""


@dataclass(frozen=True)
class Protocol:
    """Everything fixed before the first execution. Its canonical JSON digest names the run."""

    label: str
    cases: tuple[str, ...]
    arms: Mapping[str, str]  # arm name -> harness version
    k: int = 8
    n: int = 5
    t_critical: float = D37_T_CRITICAL_K8
    z_pairing: float = D37_Z_PAIRING

    def __post_init__(self) -> None:
        if set(self.arms) != set(ARMS):
            raise ValueError(f"arms must name exactly {ARMS}")
        if self.k < 2 or self.n < 2:
            raise ValueError("k and n must each be at least 2")
        if not self.cases or len(set(self.cases)) != len(self.cases):
            raise ValueError("cases must be a nonempty list of distinct ids")
        check_constants(self.k, self.t_critical, self.z_pairing)

    @classmethod
    def for_k(
        cls, label: str, cases: Sequence[str], arms: Mapping[str, str], k: int = 8, n: int = 5
    ) -> Protocol:
        """A protocol at the frozen D37 constants, or at this module's quantiles when k is not 8."""
        t_crit = D37_T_CRITICAL_K8 if k == 8 else _dist.t_isf(ALPHA_GAIN, k - 1)
        return cls(label, tuple(cases), dict(arms), k, n, t_crit, D37_Z_PAIRING)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cases": list(self.cases),
            "arms": dict(self.arms),
            "k": self.k,
            "n": self.n,
            "t_critical": self.t_critical,
            "z_pairing": self.z_pairing,
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Protocol:
        try:
            return cls(
                str(doc["label"]),
                tuple(str(c) for c in doc["cases"]),
                {str(a): str(v) for a, v in doc["arms"].items()},
                int(doc["k"]),
                int(doc["n"]),
                float(doc["t_critical"]),
                float(doc["z_pairing"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise BundleError(f"protocol: {exc}") from exc

    @property
    def digest(self) -> str:
        return sha256_hex(canonical_json(self.to_json()))


def check_constants(k: int, t_critical: float, z_pairing: float) -> None:
    """Refuse a declared threshold that is not the test's own: a bundle cannot carry a laxer rule."""
    want_t = _dist.t_isf(ALPHA_GAIN, k - 1)
    if abs(t_critical - want_t) > CONSTANT_TOLERANCE:
        raise ValueError(f"t_critical {t_critical!r} is not t(0.95, {k - 1}) = {want_t!r}")
    want_z = _dist.norm_isf(ALPHA_PAIRING / 3)
    if abs(z_pairing - want_z) > CONSTANT_TOLERANCE:
        raise ValueError(f"z_pairing {z_pairing!r} is not z(1 - 0.01 / 3) = {want_z!r}")


def _u64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def arm_seed(protocol: Protocol, session: int, arm: str, rep: int) -> str:
    """The seed of (session, arm, rep): the first 8 bytes of SHA-256 of `<label>/<session>/<arm>/<rep>`.

    It is written as a decimal string because it can exceed 2**53, past which a JSON number loses digits.
    """
    return str(_u64(f"{protocol.label}/{session}/{arm}/{rep}"))


@dataclass(frozen=True)
class Slot:
    """One scheduled execution: its position in the session's schedule and the endpoint it was assigned."""

    session: int
    position: int
    arm: str
    case_id: str
    rep: int
    seed: str
    endpoint: str
    attempt: int = 1


def session_schedule(protocol: Protocol, session: int, endpoints: Sequence[str]) -> list[Slot]:
    """Every (arm, case, rep) once, ordered and given endpoints by the session's predeclared seed."""
    if not endpoints:
        raise ValueError("a session needs at least one endpoint")
    rng = random.Random(_u64(f"{protocol.label}/{session}/schedule"))
    order = [(a, c, r) for r in range(protocol.n) for c in protocol.cases for a in ARMS]
    rng.shuffle(order)
    return [
        Slot(session, i, a, c, r, arm_seed(protocol, session, a, r), endpoints[rng.randrange(len(endpoints))])
        for i, (a, c, r) in enumerate(order)
    ]


def schedule_digest(slots: Sequence[Slot]) -> str:
    return sha256_hex(canonical_json([asdict(s) for s in slots]))


@dataclass(frozen=True)
class Outcome:
    """What the executor reports for one slot. `passed` is None when the execution was lost or invalid."""

    passed: bool | None
    run_id: str = ""
    note: str = ""


Executor = Callable[[Slot], Outcome]


@dataclass(frozen=True)
class ExecutionRecord:
    """One attempt as the bundle stores it. Times are seconds since the session started."""

    session: int
    position: int
    arm: str
    case_id: str
    rep: int
    seed: str
    endpoint: str
    attempt: int
    dispatched_s: float
    completed_s: float
    passed: bool | None
    run_id: str = ""
    note: str = ""


def run_session(
    protocol: Protocol,
    session: int,
    execute: Executor,
    endpoints: Sequence[str],
    *,
    workers: int = 1,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[Slot], list[ExecutionRecord]]:
    """Run one session's schedule. A lost execution is rerun once, in a slot drawn at random from the rest of
    the schedule (the session's own seed places it), and no outcome is read until the session ends.

    With `workers` above 1 the schedule is dispatched in order to a pool of that size, so executions overlap
    as they would on a serving pool; the record keeps each one's dispatch and completion time.
    """
    slots = session_schedule(protocol, session, endpoints)
    rng = random.Random(_u64(f"{protocol.label}/{session}/reruns"))
    start = clock()
    queue: list[Slot] = list(slots)
    records: list[ExecutionRecord] = []

    def one(slot: Slot) -> ExecutionRecord:
        dispatched = clock() - start
        try:
            out = execute(slot)
        except Exception as exc:  # an executor failure is a lost execution, recorded, never raised
            out = Outcome(None, note=f"executor error: {type(exc).__name__}: {exc}"[:300])
        return ExecutionRecord(**asdict(slot), dispatched_s=round(dispatched, 6),
                               completed_s=round(clock() - start, 6), passed=out.passed, run_id=out.run_id,
                               note=out.note)  # fmt: skip

    position = len(slots)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        i = 0
        while i < len(queue):
            batch = queue[i : i + max(1, workers)]
            i += len(batch)
            for rec in pool.map(one, batch):
                records.append(rec)
                if rec.passed is None and rec.attempt == 1:
                    retry = Slot(rec.session, position, rec.arm, rec.case_id, rec.rep, rec.seed,
                                 endpoints[rng.randrange(len(endpoints))], attempt=2)  # fmt: skip
                    position += 1
                    queue.insert(rng.randint(i, len(queue)), retry)
    return slots, records


def run_protocol(
    protocol: Protocol,
    execute: Executor,
    out_dir: Path,
    *,
    endpoints_for: Callable[[int], Sequence[str]],
    workers: int = 1,
    on_session_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run all k sessions on one host (or in one Slurm allocation), one after another; write the bundle.

    `endpoints_for(s)` names session s's serving pool; `on_session_start(s)` is called before it starts,
    which is where a deployment that serves fresh per session is restarted. The bundle holds `manifest.json`,
    `executions.jsonl` (every attempt, in completion order per session) and `verdict.json`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions: list[dict[str, Any]] = []
    with (out_dir / "executions.jsonl").open("w", encoding="utf-8") as log:
        for s in range(protocol.k):
            if on_session_start is not None:
                on_session_start(s)
            pool = list(endpoints_for(s))
            slots, records = run_session(protocol, s, execute, pool, workers=workers)
            for rec in records:
                log.write(canonical_json(asdict(rec)).decode("utf-8") + "\n")
            log.flush()
            sessions.append({"session": s, "endpoints": pool, "schedule_digest": schedule_digest(slots)})
    manifest = {"schema": SCHEMA, "protocol": protocol.to_json(), "protocol_digest": protocol.digest,
                "sessions": sessions}  # fmt: skip
    (out_dir / "manifest.json").write_bytes(canonical_json(manifest))
    verdict = evaluate_bundle(out_dir)
    (out_dir / "verdict.json").write_bytes(canonical_json(verdict))
    return verdict


# ---------------------------------------------------------------------------------------------------------
# The test, from the bundle alone
# ---------------------------------------------------------------------------------------------------------


@dataclass
class _Cell:
    """The final valid outcome of one (session, arm, case, rep), with its times and endpoint."""

    passed: bool
    dispatched_s: float
    completed_s: float
    endpoint: str


@dataclass
class Data:
    """The bundle read into the shape the test needs; every cell is the attempt that counts."""

    protocol: Protocol
    cells: dict[tuple[int, str, str, int], _Cell] = field(default_factory=dict)
    invalid: list[str] = field(default_factory=list)
    attempts: int = 0
    reruns: int = 0


def _read_records(path: Path) -> list[ExecutionRecord]:
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(ExecutionRecord(**json.loads(line)))
        except (TypeError, json.JSONDecodeError) as exc:
            raise BundleError(f"executions.jsonl line {n}: {exc}") from exc
    return out


def load_bundle(bundle: Path) -> Data:
    """Read and cross-check a bundle: the protocol digest, each session's schedule digest from its own
    predeclared seed and recorded endpoints, and one first attempt per slot with at most one rerun."""
    try:
        manifest = json.loads((bundle / "manifest.json").read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"manifest.json: {exc}") from exc
    if manifest.get("schema") != SCHEMA:
        raise BundleError(f"unsupported bundle schema {manifest.get('schema')!r}")
    protocol = Protocol.from_json(manifest.get("protocol", {}))
    if manifest.get("protocol_digest") != protocol.digest:
        raise BundleError("protocol_digest does not match the protocol")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or [s.get("session") for s in sessions] != list(range(protocol.k)):
        raise BundleError("manifest must list sessions 0 to k - 1 in order")
    expected: dict[tuple[int, str, str, int], Slot] = {}
    for s in sessions:
        slots = session_schedule(protocol, s["session"], s["endpoints"])
        if schedule_digest(slots) != s.get("schedule_digest"):
            raise BundleError(f"session {s['session']}: schedule is not the predeclared one")
        expected.update({(x.session, x.arm, x.case_id, x.rep): x for x in slots})
    data = Data(protocol)
    first: dict[tuple[int, str, str, int], ExecutionRecord] = {}
    second: dict[tuple[int, str, str, int], ExecutionRecord] = {}
    for rec in _read_records(bundle / "executions.jsonl"):
        key = (rec.session, rec.arm, rec.case_id, rec.rep)
        slot = expected.get(key)
        if slot is None or rec.seed != slot.seed:
            raise BundleError(f"execution {key} is not in the schedule or carries another seed")
        target = first if rec.attempt == 1 else second if rec.attempt == 2 else None
        if target is None or key in target:
            raise BundleError(f"execution {key}: attempt {rec.attempt} is repeated or not allowed")
        if rec.attempt == 1 and (rec.position != slot.position or rec.endpoint != slot.endpoint):
            raise BundleError(f"execution {key}: position or endpoint differs from the schedule")
        target[key] = rec
        data.attempts += 1
    data.reruns = len(second)
    for key in expected:
        first_try = first.get(key)
        if first_try is None:
            data.invalid.append(f"{key}: never run")
            continue
        counted = first_try
        if first_try.passed is None:
            rerun = second.get(key)
            if rerun is None or rerun.passed is None:
                data.invalid.append(f"{key}: lost twice" if rerun is not None else f"{key}: lost, not rerun")
                continue
            counted = rerun
        elif key in second:
            raise BundleError(f"execution {key}: a rerun of a valid execution")
        data.cells[key] = _Cell(
            bool(counted.passed), counted.dispatched_s, counted.completed_s, counted.endpoint
        )
    return data


def _count(data: Data, s: int, arm: str, case: str) -> int:
    return sum(1 for r in range(data.protocol.n) if data.cells[(s, arm, case, r)].passed)


def _hypergeom_low_tail(a: int, b: int, n1: int, n2: int) -> float:
    """P(X <= a) for X the first count, conditional on the total a + b, n1 against n2 runs (exact)."""
    t = a + b
    lo = max(0, t - n2)
    num = sum(math.comb(n1, x) * math.comb(n2, t - x) for x in range(lo, a + 1))
    return num / math.comb(n1 + n2, t)


def _mann_whitney_z(cand: Sequence[float], ref: Sequence[float]) -> float:
    pooled = sorted([(v, 1) for v in cand] + [(v, 0) for v in ref])
    n1, n2 = len(cand), len(ref)
    total = n1 + n2
    rank_sum = 0.0
    ties = 0.0
    i = 0
    while i < total:
        j = i
        while j + 1 < total and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        rank_sum += mid * sum(flag for _, flag in pooled[i : j + 1])
        size = j - i + 1
        ties += size**3 - size
        i = j + 1
    u = rank_sum - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((total + 1) - ties / (total * (total - 1)))
    return (u - n1 * n2 / 2.0) / math.sqrt(var) if var > 0 else 0.0


def _endpoint_p(data: Data) -> float:
    cand: dict[str, int] = {}
    ref: dict[str, int] = {}
    for (_, arm, _, _), cell in data.cells.items():
        side = cand if arm == "candidate" else ref
        side[cell.endpoint] = side.get(cell.endpoint, 0) + 1
    cols = sorted(set(cand) | set(ref))
    rows = ([cand.get(e, 0) for e in cols], [ref.get(e, 0) for e in cols])
    total = sum(map(sum, rows))
    x2 = 0.0
    for row in rows:
        rsum = sum(row)
        for j, e in enumerate(cols):
            exp = rsum * (cand.get(e, 0) + ref.get(e, 0)) / total
            x2 += (row[j] - exp) ** 2 / exp if exp > 0 else 0.0
    return _dist.chi2_sf(x2, max(len(cols) - 1, 1))


def _time_p(data: Data) -> float:
    p = data.protocol
    zs = []
    for s in range(p.k):
        cand = [c.completed_s for (ss, a, _, _), c in data.cells.items() if ss == s and a == "candidate"]
        ref = [c.completed_s for (ss, a, _, _), c in data.cells.items() if ss == s and a != "candidate"]
        zs.append(_mann_whitney_z(cand, ref))
    return 2.0 * _dist.norm_sf(abs(sum(zs) / math.sqrt(p.k)))


def _trend_p(data: Data) -> float:
    p = data.protocol
    num = den = 0.0
    for s in range(p.k):
        for case in p.cases:
            pts = [(data.cells[(s, a, case, r)].dispatched_s, float(data.cells[(s, a, case, r)].passed))
                   for a in UNCHANGED for r in range(p.n)]  # fmt: skip
            m = len(pts)
            tbar = sum(t for t, _ in pts) / m
            ybar = sum(y for _, y in pts) / m
            num += sum((t - tbar) * (y - ybar) for t, y in pts)
            den += sum((t - tbar) ** 2 for t, _ in pts) * sum((y - ybar) ** 2 for _, y in pts) / (m - 1)
    z = num / math.sqrt(den) if den > 0 else 0.0
    return 2.0 * _dist.norm_sf(abs(z))


def _pairing_z(data: Data) -> float:
    p = data.protocol
    best = -math.inf
    for a, b in ((0, 1), (0, 2), (1, 2)):
        num = den = 0.0
        for s in range(p.k):
            for case in p.cases:
                xa = [data.cells[(s, UNCHANGED[a], case, r)].passed for r in range(p.n)]
                xb = [data.cells[(s, UNCHANGED[b], case, r)].passed for r in range(p.n)]
                ca, cb = sum(xa), sum(xb)
                both = sum(1 for u, v in zip(xa, xb, strict=True) if u and v)
                num += both - ca * cb / p.n
                den += ca * cb * (p.n - ca) * (p.n - cb) / (p.n * p.n * (p.n - 1.0))
        best = max(best, num / math.sqrt(den) if den > 0 else 0.0)
    return best


def evaluate(data: Data) -> dict[str, Any]:
    """The D37 verdict and every statistic it rests on, from the read bundle."""
    p = data.protocol
    out: dict[str, Any] = {"protocol_digest": p.digest, "attempts": data.attempts, "reruns": data.reruns}
    if data.invalid:
        return out | {"verdict": "inconclusive", "reasons": ["invalid_execution"], "invalid": data.invalid}
    diffs: dict[str, list[float]] = {}
    t_stats: dict[str, float | None] = {}
    degenerate = False
    for ref in UNCHANGED:
        d = [sum(_count(data, s, "candidate", c) - _count(data, s, ref, c) for c in p.cases) / p.n
             for s in range(p.k)]  # fmt: skip
        mean = sum(d) / p.k
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (p.k - 1))
        diffs[ref] = d
        t_stats[ref] = mean / (sd / math.sqrt(p.k)) if sd > 0 else None
        degenerate |= sd == 0
    harm = {}
    for c in p.cases:
        cand = sum(_count(data, s, "candidate", c) for s in range(p.k))
        pooled = sum(_count(data, s, a, c) for s in range(p.k) for a in UNCHANGED)
        harm[c] = _hypergeom_low_tail(cand, pooled, p.k * p.n, 3 * p.k * p.n)
    harmed = sorted(c for c, v in harm.items() if v <= ALPHA_HARM)
    p_e, p_t, p_tr, z_p = _endpoint_p(data), _time_p(data), _trend_p(data), _pairing_z(data)
    checks = {
        "E": {"p": p_e, "fires": p_e < ALPHA_ENDPOINT},
        "T*": {"p_time": p_t, "p_trend": p_tr, "fires": p_t < ALPHA_TIME and p_tr < ALPHA_TREND},
        "P": {"z": z_p, "fires": z_p >= p.z_pairing},
    }
    smallest = min((t for t in t_stats.values() if t is not None), default=None)
    gain = not degenerate and smallest is not None and smallest >= p.t_critical
    reasons = [f"log_check_{n}" for n, v in checks.items() if v["fires"]]
    if degenerate:
        reasons.append("zero_sd")
    out |= {
        "differences": diffs,
        "t": t_stats,
        "t_min": smallest,
        "t_critical": p.t_critical,
        "harm_screen": {"p": harm, "harmed_cases": harmed},
        "log_checks": checks,
    }
    if reasons:
        return out | {"verdict": "inconclusive", "reasons": reasons}
    if not gain:
        return out | {"verdict": "not_admitted", "reasons": ["no_gain_against_every_unchanged_arm"]}
    if harmed:
        return out | {"verdict": "not_admitted", "reasons": ["harm_screen"]}
    return out | {"verdict": "admitted", "reasons": []}


def evaluate_bundle(bundle: Path) -> dict[str, Any]:
    return evaluate(load_bundle(bundle))


def verify_bundle(bundle: Path) -> tuple[bool, dict[str, Any]]:
    """Recompute the verdict and compare it with the bundle's `verdict.json`, field for field."""
    fresh = evaluate_bundle(bundle)
    try:
        stored = json.loads((bundle / "verdict.json").read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"verdict.json: {exc}") from exc
    return canonical_json(stored) == canonical_json(fresh), fresh


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ratchet.improve.sessions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    vf = sub.add_parser("verify", help="recompute a session bundle's verdict offline and compare it")
    vf.add_argument("bundle", type=Path)
    a = ap.parse_args(argv)
    try:
        same, fresh = verify_bundle(a.bundle)
    except BundleError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    summary = {"matches_stored_verdict": same, "verdict": fresh["verdict"], "reasons": fresh["reasons"],
               "t_min": fresh.get("t_min"), "t_critical": fresh.get("t_critical")}  # fmt: skip
    print(json.dumps(summary, indent=2))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
