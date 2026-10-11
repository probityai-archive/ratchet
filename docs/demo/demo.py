"""Presenter's console for the 3-minute demo: live commands that fall back to labelled recorded output.

    demo.py list                    the steps, their time windows and whether a recording exists
    demo.py preflight               check environment, store, placeholders and recording freshness
    demo.py snapshot [ID ...]       run steps now and store their output (all steps when none named;
                                    a step marked side_effect runs only when named)
    demo.py run ID                  run a step live; if it fails or times out, show the stored output
    demo.py recorded ID             show the stored output only
    demo.py follow FILE             print FILE as it grows (ctrl-c or 25 s ends it)
    demo.py evolve-bg [N]           start `ratchet evolve -n N` in the background (log: out/evolve.log)

Every screen this prints carries a one-line label: LIVE with the time, or RECORDED with the time the
stored output was captured and why the live run was not used. A recording is never shown unlabelled.
Run from the repository root with the virtualenv active (so `ratchet` resolves).
"""

from __future__ import annotations

import fcntl
import json
import os
import pty
import re
import select
import shutil
import struct
import subprocess
import sys
import termios
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STORE = Path(os.environ.get("RATCHET_DEMO_RECORDED", HERE / "recorded"))
OUT = HERE / "out"
PT = ZoneInfo("America/Los_Angeles")
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
FRESH_HOURS = 3.0
DIM, BOLD_YELLOW, BOLD_GREEN, RESET = "\x1b[2m", "\x1b[1;33m", "\x1b[1;32m", "\x1b[0m"


def stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, PT).strftime("%Y-%m-%d %H:%M:%S PT")


def load_ids() -> dict[str, str]:
    ids: dict[str, str] = {}
    path = HERE / "ids.env"
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                ids[key.strip()] = value.strip().strip("'\"")
    return ids


def resolve(text: str, ids: dict[str, str]) -> tuple[str, list[str]]:
    missing: list[str] = []

    def sub(m: re.Match[str]) -> str:
        value = ids.get(m.group(1)) or os.environ.get(m.group(1))
        if not value:
            missing.append(m.group(1))
            return m.group(0)
        return value

    return PLACEHOLDER.sub(sub, text), missing


def steps() -> list[dict[str, Any]]:
    return list(
        tomllib.loads(
            Path(os.environ.get("RATCHET_DEMO_STEPS", HERE / "steps.demo.toml")).read_text(encoding="utf-8")
        )["step"]
    )


def find(step_id: str) -> dict[str, Any]:
    for s in steps():
        if s["id"] == step_id:
            return s
    raise SystemExit(f"no step {step_id!r}; try `demo.py list`")


def meta_path(step_id: str) -> Path:
    return STORE / f"{step_id}.json"


def load_recording(step_id: str) -> tuple[bytes, dict[str, Any]] | None:
    meta, raw = meta_path(step_id), STORE / f"{step_id}.ansi"
    if not (meta.exists() and raw.exists()):
        return None
    return raw.read_bytes(), json.loads(meta.read_text(encoding="utf-8"))


def capture(cmd: str, timeout: float, cols: int, rows: int) -> tuple[bytes, int | None]:
    """Run cmd on a pseudo-terminal of the given size and keep its raw output, colours included."""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    env = {**os.environ, "TERM": "xterm-256color", "COLUMNS": str(cols), "LINES": str(rows)}
    proc = subprocess.Popen(cmd, shell=True, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    rc: int | None = None
    while True:
        if time.monotonic() > deadline:
            proc.kill()
            proc.wait()
            break
        ready, _, _ = select.select([master], [], [], 0.2)
        if ready:
            try:
                data = os.read(master, 65536)
            except OSError:
                data = b""
            if data:
                chunks.append(data)
                continue
        if proc.poll() is not None:
            rc = proc.returncode
            break
    os.close(master)
    return b"".join(chunks), rc


def snapshot(step: dict[str, Any], ids: dict[str, str]) -> bool:
    cmd, missing = resolve(step.get("snap", step["cmd"]), ids)
    if missing:
        print(f"  {step['id']:<12} SKIPPED, unresolved: {', '.join(missing)}")
        return False
    size = shutil.get_terminal_size((120, 36))
    started = time.time()
    data, rc = capture(cmd, float(step.get("timeout", 60)) * 2, size.columns, size.lines)
    if rc != 0:
        print(f"  {step['id']:<12} FAILED rc={rc}; the previous recording (if any) is kept")
        return False
    STORE.mkdir(exist_ok=True)
    (STORE / f"{step['id']}.ansi").write_bytes(data)
    meta = {
        "id": step["id"],
        "cmd": cmd,
        "captured_at": started,
        "captured": stamp(started),
        "bytes": len(data),
    }
    meta_path(step["id"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  {step['id']:<12} recorded {meta['captured']}  {len(data)} bytes")
    return True


def label(text: str, style: str) -> None:
    sys.stdout.write(f"{style}{text}{RESET}\n")
    sys.stdout.flush()


def show_recorded(step: dict[str, Any], why: str) -> int:
    found = load_recording(step["id"])
    if found is None:
        label(f"[NO RECORDING for {step['id']}] live run failed: {why}. Run `demo.py snapshot`.", BOLD_YELLOW)
        return 1
    data, meta = found
    label(f"[RECORDED {meta['captured']}] stored output of: {meta['cmd']}  ({why})", BOLD_YELLOW)
    sys.stdout.flush()
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    return 0


def run(step_id: str) -> int:
    step = find(step_id)
    cmd, missing = resolve(step["cmd"], load_ids())
    if missing:
        return show_recorded(step, f"unresolved {', '.join(missing)}")
    mark = f" {step['mark']}" if step.get("mark") else ""
    label(f"[LIVE{mark} {stamp(time.time())}] {cmd}", BOLD_GREEN)
    try:
        rc = subprocess.run(cmd, shell=True, timeout=float(step.get("timeout", 60))).returncode
    except subprocess.TimeoutExpired:
        return show_recorded(step, f"live run exceeded {step.get('timeout', 60)} s")
    if rc != 0:
        return show_recorded(step, f"live run exited {rc}")
    return 0


def follow(path: str) -> int:
    target = Path(path)
    if not target.exists():
        print(f"{path} does not exist: start `demo.py evolve-bg` first", file=sys.stderr)
        return 1
    deadline = time.monotonic() + 25
    with target.open("rb") as fh:
        fh.seek(max(0, target.stat().st_size - 6000))
        while time.monotonic() < deadline:
            chunk = fh.read(4096)
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            else:
                time.sleep(0.2)
    return 0


def evolve_bg(generations: int) -> int:
    OUT.mkdir(exist_ok=True)
    log = OUT / "evolve.log"
    if log.exists() and log.stat().st_size and not (OUT / "evolve.done").exists():
        print(
            f"{log} exists and no evolve.done marker: an earlier run may still be going. "
            "Not starting a second."
        )
        return 1
    (OUT / "evolve.done").unlink(missing_ok=True)
    done = OUT / "evolve.done"
    script = (
        f"timeout 1500 ratchet evolve -n {generations} > {log} 2>&1; "
        f"rc=$?; echo EXIT=$rc >> {log}; touch {done}"
    )
    subprocess.Popen(["sh", "-c", script], stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"evolve started at {stamp(time.time())}; log {log}; deadline 25 min; marker {OUT / 'evolve.done'}")
    return 0


def list_steps() -> int:
    ids = load_ids()
    for s in steps():
        _, missing = resolve(s["cmd"], ids)
        rec = load_recording(s["id"])
        state = f"recorded {rec[1]['captured']}" if rec else "no recording"
        print(
            f"{s['window']:<10} {s['id']:<12} {state:<34} "
            f"{'UNRESOLVED ' + ','.join(missing) if missing else ''}"
        )
    return 0


def preflight() -> int:
    ids, bad = load_ids(), 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        bad += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    check("ratchet on PATH", shutil.which("ratchet") is not None)
    check("RATCHET_MONGODB_URI set", bool(os.environ.get("RATCHET_MONGODB_URI")))
    for var in ("RATCHET_BASE_URL", "RATCHET_API_KEY", "RATCHET_MODEL", "RATCHET_IMPROVER_URI"):
        check(f"{var} set", bool(os.environ.get(var)))
    check("docs/demo/ids.env exists (run docs/demo/derive_ids.py)", (HERE / "ids.env").exists())
    live = subprocess.run("ratchet show lineage", shell=True, capture_output=True, timeout=30, check=False)
    check("store answers `ratchet show lineage`", live.returncode == 0, f"rc={live.returncode}")
    now = time.time()
    for s in steps():
        _, missing = resolve(s["cmd"], ids)
        rec = load_recording(s["id"])
        age = (now - rec[1]["captured_at"]) / 3600 if rec else None
        check(f"step {s['id']}: placeholders", not missing, ",".join(missing))
        fresh = age is not None and age < FRESH_HOURS
        detail = "none" if age is None else f"{age:.1f} h old"
        if s.get("side_effect") and not fresh:
            print(
                f"WARN  step {s['id']}: no fresh recording ({detail}); "
                "it has a side effect, so a bare snapshot skips it"
            )
        else:
            check(f"step {s['id']}: fresh recording", fresh, detail)
    print(f"\n{bad} failing check(s)")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    os.chdir(REPO)
    if cmd == "list":
        return list_steps()
    if cmd == "preflight":
        return preflight()
    if cmd == "snapshot":
        ids = load_ids()
        chosen = [find(i) for i in rest] if rest else [s for s in steps() if not s.get("side_effect")]
        results = [snapshot(s, ids) for s in chosen]
        return 0 if all(results) else 1
    if cmd in ("run", "recorded") and rest:
        return run(rest[0]) if cmd == "run" else show_recorded(find(rest[0]), "shown on request")
    if cmd == "follow" and rest:
        return follow(rest[0])
    if cmd == "evolve-bg":
        return evolve_bg(int(rest[0]) if rest else 1)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
