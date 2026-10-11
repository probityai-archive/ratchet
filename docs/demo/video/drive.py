"""Run the steps of a recording for real, inside the terminal that asciinema is capturing.

Every command is executed (nothing is replayed from a file): the driver only paces the typing so the
viewer can read the command, then lets the command write to the terminal itself. It writes a timeline
(JSON) of when each step started and ended, which `render.py` uses to place captions.

    drive.py STEPS.toml [--ids FILE] [--variant 45] [--timeline OUT.json] [--check | --warm]

`--check` resolves every {{NAME}} in the steps and exits 2 if any is missing, without running anything.
A step is a table in the TOML file:

    [[step]]
    id = "matrix"            # unique
    cmd = "ratchet replay {{REJECTED}} {{CLEARED}}"
    cmd_45 = "..."           # optional: a different command for the narrow 4:5 terminal
    warm = true              # run once silently before recording (false if it has a side effect)
    hold = 3.0               # seconds the finished screen stays up
    expect_rc = [0]          # exit codes that count as correct (default [0])
    timeout = 90             # seconds before the command is killed (default 120)
    captions = [ { at = 0.0, text = "..." } ]   # at = seconds after the step starts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

SENTINEL = "\x1b]0;ratchet-demo-start\x07"
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
PROMPT = "\x1b[1;32m$\x1b[0m "
TYPE_DELAY = 0.028


def load_ids(path: str | None) -> dict[str, str]:
    """KEY=VALUE lines; '#' starts a comment. The process environment fills any key the file lacks."""
    ids: dict[str, str] = {}
    if path:
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                ids[key.strip()] = value.strip().strip("'\"")
    return ids


def resolve(text: str, ids: dict[str, str]) -> tuple[str, list[str]]:
    missing: list[str] = []

    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        value = ids.get(key) or os.environ.get(key)
        if not value:
            missing.append(key)
            return match.group(0)
        return value

    return PLACEHOLDER.sub(sub, text), missing


def load_steps(
    path: Path, ids: dict[str, str], variant: str | None
) -> tuple[list[dict[str, Any]], list[str]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    steps = data.get("step", [])
    if not steps:
        raise SystemExit(f"{path}: no [[step]] tables")
    missing: list[str] = []
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for step in steps:
        if step["id"] in seen:
            raise SystemExit(f"{path}: duplicate step id {step['id']!r}")
        seen.add(step["id"])
        cmd, miss = resolve(step.get(f"cmd_{variant}", step["cmd"]) if variant else step["cmd"], ids)
        missing += miss
        captions = []
        for cap in step.get("captions", []):
            text, miss = resolve(cap["text"], ids)
            missing += miss
            captions.append({**cap, "text": text})
        out.append({**step, "cmd": cmd, "captions": captions})
    return out, sorted(set(missing))


def type_out(text: str) -> None:
    sys.stdout.write(PROMPT)
    sys.stdout.flush()
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(TYPE_DELAY)
    time.sleep(0.35)
    sys.stdout.write("\r\n")
    sys.stdout.flush()


def run_step(step: dict[str, Any], env: dict[str, str], origin: float) -> dict[str, Any]:
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()
    started = time.monotonic()
    type_out(step["cmd"])
    ran = time.monotonic()
    try:
        rc = subprocess.run(
            step["cmd"], shell=True, env=env, timeout=float(step.get("timeout", 120))
        ).returncode
        status = "ok" if rc in step.get("expect_rc", [0]) else "unexpected_rc"
    except subprocess.TimeoutExpired:
        rc, status = -1, "timeout"
    done = time.monotonic()
    time.sleep(float(step.get("hold", 2.0)))
    ended = time.monotonic()
    return {
        "id": step["id"],
        "cmd": step["cmd"],
        "rc": rc,
        "status": status,
        "t_start": started - origin,
        "t_run": ran - origin,
        "t_done": done - origin,
        "t_end": ended - origin,
        "captions": step.get("captions", []),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", type=Path)
    ap.add_argument("--ids")
    ap.add_argument("--timeline", type=Path)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--variant", help="use cmd_<variant> where a step has one (the narrow 4:5 cut uses '45')")
    ap.add_argument("--warm", action="store_true", help="run each command once, silently, to warm caches")
    args = ap.parse_args()

    steps, missing = load_steps(args.steps, load_ids(args.ids), args.variant)
    if missing:
        print(
            "unresolved placeholders (fill them from measured results): " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    if args.check:
        print(f"{len(steps)} steps, all placeholders resolved")
        return 0
    if args.warm:
        for step in steps:
            if step.get("warm", True):
                subprocess.run(
                    step["cmd"],
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=float(step.get("timeout", 120)),
                    check=False,
                )
        print(f"warmed {len(steps)} commands")
        return 0
    if args.timeline is None:
        print("--timeline is required unless --check", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["PS1"] = "$ "
    env.pop("NO_COLOR", None)
    sys.stdout.write(SENTINEL)
    sys.stdout.flush()
    origin = time.monotonic()
    started_utc = time.time()
    results = [run_step(s, env, origin) for s in steps]
    time.sleep(0.5)
    timeline = {"started_utc": started_utc, "total_s": time.monotonic() - origin, "steps": results}
    args.timeline.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
