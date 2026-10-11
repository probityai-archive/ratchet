"""Gate a rendered video before it is used: `check.py OUT_DIR --format 169|45 [--submission]`.

Fails (exit 1) when any of these is false, and prints every failure, not just the first:
  - the video is h264, yuv420p, 30 fps, the expected canvas, faststart (moov before mdat), under 100 MB
  - the duration is at most 60.0 s (submission) or the manifest says it is a labelled pipeline test
  - every driver step exited with an expected code (no unexpected_rc, no timeout)
  - no caption holds an unresolved {{PLACEHOLDER}} and none is longer than 120 characters
  - the manifest's sha256 values match the files on disk
  - with --submission: the manifest has no test label
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED = {"169": (1920, 1080), "45": (1080, 1350)}


def probe(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    stream: dict[str, Any] = json.loads(out)["streams"][0]
    return stream


def faststart(path: Path) -> bool:
    head = path.read_bytes()[:1_000_000]
    moov, mdat = head.find(b"moov"), head.find(b"mdat")
    return moov != -1 and (mdat == -1 or moov < mdat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--format", choices=sorted(EXPECTED), required=True)
    ap.add_argument("--submission", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((args.out / f"manifest-{args.format}.json").read_text(encoding="utf-8"))
    mp4 = args.out / f"ratchet-{args.format}.mp4"
    failures: list[str] = []
    s = probe(mp4)
    if s["codec_name"] != "h264" or s["pix_fmt"] != "yuv420p":
        failures.append(f"codec/pixel format {s['codec_name']}/{s['pix_fmt']}, want h264/yuv420p")
    if s["r_frame_rate"] != "30/1":
        failures.append(f"frame rate {s['r_frame_rate']}, want 30/1")
    if (int(s["width"]), int(s["height"])) != EXPECTED[args.format]:
        failures.append(f"canvas {s['width']}x{s['height']}, want {EXPECTED[args.format]}")
    if not faststart(mp4):
        failures.append("moov atom is not before mdat (no faststart)")
    if mp4.stat().st_size > 100_000_000:
        failures.append(f"file is {mp4.stat().st_size} bytes, over 100 MB")
    if manifest["duration_s"] > 60.0:
        failures.append(f"duration {manifest['duration_s']} s is over the 60 s limit")
    for step in manifest["steps"]:
        if step["status"] != "ok":
            failures.append(f"step {step['id']} ended {step['status']} (rc {step['rc']})")
    for text in manifest["captions"]:
        if "{{" in text:
            failures.append(f"unresolved placeholder in caption: {text[:60]}")
        if len(text) > 120:
            failures.append(f"caption is {len(text)} characters (limit 120): {text[:40]}...")
    for name, digest in manifest["files"].items():
        path = args.out / name
        if not path.exists():
            failures.append(f"{name} is listed in the manifest but missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(f"{name} changed after it was recorded")
    if args.submission and manifest["label"]:
        failures.append(f"manifest carries a label ({manifest['label']!r}): this is a test render")

    for f in failures:
        print("FAIL", f)
    print(f"{mp4.name}: {manifest['duration_s']} s, {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
