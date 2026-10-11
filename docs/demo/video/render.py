"""Turn a recorded terminal session (asciinema cast + driver timeline) into the submission video.

    render.py --cast run.cast --timeline run.json --format 169|45 --out DIR [--label TEXT] [--max-seconds 60]

What it makes in DIR:
    ratchet-<format>.mp4   h264, yuv420p, 30 fps, faststart, banner on top, captions below the terminal
    ratchet-<format>.vtt   the same captions as WebVTT
    ratchet-<format>.ass   the caption file that was burned in
    contact-<format>.png   one frame per step, for the visual check
    manifest-<format>.json what was recorded, when, with which tools, and the sha256 of every file

The terminal pixels come only from the cast. Nothing is drawn into the terminal area: the banner states
that the run is recorded and when, and the captions are text on the padding around it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

TOOLS = Path.home() / ".local/share/ratchet-demo-tools"
BG = "0d1117"
FPS = 30
GLYPH_W = 0.6  # JetBrains Mono advance, in em
LINE_H = 1.2
SENTINEL_TITLE = "ratchet-demo-start"

# canvas, banner height, caption height, terminal font size (px), caption font size (px), banner font size
FORMATS: dict[str, dict[str, int]] = {
    "169": {"w": 1920, "h": 1080, "banner": 64, "caption": 230, "font": 24, "cap_font": 54, "ban_font": 28},
    "45": {"w": 1080, "h": 1350, "banner": 76, "caption": 360, "font": 28, "cap_font": 56, "ban_font": 30},
}


def terminal_size(fmt: dict[str, int]) -> tuple[int, int]:
    """Columns and rows of the recorded terminal so that it fills the area between banner and captions."""
    area_h = fmt["h"] - fmt["banner"] - fmt["caption"]
    return int(fmt["w"] // (fmt["font"] * GLYPH_W)), int(area_h // (fmt["font"] * LINE_H))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cast_events(path: Path) -> list[tuple[float, str, str]]:
    """(absolute seconds, code, data) for a v2 or v3 cast."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    relative = header.get("version") == 3
    now, out = 0.0, []
    for raw in lines[1:]:
        if not raw.strip():
            continue
        t, code, data = json.loads(raw)
        now = now + t if relative else t
        out.append((now, code, data))
    return out


def sentinel_time(events: list[tuple[float, str, str]]) -> float:
    for t, code, data in events:
        if code == "o" and SENTINEL_TITLE in data:
            return t
    raise SystemExit("the cast has no start marker: it was not recorded through drive.py")


def ass_time(seconds: float) -> str:
    cs = round(seconds * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def vtt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def beats(timeline: dict[str, Any]) -> list[tuple[float, float, str]]:
    """Caption beats on the video clock: each lasts until the next one, the last until its step ends."""
    out: list[tuple[float, float, str]] = []
    for step in timeline["steps"]:
        marks = sorted(step["captions"], key=lambda c: c["at"])
        for i, cap in enumerate(marks):
            start = step["t_start"] + cap["at"]
            end = step["t_start"] + marks[i + 1]["at"] if i + 1 < len(marks) else step["t_end"]
            out.append((start, end, cap["text"]))
    return out


def wrap(text: str, width_px: int, font_px: int) -> str:
    per_line = max(12, int(width_px / (font_px * 0.54)))
    return "\\N".join(textwrap.wrap(text, per_line))


def banner_text(timeline: dict[str, Any], label: str | None) -> str:
    if label:
        return label
    when = datetime.fromtimestamp(timeline["started_utc"], UTC).astimezone(ZoneInfo("America/Los_Angeles"))
    return f"RECORDED RUN  {when:%Y-%m-%d %H:%M} PT  |  real commands, unedited"


def write_ass(
    path: Path, fmt: dict[str, int], timeline: dict[str, Any], label: str | None, end: float
) -> list[str]:
    w, h = fmt["w"], fmt["h"]
    cap_y = h - fmt["caption"] // 2
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {w}",
        f"PlayResY: {h}",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
        "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Caption,Inter,{fmt['cap_font']},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,0,0,5,40,40,0,1",
        f"Style: Banner,Inter,{fmt['ban_font']},&H0038C8FF,&H0038C8FF,&H00000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,0,0,5,20,20,0,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
        f"Dialogue: 0,{ass_time(0)},{ass_time(end)},Banner,,0,0,0,,{{\\pos({w // 2},{fmt['banner'] // 2})}}"
        + banner_text(timeline, label).replace("|", "\\N" if w < 1400 else "|"),
    ]
    texts: list[str] = []
    for start, stop, text in beats(timeline):
        texts.append(text)
        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(stop)},Caption,,0,0,0,,"
            f"{{\\pos({w // 2},{cap_y})}}{wrap(text, w - 80, fmt['cap_font'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return texts


def write_vtt(path: Path, timeline: dict[str, Any]) -> None:
    out = ["WEBVTT", ""]
    for start, stop, text in beats(timeline):
        out += [f"{vtt_time(start)} --> {vtt_time(stop)}", text, ""]
    path.write_text("\n".join(out), encoding="utf-8")


def run(cmd: list[str]) -> None:
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        sys.stderr.write(done.stderr[-3000:])
        raise SystemExit(f"command failed ({done.returncode}): {' '.join(cmd[:3])} ...")


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(out.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cast", type=Path, required=True)
    ap.add_argument("--timeline", type=Path, required=True)
    ap.add_argument("--format", choices=sorted(FORMATS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", help="replaces the banner; use for pipeline tests")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    args = ap.parse_args()

    fmt = FORMATS[args.format]
    timeline = json.loads(args.timeline.read_text(encoding="utf-8"))
    events = cast_events(args.cast)
    shift = sentinel_time(events)  # driver clock zero, on the cast clock
    cols, rows = terminal_size(fmt)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"ratchet-{args.format}"

    # The driver clock starts when the start marker was written; the video starts at the cast start.
    # Shift every step so captions sit on the video clock.
    for step in timeline["steps"]:
        for key in ("t_start", "t_run", "t_done", "t_end"):
            step[key] += shift
    total = max(s["t_end"] for s in timeline["steps"]) + 0.5

    gif = args.out / f"terminal-{args.format}.gif"
    run(
        [
            str(TOOLS / "agg"),
            "--cols",
            str(cols),
            "--rows",
            str(rows),
            "--font-size",
            str(fmt["font"]),
            "--line-height",
            str(LINE_H),
            "--font-family",
            "JetBrains Mono,DejaVu Sans Mono",
            "--theme",
            "github-dark",
            "--fps-cap",
            "25",
            "--idle-time-limit",
            "999",
            "--last-frame-duration",
            "1",
            "--no-loop",
            str(args.cast),
            str(gif),
        ]
    )

    texts = write_ass(stem.with_suffix(".ass"), fmt, timeline, args.label, total)
    write_vtt(stem.with_suffix(".vtt"), timeline)

    area_h = fmt["h"] - fmt["banner"] - fmt["caption"]
    mp4 = stem.with_suffix(".mp4")
    fonts = Path.home() / ".local/share/fonts"
    graph = (
        f"color=c=0x{BG}:s={fmt['w']}x{fmt['h']}:r={FPS}[bg];"
        f"[0:v]fps={FPS},scale={fmt['w']}:{area_h}:force_original_aspect_ratio=decrease:flags=lanczos[t];"
        f"[bg][t]overlay=x=(W-w)/2:y={fmt['banner']}+({area_h}-h)/2:eof_action=repeat:shortest=0[v];"
        f"[v]ass={stem.with_suffix('.ass')}:fontsdir={fonts}[out]"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(gif),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-an",
            "-t",
            f"{total:.2f}",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            "-movflags",
            "+faststart",
            str(mp4),
        ]
    )
    gif.unlink()

    contact = args.out / f"contact-{args.format}.png"
    picks = [s["t_done"] + min(1.0, (s["t_end"] - s["t_done"]) / 2) for s in timeline["steps"]]
    sel = "+".join(f"eq(n\\,{max(0, round(p * FPS))})" for p in picks)
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(mp4),
            "-vf",
            f"select='{sel}',scale=iw/2:ih/2,tile={min(len(picks), 3)}x{(len(picks) + 2) // 3}",
            "-frames:v",
            "1",
            str(contact),
        ]
    )

    manifest = {
        "format": args.format,
        "canvas": [fmt["w"], fmt["h"]],
        "terminal": {"cols": cols, "rows": rows, "font_px": fmt["font"]},
        "recorded_utc": datetime.fromtimestamp(timeline["started_utc"], UTC).isoformat(),
        "duration_s": round(probe_duration(mp4), 2),
        "label": args.label,
        "steps": [{k: s[k] for k in ("id", "cmd", "rc", "status")} for s in timeline["steps"]],
        "captions": texts,
        "tools": {
            "agg": sha256(TOOLS / "agg")[:16],
            "asciinema": sha256(TOOLS / "asciinema")[:16],
            "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split(
                "\n"
            )[0],
        },
        "files": {
            p.name: sha256(p) for p in (args.cast, mp4, stem.with_suffix(".vtt"), stem.with_suffix(".ass"))
        },
    }
    (args.out / f"manifest-{args.format}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{mp4}  {manifest['duration_s']} s  {fmt['w']}x{fmt['h']}  terminal {cols}x{rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
