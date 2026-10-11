#!/bin/sh
# Install the two pinned recording tools (asciinema, agg) into ~/.local/share/ratchet-demo-tools and
# check the rest of the toolchain. Safe to run twice. POSIX sh.
set -eu

TOOLS="${RATCHET_DEMO_TOOLS:-$HOME/.local/share/ratchet-demo-tools}"
AGG_TAG=v1.9.0
AGG_SHA=f111e315cd71056b116302342553dd765b7297579ed511f111d0cedb442aeda6
CAST_TAG=v3.2.1
CAST_SHA=1b405bbda565b33c3c4718de67fedc3535580603c0694b1ff3fb04f363430a20

fetch() { # name repo tag asset sha
    dest="$TOOLS/$1"
    if [ -x "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$5" ]; then
        echo "ok      $1 (pinned sha256 matches)"
        return 0
    fi
    mkdir -p "$TOOLS"
    gh release download "$3" -R "$2" -p "$4" -O "$dest.part" --clobber
    got=$(sha256sum "$dest.part" | cut -d' ' -f1)
    if [ "$got" != "$5" ]; then
        rm -f "$dest.part"
        echo "REFUSED $1: sha256 $got is not the pinned $5" >&2
        exit 1
    fi
    chmod +x "$dest.part"
    mv "$dest.part" "$dest"
    echo "fetched $1 $3"
}

fetch agg asciinema/agg "$AGG_TAG" agg-x86_64-unknown-linux-gnu "$AGG_SHA"
fetch asciinema asciinema/asciinema "$CAST_TAG" asciinema-x86_64-unknown-linux-gnu "$CAST_SHA"

missing=0
for tool in ffmpeg ffprobe python3 fc-list; do
    command -v "$tool" >/dev/null 2>&1 || { echo "MISSING $tool" >&2; missing=1; }
done
ffmpeg -hide_banner -filters 2>/dev/null | grep -q ' subtitles ' || { echo "MISSING ffmpeg subtitles filter (libass)" >&2; missing=1; }
fc-list 2>/dev/null | grep -q 'JetBrains Mono' || { echo "MISSING font JetBrains Mono" >&2; missing=1; }
fc-list 2>/dev/null | grep -q 'Inter' || { echo "MISSING font Inter" >&2; missing=1; }
[ "$missing" -eq 0 ] || exit 1
echo "toolchain ready: $TOOLS"
