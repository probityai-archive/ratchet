#!/bin/sh
# Add a narration track to a finished video without touching its picture.
#   narrate.sh ratchet-169.mp4 narration.m4a   (any audio ffmpeg reads) -> ratchet-169-narrated.mp4
# The video stream is copied, so the recorded pixels and the caption timing stay byte-identical.
# The audio is cut to the video length. POSIX sh.
set -eu
[ $# -eq 2 ] || { echo "usage: narrate.sh VIDEO.mp4 AUDIO" >&2; exit 2; }
out="${1%.mp4}-narrated.mp4"
ffmpeg -y -v error -i "$1" -i "$2" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 160k -shortest -movflags +faststart "$out"
echo "$out"
