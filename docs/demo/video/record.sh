#!/bin/sh
# One command from steps to submission video.
#
#   docs/demo/video/record.sh                      real submission: steps.submission.toml, both formats
#   docs/demo/video/record.sh --test               pipeline test on steps.test.toml, banner says TEST
#   docs/demo/video/record.sh --steps FILE --ids FILE --format 169|45|both --out DIR
#
# Order: resolve placeholders (refuses if any is unresolved) -> record each format in a headless
# terminal of the right size -> render -> check. POSIX sh.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
TOOLS="${RATCHET_DEMO_TOOLS:-$HOME/.local/share/ratchet-demo-tools}"
STEPS="$HERE/steps.submission.toml"
IDS="$HERE/../ids.env"
FORMAT=both
TEST=0
OUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --test) TEST=1; STEPS="$HERE/steps.test.toml"; IDS=""; shift ;;
        --steps) STEPS="$2"; shift 2 ;;
        --ids) IDS="$2"; shift 2 ;;
        --format) FORMAT="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
[ -n "$OUT" ] || OUT="$REPO/docs/demo/out/$STAMP"
[ "$TEST" -eq 0 ] || OUT="${OUT}-test"
mkdir -p "$OUT"

"$HERE/setup.sh" >/dev/null
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"

IDARG=""
[ -z "$IDS" ] || { [ -f "$IDS" ] || { echo "no ids file at $IDS (copy ids.example.env and fill it from measured results)" >&2; exit 2; }; IDARG="--ids $IDS"; }
# shellcheck disable=SC2086
python3 "$HERE/drive.py" "$STEPS" $IDARG --check

formats="$FORMAT"
[ "$FORMAT" != both ] || formats="169 45"
for f in $formats; do
    case "$f" in
        169) size=$(python3 -c "import sys; sys.path.insert(0,'$HERE'); import render; c,r=render.terminal_size(render.FORMATS['169']); print(f'{c}x{r}')") ;;
        45)  size=$(python3 -c "import sys; sys.path.insert(0,'$HERE'); import render; c,r=render.terminal_size(render.FORMATS['45']); print(f'{c}x{r}')") ;;
        *) echo "format must be 169, 45 or both" >&2; exit 2 ;;
    esac
    cast="$OUT/run-$f.cast"
    timeline="$OUT/run-$f.json"
    rm -f "$cast" "$timeline"
    # shellcheck disable=SC2086
    python3 "$HERE/drive.py" "$STEPS" $IDARG --variant "$f" --warm
    set +e
    # shellcheck disable=SC2086
    "$TOOLS/asciinema" rec --headless --overwrite --quiet --return --window-size "$size" \
        --command "python3 $HERE/drive.py $STEPS $IDARG --variant $f --timeline $timeline" "$cast"
    rc=$?
    set -e
    [ -f "$timeline" ] || { echo "recording $f produced no timeline (driver rc $rc)" >&2; exit 1; }
    label=""
    [ "$TEST" -eq 0 ] || label="PIPELINE TEST, NOT A SUBMISSION CAPTURE"
    if [ -n "$label" ]; then
        python3 "$HERE/render.py" --cast "$cast" --timeline "$timeline" --format "$f" --out "$OUT" --label "$label" --max-seconds 600
    else
        python3 "$HERE/render.py" --cast "$cast" --timeline "$timeline" --format "$f" --out "$OUT"
    fi
    if [ "$TEST" -eq 1 ]; then
        python3 "$HERE/check.py" "$OUT" --format "$f"
    else
        python3 "$HERE/check.py" "$OUT" --format "$f" --submission
    fi
done
echo "done: $OUT"
