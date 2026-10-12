#!/bin/bash
# Run the reviewer sequence of docs/RUNNING.md against a built demo image, in a fresh container with no
# network and no host mount, print every command with its output, and fail unless each step gives the
# answer the paper quotes.  Usage: docker/demo/check.sh [IMAGE]   (default ratchet-demo:dev)
set -uo pipefail

IMAGE=${1:-ratchet-demo:dev}
NAME=ratchet-demo-check-$$
PEM=/data/ratchet-demo/ratchet_demo-decision-key.pub
BAD=13881cee67fe
GOOD=ea3bd812059a
failures=0
out=$(mktemp)
trap 'docker rm -f "$NAME" >/dev/null 2>&1; rm -f "$out"' EXIT

# step EXPECTED_EXIT PATTERN COMMAND...: run, echo, and require the exit code and a line matching PATTERN
step() {
    local want=$1 pattern=$2
    shift 2
    echo "\$ $*"
    local start=$SECONDS
    "$@" >"$out" 2>&1
    local rc=$?
    cat "$out"
    echo "[exit $rc, $((SECONDS - start)) s]"
    echo
    if [ "$rc" -ne "$want" ] || ! grep -qE -- "$pattern" "$out"; then
        echo "CHECK FAILED: wanted exit $want and a line matching /$pattern/" >&2
        failures=$((failures + 1))
    fi
}

step 0 . docker run -d --rm --network none --name "$NAME" "$IMAGE"
step 0 "admitted decision +dfdfffeeb388->$GOOD admitted=True" docker exec "$NAME" ratchet demo double-refund
step 0 "REJECTED +dfdfffeeb388 -> $BAD" docker exec "$NAME" ratchet show decision "$BAD"
step 0 "ADMITTED +dfdfffeeb388 -> $GOOD" docker exec "$NAME" ratchet show decision "$GOOD"
step 0 "VERDICT +REJECTED +ADMITTED" docker exec "$NAME" ratchet replay "$BAD" "$GOOD"
step 0 "VERIFIED +dfdfffeeb388->$GOOD" docker exec "$NAME" ratchet verify "$GOOD" --pubkey-file "$PEM"
step 0 "VERIFIED +dfdfffeeb388->$BAD" docker exec "$NAME" ratchet verify "$BAD" --pubkey-file "$PEM"
step 1 "INVALID +dfdfffeeb388->$GOOD" docker exec "$NAME" ratchet verify "$GOOD"

if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed" >&2
    exit 1
fi
echo "all demo checks passed"
