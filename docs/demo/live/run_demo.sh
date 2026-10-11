#!/bin/sh
# Live stage runner for the Round 1 demo. Every step runs the real command against MongoDB Atlas.
#
#   docs/demo/live/run_demo.sh          press Enter to run each step; the screen clears between steps
#   docs/demo/live/run_demo.sh --warm   run every step once with output hidden, print each step's time
#
# Read-only: every command reads stored traces and decisions; none writes to Atlas or calls a model.
# The connection string comes from RATCHET_MONGODB_URI, else the keyring entry
# `service mongodb account hackathon-uri`. Ids come from docs/demo/ids.env, read as data (never sourced:
# its values contain `->` and spaces, which a shell would run as redirections and commands).

ROOT=$(cd "$(dirname "$0")/../../.." && pwd) || exit 1
cd "$ROOT" || exit 1
IDS="$ROOT/docs/demo/ids.env"

if [ -x "$ROOT/.venv/bin/ratchet" ]; then
    RATCHET="$ROOT/.venv/bin/ratchet"
else
    RATCHET=ratchet
fi

if [ -z "$RATCHET_MONGODB_URI" ]; then
    RATCHET_MONGODB_URI=$(secret-tool lookup service mongodb account hackathon-uri 2>/dev/null)
fi
if [ -z "$RATCHET_MONGODB_URI" ]; then
    echo "No Atlas connection string: set RATCHET_MONGODB_URI or store it in the keyring." >&2
    exit 1
fi
export RATCHET_MONGODB_URI

idv() {
    v=$(sed -n "s/^$1=//p" "$IDS" | head -n 1)
    if [ -z "$v" ]; then
        echo "docs/demo/ids.env has no $1" >&2
        exit 1
    fi
    printf '%s' "$v"
}

DEMO_DB=$(idv RATCHET_DB)
BAD_TRACE=$(idv BAD_TRACE)
REJECTED_CAND=$(idv REJECTED_CAND)
CLEARED_CAND=$(idv CLEARED_CAND)
ADMITTED_CAND=$(idv ADMITTED_CAND)
AUTHORITY_DB=$(idv AUTHORITY_DB)
AUTHORITY_DECISION=$(idv AUTHORITY_DECISION)
V3_DB=ratchet_v3_full
V3_DECISION=b972467a3b9a

MODE=live
case "$1" in
    --warm) MODE=warm ;;
    "") ;;
    *) echo "usage: $0 [--warm]" >&2; exit 2 ;;
esac

now() { date +%s.%N; }

# step <number> <header> <database or -> <command ...>
step() {
    n=$1; header=$2; db=$3
    shift 3
    if [ "$MODE" = warm ]; then
        t0=$(now)
        if [ "$db" = - ]; then
            "$@" >/dev/null 2>&1
        else
            RATCHET_DB=$db "$@" >/dev/null 2>&1
        fi
        rc=$?
        t1=$(now)
        awk -v n="$n" -v rc="$rc" -v a="$t0" -v b="$t1" -v h="$header" \
            'BEGIN { printf "step %s  rc=%s  %5.1f s  %s\n", n, rc, b - a, h }'
        [ "$rc" -eq 0 ] || FAILED=1
        return
    fi
    clear
    printf '\033[1m%s\033[0m\n\n' "$header"
    if [ "$db" = - ]; then
        "$@"
    else
        RATCHET_DB=$db "$@"
    fi
    printf '\n'
    if [ "$n" -lt 7 ]; then
        read -r _
    fi
}

FAILED=0
if [ "$MODE" = live ]; then
    clear
    printf '\033[1mRatchet: how an agent'"'"'s safety rules are made and proven\033[0m\n\n(press Enter)\n'
    read -r _
fi

step 1 "How it works: one support case. The reply to a refund is lost, the agent retries, the customer is refunded twice." \
    "$DEMO_DB" "$RATCHET" show trace "$BAD_TRACE"
step 2 "Two candidate safety rules, rerun over 8 recorded conversations. No AI calls." \
    "$DEMO_DB" "$RATCHET" replay "$REJECTED_CAND" "$CLEARED_CAND"
step 3 "The rule that passed, tested on cases the improver never saw." \
    "$DEMO_DB" "$RATCHET" show decision "$ADMITTED_CAND"
step 4 "Anyone can check the signed decision." \
    "$DEMO_DB" "$RATCHET" verify "$ADMITTED_CAND"
step 5 "A change that gives the agent a new tool is refused." \
    "$AUTHORITY_DB" "$RATCHET" show decision "$AUTHORITY_DECISION"
step 6 "Now a real model: gpt-oss-120b, 3,800 runs." \
    "$V3_DB" "$RATCHET" show decision "$V3_DECISION"
step 7 "Real data: one safety rule over 629,192 actions from real coding-agent sessions." \
    - cat docs/results/hooks-replay.md

if [ "$MODE" = warm ]; then
    if [ "$FAILED" -ne 0 ]; then
        echo "WARM FAILED: a step returned non-zero; run it by hand before going on stage." >&2
        exit 1
    fi
    echo "warm: all 7 steps ok"
fi
