#!/usr/bin/env bash
# Every CI job that can run on the box: lint, tests with Atlas, the case-suite jobs, the records test, the demo image.
ATLAS=1 scripts/l7-scratch/ci.sh; a=$?
scripts/l7-scratch/records.sh; b=$?
if [ "${IMAGE:-1}" = 1 ]; then
  echo "=== STEP demo image"; docker build -q -f docker/demo.Dockerfile -t ratchet-demo:l7 . > /tmp/l7-image.log 2>&1 && docker/demo/check.sh ratchet-demo:l7; c=$?; tail -30 /tmp/l7-image.log; echo "=== RC $c demo-image"; docker image rm -f ratchet-demo:l7 >/dev/null 2>&1
else c=0; fi
echo "=== ALL ci=$a records=$b image=$c"
[ $a -eq 0 ] && [ $b -eq 0 ] && [ $c -eq 0 ]
