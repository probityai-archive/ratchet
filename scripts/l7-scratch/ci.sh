#!/usr/bin/env bash
# Mirror of the ci.yml lint and test jobs, for the box. Each step's status is printed; the script exits nonzero if any failed.
set -u
export UV_PYTHON=3.14 UV_LOCKED=1
fail=0
step() { echo "=== STEP $*"; "$@"; rc=$?; echo "=== RC $rc $1 ${2:-}"; [ $rc -eq 0 ] || fail=1; }
step uv sync --all-groups --all-extras
if [ "${ATLAS:-0}" = 1 ]; then
  docker rm -f l7-atlas >/dev/null 2>&1
  docker run -d --name l7-atlas --hostname ratchet-atlas --ulimit nofile=1048576:1048576 -e MONGODB_INITDB_ROOT_USERNAME=ratchet -e MONGODB_INITDB_ROOT_PASSWORD=ratchet-local-dev -p 127.0.0.1:27018:27017 mongodb/mongodb-atlas-local:8.3.9-20260928T091948Z >/dev/null
  for i in $(seq 1 90); do docker exec l7-atlas mongosh -u ratchet -p ratchet-local-dev --quiet --eval "db.getSiblingDB('healthprobe').nothing.aggregate([{\$listSearchIndexes:{}}]).toArray().length" >/dev/null 2>&1 && break; sleep 5; done
fi
if [ "${LINT:-1}" = 1 ]; then
step uv run ruff check --extend-exclude scripts/l7-scratch
step uv run ruff format --check --extend-exclude scripts/l7-scratch
step uv run mypy src
step uv run mypy --config-file probity-ratchet-cases/pyproject.toml probity-ratchet-cases/src
step python3 scripts/readme-lint.py README.md
step uv run python scripts/check-docs.py
fi
casestest() { (cd probity-ratchet-cases && uv run python -m pytest -q -W error -p no:cacheprovider); }
step casestest
step uv run --with pytest-timeout python -m pytest -q -W error --timeout=120 -rfEs --tb=short -m "not sandbox and not live and not records" ${PYTEST_ARGS:-}
if [ "${ATLAS:-0}" = 1 ]; then docker rm -f l7-atlas >/dev/null 2>&1; fi
echo "=== FAIL $fail"
exit $fail
