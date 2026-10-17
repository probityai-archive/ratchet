#!/usr/bin/env bash
export UV_PYTHON=3.14 UV_LOCKED=1
RATCHET_RECORDS_DIR="$PWD/scripts/l7-scratch/records" uv run python -m pytest -q -W error -s -m records tests/records ${RECORDS_ARGS:-}
echo "=== RC $? records"
