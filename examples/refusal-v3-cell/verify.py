"""Verify the signed refusal of candidate b972467a3b9a offline.

Run from the repository root: uv run python examples/refusal-v3-cell/verify.py

The verifier recounts and regrades the 80 parent and candidate executions from their traces. This script
then reads the signed decision and prints what it says. Exit status is 0 only when every check holds.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results" / "v3-cell" / "v3-cell-results"
DSSE = RESULTS / "v3-cell.dsse"
DSSE_SHA256 = "49f5e8ecc0bb1e25e294090f6baf561aaa7337c5c9801b17c03076142541d187"
DECISION_ID = "v3:dfdfffeeb388->b972467a3b9a:cd7b60eb8317"


def fail(message: str) -> int:
    print(f"FAILED: {message}", file=sys.stderr)
    return 1


def main() -> int:
    digest = hashlib.sha256(DSSE.read_bytes()).hexdigest()
    if digest != DSSE_SHA256:
        return fail(f"signed results file digest {digest} is not the pinned {DSSE_SHA256}")
    run = subprocess.run(
        [sys.executable, "-m", "ratchet.verify.results", str(DSSE), "--case-suite"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if run.returncode != 0 or "1 of 1 decisions verified" not in run.stdout:
        return fail(f"offline verification did not pass:\n{run.stdout}{run.stderr}")
    regraded = re.search(r'"regraded": (\d+)', run.stdout)
    if regraded is None or int(regraded.group(1)) != 80:
        return fail("the verifier did not regrade 80 executions")
    payload = json.loads(base64.b64decode(json.loads(DSSE.read_text())["payload"]))
    (decision,) = payload["decisions"]
    if decision["decision_id"] != DECISION_ID:
        return fail(f"unexpected decision {decision['decision_id']}")
    if decision["admitted"] or "cohort_qualified" not in decision["failed"]:
        return fail("the signed decision is not a refusal for an unqualified cohort")
    print("regraded 80 executions from their traces")
    print("decision verified, candidate refused, reason: identity controls breached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
