"""Show that the improver's database identity cannot read the held-out cases.

    RATCHET_IMPROVER_URI=... [RATCHET_DB=ratchet] python3 docs/demo/denied_read.py

Connects as the least-privilege improver user and tries to read the held-out cases. Prints the server's
own refusal and exits 0 when the read is denied; exits 1, loudly, if it succeeds. The connection string
is never printed.
"""

from __future__ import annotations

import os
import sys

from pymongo.errors import OperationFailure

from ratchet.store import Store

UNAUTHORIZED = 13


def main() -> int:
    uri = os.environ.get("RATCHET_IMPROVER_URI")
    if not uri:
        print("RATCHET_IMPROVER_URI is not set (the improver's connection string)", file=sys.stderr)
        return 2
    db = os.environ.get("RATCHET_DB", "ratchet")
    with Store(uri, db) as improver:
        try:
            cases = improver.cases("heldout")
        except OperationFailure as exc:
            if exc.code != UNAUTHORIZED:
                print(f"unexpected server error {exc.code}: {exc}", file=sys.stderr)
                return 2
            print(f"improver identity -> {db}_heldout.cases: DENIED (code {exc.code})")
            print(f"server said: {exc.details.get('errmsg', exc) if exc.details else exc}")
            return 0
    print(f"READ SUCCEEDED: {len(cases)} held-out cases visible to the improver. Isolation is broken.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
