"""Create the least-privilege database user the improver connects as, through the Atlas Admin API.

The improver reads and writes the working database (traces, failures, harness versions, decisions) but has
no privilege at all on the held-out database, so it cannot read the cases it is judged on. Atlas custom
roles only grant, so "denied" means "never granted". Keys come from the environment:

    ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY, ATLAS_GROUP_ID, RATCHET_IMPROVER_PASSWORD
"""

from __future__ import annotations

import os
import sys
from typing import Any

import httpx

API = "https://cloud.mongodb.com/api/atlas/v2"
ACCEPT = "application/vnd.atlas.2024-08-05+json"
IMPROVER_ROLE = "ratchetImprover"
IMPROVER_USER = "ratchet-improver"
_EXISTS = 409


def improver_role_body(db: str) -> dict[str, Any]:
    return {"roleName": IMPROVER_ROLE, "actions": [], "inheritedRoles": [{"db": db, "role": "readWrite"}]}


def improver_user_body(group_id: str, password: str) -> dict[str, Any]:
    return {
        "groupId": group_id,
        "databaseName": "admin",
        "username": IMPROVER_USER,
        "password": password,
        "roles": [{"databaseName": "admin", "roleName": IMPROVER_ROLE}],
    }


def _post(client: httpx.Client, path: str, body: dict[str, Any]) -> str:
    resp = client.post(f"{API}{path}", json=body, headers={"Accept": ACCEPT})
    if resp.status_code == _EXISTS:
        return "exists"
    if resp.status_code >= 300:
        # the body can echo request fields; report the Atlas error code only
        raise RuntimeError(f"Atlas API {path} failed: HTTP {resp.status_code} {resp.json().get('errorCode')}")
    return "created"


def create_improver(client: httpx.Client, group_id: str, db: str, password: str) -> tuple[str, str]:
    """Create the role and the user. Returns ("created" | "exists") for each; safe to re-run."""
    role = _post(client, f"/groups/{group_id}/customDBRoles/roles", improver_role_body(db))
    user = _post(client, f"/groups/{group_id}/databaseUsers", improver_user_body(group_id, password))
    return role, user


def main() -> int:
    env = os.environ
    missing = [
        k
        for k in ("ATLAS_PUBLIC_KEY", "ATLAS_PRIVATE_KEY", "ATLAS_GROUP_ID", "RATCHET_IMPROVER_PASSWORD")
        if not env.get(k)
    ]
    if missing:
        print(f"error: set {', '.join(missing)}", file=sys.stderr)
        return 2
    auth = httpx.DigestAuth(env["ATLAS_PUBLIC_KEY"], env["ATLAS_PRIVATE_KEY"])
    with httpx.Client(auth=auth, timeout=60.0) as client:
        role, user = create_improver(
            client,
            env["ATLAS_GROUP_ID"],
            env.get("RATCHET_MONGODB_DB", "ratchet"),
            env["RATCHET_IMPROVER_PASSWORD"],
        )
    print(f"role {IMPROVER_ROLE}: {role}; user {IMPROVER_USER}: {user}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
