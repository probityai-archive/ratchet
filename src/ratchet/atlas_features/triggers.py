"""Probe 3: an Atlas Trigger that turns each inserted failure into one sanitized failure-memory document.

    uv run python -m ratchet.atlas_features.triggers

Creates (idempotently) an App Services app, links Cluster0, uploads `failure_memory.js` as a function and a
database trigger on `failures_probe` inserts, then proves: the memory document appears and is sanitized, a
second insert of the same failure and a duplicate delivery of the same change event write nothing new, and the
sanitized text is found by meaning through an autoEmbed index with no embedding key in the trigger.

The App Services Admin API accepts the project API key: `POST /auth/providers/mongodb-cloud/login` returns a
bearer token. Nothing here needs the operator.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from ratchet.atlas_features.common import (
    CLUSTER,
    GROUP_ID,
    PROBE_DB,
    ProbeResult,
    connect,
    probe_db,
    secret,
    timed,
    wait_queryable,
)

ADMIN = "https://services.cloud.mongodb.com/api/admin/v3.0"
APP_NAME = "ratchet-featureprobe"
FUNCTION_NAME = "failureMemory"
TRIGGER_NAME = "failure_to_memory"
SOURCE = "failures_probe"
MEMORY = "failure_memory"
MEMORY_INDEX = "memory_auto"
FUNCTION_SOURCE = Path(__file__).with_name("failure_memory.js")


class AppServicesError(RuntimeError):
    """The Admin API refused a request. Carries the method, path, status and body; never a token."""


class AppServices:
    """A minimal App Services Admin API client authenticated with the Atlas project API key."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._http = client or httpx.Client(timeout=60.0)
        pub = secret("ATLAS_PUBLIC_KEY", "atlas", "hackathon-public-key")
        pri = secret("ATLAS_PRIVATE_KEY", "atlas", "hackathon-private-key")
        resp = self._http.post(
            f"{ADMIN}/auth/providers/mongodb-cloud/login", json={"username": pub, "apiKey": pri}
        )
        if resp.status_code != 200:
            raise AppServicesError(f"login -> HTTP {resp.status_code}: {resp.text[:300]}")
        self._headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

    def call(self, method: str, path: str, body: Any = None, ok: tuple[int, ...] = (200, 201, 204)) -> Any:
        resp = self._http.request(method, f"{ADMIN}{path}", headers=self._headers, json=body)
        if resp.status_code not in ok:
            raise AppServicesError(f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:500]}")
        return resp.json() if resp.content else None

    def close(self) -> None:
        self._http.close()


def app_path(app_id: str) -> str:
    return f"/groups/{GROUP_ID}/apps/{app_id}"


def ensure_app(svc: AppServices) -> dict[str, Any]:
    for app in svc.call("GET", f"/groups/{GROUP_ID}/apps"):
        if app["name"] == APP_NAME:
            return dict(app)
    body = {"name": APP_NAME, "location": "US-VA", "deployment_model": "LOCAL"}
    return dict(svc.call("POST", f"/groups/{GROUP_ID}/apps", body))


def ensure_service(svc: AppServices, app_id: str) -> str:
    base = f"{app_path(app_id)}/services"
    for s in svc.call("GET", base):
        if s["name"] == "mongodb-atlas":
            return str(s["_id"])
    body = {
        "name": "mongodb-atlas",
        "type": "mongodb-atlas",
        "config": {"clusterName": CLUSTER, "readPreference": "primary", "wireProtocolEnabled": False},
    }
    return str(svc.call("POST", base, body)["_id"])


def ensure_function(svc: AppServices, app_id: str) -> str:
    base = f"{app_path(app_id)}/functions"
    body = {
        "name": FUNCTION_NAME,
        "source": FUNCTION_SOURCE.read_text(),
        "run_as_system": True,
        "private": False,
    }
    for f in svc.call("GET", base):
        if f["name"] == FUNCTION_NAME:
            svc.call("PUT", f"{base}/{f['_id']}", body)
            return str(f["_id"])
    return str(svc.call("POST", base, body)["_id"])


def trigger_body(service_id: str, function_id: str) -> dict[str, Any]:
    return {
        "name": TRIGGER_NAME,
        "type": "DATABASE",
        "function_id": function_id,
        "disabled": False,
        "config": {
            "service_id": service_id,
            "database": PROBE_DB,
            "collection": SOURCE,
            "operation_types": ["INSERT"],
            "full_document": True,
            "unordered": False,
            "skip_catchup_events": False,
        },
    }


def ensure_trigger(svc: AppServices, app_id: str, service_id: str, function_id: str) -> str:
    base = f"{app_path(app_id)}/triggers"
    body = trigger_body(service_id, function_id)
    for t in svc.call("GET", base):
        if t["name"] == TRIGGER_NAME:
            svc.call("PUT", f"{base}/{t['_id']}", body)
            return str(t["_id"])
    return str(svc.call("POST", base, body)["_id"])


def execute_function(svc: AppServices, app_id: str, event: dict[str, Any]) -> Any:
    """Run the function by hand on a change event: the way to deliver the same event twice."""
    body = {"name": FUNCTION_NAME, "arguments": [event]}
    return svc.call("POST", f"{app_path(app_id)}/debug/execute_function?run_as_system=true", body)


def ensure_memory_index(db: Database[dict[str, Any]]) -> None:
    if MEMORY not in db.list_collection_names():
        db.create_collection(MEMORY)
    coll = db[MEMORY]
    if MEMORY_INDEX not in {i["name"] for i in coll.list_search_indexes()}:
        definition = {
            "fields": [{"type": "autoEmbed", "modality": "text", "path": "summary", "model": "voyage-4"}]
        }
        coll.create_search_index(
            SearchIndexModel(definition=definition, name=MEMORY_INDEX, type="vectorSearch")
        )
    wait_queryable(db, MEMORY, [MEMORY_INDEX], timeout_s=600)


def wait_for(coll: Collection[dict[str, Any]], doc_id: str, timeout_s: float = 60.0) -> float | None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if coll.count_documents({"_id": doc_id}) == 1:
            return round(time.monotonic() - start, 2)
        time.sleep(0.5)
    return None


def prove(db: Database[dict[str, Any]], svc: AppServices, app_id: str) -> dict[str, Any]:
    """Insert one failure; check trigger latency, sanitization, dedupe and duplicate delivery."""
    src, mem = db[SOURCE], db[MEMORY]
    failure_id = f"probe-{uuid.uuid4().hex[:8]}"
    summary = (
        "Charge ch_204 for order 7781 refunded $1,250.00 twice; customer bob@example.com was paid twice."
    )
    src.insert_one({"failure_id": failure_id, "case_family": "refund_lost_reply", "summary": summary})
    mem_id = f"fm:{failure_id}"
    appeared = wait_for(mem, mem_id)
    stored = mem.find_one({"_id": mem_id}) or {}
    src.insert_one({"failure_id": failure_id, "case_family": "refund_lost_reply", "summary": summary})
    time.sleep(8)  # let the second delivery run
    after_second_insert = mem.count_documents({"failure_id": failure_id})
    event = {"operationType": "insert", "fullDocument": {"failure_id": failure_id, "summary": summary}}
    first = execute_function(svc, app_id, event)
    second = execute_function(svc, app_id, event)
    after_redelivery = mem.count_documents({"failure_id": failure_id})
    return {
        "failure_id": failure_id,
        "memory_appeared_s": appeared,
        "stored_summary": stored.get("summary"),
        "sanitized": all(t not in str(stored.get("summary")) for t in ("ch_204", "7781", "1,250", "bob@")),
        "count_after_second_insert_of_same_failure": after_second_insert,
        "redelivery_results": [first, second],
        "count_after_duplicate_delivery": after_redelivery,
        "cleanup": mem.delete_many({"failure_id": failure_id}).deleted_count
        + src.delete_many({"failure_id": failure_id}).deleted_count,
    }


def retrieval_check(db: Database[dict[str, Any]]) -> dict[str, Any]:
    """A fresh failure, written through the trigger, is found by meaning from the autoEmbed index."""
    failure_id = f"probe-{uuid.uuid4().hex[:8]}"
    db[SOURCE].insert_one(
        {
            "failure_id": failure_id,
            "summary": "Refund acknowledgement lost on the network so the money went out twice.",
        }
    )
    start = time.monotonic()
    found: float | None = None
    while time.monotonic() - start < 120:
        hits = list(
            db[MEMORY].aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": MEMORY_INDEX,
                            "path": "summary",
                            "query": "customer received the same refund two times",
                            "numCandidates": 20,
                            "limit": 1,
                        }
                    },
                    {"$project": {"failure_id": 1}},
                ]
            )
        )
        if hits and hits[0].get("failure_id") == failure_id:
            found = round(time.monotonic() - start, 1)
            break
        time.sleep(1.0)
    db[MEMORY].delete_many({"failure_id": failure_id})
    db[SOURCE].delete_many({"failure_id": failure_id})
    return {"failure_id": failure_id, "searchable_by_meaning_s": found}


def run() -> ProbeResult:
    client = connect()
    svc: AppServices | None = None
    try:
        svc = AppServices()
        db = probe_db(client)
        with timed() as elapsed:
            app = ensure_app(svc)
            app_id = str(app["_id"])
            service_id = ensure_service(svc, app_id)
            function_id = ensure_function(svc, app_id)
            trigger_id = ensure_trigger(svc, app_id, service_id, function_id)
            if SOURCE not in db.list_collection_names():
                db.create_collection(SOURCE)
            ensure_memory_index(db)
            proof = prove(db, svc, app_id)
            retrieval = retrieval_check(db)
        ok = (
            proof["memory_appeared_s"] is not None
            and proof["sanitized"]
            and proof["count_after_second_insert_of_same_failure"] == 1
            and proof["count_after_duplicate_delivery"] == 1
            and retrieval["searchable_by_meaning_s"] is not None
        )
        return ProbeResult(
            probe="triggers",
            works=bool(ok),
            summary="trigger wrote one sanitized memory per failure" if ok else "trigger proof failed",
            latency_s=elapsed(),
            pipeline={"trigger": trigger_body("<service_id>", "<function_id>")},
            data={
                "app_id": app_id,
                "client_app_id": app.get("client_app_id"),
                "trigger_id": trigger_id,
                "proof": proof,
                "retrieval": retrieval,
            },
            error=None if ok else "see data.proof",
        )
    except AppServicesError as exc:
        return ProbeResult(
            probe="triggers", works=False, summary="App Services Admin API refused", error=str(exc)
        )
    finally:
        if svc is not None:
            svc.close()
        client.close()


def main() -> int:
    result = run()
    path = result.save()
    print(f"{result.probe}: works={result.works} {result.summary} -> {path}")
    if result.error:
        print(result.error)
    return 0 if result.works else 1


if __name__ == "__main__":
    sys.exit(main())
