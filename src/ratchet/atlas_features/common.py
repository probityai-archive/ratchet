"""Shared plumbing for the Atlas probes: connection, secrets, Voyage embeddings, timing, result records.

Secrets come from the environment or the system keyring and are never printed, logged or stored in a result.
Every probe works in the `ratchet_probes` database so it never collides with the store lane.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import httpx
from pymongo import MongoClient
from pymongo.database import Database

from ratchet.store.embed import CachedEmbedder, Embedder, VoyageEmbedder

PROBE_DB = "ratchet_probes"
GROUP_ID = "6acbd00b1b25c55279b6511b"
CLUSTER = "Cluster0"
ATLAS_API = "https://cloud.mongodb.com/api/atlas/v2"
ATLAS_ACCEPT = "application/vnd.atlas.2024-08-05+json"
ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "research" / "_raw" / "atlas-probes" / "results"
FAST_FAIL_MS = 2000


class ProbeUnavailableError(RuntimeError):
    """The sandbox cannot be reached or a secret is missing. Carries a message safe to print."""


def keyring(service: str, account: str) -> str:
    """Read one secret from the GNOME keyring; empty string when absent."""
    try:
        out = subprocess.run(
            ["secret-tool", "lookup", "service", service, "account", account],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip()


def secret(env: str, service: str, account: str) -> str:
    """Environment first, keyring second. Raises with the names tried, never a value."""
    value = os.environ.get(env, "") or keyring(service, account)
    if not value:
        raise ProbeUnavailableError(
            f"secret missing: set {env} or store keyring service={service} account={account}"
        )
    return value


def atlas_admin_client() -> httpx.Client:
    """An httpx client with digest auth for the Atlas Admin API."""
    pub = secret("ATLAS_PUBLIC_KEY", "atlas", "hackathon-public-key")
    pri = secret("ATLAS_PRIVATE_KEY", "atlas", "hackathon-private-key")
    return httpx.Client(
        auth=httpx.DigestAuth(pub, pri),
        headers={"Accept": ATLAS_ACCEPT},
        timeout=30.0,
        base_url=ATLAS_API,
    )


def atlas_get(path: str, *, accept: str | None = None) -> httpx.Response:
    headers = {"Accept": accept} if accept else {}
    with atlas_admin_client() as client:
        return client.get(path, headers=headers)


def cluster_srv_host() -> str:
    resp = atlas_get(f"/groups/{GROUP_ID}/clusters/{CLUSTER}")
    resp.raise_for_status()
    srv = str(resp.json()["connectionStrings"]["standardSrv"])
    return srv.removeprefix("mongodb+srv://")


def probe_uri() -> str:
    """The sandbox URI: env override, then the store lane's stored URI, then built from the cluster host."""
    explicit = os.environ.get("RATCHET_PROBES_URI", "") or keyring("mongodb", "hackathon-uri")
    if explicit:
        return explicit
    password = secret("RATCHET_DB_PASSWORD", "mongodb", "hackathon-db-ratchet")
    return f"mongodb+srv://ratchet:{quote_plus(password)}@{cluster_srv_host()}/?retryWrites=true&w=majority"


def connect(timeout_ms: int = FAST_FAIL_MS) -> MongoClient[dict[str, Any]]:
    """Connect and ping; fail fast with a message that names the cause, never a secret."""
    client: MongoClient[dict[str, Any]] = MongoClient(
        probe_uri(), serverSelectionTimeoutMS=timeout_ms, tz_aware=True
    )
    try:
        client.admin.command("ping")
    except Exception as exc:
        client.close()
        raise ProbeUnavailableError(
            f"Atlas sandbox unreachable within {timeout_ms} ms ({type(exc).__name__}); "
            "check the cluster state and the IP access list"
        ) from exc
    return client


def probe_db(client: MongoClient[dict[str, Any]]) -> Database[dict[str, Any]]:
    return client[PROBE_DB]


@contextmanager
def embedder(db: Database[dict[str, Any]]) -> Iterator[Embedder]:
    """Voyage embeddings through the store lane's client, cached in the probe database; closes the client."""
    key = secret("VOYAGE_API_KEY", "voyageai", "api-key")
    voyage = VoyageEmbedder(api_key=key)
    try:
        yield CachedEmbedder(voyage, db["embedding_cache"])
    finally:
        voyage.close()


@contextmanager
def timed() -> Iterator[Callable[[], float]]:
    """Yield a function returning elapsed seconds since entry."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


@dataclass
class ProbeResult:
    """One probe outcome, JSON-serialisable. `error` carries the server response verbatim on failure."""

    probe: str
    works: bool
    summary: str
    latency_s: float = 0.0
    pipeline: Any = None
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def save(self, directory: Path = RESULTS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.probe}.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str, sort_keys=True) + "\n")
        return path


def wait_queryable(
    db: Database[dict[str, Any]], collection: str, names: list[str], timeout_s: float = 240.0
) -> float:
    """Block until the named search indexes are queryable; return seconds waited."""
    start = time.monotonic()
    while True:
        have = {i["name"]: i for i in db[collection].list_search_indexes()}
        missing = [n for n in names if n not in have]
        if not missing and all(have[n].get("queryable") for n in names):
            return time.monotonic() - start
        if time.monotonic() - start > timeout_s:
            states = {n: (have[n].get("status") if n in have else "absent") for n in names}
            raise TimeoutError(f"search indexes not queryable after {timeout_s:.0f}s: {states}")
        time.sleep(2.0)
