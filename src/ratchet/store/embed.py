"""Text embeddings: a Voyage AI client and a content-addressed cache in MongoDB.

The cache key is the SHA-256 of the model, the input type and the text, so a repeated run never pays for
the same text twice and a model change can never serve a vector of the wrong size or meaning.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from pymongo import UpdateOne
from pymongo.collection import Collection

InputType = Literal["document", "query"]

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
RERANK_URL = "https://api.voyageai.com/v1/rerank"
DEFAULT_MODEL = "voyage-4-large"
DEFAULT_RERANK_MODEL = "rerank-2.5"
EMBEDDING_DIMS = 1024
_MAX_BATCH = 128
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class EmbeddingError(RuntimeError):
    """The embedding service refused or failed a request. Never carries the API key."""


class MissingKeyError(EmbeddingError):
    """No Voyage API key in the environment."""


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]: ...


class Reranker(Protocol):
    model: str

    def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[tuple[int, float]]:
        """(index into documents, relevance score) pairs, best first."""
        ...


class _VoyageHTTP:
    """Authenticated JSON POSTs to Voyage with bounded retries. The key is never logged or raised."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        url: str,
        *,
        client: httpx.Client | None,
        retries: int,
        backoff_s: float,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("VOYAGE_API_KEY", "")
        if not key:
            raise MissingKeyError("VOYAGE_API_KEY is not set")
        self.model = model
        self._key = key
        self._client = client or httpx.Client(timeout=60.0)
        self._retries = retries
        self._backoff_s = backoff_s
        self._url = url

    def close(self) -> None:
        self._client.close()

    def _post_with_retry(self, body: dict[str, Any]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last = "no attempt made"
        for attempt in range(self._retries + 1):
            try:
                resp = self._client.post(self._url, json=body, headers=headers)
            except httpx.TransportError as exc:
                last = type(exc).__name__
            else:
                if resp.status_code == 200:
                    return resp
                last = f"HTTP {resp.status_code}"
                if resp.status_code not in _RETRY_STATUS:
                    raise EmbeddingError(f"voyage request failed: {last}")
            if attempt < self._retries:
                time.sleep(self._backoff_s * 2**attempt)
        raise EmbeddingError(f"voyage request failed after retries: {last}")


class VoyageEmbedder(_VoyageHTTP):
    """Voyage AI embeddings over HTTPS. The key is read from `VOYAGE_API_KEY`."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        client: httpx.Client | None = None,
        retries: int = 4,
        backoff_s: float = 1.0,
        url: str = VOYAGE_URL,
    ) -> None:
        super().__init__(api_key, model, url, client=client, retries=retries, backoff_s=backoff_s)

    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            out.extend(self._embed_batch(list(texts[start : start + _MAX_BATCH]), input_type))
        return out

    def _embed_batch(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        body = {"input": batch, "model": self.model, "input_type": input_type, "truncation": True}
        resp = self._post_with_retry(body)
        rows = sorted(resp.json()["data"], key=lambda r: r["index"])
        vectors = [list(map(float, r["embedding"])) for r in rows]
        if len(vectors) != len(batch) or any(len(v) != EMBEDDING_DIMS for v in vectors):
            raise EmbeddingError(f"unexpected embedding shape from model {self.model}")
        return vectors


class VoyageReranker(_VoyageHTTP):
    """Voyage cross-encoder reranking: reads query and document together, so it orders a short list better
    than vector distance can."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_RERANK_MODEL,
        *,
        client: httpx.Client | None = None,
        retries: int = 4,
        backoff_s: float = 1.0,
        url: str = RERANK_URL,
    ) -> None:
        super().__init__(api_key, model, url, client=client, retries=retries, backoff_s=backoff_s)

    def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        body = {"query": query, "documents": list(documents), "model": self.model, "top_k": top_k}
        rows = self._post_with_retry(body).json()["data"]
        return [(int(r["index"]), float(r["relevance_score"])) for r in rows]


def cache_key(model: str, input_type: InputType, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{input_type}\x00{text}".encode()).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


class CachedEmbedder:
    """Wraps an embedder with a MongoDB-backed cache; only misses reach the inner embedder."""

    def __init__(self, inner: Embedder, collection: Collection[dict[str, Any]]) -> None:
        self._inner = inner
        self._coll = collection
        self.model = inner.model
        self.stats = CacheStats()

    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        keys = [cache_key(self.model, input_type, t) for t in texts]
        found: dict[str, list[float]] = {
            d["_id"]: d["embedding"] for d in self._coll.find({"_id": {"$in": list(set(keys))}})
        }
        missing: dict[str, str] = {k: t for k, t in zip(keys, texts, strict=True) if k not in found}
        self.stats.hits += sum(1 for k in keys if k in found)
        self.stats.misses += len(missing)
        if missing:
            vectors = self._inner.embed(list(missing.values()), input_type)
            ops = [
                UpdateOne(
                    {"_id": k},
                    {"$setOnInsert": {"model": self.model, "input_type": input_type, "embedding": v}},
                    upsert=True,
                )
                for k, v in zip(missing, vectors, strict=True)
            ]
            self._coll.bulk_write(ops, ordered=False)
            found.update(zip(missing, vectors, strict=True))
        return [found[k] for k in keys]
