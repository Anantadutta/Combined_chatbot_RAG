"""Qdrant vector store.

Local mode (a folder on disk, no server) by default; set QDRANT_URL (+ QDRANT_API_KEY) to use
Qdrant Cloud or a Qdrant server - useful for cloud deployment.
Each point = one chunk; payload = chunk text + all chunk metadata, so we can filter on
status, category, filename, doc_id, entity_ids, ...
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

Filters = dict[str, Any]  # {"status": ["current", "unspecified"], "category": "policy", ...}


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def payload_matches(payload: dict, filters: Filters | None) -> bool:
    """Python mirror of the Qdrant filter (used by BM25 so both retrievers obey the same filter)."""
    for key, wanted in (filters or {}).items():
        value = payload.get(key)
        wanted_set = set(wanted) if isinstance(wanted, (list, tuple, set)) else {wanted}
        values = set(value) if isinstance(value, list) else {value}
        if not values & wanted_set:
            return False
    return True


class VectorStore:
    def __init__(self, collection: str, dim: int, path: str | None = None, url: str | None = None,
                 api_key: str | None = None):
        from qdrant_client import QdrantClient

        if url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
            self.local = False
        elif path in (None, ":memory:"):
            self.client = QdrantClient(location=":memory:")
            self.local = True
        else:
            self.client = QdrantClient(path=path)
            self.local = True
        self.collection = collection
        self.dim = dim
        self._lock = threading.RLock()  # local-mode Qdrant is not thread-safe
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        with self._lock:
            if self.client.collection_exists(self.collection):
                size = self.client.get_collection(self.collection).config.params.vectors.size
                if size == self.dim:
                    return
                log.warning("collection %s has dim %s, embedder has %s - recreating (re-index needed)",
                            self.collection, size, self.dim)
                self.client.delete_collection(self.collection)
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )
            if not self.local:  # payload indexes speed up filtering on a server
                for field in ("doc_id", "status", "category", "filename", "file_type", "entity_ids"):
                    self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)

    # ------------------------------------------------------------------ writes
    def upsert(self, payloads: list[dict], vectors: np.ndarray) -> None:
        from qdrant_client import models

        if len(payloads) != len(vectors):
            raise ValueError("payloads and vectors differ in length")
        points = [
            models.PointStruct(id=point_id(p["chunk_id"]), vector=[float(x) for x in v], payload=p)
            for p, v in zip(payloads, vectors)
        ]
        with self._lock:
            for i in range(0, len(points), 256):
                self.client.upsert(self.collection, points=points[i : i + 256], wait=True)

    def delete_doc(self, doc_id: str) -> None:
        from qdrant_client import models

        flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        with self._lock:
            self.client.delete(self.collection, points_selector=models.FilterSelector(filter=flt), wait=True)

    # ------------------------------------------------------------------ reads
    def _qdrant_filter(self, filters: Filters | None):
        from qdrant_client import models

        if not filters:
            return None
        must = []
        for key, wanted in filters.items():
            if isinstance(wanted, (list, tuple, set)):
                must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(wanted))))
            else:
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=wanted)))
        return models.Filter(must=must)

    def search(self, vector: np.ndarray, limit: int, filters: Filters | None = None) -> list[tuple[dict, float]]:
        with self._lock:
            result = self.client.query_points(
                self.collection,
                query=[float(x) for x in vector],
                limit=limit,
                query_filter=self._qdrant_filter(filters),
                with_payload=True,
            )
        return [(p.payload, float(p.score)) for p in result.points]

    def all_payloads(self) -> list[dict]:
        out: list[dict] = []
        offset = None
        with self._lock:
            while True:
                points, offset = self.client.scroll(
                    self.collection, limit=512, offset=offset, with_payload=True, with_vectors=False
                )
                out.extend(p.payload for p in points)
                if offset is None:
                    break
        return out

    def count(self) -> int:
        with self._lock:
            return self.client.count(self.collection, exact=True).count

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # pragma: no cover
            pass
