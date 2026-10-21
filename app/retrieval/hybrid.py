"""Hybrid retrieval: dense (Qdrant vectors) + sparse (BM25), merged with Reciprocal Rank Fusion.

RRF(d) = sum over retrievers r of 1 / (RRF_K + rank_r(d))      (RRF_K = 60)

RRF only uses ranks, so it needs no score calibration between cosine similarity and BM25.
Both retrievers apply the same metadata filter.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ..ingestion.models import Chunk
from .bm25 import BM25Index
from .embeddings import Embedder
from .vector_store import Filters, VectorStore, payload_matches

RRF_K = 60


@dataclass
class Candidate:
    chunk_id: str
    payload: dict
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf: float = 0.0
    rerank_score: float | None = None
    features: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf

    def brief(self) -> dict:
        p = self.payload
        return {
            "chunk_id": self.chunk_id,
            "filename": p.get("filename"),
            "location": p.get("location"),
            "status": p.get("status"),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rrf": round(self.rrf, 5),
            "score": round(self.score, 4),
        }


def chunk_payload(chunk: Chunk) -> dict:
    return {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "text": chunk.text,
            "embed_text": chunk.embed_text, **chunk.metadata}


class HybridRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store
        self.bm25 = BM25Index()
        self.payloads: dict[str, dict] = {}
        self._lock = threading.RLock()
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the BM25 index from what is stored in the vector DB (the source of truth)."""
        payloads = self.store.all_payloads()
        payloads.sort(key=lambda p: p["chunk_id"])
        bm25 = BM25Index()
        bm25.build([(p["chunk_id"], p["embed_text"]) for p in payloads])
        with self._lock:
            self.payloads = {p["chunk_id"]: p for p in payloads}
            self.bm25 = bm25

    # ------------------------------------------------------------------ indexing
    def index_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and store chunks. Existing chunks of the same documents are replaced."""
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.embed_text for c in chunks])
        for doc_id in sorted({c.doc_id for c in chunks}):
            self.store.delete_doc(doc_id)
        self.store.upsert([chunk_payload(c) for c in chunks], vectors)
        self.refresh()
        return len(chunks)

    def delete_doc(self, doc_id: str) -> None:
        self.store.delete_doc(doc_id)
        self.refresh()

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 20, filters: Filters | None = None) -> list[Candidate]:
        with self._lock:
            payloads, bm25 = self.payloads, self.bm25
        if not payloads:
            return []

        candidates: dict[str, Candidate] = {}

        qvec = self.embedder.embed_query(query)
        for rank, (payload, score) in enumerate(self.store.search(qvec, k, filters), start=1):
            cid = payload["chunk_id"]
            c = candidates.setdefault(cid, Candidate(cid, payload))
            c.dense_score, c.dense_rank = score, rank
            c.rrf += 1.0 / (RRF_K + rank)

        allowed = {cid for cid, p in payloads.items() if payload_matches(p, filters)} if filters else None
        for rank, (cid, score) in enumerate(bm25.search(query, k, allowed), start=1):
            c = candidates.setdefault(cid, Candidate(cid, payloads[cid]))
            c.bm25_score, c.bm25_rank = score, rank
            c.rrf += 1.0 / (RRF_K + rank)

        return sorted(candidates.values(), key=lambda c: (-c.rrf, c.chunk_id))
