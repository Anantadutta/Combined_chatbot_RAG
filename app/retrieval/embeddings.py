"""Embedders.

GeminiEmbedder - gemini-embedding-001 (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY task types),
                 vectors L2-normalised because we request fewer than 3072 dimensions.
HashEmbedder   - deterministic offline embedder (feature hashing of words, word pairs and
                 character trigrams). No network; used in tests and when no API key is set.
CachedEmbedder - wraps either one and caches document vectors on disk by content hash, so
                 re-indexing unchanged chunks costs no API calls.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        toks = tokenize(text)
        feats = [(t, 1.0) for t in toks]
        feats += [(f"{a}_{b}", 0.7) for a, b in zip(toks, toks[1:])]
        for t in toks:
            padded = f"#{t}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.3) for i in range(len(padded) - 2)]
        return feats

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for feat, weight in self._features(text):
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += weight if (h >> 63) & 1 else -weight
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class GeminiEmbedder:
    def __init__(self, client, model: str = "gemini-embedding-001", dim: int = 768):
        self.client = client
        self.model = model
        self.dim = dim
        self.name = f"gemini:{model}:{dim}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalise(np.array(self.client.embed(texts, "RETRIEVAL_DOCUMENT", self.model, self.dim)))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalise(np.array(self.client.embed([text], "RETRIEVAL_QUERY", self.model, self.dim)))[0]


class CachedEmbedder:
    """Disk cache for document embeddings: {sha1(model|text): vector}."""

    def __init__(self, inner: Embedder, cache_dir: str | Path | None):
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self._lock = threading.Lock()
        self._path = None
        self._cache: dict[str, list[float]] = {}
        if cache_dir:
            safe = "".join(c if c.isalnum() else "_" for c in inner.name)
            self._path = Path(cache_dir) / f"embeddings_{safe}.json"
            if self._path.exists():
                try:
                    self._cache = json.loads(self._path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    self._cache = {}

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.inner.name}|{text}".encode("utf-8")).hexdigest()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            fresh = self.inner.embed_documents([texts[i] for i in missing])
            with self._lock:
                for i, vec in zip(missing, fresh):
                    self._cache[keys[i]] = [round(float(x), 6) for x in vec]
                self._save()
        self.last_misses = len(missing)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.array([self._cache[k] for k in keys], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed_query(text)

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache), encoding="utf-8")
        tmp.replace(self._path)
