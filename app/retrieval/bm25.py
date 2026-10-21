"""BM25 (Okapi) keyword index, written from scratch.

score(q, d) = sum over query terms t of
    idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
"""
from __future__ import annotations

import math
from collections import Counter

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tfs: list[Counter] = []
        self.lengths: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def build(self, docs: list[tuple[str, str]]) -> None:
        self.ids = [d[0] for d in docs]
        self.tfs = [Counter(tokenize(text)) for _, text in docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        n = len(docs)
        self.avgdl = (sum(self.lengths) / n) if n else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, limit: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        if not terms or not self.ids:
            return []
        scored = []
        for i, tf in enumerate(self.tfs):
            if allowed is not None and self.ids[i] not in allowed:
                continue
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            if s > 0:
                scored.append((self.ids[i], s))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]
