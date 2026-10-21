"""The RAG engine: ingest -> index -> retrieve (hybrid) -> rerank -> generate (cited answer)."""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings, load_settings
from ..ingestion import IngestResult, ingest_bytes, ingest_directory
from ..retrieval import (CachedEmbedder, GeminiEmbedder, GeminiReranker, HashEmbedder, HybridRetriever,
                         LocalReranker, NoReranker, VectorStore)
from ..retrieval.hybrid import Candidate
from ..retrieval.text import tokenize
from .prompts import SYSTEM_PROMPT, build_prompt

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
NO_ANSWER = "I could not find this in the NexaCorp documents."
CITATION = re.compile(r"\[(S\d+)\]")


@dataclass
class Citation:
    label: str
    filename: str
    location: str
    status: str
    chunk_id: str
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    confidence: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    model: str | None = None
    reasoning: str | None = None  # model scratchpad, never shown in the answer

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self, show_sources: bool = False) -> str:
        lines = [self.answer, ""]
        for c in self.citations:
            lines.append(f"  [{c.label}] {c.filename} - {c.location} ({c.status})")
        if show_sources:
            lines += ["", "Retrieved (after rerank):"]
            lines += [f"  {r['score']:.3f}  {r['filename']} - {r['location']} ({r['status']})" for r in self.retrieved]
        t = self.timings_ms
        lines.append(f"\n({t.get('total', 0):.0f} ms: retrieve {t.get('retrieve', 0):.0f}, "
                     f"rerank {t.get('rerank', 0):.0f}, generate {t.get('generate', 0):.0f}"
                     f"{', model ' + self.model if self.model else ''})")
        return "\n".join(lines)


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Parse the model's JSON, tolerating code fences or text around it."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    for candidate in (text, (re.search(r"\{.*\}", text, re.S) or [None])[0]):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and "answer" in data:
                return data
        except json.JSONDecodeError:
            continue
    # not JSON: treat the whole reply as the answer text
    return {"answer": raw.strip(), "citations": CITATION.findall(raw), "abstained": False, "confidence": "low"}


class RAGEngine:
    def __init__(self, settings: Settings | None = None, *, embedder=None, llm=None, store: VectorStore | None = None):
        self.settings = settings or load_settings()
        s = self.settings
        if llm is None and s.llm_provider == "gemini":
            from ..llm import GeminiClient

            llm = GeminiClient(s.gemini_api_key, s.gemini_model, s.gemini_fallback_model,
                               s.request_timeout_s, s.max_retries)
        self.llm = llm

        if embedder is None:
            if s.embedding_provider == "gemini":
                if llm is None:
                    from ..llm import GeminiClient

                    client = GeminiClient(s.gemini_api_key, s.gemini_model, None, s.request_timeout_s, s.max_retries)
                else:
                    client = llm
                embedder = GeminiEmbedder(client, s.embedding_model, s.embedding_dim)
            else:
                embedder = HashEmbedder()
            cache = None if s.qdrant_path == ":memory:" else s.index_dir
            embedder = CachedEmbedder(embedder, cache)
        self.embedder = embedder

        self.store = store or VectorStore(s.collection, embedder.dim,
                                          path=None if s.qdrant_url else s.resolved_qdrant_path,
                                          url=s.qdrant_url, api_key=s.qdrant_api_key)
        self.retriever = HybridRetriever(self.embedder, self.store)
        if s.reranker == "gemini" and self.llm is not None:
            self.reranker = GeminiReranker(self.llm)
        elif s.reranker == "none":
            self.reranker = NoReranker()
        else:
            self.reranker = LocalReranker()
        self._index_lock = threading.Lock()

    # ------------------------------------------------------------------ indexing
    def index_results(self, results: list[IngestResult]) -> int:
        chunks = [c for r in results if r.status == "ok" for c in r.chunks]
        with self._index_lock:
            return self.retriever.index_chunks(chunks)

    def index_directory(self, path: str | Path | None = None) -> list[IngestResult]:
        results = ingest_directory(path or self.settings.data_dir)
        self.index_results(results)
        return results

    def index_file(self, data: bytes, filename: str) -> IngestResult:
        result = ingest_bytes(data, filename)
        if result.status == "ok":
            self.index_results([result])
        return result

    def delete_document(self, doc_id: str) -> None:
        with self._index_lock:
            self.retriever.delete_doc(doc_id)

    def stats(self) -> dict[str, Any]:
        docs: dict[str, dict] = {}
        for p in self.retriever.payloads.values():
            d = docs.setdefault(p["doc_id"], {"doc_id": p["doc_id"], "filename": p["filename"],
                                              "status": p["status"], "chunks": 0})
            d["chunks"] += 1
        return {"chunks": len(self.retriever.payloads), "documents": sorted(docs.values(), key=lambda d: d["filename"]),
                **self.settings.describe()}

    # ------------------------------------------------------------------ answering
    def retrieve(self, question: str, filters: dict | None = None, top_k: int | None = None) -> tuple[list[Candidate], dict]:
        t0 = time.perf_counter()
        candidates = self.retriever.search(question, self.settings.candidate_k, filters)
        t1 = time.perf_counter()
        top = self.reranker.rerank(question, candidates, top_k or self.settings.top_k)
        t2 = time.perf_counter()
        return top, {"retrieve": (t1 - t0) * 1000, "rerank": (t2 - t1) * 1000}

    def ask(self, question: str, filters: dict | None = None, top_k: int | None = None) -> Answer:
        start = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise ValueError("question is empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"question is longer than {MAX_QUESTION_CHARS} characters")

        top, timings = self.retrieve(question, filters, top_k)
        sources = [(f"S{i}", c) for i, c in enumerate(top, start=1)]

        t_gen = time.perf_counter()
        model = None
        if not top:
            data = {"answer": NO_ANSWER, "citations": [], "abstained": True, "confidence": "high"}
        elif self.llm is None:
            data = self._extractive_answer(question, sources)
        else:
            try:
                raw = self.llm.generate(SYSTEM_PROMPT, build_prompt(question, sources, dt.date.today().isoformat()),
                                        temperature=self.settings.temperature)
                data = parse_llm_json(raw)
                model = getattr(self.llm, "last_model_used", None)
            except Exception as exc:  # Gemini down / quota exhausted: degrade instead of crashing
                log.error("generation failed, returning best passage instead: %s", exc)
                data = self._extractive_answer(question, sources)
                if not data["abstained"]:
                    data["answer"] = ("(The AI model is unavailable right now, so this is the most relevant passage "
                                      "from the documents.) " + data["answer"])
                model = "unavailable-fallback"
        timings["generate"] = (time.perf_counter() - t_gen) * 1000

        answer = self._finalise(question, data, sources)
        answer.model = model
        timings["total"] = (time.perf_counter() - start) * 1000
        answer.timings_ms = {k: round(v, 1) for k, v in timings.items()}
        return answer

    def _finalise(self, question: str, data: dict, sources: list[tuple[str, Candidate]]) -> Answer:
        by_label = {label: c for label, c in sources}
        text = str(data.get("answer") or "").strip()
        abstained = bool(data.get("abstained")) or not text
        # citations = those listed + those used inline, keeping only labels that really exist
        listed = [str(x).strip("[] ").upper() for x in (data.get("citations") or [])]
        labels = list(dict.fromkeys(listed + CITATION.findall(text)))
        valid = [l for l in labels if l in by_label]
        # drop inline references to sources that do not exist (hallucinated labels)
        text = CITATION.sub(lambda m: m.group(0) if m.group(1) in by_label else "", text).strip()
        confidence = str(data.get("confidence") or "medium").lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        if not abstained and not valid:
            confidence = "low"  # an answer with no valid citation is not grounded
        citations = [
            Citation(l, by_label[l].payload["filename"], by_label[l].payload["location"],
                     by_label[l].payload["status"], by_label[l].chunk_id, by_label[l].payload["text"][:300])
            for l in valid
        ]
        return Answer(question=question, answer=text or NO_ANSWER, abstained=abstained, confidence=confidence,
                      citations=citations, retrieved=[c.brief() for _, c in sources],
                      reasoning=str(data.get('reasoning') or '').strip() or None)

    def _extractive_answer(self, question: str, sources: list[tuple[str, Candidate]]) -> dict:
        """Offline mode (no LLM): return the best authoritative passage verbatim, or abstain."""
        q_terms = set(tokenize(question))
        best = None
        for label, c in sources:
            terms = set(tokenize(c.payload.get("embed_text", "")))
            coverage = len(q_terms & terms) / len(q_terms) if q_terms else 0
            if c.payload.get("authoritative", True) and (best is None or coverage > best[0]):
                best = (coverage, label, c)
        if best is None or best[0] < 0.5:
            return {"answer": NO_ANSWER, "citations": [], "abstained": True, "confidence": "low"}
        _, label, c = best
        return {"answer": f"{c.payload['text']} [{label}]", "citations": [label], "abstained": False,
                "confidence": "low"}

    def close(self) -> None:
        self.store.close()
