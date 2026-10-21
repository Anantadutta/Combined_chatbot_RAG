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
from ..ingestion import SUPPORTED_EXTENSIONS, IngestResult, ingest_bytes, ingest_directory
from ..ingestion.metadata import doc_id_for
from ..ingestion.pipeline import sha256_of
from ..retrieval import (CachedEmbedder, GeminiEmbedder, GeminiReranker, HashEmbedder, HybridRetriever,
                         LocalReranker, NoReranker, VectorStore)
from ..retrieval.hybrid import Candidate
from ..retrieval.text import tokenize
from .conversation import Turn, history_text, rewrite_question
from .guardrails import (Intent, answer_flags, apply_authority_policy, build_supersession_links, detect_intent,
                         leaks_system_prompt)
from .prompts import SYSTEM_PROMPT, build_prompt

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
NO_ANSWER = "I could not find this in the NexaCorp documents."
REFUSAL = "I can only answer questions about NexaCorp documents, and I can't share my internal instructions."
NON_AUTHORITATIVE_NOTE = "Note: this is based only on a superseded or draft document, which is not in force. "
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
    flags: list[str] = field(default_factory=list)
    reasoning: str | None = None  # model scratchpad: kept for debugging (--json), never shown in the answer
    standalone_question: str | None = None  # follow-up rewritten for retrieval (multi-turn chat)
    rewrite: str = "none"  # how the follow-up was rewritten: none | llm | heuristic

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self, show_sources: bool = False) -> str:
        lines = [self.answer, ""]
        if self.rewrite != "none" and self.standalone_question:
            lines.insert(0, f"(interpreted as: {self.standalone_question})")
        if self.flags:
            lines.insert(1, f"[flags: {', '.join(self.flags)}]")
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
        self._index_lock = threading.RLock()  # one writer at a time; readers never wait for it

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

    # ------------------------------------------------------------------ document library
    # Sources of truth: data/raw (the corpus) + data/uploads (added through the API/UI; a file there
    # overrides a corpus file with the same name). Deleting a corpus file records a "tombstone" so that
    # a later reindex does not bring it back.
    @property
    def _tombstone_file(self) -> Path | None:
        return None if self.settings.qdrant_path == ":memory:" else Path(self.settings.index_dir) / "deleted.json"

    def _tombstones(self) -> set[str]:
        f = self._tombstone_file
        if f is None:
            return getattr(self, "_mem_tombstones", set())
        try:
            return set(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return set()

    def _save_tombstones(self, names: set[str]) -> None:
        f = self._tombstone_file
        if f is None:
            self._mem_tombstones = set(names)
            return
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(sorted(names), indent=2), encoding="utf-8")

    def source_files(self) -> dict[str, Path]:
        """filename -> path of every document that should be in the index."""
        files: dict[str, Path] = {}
        for folder in (self.settings.data_dir, self.settings.uploads_dir):  # uploads win on name clashes
            d = Path(folder)
            if d.is_dir():
                for p in sorted(d.rglob("*")):
                    if p.is_file() and not p.name.startswith("."):
                        files[p.name] = p
        dead = self._tombstones()
        return {name: p for name, p in files.items() if name not in dead}

    def documents(self) -> list[dict[str, Any]]:
        docs: dict[str, dict] = {}
        uploads = Path(self.settings.uploads_dir)
        for p in list(self.retriever.payloads.values()):
            d = docs.get(p["doc_id"])
            if d is None:
                d = docs[p["doc_id"]] = {
                    "doc_id": p["doc_id"], "filename": p["filename"], "file_type": p.get("file_type"),
                    "status": p.get("status"), "authoritative": p.get("authoritative"),
                    "category": p.get("category"), "title": p.get("title"),
                    "effective_date": p.get("effective_date"), "superseded_date": p.get("superseded_date"),
                    "sha256": p.get("doc_sha256"), "chunks": 0,
                    "source": "upload" if (uploads / p["filename"]).is_file() else "corpus",
                }
            d["chunks"] += 1
        links = self.supersession_links()
        for d in docs.values():
            if d["doc_id"] in links:
                d["superseded_by"] = links[d["doc_id"]]["by"]
        return sorted(docs.values(), key=lambda d: d["filename"].lower())

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return next((d for d in self.documents() if d["doc_id"] == doc_id), None)

    def add_document(self, data: bytes, filename: str, expect_doc_id: str | None = None) -> dict[str, Any]:
        """Upload or update one file. Returns the ingestion report plus an `action`:

        created | updated | unchanged (same name, same bytes) | duplicate (same bytes, other name) |
        failed (malformed / empty / too large) | skipped (unsupported type) | rejected (wrong doc id).
        """
        filename = Path(str(filename or "")).name.strip()
        doc_id = doc_id_for(filename) if filename else ""
        if expect_doc_id is not None and doc_id != expect_doc_id:
            return {"filename": filename, "doc_id": doc_id, "action": "rejected", "status": "failed",
                    "error": f"file name must match the document being updated ({expect_doc_id})", "n_chunks": 0}
        with self._index_lock:
            existing = self.get_document(doc_id) if doc_id else None
            sha = sha256_of(data)
            if existing and existing["sha256"] == sha:
                return {"filename": filename, "doc_id": doc_id, "action": "unchanged", "status": "ok",
                        "sha256": sha, "n_chunks": existing["chunks"], "doc_status": existing["status"],
                        "error": None}
            others = {d["sha256"]: d["filename"] for d in self.documents() if d["doc_id"] != doc_id and d["sha256"]}
            result = ingest_bytes(data, filename or "(no name)", known_hashes=others)
            report = result.summary()
            if result.status != "ok":
                report["action"] = result.status
                return report
            self.index_results([result])
            self._store_upload(filename, data)
            dead = self._tombstones()
            if filename in dead:
                self._save_tombstones(dead - {filename})
            report["action"] = "updated" if existing else "created"
            return report

    def _store_upload(self, filename: str, data: bytes) -> None:
        folder = Path(self.settings.uploads_dir)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f".{filename}.tmp"
        tmp.write_bytes(data)
        tmp.replace(folder / filename)

    def delete_document(self, doc_id: str) -> dict[str, Any] | None:
        """Remove a document from the index (and its uploaded file). Returns what was deleted, or None."""
        with self._index_lock:
            doc = self.get_document(doc_id)
            if doc is None:
                return None
            self.retriever.delete_doc(doc_id)
            upload = Path(self.settings.uploads_dir) / doc["filename"]
            if upload.is_file():
                upload.unlink()
            if (Path(self.settings.data_dir) / doc["filename"]).is_file():
                self._save_tombstones(self._tombstones() | {doc["filename"]})
            return doc

    def reindex(self) -> list[IngestResult]:
        """Rebuild the index from data/raw + data/uploads (minus deleted files). Embeddings come from cache."""
        with self._index_lock:
            known: dict[str, str] = {}
            files = self.source_files()
            results = [ingest_bytes(p.read_bytes(), name, known) for name, p in sorted(files.items())]
            keep = {r.doc_id for r in results if r.status == "ok"}
            for d in self.documents():
                if d["doc_id"] not in keep:
                    self.retriever.delete_doc(d["doc_id"])
            self.index_results(results)
            return results

    def stats(self) -> dict[str, Any]:
        docs = self.documents()
        by_status: dict[str, int] = {}
        for d in docs:
            by_status[d["status"]] = by_status.get(d["status"], 0) + 1
        return {"chunks": len(self.retriever.payloads), "n_documents": len(docs), "by_status": by_status,
                "documents": docs, "supported_types": sorted(SUPPORTED_EXTENSIONS), **self.settings.describe()}

    # ------------------------------------------------------------------ answering
    def retrieve(self, question: str, filters: dict | None = None, top_k: int | None = None,
                 intent: Intent | None = None) -> tuple[list[Candidate], dict]:
        """Hybrid search -> rerank all candidates -> authority policy -> top_k sources."""
        intent = intent or detect_intent(question)
        if filters and any(k in filters for k in ("status", "filename", "doc_id")):
            # the caller explicitly chose which documents to search: respect that choice
            intent = Intent(historical=True, drafts=True, injection=intent.injection)
        t0 = time.perf_counter()
        candidates = self.retriever.search(question, self.settings.candidate_k, filters)
        t1 = time.perf_counter()
        ranked = self.reranker.rerank(question, candidates, len(candidates))
        policy = apply_authority_policy(ranked, intent, top_k or self.settings.top_k)
        t2 = time.perf_counter()
        return policy.sources, {"retrieve": (t1 - t0) * 1000, "rerank": (t2 - t1) * 1000}

    def supersession_links(self) -> dict[str, dict]:
        payloads = self.retriever.payloads
        if getattr(self, "_links_for", None) is not payloads:  # recompute after every (re)index
            self._links = build_supersession_links(list(payloads.values()))
            self._links_for = payloads
        return self._links

    def ask(self, question: str, filters: dict | None = None, top_k: int | None = None,
            history: list[Turn] | None = None) -> Answer:
        start = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise ValueError("question is empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"question is longer than {MAX_QUESTION_CHARS} characters")

        # multi-turn: resolve "it / that / the previous one" before searching
        timings: dict[str, float] = {}
        standalone, method = question, "none"
        if history and self.settings.rewrite_follow_ups:
            t_rw = time.perf_counter()
            standalone, method = rewrite_question(question, history, self.llm)
            timings["rewrite"] = (time.perf_counter() - t_rw) * 1000

        intent = detect_intent(standalone)
        intent.injection = intent.injection or detect_intent(question).injection
        top, t_retrieval = self.retrieve(standalone, filters, top_k, intent)
        timings.update(t_retrieval)
        sources = [(f"S{i}", c) for i, c in enumerate(top, start=1)]

        t_gen = time.perf_counter()
        model = None
        if not top:
            data = {"answer": NO_ANSWER, "citations": [], "abstained": True, "confidence": "high"}
        elif self.llm is None:
            data = self._extractive_answer(standalone, sources)
        else:
            try:
                asked = question if method == "none" else f"{question}\n(standalone form: {standalone})"
                prompt = build_prompt(asked, sources, dt.date.today().isoformat(), self.supersession_links(),
                                      injection_suspected=intent.injection,
                                      history=history_text(history) if history else None)
                raw = self.llm.generate(SYSTEM_PROMPT, prompt, temperature=self.settings.temperature)
                data = parse_llm_json(raw)
                model = getattr(self.llm, "last_model_used", None)
            except Exception as exc:  # Gemini down / quota exhausted: degrade instead of crashing
                log.error("generation failed, returning best passage instead: %s", exc)
                data = self._extractive_answer(standalone, sources)
                if not data["abstained"]:
                    data["answer"] = ("(The AI model is unavailable right now, so this is the most relevant passage "
                                      "from the documents.) " + data["answer"])
                model = "unavailable-fallback"
        timings["generate"] = (time.perf_counter() - t_gen) * 1000

        answer = self._finalise(question, data, sources, intent)
        answer.model = model
        answer.standalone_question = standalone if method != "none" else None
        answer.rewrite = method
        timings["total"] = (time.perf_counter() - start) * 1000
        answer.timings_ms = {k: round(v, 1) for k, v in timings.items()}
        return answer

    def _finalise(self, question: str, data: dict, sources: list[tuple[str, Candidate]],
                  intent: Intent | None = None) -> Answer:
        intent = intent or Intent()
        by_label = {label: c for label, c in sources}
        flags: list[str] = []
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
        reasoning = str(data.get("reasoning") or "").strip() or None

        if intent.injection:
            flags.append("injection_attempt_in_question")
        if any(c.payload.get("contains_injection") for c in by_label.values()):
            flags.append("injection_text_in_sources")

        if leaks_system_prompt(text):  # output guard: never reveal the hidden instructions
            flags.append("blocked_prompt_leak")
            text, valid, abstained, confidence, reasoning = REFUSAL, [], True, "high", None
        elif not abstained and not valid:
            # grounding guard: an answer that cites nothing is not shown as fact
            flags.append("ungrounded_answer_suppressed")
            text, abstained, confidence = NO_ANSWER, True, "low"
        elif abstained:
            valid = [l for l in valid if l in CITATION.findall(text)]  # keep citations only if used in the text

        statuses = [by_label[l].payload.get("status") for l in valid]
        extra = answer_flags(statuses, intent) if not abstained else []
        if "relies_on_non_authoritative" in extra:
            text, confidence = NON_AUTHORITATIVE_NOTE + text, "low"
        flags += extra

        citations = [
            Citation(l, by_label[l].payload["filename"], by_label[l].payload["location"],
                     by_label[l].payload["status"], by_label[l].chunk_id, by_label[l].payload["text"][:300])
            for l in valid
        ]
        return Answer(question=question, answer=text or NO_ANSWER, abstained=abstained, confidence=confidence,
                      citations=citations, retrieved=[c.brief() for _, c in sources], flags=flags,
                      reasoning=reasoning)

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
