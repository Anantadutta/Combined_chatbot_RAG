"""FastAPI backend: chat, single questions, document upload/update/delete/reindex, ingestion status.

    uvicorn app.api.server:app --port 8000        (or: python -m app.cli serve)

Endpoints (all JSON; interactive docs at /docs):
    GET    /                      web chat + upload UI
    GET    /health                liveness + index size
    GET    /api/stats             settings, document counts by status
    POST   /api/ask               one question, no memory          {"question", "filters"?, "top_k"?}
    POST   /api/chat              multi-turn question               {"question", "session_id"?, "filters"?}
    GET    /api/chat/{sid}        conversation history
    DELETE /api/chat/{sid}        forget a conversation
    GET    /api/documents         indexed documents with status / chunks
    GET    /api/documents/{id}    one document
    POST   /api/documents         upload one or more files (multipart field "files")
    PUT    /api/documents/{id}    replace a document with a new version (multipart field "file")
    DELETE /api/documents/{id}    delete a document from the index
    POST   /api/reindex           rebuild the index in the background -> job id
    GET    /api/jobs[/{id}]       ingestion / reindex job status

Write endpoints (upload/update/delete/reindex) require the header X-Admin-Token when ADMIN_TOKEN
is set. Run ONE server process with local Qdrant (it locks its folder); use QDRANT_URL to scale out.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..ingestion.pipeline import MAX_FILE_BYTES
from ..rag import RAGEngine
from ..rag.conversation import ConversationStore, Turn
from ..rag.engine import MAX_QUESTION_CHARS

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"
MAX_FILES_PER_UPLOAD = 20
FILTER_KEYS = {"status", "category", "filename", "doc_id", "file_type", "entity_ids"}


# ---------------------------------------------------------------- request / response models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    filters: dict[str, Any] | None = Field(None, description='e.g. {"status": ["current"], "category": "policy"}')
    top_k: int | None = Field(None, ge=1, le=20)


class ChatRequest(AskRequest):
    session_id: str | None = Field(None, max_length=64, description="omit to start a new conversation")


# ---------------------------------------------------------------- ingestion jobs
class JobRegistry:
    """Status of uploads and reindex runs (kept in memory, newest last, bounded)."""

    def __init__(self, keep: int = 100):
        self.keep = keep
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def start(self, kind: str, **info) -> dict:
        job = {"job_id": uuid.uuid4().hex[:12], "kind": kind, "state": "running", "started_at": time.time(),
               "finished_at": None, "seconds": None, "results": [], "error": None, **info}
        with self._lock:
            self._jobs[job["job_id"]] = job
            while len(self._jobs) > self.keep:
                self._jobs.popitem(last=False)
        return job

    def finish(self, job: dict, results: list[dict] | None = None, error: str | None = None) -> dict:
        with self._lock:
            job["results"] = results or []
            job["error"] = error
            job["state"] = "failed" if error else "done"
            job["finished_at"] = time.time()
            job["seconds"] = round(job["finished_at"] - job["started_at"], 3)
            job["summary"] = _count_actions(job["results"])
        return job

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[dict]:
        with self._lock:
            return list(reversed(self._jobs.values()))

    def running(self, kind: str) -> dict | None:
        with self._lock:
            return next((j for j in self._jobs.values() if j["kind"] == kind and j["state"] == "running"), None)


def _count_actions(results: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        key = r.get("action") or r.get("status") or "unknown"
        out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------- app factory
def create_app(engine: RAGEngine | None = None) -> FastAPI:
    """Build the app. Tests pass their own engine; the server builds one from .env at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        own = engine is None
        app.state.engine = engine or RAGEngine()
        if own and app.state.engine.store.count() == 0:
            log.warning("index is empty - indexing %s", app.state.engine.settings.data_dir)
            app.state.engine.reindex()
        yield
        if own:
            app.state.engine.close()

    app = FastAPI(title="NexaCorp RAG Assistant", version="0.4.0", lifespan=lifespan,
                  description="Grounded, cited answers over NexaCorp documents.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.conversations = ConversationStore()
    app.state.jobs = JobRegistry()
    if engine is not None:
        app.state.engine = engine

    def get_engine(request: Request) -> RAGEngine:
        return request.app.state.engine

    def require_admin(request: Request, x_admin_token: str | None = Header(None)) -> None:
        token = request.app.state.engine.settings.admin_token
        if token and x_admin_token != token:
            raise HTTPException(401, "missing or wrong X-Admin-Token header")

    def clean_filters(filters: dict | None) -> dict | None:
        if not filters:
            return None
        bad = set(filters) - FILTER_KEYS
        if bad:
            raise HTTPException(422, f"unknown filter field(s): {sorted(bad)}; allowed: {sorted(FILTER_KEYS)}")
        return {k: v for k, v in filters.items() if v not in (None, "", [])} or None

    def run_question(eng: RAGEngine, req: AskRequest, history: list[Turn] | None = None):
        try:
            return eng.ask(req.question, filters=clean_filters(req.filters), top_k=req.top_k, history=history)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):  # never leak a stack trace to the client
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal error - see server log"})

    # ------------------------------------------------------------ pages / health
    @app.get("/", include_in_schema=False)
    def index_page():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health(eng: RAGEngine = Depends(get_engine)):
        return {"status": "ok", "chunks": len(eng.retriever.payloads), "documents": len(eng.documents()),
                "llm": eng.settings.llm_provider}

    @app.get("/api/stats")
    def stats(eng: RAGEngine = Depends(get_engine)):
        return eng.stats()

    # ------------------------------------------------------------ questions
    @app.post("/api/ask")
    def ask(req: AskRequest, eng: RAGEngine = Depends(get_engine)):
        return run_question(eng, req).to_dict()

    @app.post("/api/chat")
    def chat(req: ChatRequest, request: Request, eng: RAGEngine = Depends(get_engine)):
        store: ConversationStore = request.app.state.conversations
        sid = req.session_id or store.new_id()
        history = store.history(sid)
        answer = run_question(eng, req, history)
        store.append(sid, Turn(question=req.question, answer=answer.answer,
                               standalone_question=answer.standalone_question,
                               sources=list(dict.fromkeys(c.filename for c in answer.citations)),
                               abstained=answer.abstained))
        return {"session_id": sid, "turn": len(history) + 1, **answer.to_dict()}

    @app.get("/api/chat/{session_id}")
    def chat_history(session_id: str, request: Request):
        return {"session_id": session_id,
                "turns": [t.to_dict() for t in request.app.state.conversations.history(session_id)]}

    @app.delete("/api/chat/{session_id}")
    def chat_clear(session_id: str, request: Request):
        return {"session_id": session_id, "cleared": request.app.state.conversations.clear(session_id)}

    # ------------------------------------------------------------ documents
    @app.get("/api/documents")
    def list_documents(eng: RAGEngine = Depends(get_engine)):
        docs = eng.documents()
        return {"count": len(docs), "documents": docs}

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: str, eng: RAGEngine = Depends(get_engine)):
        doc = eng.get_document(doc_id)
        if doc is None:
            raise HTTPException(404, f"no document with id '{doc_id}'")
        return doc

    async def read_upload(f: UploadFile) -> bytes:
        return await f.read(MAX_FILE_BYTES + 1)  # ingestion rejects > MAX_FILE_BYTES with a clear message

    @app.post("/api/documents", dependencies=[Depends(require_admin)])
    async def upload(request: Request, files: list[UploadFile] = File(...)):
        eng: RAGEngine = request.app.state.engine
        if len(files) > MAX_FILES_PER_UPLOAD:
            raise HTTPException(413, f"at most {MAX_FILES_PER_UPLOAD} files per upload")
        job = request.app.state.jobs.start("upload", files=[f.filename for f in files])
        results = []
        for f in files:
            data = await read_upload(f)
            # parsing + embedding is blocking work: run it off the event loop
            results.append(await run_in_threadpool(eng.add_document, data, f.filename or ""))
        request.app.state.jobs.finish(job, results)
        ok = any(r["action"] in ("created", "updated", "unchanged", "duplicate") for r in results)
        return JSONResponse(status_code=200 if ok else 422, content=job)

    @app.put("/api/documents/{doc_id}", dependencies=[Depends(require_admin)])
    async def update(doc_id: str, request: Request, file: UploadFile = File(...)):
        eng: RAGEngine = request.app.state.engine
        job = request.app.state.jobs.start("update", files=[file.filename], doc_id=doc_id)
        result = await run_in_threadpool(eng.add_document, await read_upload(file), file.filename or "", doc_id)
        request.app.state.jobs.finish(job, [result])
        code = 200 if result["action"] in ("created", "updated", "unchanged") else 422
        return JSONResponse(status_code=code, content=job)

    @app.delete("/api/documents/{doc_id}", dependencies=[Depends(require_admin)])
    def delete(doc_id: str, eng: RAGEngine = Depends(get_engine)):
        doc = eng.delete_document(doc_id)
        if doc is None:
            raise HTTPException(404, f"no document with id '{doc_id}'")
        return {"deleted": True, "doc_id": doc_id, "filename": doc["filename"], "chunks_removed": doc["chunks"]}

    @app.post("/api/reindex", status_code=202, dependencies=[Depends(require_admin)])
    def reindex(request: Request, wait: bool = False):
        jobs: JobRegistry = request.app.state.jobs
        if (running := jobs.running("reindex")) is not None:
            return JSONResponse(status_code=409, content={"detail": "a reindex is already running", **running})
        eng: RAGEngine = request.app.state.engine
        job = jobs.start("reindex")

        def work():
            try:
                results = [{**r.summary(), "action": r.status} for r in eng.reindex()]
                jobs.finish(job, results)
            except Exception as exc:
                log.exception("reindex failed")
                jobs.finish(job, error=f"{type(exc).__name__}: {exc}")

        if wait:
            work()
            return JSONResponse(status_code=200, content=job)
        threading.Thread(target=work, name="reindex", daemon=True).start()
        return job

    @app.get("/api/jobs")
    def list_jobs(request: Request):
        return {"jobs": request.app.state.jobs.all()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        job = request.app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job with id '{job_id}'")
        return job

    return app


app = create_app()
