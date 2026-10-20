"""Ingestion pipeline: validate -> parse -> metadata -> chunk, with a status report per file.

A bad file never crashes a batch: it comes back as an IngestResult with status "failed"
(corrupt / empty / malformed), "skipped" (unsupported type) or "duplicate" (same bytes as a
file already ingested).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chunker import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_SENTENCES, chunk_document
from .metadata import build_document_metadata, doc_id_for
from .models import Chunk, ParseError
from .parsers import SUPPORTED_EXTENSIONS, parse_bytes

MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB


@dataclass
class IngestResult:
    filename: str
    status: str  # "ok" | "failed" | "skipped" | "duplicate"
    doc_id: str
    sha256: str = ""
    error: str | None = None
    seconds: float = 0.0
    document: dict[str, Any] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def summary(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "status": self.status,
            "doc_id": self.doc_id,
            "sha256": self.sha256,
            "n_chunks": self.n_chunks,
            "doc_status": self.document.get("status"),
            "error": self.error,
            "seconds": round(self.seconds, 4),
        }


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest_bytes(
    data: bytes,
    filename: str,
    known_hashes: dict[str, str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> IngestResult:
    """Ingest one file's bytes. `known_hashes` maps sha256 -> filename for deduplication."""
    start = time.perf_counter()
    filename = Path(filename).name  # never trust directory parts of an uploaded name
    result = IngestResult(filename=filename, status="failed", doc_id=doc_id_for(filename))

    def done(status: str, error: str | None = None) -> IngestResult:
        result.status, result.error = status, error
        result.seconds = time.perf_counter() - start
        return result

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return done("skipped", f"unsupported file type '{ext or '(none)'}'")
    if len(data) > MAX_FILE_BYTES:
        return done("failed", f"file too large ({len(data)} bytes > {MAX_FILE_BYTES})")

    result.sha256 = sha256_of(data)
    if known_hashes is not None:
        original = known_hashes.get(result.sha256)
        if original is not None and original != filename:
            return done("duplicate", f"identical content already ingested as {original}")

    try:
        parsed = parse_bytes(data, filename)
    except ParseError as exc:
        return done("failed", str(exc))
    except Exception as exc:  # unexpected parser crash: report, don't propagate
        return done("failed", f"{filename}: unexpected parse error: {type(exc).__name__}: {exc}")

    result.document = build_document_metadata(parsed, result.sha256)
    result.chunks = chunk_document(parsed, result.document, max_chars, overlap_sentences)
    if not result.chunks:
        return done("failed", f"{filename}: produced no chunks")
    if known_hashes is not None:
        known_hashes[result.sha256] = filename
    return done("ok")


def ingest_path(path: str | Path, known_hashes: dict[str, str] | None = None, **kwargs: Any) -> IngestResult:
    path = Path(path)
    return ingest_bytes(path.read_bytes(), path.name, known_hashes, **kwargs)


def ingest_directory(directory: str | Path, **kwargs: Any) -> list[IngestResult]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    known: dict[str, str] = {}
    files = sorted(p for p in directory.rglob("*") if p.is_file() and not p.name.startswith("."))
    return [ingest_path(p, known, **kwargs) for p in files]
