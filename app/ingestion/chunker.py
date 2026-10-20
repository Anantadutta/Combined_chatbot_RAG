"""Structure-aware chunking.

Rules:
  * A chunk never crosses a section boundary (page paragraph, heading section, table row,
    JSON record), so every chunk has one precise citation location.
  * Records (table / CSV / JSON rows) are kept whole - splitting a row would separate a value
    from its column name.
  * Long prose sections are split on sentence boundaries into chunks of at most `max_chars`,
    with `overlap_sentences` sentences repeated between neighbouring chunks.
  * Each chunk gets an `embed_text` = contextual header + text, so a chunk such as
    "Milestone 2: integration due 2026-10-28" is still findable by "Orion".
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from .metadata import find_entity_ids, looks_like_injection
from .models import Chunk, ParsedDocument, Section

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP_SENTENCES = 1

# split after . ! ? or a line break, but not inside "e.g." / decimals / "v2.1"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Split one over-long sentence on word boundaries."""
    words, parts, cur = sentence.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            parts.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        parts.append(cur)
    return parts


def split_text(text: str, max_chars: int, overlap_sentences: int) -> list[str]:
    """Greedy sentence packing with sentence overlap."""
    if len(text) <= max_chars:
        return [text]
    sentences: list[str] = []
    for s in split_sentences(text):
        sentences.extend(_hard_split(s, max_chars) if len(s) > max_chars else [s])

    pieces: list[str] = []
    cur: list[str] = []
    for s in sentences:
        if cur and len(" ".join(cur + [s])) > max_chars:
            pieces.append(" ".join(cur))
            keep = cur[-overlap_sentences:] if overlap_sentences else []
            # only keep the overlap if it leaves room for the new sentence
            cur = keep if len(" ".join(keep + [s])) <= max_chars else []
        cur.append(s)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _header(doc_meta: dict[str, Any], section: Section) -> str:
    parts = [f"Document: {doc_meta['filename']}"]
    if doc_meta.get("title") and doc_meta["title"] != doc_meta["filename"]:
        parts.append(f"Title: {doc_meta['title']}")
    parts.append(f"Status: {doc_meta['status']}")
    if section.heading and section.heading != doc_meta.get("title"):
        parts.append(f"Section: {section.heading}")
    parts.append(f"Location: {section.location}")
    return " | ".join(parts)


def chunk_document(
    doc: ParsedDocument,
    doc_meta: dict[str, Any],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[Chunk]:
    doc_id = doc_meta["doc_id"]
    chunks: list[Chunk] = []
    seen: set[str] = set()

    for section_index, section in enumerate(doc.sections):
        pieces = [section.text] if section.kind == "record" else split_text(section.text, max_chars, overlap_sentences)
        for part_index, piece in enumerate(pieces):
            content_hash = hashlib.sha1(piece.encode("utf-8")).hexdigest()
            if content_hash in seen:  # exact duplicate text inside the same document
                continue
            seen.add(content_hash)

            location = section.location
            if len(pieces) > 1:
                location = f"{location} (part {part_index + 1}/{len(pieces)})"
            header = _header(doc_meta, Section(piece, location, section.loc, section.heading, section.kind))

            metadata = {
                # document-level (copied so the vector store can filter on them)
                "doc_id": doc_id,
                "filename": doc_meta["filename"],
                "file_type": doc_meta["file_type"],
                "title": doc_meta["title"],
                "category": doc_meta["category"],
                "status": doc_meta["status"],
                "authoritative": doc_meta["authoritative"],
                "effective_date": doc_meta["effective_date"],
                "version": doc_meta["version"],
                "doc_sha256": doc_meta["sha256"],
                # chunk-level
                "location": location,
                "loc": section.loc,
                "heading": section.heading,
                "kind": section.kind,
                "section_index": section_index,
                "part": part_index + 1,
                "parts": len(pieces),
                "chunk_index": len(chunks),
                "entity_ids": find_entity_ids(piece),
                "content_sha1": content_hash,
                "contains_injection": looks_like_injection(piece),
                "n_chars": len(piece),
            }
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{len(chunks):04d}",
                    doc_id=doc_id,
                    text=piece,
                    embed_text=f"{header}\n{piece}",
                    metadata=metadata,
                )
            )
    return chunks
