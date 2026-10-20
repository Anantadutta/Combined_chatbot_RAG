"""Data structures shared by the ingestion pipeline.

Flow:  raw file bytes --parser--> ParsedDocument (list of Sections)
                     --metadata--> document metadata (status, dates, ...)
                     --chunker--> list of Chunks (ready to embed/index)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Section:
    """One structural unit of a document: a paragraph, a page block, a table row, a JSON record...

    `location` is the human-readable citation ("page 1", "sheet 'Projects', row 3").
    `loc` holds the same information in machine-readable form ({"page": 1}).
    """

    text: str
    location: str
    loc: dict[str, Any] = field(default_factory=dict)
    heading: str | None = None  # nearest heading / section title, if any
    kind: str = "text"  # "text" (prose) or "record" (table row, CSV row, JSON object)


@dataclass
class ParsedDocument:
    filename: str
    file_type: str  # pdf, docx, xlsx, csv, json, html, txt, md
    title: str
    sections: list[Section]

    @property
    def full_text(self) -> str:
        return "\n".join(s.text for s in self.sections)


@dataclass
class Chunk:
    chunk_id: str  # "<doc_id>::<index>", stable across re-ingestion of the same file
    doc_id: str
    text: str  # the raw chunk text (shown to the LLM and in citations)
    embed_text: str  # text + contextual header (document, status, section) used for embedding/BM25
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ParseError(Exception):
    """Raised when a file cannot be parsed (corrupt, empty, wrong format)."""
