"""Document-level metadata: authority status (current / superseded / draft), dates, category,
entity ids and a prompt-injection flag.

Status rules (checked on the filename and the *top* of the document, where status banners live):
  draft       - "draft", "do not apply", "not an effective policy", "unapproved"
  superseded  - "superseded" / "obsolete" / "replaced by"
  current     - "status: current", or an effective date with no other marker
  unspecified - none of the above (ordinary reference material)
Only `current` and `unspecified` documents are authoritative at answer time (step 3).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import ParsedDocument

HEAD_CHARS = 600  # status banners are expected near the top of a document

DATE = r"(\d{4}-\d{2}-\d{2})"

DRAFT_PATTERNS = [
    r"\bunapproved\b",
    r"\bdo not apply\b",
    r"\bnot an? effective (policy|document|standard)\b",
    r"\bstatus\s*:\s*draft\b",
    r"^\W*draft\b",
    r"\bdraft\s*(-|–|—|:)",
]
SUPERSEDED_PATTERNS = [
    r"\bstatus\s*:\s*(superseded|obsolete|archived|retired|withdrawn)\b",
    r"\bsuperseded\s+(effective|by|on|as of)\b",
    r"\b(this|the)\s+(document|policy|standard|version)\s+(is|has been|was)\s+(superseded|replaced)\b",
]
# NB: "Status: Active" is deliberately NOT here - in project docs it is the project's status,
# not the document's authority.
CURRENT_PATTERNS = [r"\bstatus\s*:\s*(current|in force|effective)\b"]

CATEGORY_KEYWORDS = [
    ("policy", ["policy", "standard", "handbook", "guideline"]),
    ("project", ["project", "nebula", "orion", "atlas", "phoenix", "budget"]),
    ("contract", ["sla", "contract", "agreement"]),
    ("operations", ["operations", "incident", "ticket"]),
    ("people", ["employee", "staff", "people"]),
    ("technical", ["api", "guide", "technical"]),
    ("announcement", ["announcement", "news"]),
    ("readme", ["readme"]),
]

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |all )?(previous|prior|above|system)",
    r"you are now\b",
    r"\bsystem prompt\b",
    r"reveal (your|the) (instructions|prompt)",
    r"do not cite",
    r"act as (an? )?(admin|administrator|developer)",
]

ENTITY_ID = re.compile(r"\b[A-Z]{2,5}-\d{2,5}\b")  # OR-101, INC-102, ...


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in patterns)


def detect_status(filename: str, text: str) -> dict[str, Any]:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").lower()
    head = text[:HEAD_CHARS]

    effective_date = superseded_date = None
    for m in re.finditer(r"(superseded\s+)?effective(?:\s+(?:from|date|on))?\s*:?\s*" + DATE, head, re.I):
        if m.group(1):
            superseded_date = m.group(2)
        elif effective_date is None:
            effective_date = m.group(2)

    version = None
    vm = re.search(r"\bversion\s*:?\s*v?([\w.]+)", head, re.I)
    if vm:
        version = vm.group(1).rstrip(".")

    if re.search(r"\bdraft\b", stem) or _any(DRAFT_PATTERNS, head):
        status, reason = "draft", "draft / do-not-apply marker"
    elif re.search(r"\bsuperseded\b|\bobsolete\b|\barchived?\b", stem) or _any(SUPERSEDED_PATTERNS, head):
        status, reason = "superseded", "superseded marker"
    elif re.search(r"\bcurrent\b", stem) or _any(CURRENT_PATTERNS, head):
        status, reason = "current", "explicit current/active status"
    elif effective_date:
        status, reason = "current", "has an effective date and no superseded/draft marker"
    else:
        status, reason = "unspecified", "no status marker (reference material)"

    return {
        "status": status,
        "status_reason": reason,
        "authoritative": status in ("current", "unspecified"),
        "effective_date": effective_date,
        "superseded_date": superseded_date,
        "version": version,
    }


def detect_category(filename: str, title: str) -> str:
    haystack = f"{filename} {title}".lower()
    for category, words in CATEGORY_KEYWORDS:
        if any(w in haystack for w in words):
            return category
    return "general"


def find_entity_ids(text: str) -> list[str]:
    return sorted(set(ENTITY_ID.findall(text)))


def looks_like_injection(text: str) -> bool:
    return _any(INJECTION_PATTERNS, text)


def doc_id_for(filename: str) -> str:
    """Stable id from the filename, so re-uploading a file *replaces* its old version."""
    stem = Path(filename).name.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-")


def build_document_metadata(doc: ParsedDocument, sha256: str) -> dict[str, Any]:
    full = doc.full_text
    meta: dict[str, Any] = {
        "doc_id": doc_id_for(doc.filename),
        "filename": doc.filename,
        "file_type": doc.file_type,
        "title": doc.title,
        "category": detect_category(doc.filename, doc.title),
        "sha256": sha256,
        "n_sections": len(doc.sections),
        "n_chars": len(full),
        "entity_ids": find_entity_ids(full),
        "contains_injection": looks_like_injection(full),
    }
    meta.update(detect_status(doc.filename, full))
    return meta
