"""Tokenisation shared by BM25, the hash embedder and the local reranker."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it
    its me my of on or our please should so tell than that the their them then there these they this those
    to us was we were what when where which who whom why will with would you your about any all also am
    give list show much many""".split()
)

# A few domain abbreviations, expanded in both queries and documents.
ABBREVIATIONS = {
    "wfh": ["work", "home"],
    "pm": ["project", "manager"],
    "mfa": ["multi", "factor", "authentication"],
    "sla": ["service", "level"],
    "slas": ["service", "level"],
    "qa": ["quality", "assurance"],
}

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
ENTITY_ID = re.compile(r"\b[A-Za-z]{2,5}-\d{2,5}\b")


def stem(token: str) -> str:
    """Very light stemming: plurals and a few suffixes (days->day, policies->policy)."""
    if len(token) <= 3 or token[0].isdigit():
        return token
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith(("ss", "us", "is")):
                return token
            return token[: len(token) - len(suffix)] + repl
    return token


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if not keep_stopwords and tok in STOPWORDS:
            continue
        out.append(tok if any(c in tok for c in "-_.") else stem(tok))
        if any(c in tok for c in "-_."):  # "or-101" also yields "or", "101"; "api/v2" style parts
            out.extend(stem(p) for p in re.split(r"[-_.]", tok) if p and p not in STOPWORDS)
        out.extend(ABBREVIATIONS.get(tok, []))
    return out


def entity_ids(text: str) -> set[str]:
    return {m.upper() for m in ENTITY_ID.findall(text)}
