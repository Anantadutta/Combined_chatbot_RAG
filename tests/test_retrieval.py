"""Retrieval tests (offline: hash embeddings + in-memory Qdrant, no API key needed)."""
import pytest

from app.config import load_settings
from app.rag import RAGEngine
from app.retrieval import BM25Index, LocalReranker
from app.retrieval.embeddings import CachedEmbedder, HashEmbedder
from app.retrieval.hybrid import Candidate
from app.retrieval.text import entity_ids, stem, tokenize
from app.retrieval.vector_store import payload_matches


def offline_engine(**kw):
    s = load_settings(llm_provider="offline", embedding_provider="hash", qdrant_path=":memory:", qdrant_url=None,
                      reranker="local", **kw)
    return RAGEngine(s)


@pytest.fixture(scope="module")
def engine(raw_dir):
    e = offline_engine()
    e.index_directory(raw_dir)
    yield e
    e.close()


# ---------------------------------------------------------------- text + BM25 units
def test_tokenize_ids_stems_and_abbreviations():
    toks = tokenize("Policies for WFH days on OR-101")
    assert "policy" in toks and "day" in toks
    assert "or-101" in toks and "101" in toks
    assert {"work", "home"} <= set(toks)
    assert "for" not in toks  # stopword
    assert stem("process") == "process" and stem("status") == "status"


def test_entity_ids():
    assert entity_ids("what about inc-102 and NE-303?") == {"INC-102", "NE-303"}


def test_bm25_ranks_rare_term_first():
    idx = BM25Index()
    idx.build([("a", "annual leave is 22 days"), ("b", "rate limit 120 requests"), ("c", "leave the office")])
    assert idx.search("rate limit", 3)[0][0] == "b"
    assert idx.search("annual leave", 3)[0][0] == "a"
    assert idx.search("annual leave", 3, allowed={"c"}) == [("c", pytest.approx(idx.search("leave", 3, {"c"})[0][1]))]
    assert idx.search("zebra", 3) == []


def test_payload_filter_logic():
    p = {"status": "current", "entity_ids": ["OR-101", "NE-303"]}
    assert payload_matches(p, {"status": ["current", "unspecified"]})
    assert payload_matches(p, {"entity_ids": "NE-303"})
    assert not payload_matches(p, {"status": "draft"})


# ---------------------------------------------------------------- hybrid retrieval on the corpus
@pytest.mark.parametrize(
    "question,expected_file",
    [
        ("How many days per week can I work from home?", "HR_Policy_2026_Current.pdf"),
        ("Was INC-102 acknowledged within the SLA?", "Operations_Notes.txt"),
        ("What is the P1 response time in the Cobalt Logistics contract?", "Client_SLAs.json"),
        ("Why was Project Atlas delayed?", "Project_Atlas.docx"),
        ("What is the API rate limit?", "API_Guide.html"),
        ("Is there a policy on cryptocurrency expense reimbursements?", "Announcements.md"),
        ("What budget increase was approved for OR-101?", "Projects_and_Budgets.xlsx"),
        ("Which skill does Arjun Patel have?", "Employees.csv"),
        ("Is MFA mandatory?", "Security_Standard.pdf"),
    ],
)
def test_expected_document_is_retrieved_first(engine, question, expected_file):
    top, _ = engine.retrieve(question)
    assert top[0].payload["filename"] == expected_file, [c.brief() for c in top]


def test_both_retrievers_contribute(engine):
    cands = engine.retriever.search("INC-102 acknowledged", 20)
    best = cands[0]
    assert best.dense_rank is not None and best.bm25_rank is not None
    assert best.rrf == pytest.approx(1 / (60 + best.dense_rank) + 1 / (60 + best.bm25_rank))


def test_metadata_filter_excludes_non_authoritative(engine):
    top, _ = engine.retrieve("work from home days per week", filters={"status": ["current", "unspecified"]})
    statuses = {c.payload["status"] for c in top}
    assert "superseded" not in statuses and "draft" not in statuses
    top, _ = engine.retrieve("work from home", filters={"filename": "Conflicting_Draft.txt"})
    assert {c.payload["filename"] for c in top} == {"Conflicting_Draft.txt"}


def test_current_policy_outranks_superseded_and_draft_is_excluded(engine):
    top, _ = engine.retrieve("work from home allowance")
    order = [c.payload["filename"] for c in top]
    assert order[0] == "HR_Policy_2026_Current.pdf"
    assert order[-1] == "HR_Policy_2025_Superseded.pdf"  # kept only as trailing background
    assert "Conflicting_Draft.txt" not in order


def test_draft_and_history_questions_keep_those_documents(engine):
    drafts = [c.payload["filename"] for c in engine.retrieve("Is there a draft proposal to change WFH days?")[0]]
    assert "Conflicting_Draft.txt" in drafts
    history = [c.payload["filename"] for c in engine.retrieve("What was the previous work from home policy?")[0]]
    assert "HR_Policy_2025_Superseded.pdf" in history and "Conflicting_Draft.txt" not in history


def test_reranker_entity_boost():
    a = Candidate("a", {"embed_text": "budget table", "entity_ids": ["NE-303"], "status": "unspecified"}, rrf=0.03)
    b = Candidate("b", {"embed_text": "budget table", "entity_ids": ["OR-101"], "status": "unspecified"}, rrf=0.03)
    ranked = LocalReranker().rerank("budget of OR-101", [a, b], 2)
    assert ranked[0].chunk_id == "b"


# ---------------------------------------------------------------- index maintenance
def test_reindex_is_idempotent(raw_dir):
    e = offline_engine()
    e.index_directory(raw_dir)
    n = e.store.count()
    e.index_directory(raw_dir)
    assert e.store.count() == n == len(e.retriever.payloads)
    e.close()


def test_delete_and_update_document(raw_dir):
    e = offline_engine()
    e.index_directory(raw_dir)
    before = e.store.count()

    e.delete_document("nebula-notes-txt")
    assert e.store.count() == before - 1
    assert all(p["doc_id"] != "nebula-notes-txt" for p in e.retriever.payloads.values())

    r = e.index_file(b"Nebula / NE-303: launch deadline moved to 2026-12-15.", "Nebula_Notes.txt")
    assert r.status == "ok"
    top, _ = e.retrieve("Nebula launch deadline")
    assert "2026-12-15" in top[0].payload["text"]
    assert e.store.count() == before
    e.close()


def test_embedding_cache(tmp_path):
    cached = CachedEmbedder(HashEmbedder(64), tmp_path)
    cached.embed_documents(["alpha", "beta"])
    assert cached.last_misses == 2
    again = CachedEmbedder(HashEmbedder(64), tmp_path)  # reloads from disk
    again.embed_documents(["alpha", "beta", "gamma"])
    assert again.last_misses == 1
