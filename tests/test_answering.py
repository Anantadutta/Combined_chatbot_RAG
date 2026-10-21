"""Answer generation tests with a fake LLM, plus Gemini client retry/fallback logic."""
import json
from types import SimpleNamespace

import pytest

from app.config import load_settings
from app.llm.gemini import GeminiClient, LLMError
from app.rag import RAGEngine, parse_llm_json
from app.rag.prompts import SYSTEM_PROMPT, build_prompt
from app.retrieval import GeminiReranker
from app.retrieval.hybrid import Candidate


class FakeLLM:
    """Records prompts and returns a canned reply."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []
        self.last_model_used = "fake-model"

    def generate(self, system, prompt, temperature=0.1, json_mode=True, **_):
        self.prompts.append((system, prompt))
        return self.reply(prompt) if callable(self.reply) else self.reply


def engine_with(raw_dir, reply):
    s = load_settings(llm_provider="offline", embedding_provider="hash", qdrant_path=":memory:", qdrant_url=None)
    e = RAGEngine(s, llm=FakeLLM(reply))
    e.index_directory(raw_dir)
    return e


# ---------------------------------------------------------------- parsing
def test_parse_plain_and_fenced_json():
    data = {"answer": "2 days [S1]", "citations": ["S1"], "abstained": False, "confidence": "high"}
    assert parse_llm_json(json.dumps(data))["answer"] == "2 days [S1]"
    assert parse_llm_json("```json\n" + json.dumps(data) + "\n```")["citations"] == ["S1"]
    assert parse_llm_json("Sure! " + json.dumps(data) + " Hope that helps")["confidence"] == "high"


def test_parse_non_json_falls_back_to_text():
    data = parse_llm_json("Two days per week [S2].")
    assert data["answer"] == "Two days per week [S2]." and data["citations"] == ["S2"]


# ---------------------------------------------------------------- prompt
def test_prompt_labels_status_and_numbers_sources():
    cur = Candidate("x::0", {"filename": "HR_Policy_2026_Current.pdf", "location": "page 1", "status": "current",
                             "text": "WFH 2 days", "effective_date": "2026-04-01"})
    old = Candidate("y::0", {"filename": "HR_Policy_2025_Superseded.pdf", "location": "page 1",
                             "status": "superseded", "text": "WFH 3 days"})
    prompt = build_prompt("How many WFH days?", [("S1", cur), ("S2", old)], "2026-10-20")
    assert "[S1] file: HR_Policy_2026_Current.pdf | location: page 1 | status: CURRENT - authoritative" in prompt
    assert "SUPERSEDED - no longer in force" in prompt
    assert prompt.strip().endswith("QUESTION: How many WFH days?")
    assert "untrusted data" in SYSTEM_PROMPT


# ---------------------------------------------------------------- end-to-end with fake LLM
def test_answer_citations_map_to_real_sources(raw_dir):
    reply = json.dumps({"answer": "Up to 2 days per week with manager approval [S1].", "citations": ["S1"],
                        "abstained": False, "confidence": "high"})
    e = engine_with(raw_dir, reply)
    a = e.ask("How many days can I work from home?")
    assert a.citations[0].filename == "HR_Policy_2026_Current.pdf"
    assert a.citations[0].location == "page 1"
    assert a.confidence == "high" and not a.abstained
    assert a.model == "fake-model"
    assert {"retrieve", "rerank", "generate", "total"} <= a.timings_ms.keys()
    assert "SOURCES:" in e.llm.prompts[0][1]
    e.close()


def test_hallucinated_citation_removed_and_confidence_lowered(raw_dir):
    reply = json.dumps({"answer": "It is 5 days [S99].", "citations": ["S99"], "abstained": False,
                        "confidence": "high"})
    e = engine_with(raw_dir, reply)
    a = e.ask("How many days can I work from home?")
    assert a.citations == [] and "[S99]" not in a.answer and a.confidence == "low"
    e.close()


def test_abstention_passes_through(raw_dir):
    reply = json.dumps({"answer": "I could not find this in the NexaCorp documents.", "citations": [],
                        "abstained": True, "confidence": "high"})
    e = engine_with(raw_dir, reply)
    a = e.ask("What is the capital of France?")
    assert a.abstained and a.citations == []
    e.close()


def test_offline_mode_answers_and_abstains(raw_dir):
    s = load_settings(llm_provider="offline", embedding_provider="hash", qdrant_path=":memory:", qdrant_url=None)
    e = RAGEngine(s)
    e.index_directory(raw_dir)
    a = e.ask("What is the API rate limit?")
    assert "120 requests per minute" in a.answer and a.citations[0].filename == "API_Guide.html"
    assert e.ask("What is the capital of France?").abstained
    with pytest.raises(ValueError):
        e.ask("   ")
    with pytest.raises(ValueError):
        e.ask("x" * 5000)
    e.close()


def test_empty_index_abstains():
    s = load_settings(llm_provider="offline", embedding_provider="hash", qdrant_path=":memory:", qdrant_url=None)
    e = RAGEngine(s)
    assert e.ask("anything?").abstained
    e.close()


# ---------------------------------------------------------------- Gemini reranker
def test_gemini_reranker_uses_llm_grades_and_falls_back():
    cands = [Candidate(f"c{i}", {"embed_text": "budget", "text": f"t{i}", "status": "unspecified"}, rrf=0.03)
             for i in range(3)]
    llm = FakeLLM(json.dumps({"scores": [{"id": "P1", "score": 1}, {"id": "P2", "score": 2}, {"id": "P3", "score": 9}]}))
    ranked = GeminiReranker(llm).rerank("budget", cands, 3)
    assert ranked[0].payload["text"] == "t2"
    broken = GeminiReranker(FakeLLM("not json")).rerank("budget", cands, 2)
    assert len(broken) == 2  # local scores used, no exception


# ---------------------------------------------------------------- Gemini client (fake SDK)
class APIErr(Exception):
    def __init__(self, code, msg="err"):
        super().__init__(msg)
        self.code = code


def fake_sdk(generate=None, embed=None):
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate, embed_content=embed))


def test_retries_on_429_then_succeeds():
    calls, sleeps = [], []

    def generate(model, contents, config):
        calls.append(model)
        if len(calls) < 3:
            raise APIErr(429, "Resource exhausted. Please retry in 1.5s")
        return SimpleNamespace(text='{"answer": "ok"}')

    c = GeminiClient("k", "m1", None, sleep=sleeps.append, client=fake_sdk(generate=generate))
    assert c.generate("sys", "hi") == '{"answer": "ok"}'
    assert len(calls) == 3 and len(sleeps) == 2 and 1.5 <= sleeps[0] < 2.1


def test_non_retriable_error_is_raised_immediately():
    calls = []

    def generate(model, contents, config):
        calls.append(model)
        raise APIErr(400, "bad request")

    c = GeminiClient("k", "m1", None, sleep=lambda s: None, client=fake_sdk(generate=generate))
    with pytest.raises(LLMError):
        c.generate("sys", "hi")
    assert calls == ["m1"]


def test_falls_back_to_second_model_on_404():
    def generate(model, contents, config):
        if model == "m1":
            raise APIErr(404, "model not found")
        return SimpleNamespace(text="fine")

    c = GeminiClient("k", "m1", "m2", sleep=lambda s: None, client=fake_sdk(generate=generate))
    assert c.generate("sys", "hi") == "fine" and c.last_model_used == "m2"


def test_embed_batches_and_preserves_order():
    batches = []

    def embed(model, contents, config):
        batches.append(len(contents))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(t.split()[-1]), 0.0]) for t in contents])

    c = GeminiClient("k", "m", None, client=fake_sdk(embed=embed))
    vecs = c.embed([f"text {i}" for i in range(250)], "RETRIEVAL_DOCUMENT", "gemini-embedding-001", 2)
    assert batches == [100, 100, 50]
    assert [v[0] for v in vecs] == [float(i) for i in range(250)]


# ---------------------------------------------------------------- config
def test_settings_key_aliases_and_masking(monkeypatch, tmp_path):
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER", "EMBEDDING_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "AIzaTESTKEY123456")
    s = load_settings(env_file=tmp_path / "none.env")
    assert s.llm_provider == "gemini" and s.embedding_provider == "gemini"
    assert s.describe()["api_key"] == "AIza...3456"
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError):
        load_settings(env_file=tmp_path / "none.env")


def test_overloaded_primary_falls_back_after_one_retry():
    calls = []

    def generate(model, contents, config):
        calls.append(model)
        if model == "m1":
            raise APIErr(503, "The model is overloaded")
        return SimpleNamespace(text="fine")

    c = GeminiClient("k", "m1", "m2", sleep=lambda s: None, client=fake_sdk(generate=generate))
    assert c.generate("sys", "hi") == "fine"
    assert calls == ["m1", "m1", "m2"]  # one retry on the primary, then the fallback


def test_engine_degrades_gracefully_when_llm_fails(raw_dir):
    class DownLLM(FakeLLM):
        def generate(self, *a, **k):
            raise LLMError("503 overloaded")

    s = load_settings(llm_provider="offline", embedding_provider="hash", qdrant_path=":memory:", qdrant_url=None)
    e = RAGEngine(s, llm=DownLLM(None))
    e.index_directory(raw_dir)
    a = e.ask("What is the API rate limit?")
    assert "unavailable" in a.answer and "120 requests" in a.answer and a.model == "unavailable-fallback"
    e.close()
