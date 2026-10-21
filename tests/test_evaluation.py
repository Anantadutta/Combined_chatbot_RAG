"""Evaluation harness: scoring rules, CSV loading, conversations, outputs."""
import csv
import json
from pathlib import Path

import pytest

from app.evaluation import (load_questions, run_evaluation, score_answer, score_sources, write_outputs)


@pytest.mark.parametrize("answer,abstained,expected,ok", [
    ("Employees may work from home 2 days per week [S1].", False, "2", True),
    ("Employees may work from home 3 days per week.", False, "2", False),
    ("The total budget is USD 130,000 (120,000 + 10,000).", False, "130000", True),
    ("The milestone is due on October 28, 2026.", False, "2026-10-28", True),
    ("The milestone is due on 2026-10-28.", False, "2026-10-28", True),
    ("It took 27 minutes, missing the 20-minute SLA and the 15-minute standard.", False, "27 20 15", True),
    ("It took 27 minutes, missing the 20-minute SLA.", False, "27 20 15", False),
    ("Delayed by late third-party sandbox credentials.", False, "late third-party sandbox credentials", True),
    ("Delayed by staffing shortages.", False, "late third-party sandbox credentials", False),
    ("I could not find this in the NexaCorp documents.", True, "NO_ANSWER", True),
    ("Paris.", False, "NO_ANSWER", False),
    ("No cryptocurrency reimbursement policy has been published.", False, "NO_ANSWER|no published policy cryptocurrency", True),
    ("I could not find this.", True, "NO_ANSWER|no published policy cryptocurrency", True),
    ("anything", False, "", None),
])
def test_score_answer(answer, abstained, expected, ok):
    assert score_answer(answer, abstained, expected) is ok


def test_score_answer_rejects_numbers_when_abstained():
    assert score_answer("I could not find 22 days", True, "22") is False


def test_score_sources():
    assert score_sources(["HR_Policy_2026_Current.pdf"], "HR_Policy_2026_Current.pdf") is True
    assert score_sources(["Announcements.md"], "Project_Orion.docx;Announcements") is True
    assert score_sources(["README.txt"], "Project_Orion.docx") is False
    assert score_sources([], "") is None


def test_load_questions_requires_question_column(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,text\n1,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        load_questions(bad)
    good = tmp_path / "good.csv"
    good.write_text("﻿Question,Notes\nHow many WFH days?,x\n,skipped\n", encoding="utf-8")  # BOM + caps
    rows = load_questions(good)
    assert rows == [{"question": "How many WFH days?", "notes": "x", "id": "1"}]


class FakeBackend:
    def __init__(self):
        self.calls = []

    def ask(self, question, conversation_id):
        self.calls.append((question, conversation_id))
        if "boom" in question:
            raise RuntimeError("server down")
        return {"answer": "2 days [S1]", "abstained": False, "confidence": "high", "flags": [],
                "citations": [{"filename": "HR_Policy_2026_Current.pdf", "location": "page 1", "status": "current"}],
                "timings_ms": {"retrieve": 1.0, "rerank": 0.1, "generate": 5.0, "total": 6.5}, "model": "fake"}


def test_run_and_write_outputs(tmp_path):
    rows = [
        {"id": "a", "question": "WFH days?", "expected_answer": "2", "expected_sources": "HR_Policy_2026_Current.pdf",
         "category": "policy", "conversation_id": "", "notes": "keep me"},
        {"id": "b", "question": "first", "conversation_id": "c1", "expected_answer": "3"},
        {"id": "c", "question": "second", "conversation_id": "c1"},
        {"id": "d", "question": "boom", "conversation_id": ""},
    ]
    backend = FakeBackend()
    results, summary = run_evaluation(rows, backend, workers=3)
    assert [r["id"] for r in results] == ["a", "b", "c", "d"]  # input order kept
    conv = [q for q, c in backend.calls if c == "c1"]
    assert conv == ["first", "second"]  # one conversation, asked in order
    assert results[2]["turn"] == 2
    assert results[0]["answer_correct"] is True and results[1]["answer_correct"] is False
    assert results[3]["error"].startswith("RuntimeError") and results[3]["answer_correct"] is None
    assert summary["questions"] == 4 and summary["errors"] == 1 and summary["correct"] == 1
    assert summary["answer_accuracy"] == 0.5 and summary["source_accuracy"] == 1.0
    assert summary["by_category"]["policy"]["accuracy"] == 1.0

    out = write_outputs(results, summary, tmp_path / "out", {"target": "test"})
    with open(out / "results.csv", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    assert table[0]["answer_correct"] == "true" and table[0]["notes"] == "keep me"
    assert table[0]["cited_sources"] == "HR_Policy_2026_Current.pdf (page 1, current)"
    assert table[0]["total_ms"] == "6.5" and float(table[0]["wall_ms"]) >= 0
    lines = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["response"]["model"] == "fake"
    assert json.loads((out / "summary.json").read_text())["target"] == "test"


def test_local_backend_end_to_end(raw_dir, tmp_path):
    from app.config import load_settings
    from app.evaluation import LocalBackend
    from app.rag import RAGEngine

    s = load_settings(llm_provider="offline", embedding_provider="hash", qdrant_path=":memory:", qdrant_url=None)
    e = RAGEngine(s)
    e.index_directory(raw_dir)
    rows = load_questions(Path(__file__).resolve().parents[1] / "eval" / "questions.csv")
    results, summary = run_evaluation(rows, LocalBackend(e), workers=4)
    e.close()
    assert summary["questions"] == len(rows) >= 15 and summary["errors"] == 0
    by_id = {r["id"]: r for r in results}
    assert by_id["q01"]["answer_correct"] is True  # WFH: current policy wins
    assert by_id["q13"]["abstained"] is True  # out-of-scope question
    assert by_id["q14"]["flags"] and "injection" in by_id["q14"]["flags"]
    assert by_id["q17"]["standalone_question"]  # follow-up was rewritten
