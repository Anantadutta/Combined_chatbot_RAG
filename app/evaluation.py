"""Batch evaluation: a CSV of questions in -> per-question results (CSV + JSONL) and a summary out.

    python -m app.cli eval eval/questions.csv                       # run against the local engine
    python -m app.cli eval questions.csv --url https://my-app.onrender.com --workers 4   # a deployed server

INPUT CSV (UTF-8, header row; only `question` is required, unknown columns are copied through):
    id                 question id (default: row number)
    question           the question text
    expected_answer    optional answer key - see scoring below
    expected_sources   optional; file names separated by ';' (e.g. "HR_Policy_2026_Current.pdf")
    conversation_id    optional; rows with the same id are asked in file order as ONE chat
                       (follow-up questions), other rows are independent
    category           optional label, used to break the summary down

SCORING (deterministic, no LLM judge, so the same answers always get the same score):
    * expected_answer = NO_ANSWER (or ABSTAIN): correct if the assistant abstained
    * otherwise correct if it did not abstain and
        - the expected answer contains numbers -> every number appears in the answer
          ("120,000" = "120000"; a date "2026-10-28" also matches "October 28, 2026")
        - no numbers -> at least 60% of the expected keywords appear in the answer
    * alternatives separated by '|' : correct if any alternative matches
    * source_match: at least one expected source file is cited

OUTPUT (in --out, default eval/results/<timestamp>/):
    results.csv     one row per question: answer, abstained, confidence, cited sources with locations,
                    flags, standalone question (follow-ups), model, timings (rewrite / retrieve / rerank /
                    generate / total in the engine, wall = end-to-end incl. HTTP), scores, error
    results.jsonl   the same plus the full answer object (citations with snippets, retrieved chunks)
    summary.json    counts, accuracy, source accuracy, abstention rate, latency mean/p50/p95/max,
                    throughput, per-category accuracy
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .retrieval.text import tokenize

NO_ANSWER_KEYS = {"NO_ANSWER", "ABSTAIN", "NOT_FOUND"}
KEYWORD_RECALL = 0.6
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]

CSV_COLUMNS = ["id", "conversation_id", "turn", "category", "question", "standalone_question", "answer",
               "abstained", "confidence", "cited_sources", "cited_files", "flags", "model",
               "rewrite_ms", "retrieve_ms", "rerank_ms", "generate_ms", "total_ms", "wall_ms",
               "expected_answer", "answer_correct", "expected_sources", "source_match", "error"]


# ---------------------------------------------------------------- scoring
def _numbers(text: str) -> list[str]:
    """Numbers in text, normalised: '120,000' -> '120000', '2.50' -> '2.5'. ISO dates add their parts."""
    out = []
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        n = raw.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.append(n.lstrip("0") or "0")
    return out


def _answer_numbers(text: str) -> set[str]:
    nums = set(_numbers(text))
    # "October 28, 2026" also counts as 2026-10-28 -> add the month number
    for i, m in enumerate(MONTHS, start=1):
        if re.search(rf"\b{m[:3]}[a-z]*\.?\s+\d", text, re.I):  # "Oct 28", "October 28"
            nums.add(str(i))
    return nums


def _expected_numbers(text: str) -> list[str]:
    nums = []
    for raw in re.findall(r"\d{4}-\d{2}-\d{2}|\d[\d,]*(?:\.\d+)?", text):
        nums += _numbers(raw.replace("-", " ")) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw) else _numbers(raw)
    return nums


def match_alternative(answer: str, expected: str) -> bool:
    nums = _expected_numbers(expected)
    if nums:
        have = _answer_numbers(answer)
        return all(n in have for n in nums)
    want = set(tokenize(expected))
    if not want:
        return False
    return len(want & set(tokenize(answer))) / len(want) >= KEYWORD_RECALL


def score_answer(answer: str, abstained: bool, expected: str | None) -> bool | None:
    """True / False against the answer key, or None when there is no key."""
    expected = (expected or "").strip()
    if not expected:
        return None
    for alt in (a.strip() for a in expected.split("|")):
        if alt.upper() in NO_ANSWER_KEYS:
            if abstained:
                return True
        elif alt and not abstained and match_alternative(answer, alt):
            return True
    return False


def _stem(name: str) -> str:
    return re.sub(r"\.[a-z0-9]+$", "", name.strip().lower())


def score_sources(cited_files: list[str], expected_sources: str | None) -> bool | None:
    wanted = [_stem(s) for s in (expected_sources or "").split(";") if s.strip()]
    if not wanted:
        return None
    cited = {_stem(f) for f in cited_files}
    return any(w in cited for w in wanted)


# ---------------------------------------------------------------- backends
class LocalBackend:
    """Ask the in-process engine (multi-turn history kept per conversation id)."""

    def __init__(self, engine):
        from .rag.conversation import Turn

        self.engine, self._turn = engine, Turn
        self._histories: dict[str, list] = {}
        self._lock = threading.Lock()

    def ask(self, question: str, conversation_id: str | None) -> dict:
        history = self._histories.get(conversation_id, []) if conversation_id else None
        answer = self.engine.ask(question, history=history)
        if conversation_id:
            turn = self._turn(question, answer.answer, answer.standalone_question,
                              [c.filename for c in answer.citations], answer.abstained)
            with self._lock:
                self._histories[conversation_id] = (history + [turn])[-6:]
        return answer.to_dict()


class HttpBackend:
    """Ask a running server (local or deployed) through its REST API."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._sessions: dict[str, str] = {}

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    def ask(self, question: str, conversation_id: str | None) -> dict:
        if not conversation_id:
            return self._post("/api/ask", {"question": question})
        body = {"question": question, "session_id": self._sessions.get(conversation_id)}
        data = self._post("/api/chat", body)
        self._sessions[conversation_id] = data["session_id"]
        return data


# ---------------------------------------------------------------- running
def load_questions(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "question" not in [h.strip().lower() for h in reader.fieldnames]:
            raise ValueError(f"{path}: the CSV needs a header row with a 'question' column")
        rows = []
        for i, raw in enumerate(reader, start=1):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items() if k}
            if not row.get("question"):
                continue
            row.setdefault("id", "")
            row["id"] = row["id"] or str(i)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no questions found")
    return rows


def _run_one(backend, row: dict, turn: int) -> dict:
    conv = row.get("conversation_id") or None
    start = time.perf_counter()
    rec: dict[str, Any] = {k: row.get(k, "") for k in ("id", "conversation_id", "category", "question",
                                                      "expected_answer", "expected_sources")}
    rec["turn"] = turn
    try:
        a = backend.ask(row["question"], conv)
        error = None
    except Exception as exc:  # one failing question never stops the batch
        a, error = {}, f"{type(exc).__name__}: {exc}"
    wall = (time.perf_counter() - start) * 1000
    cites = a.get("citations") or []
    t = a.get("timings_ms") or {}
    cited_files = list(dict.fromkeys(c["filename"] for c in cites))
    rec.update({
        "standalone_question": a.get("standalone_question") or "",
        "answer": a.get("answer", ""),
        "abstained": a.get("abstained"),
        "confidence": a.get("confidence", ""),
        "cited_sources": "; ".join(f"{c['filename']} ({c['location']}, {c['status']})" for c in cites),
        "cited_files": "; ".join(cited_files),
        "flags": "; ".join(a.get("flags") or []),
        "model": a.get("model") or "",
        **{f"{k}_ms": round(t[k], 1) if k in t else "" for k in ("rewrite", "retrieve", "rerank", "generate", "total")},
        "wall_ms": round(wall, 1),
        "answer_correct": None if error else score_answer(a.get("answer", ""), bool(a.get("abstained")),
                                                          row.get("expected_answer")),
        "source_match": None if error else score_sources(cited_files, row.get("expected_sources")),
        "error": error or "",
        "_full": a,
        "_extra": {k: v for k, v in row.items() if k not in CSV_COLUMNS},
    })
    return rec


def run_evaluation(rows: list[dict], backend, workers: int = 1,
                   progress: Callable[[dict], None] | None = None) -> tuple[list[dict], dict]:
    """Run all questions. Rows of one conversation run in order; conversations run in parallel."""
    groups: list[list[tuple[int, dict]]] = []
    by_conv: dict[str, list] = {}
    for idx, row in enumerate(rows):
        conv = row.get("conversation_id")
        if conv:
            if conv not in by_conv:
                by_conv[conv] = []
                groups.append(by_conv[conv])
            by_conv[conv].append((idx, row))
        else:
            groups.append([(idx, row)])

    results: list[dict | None] = [None] * len(rows)

    def run_group(group):
        for turn, (idx, row) in enumerate(group, start=1):
            rec = _run_one(backend, row, turn)
            results[idx] = rec
            if progress:
                progress(rec)

    start = time.perf_counter()
    if workers <= 1:
        for g in groups:
            run_group(g)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_group, groups))
    elapsed = time.perf_counter() - start
    done = [r for r in results if r is not None]
    return done, summarise(done, elapsed, workers)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def _rate(values: list) -> float | None:
    scored = [v for v in values if v is not None]
    return round(sum(scored) / len(scored), 4) if scored else None


def summarise(results: list[dict], elapsed_s: float, workers: int) -> dict:
    ok = [r for r in results if not r["error"]]
    wall = [r["wall_ms"] for r in ok]
    total = [r["total_ms"] for r in ok if r["total_ms"] != ""]
    categories: dict[str, dict] = {}
    for r in results:
        cat = r.get("category") or "uncategorised"
        c = categories.setdefault(cat, {"questions": 0, "scores": []})
        c["questions"] += 1
        c["scores"].append(r["answer_correct"])
    return {
        "questions": len(results),
        "errors": len(results) - len(ok),
        "answered": sum(1 for r in ok if not r["abstained"]),
        "abstained": sum(1 for r in ok if r["abstained"]),
        "scored": sum(1 for r in results if r["answer_correct"] is not None),
        "answer_accuracy": _rate([r["answer_correct"] for r in results]),
        "correct": sum(1 for r in results if r["answer_correct"] is True),
        "source_accuracy": _rate([r["source_match"] for r in results]),
        "flagged_injection": sum(1 for r in ok if "injection" in r["flags"]),
        "latency_ms": {"mean": round(statistics.fmean(wall), 1) if wall else 0, "p50": _pct(wall, 0.5),
                       "p95": _pct(wall, 0.95), "max": round(max(wall), 1) if wall else 0,
                       "engine_mean": round(statistics.fmean(total), 1) if total else 0},
        "elapsed_s": round(elapsed_s, 2),
        "workers": workers,
        "throughput_qps": round(len(results) / elapsed_s, 3) if elapsed_s else 0,
        "by_category": {k: {"questions": v["questions"], "accuracy": _rate(v["scores"])}
                        for k, v in sorted(categories.items())},
        "models": sorted({r["model"] for r in ok if r["model"]}),
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def write_outputs(results: list[dict], summary: dict, out_dir: str | Path, meta: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extra_cols = sorted({k for r in results for k in r["_extra"]})
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS + extra_cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLUMNS}
            for k in ("abstained", "answer_correct", "source_match"):
                if isinstance(r.get(k), bool):
                    row[k] = str(r[k]).lower()
            row.update(r["_extra"])
            w.writerow(row)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            rec = {k: v for k, v in r.items() if not k.startswith("_")}
            rec.update(r["_extra"])
            rec["response"] = r["_full"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({**(meta or {}), **summary}, indent=2), encoding="utf-8")
    return out
