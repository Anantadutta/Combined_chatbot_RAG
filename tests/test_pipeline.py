"""Integration tests: the full ingestion pipeline over the trainee corpus and bad uploads."""
import json

from app.cli import main
from app.ingestion import SUPPORTED_EXTENSIONS, ingest_bytes, ingest_directory


def test_all_trainee_files_ingest(ingested):
    assert len(ingested) == 14
    bad = {n: r.error for n, r in ingested.items() if r.status != "ok"}
    assert not bad, bad


def test_all_eight_formats_covered(ingested):
    types = {r.document["file_type"] for r in ingested.values()}
    assert types == {"pdf", "docx", "xlsx", "csv", "json", "html", "txt", "md"}


def test_key_facts_survive_ingestion(ingested):
    all_text = "\n".join(c.text for r in ingested.values() for c in r.chunks)
    for fact in [
        "maximum 2 days per week",
        "Approved increase USD: 10000",
        "acknowledged 10:27",
        "p1_response_minutes: 20",
        "Rate limit: 120 requests per minute",
        "not staffing shortages",
        "cryptocurrency expense reimbursements",
    ]:
        assert fact in all_text, fact


def test_malformed_upload_is_reported_not_raised():
    r = ingest_bytes(b"%PDF-1.7 garbage", "policy.pdf")
    assert r.status == "failed" and "PDF" in r.error and r.chunks == []


def test_unsupported_type_skipped():
    assert ingest_bytes(b"MZ", "tool.exe").status == "skipped"


def test_oversized_file_rejected(monkeypatch):
    import app.ingestion.pipeline as pipeline

    monkeypatch.setattr(pipeline, "MAX_FILE_BYTES", 10)
    assert ingest_bytes(b"x" * 11, "big.txt").status == "failed"


def test_duplicate_content_detected(raw_dir):
    known = {}
    data = (raw_dir / "Nebula_Notes.txt").read_bytes()
    assert ingest_bytes(data, "Nebula_Notes.txt", known).status == "ok"
    assert ingest_bytes(data, "Nebula_Copy.txt", known).status == "duplicate"
    # re-ingesting the SAME filename is an update, not a duplicate
    assert ingest_bytes(data, "Nebula_Notes.txt", known).status == "ok"


def test_upload_name_cannot_escape_folder():
    r = ingest_bytes(b"hello", "../../secret/notes.txt")
    assert r.filename == "notes.txt" and r.doc_id == "notes-txt"


def test_chunk_ids_stable_across_runs(raw_dir):
    a = [c.chunk_id for r in ingest_directory(raw_dir) for c in r.chunks]
    b = [c.chunk_id for r in ingest_directory(raw_dir) for c in r.chunks]
    assert a == b and len(a) == len(set(a))


def test_cli_writes_outputs(raw_dir, tmp_path):
    assert main(["ingest", str(raw_dir), "--output", str(tmp_path)]) == 0
    lines = (tmp_path / "chunks.jsonl").read_text().splitlines()
    assert len(lines) > 20
    first = json.loads(lines[0])
    assert {"chunk_id", "text", "embed_text", "metadata"} <= first.keys()
    report = json.loads((tmp_path / "ingestion_report.json").read_text())
    assert all(r["status"] == "ok" for r in report)


def test_supported_extensions():
    assert {".pdf", ".docx", ".xlsx", ".csv", ".json", ".html", ".txt", ".md"} <= SUPPORTED_EXTENSIONS
