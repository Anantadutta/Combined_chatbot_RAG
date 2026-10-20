# NexaCorp RAG Assistant

A retrieval-augmented generation (RAG) assistant over the NexaCorp knowledge base.
It answers questions only from company documents, cites the exact file and location,
prefers current documents over superseded and draft ones, and says so when the answer
isn't in the documents.

> **Status:** Step 1 of 6 complete - document ingestion (parsing + chunking).

## Roadmap

| Step | What | Status |
|---|---|---|
| 1 | Parse 8 file types, detect document status, structure-aware chunking | ✅ done |
| 2 | Embeddings, vector DB, hybrid (BM25 + vector) retrieval, reranking, answers with citations | ⏳ |
| 3 | Current/superseded/draft handling, no-answer abstention, prompt-injection defence | ⏳ |
| 4 | FastAPI backend + chat/upload UI, multi-turn chat | ⏳ |
| 5 | Upload / update / delete / reindex, deduplication, ingestion status | ⏳ |
| 6 | Evaluation harness, Docker, cloud deployment, architecture diagram | ⏳ |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m app.cli ingest data/raw --show 5   # parse + chunk all documents
python -m pytest -q                          # run the tests
```

Ingestion writes to `data/processed/`:

- `chunks.jsonl` - one chunk per line: `chunk_id`, `text`, `embed_text`, `metadata`
- `documents.json` - document-level metadata (status, effective date, category, sha256...)
- `ingestion_report.json` - per-file result: `ok` / `failed` / `skipped` / `duplicate`, timing, error

## How ingestion works

```
file bytes ─► validate ─► parser ─► Sections ─► metadata ─► chunker ─► Chunks
            (type, size,  (per      (paragraph,  (status,     (structure-
             magic bytes,  format)   row, record  dates,       aware, with
             duplicate)              + location)  category)    context header)
```

### Parsers (`app/ingestion/parsers.py`)

| Format | Unit (Section) | Example citation location |
|---|---|---|
| PDF | paragraph per page | `page 1` |
| DOCX | paragraph / table row, in document order, under nearest heading | `paragraph 3`, `table 1, row 2` |
| XLSX | one record per row, every sheet, header row as field names | `sheet 'Budget Changes', row 2` |
| CSV | one record per row (delimiter sniffed) | `row 4` |
| JSON | one record per object, flattened | `JSON path $.contracts[1]` |
| HTML | heading sections, paragraphs, table rows (scripts stripped) | `section 'Endpoints', table 1, row 3` |
| Markdown | heading hierarchy, paragraphs (code blocks respected) | `section 'Announcements > 2026-09-15', line 4` |
| TXT | blank-line separated paragraphs | `lines 1-5` |

Rows are written as `Column: value; Column: value` so a single row is meaningful on its own.

### Document status (`app/ingestion/metadata.py`)

Status banners are read from the filename and the top of each document:

| Status | Triggered by | Authoritative? |
|---|---|---|
| `draft` | "draft", "unapproved", "do not apply", "not an effective policy" | no |
| `superseded` | "Status: superseded", "superseded effective/by ..." | no |
| `current` | "Status: current", or an effective date with no other marker | yes |
| `unspecified` | none of the above (ordinary reference data) | yes |

"Status: Active" is intentionally ignored, because in project documents it describes the
project, not the document. Every chunk also records entity ids (`OR-101`, `INC-102`) for
metadata filtering and a `contains_injection` flag for text that looks like instructions
to the AI.

### Chunking (`app/ingestion/chunker.py`)

- A chunk never crosses a section boundary, so each chunk has exactly one citation.
- Table/CSV/JSON records are never split.
- Long prose is split on sentence boundaries (max 1000 characters, 1 sentence overlap);
  the parts are labelled `page 3 (part 2/4)`.
- `embed_text` = a context header (document, title, status, section, location) plus the text, so
  "Milestone 2: integration due 2026-10-28" can still be found by searching for "Orion".
- `chunk_id` = `<doc_id>::<n>`; `doc_id` comes from the filename, so re-uploading a file replaces it.

### Robustness

Corrupt, empty, binary, oversized (>20 MB) or unsupported files are reported in the ingestion
report and never crash a batch. Identical content uploaded under a different name is marked
`duplicate`. Uploaded filenames are stripped of directory parts.

## Project layout

```
app/
  cli.py                  command line (python -m app.cli ...)
  ingestion/
    models.py             Section, ParsedDocument, Chunk
    parsers.py            one parser per format
    metadata.py           status, dates, category, entity ids, injection flag
    chunker.py            structure-aware chunking
    pipeline.py           validate -> parse -> chunk, per-file result
data/raw/                 the NexaCorp source documents
tests/                    unit + integration tests (pytest)
```
