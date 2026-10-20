"""Unit tests: structure-aware chunking."""
from app.ingestion.chunker import chunk_document, split_text
from app.ingestion.metadata import build_document_metadata
from app.ingestion.models import ParsedDocument, Section


def make_doc(sections):
    doc = ParsedDocument("t.txt", "txt", "Test Doc", sections)
    return doc, build_document_metadata(doc, "0" * 64)


def test_short_text_is_one_piece():
    assert split_text("One. Two.", 100, 1) == ["One. Two."]


def test_long_text_respects_max_and_overlaps():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(40)]
    pieces = split_text(" ".join(sentences), 200, 1)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    for a, b in zip(pieces, pieces[1:]):  # last sentence of a piece starts the next
        assert b.startswith(a.split(". ")[-1].rstrip("."))


def test_giant_sentence_is_hard_split():
    pieces = split_text("word " * 500, 100, 1)
    assert all(len(p) <= 100 for p in pieces)


def test_records_never_split_and_sections_never_merged():
    long_record = "; ".join(f"col{i}: value{i}" for i in range(200))
    doc, meta = make_doc(
        [
            Section("Short prose.", "line 1"),
            Section(long_record, "row 2", kind="record"),
        ]
    )
    chunks = chunk_document(doc, meta, max_chars=100)
    assert len(chunks) == 2
    assert chunks[1].text == long_record
    assert chunks[1].metadata["location"] == "row 2"


def test_split_parts_are_labelled():
    doc, meta = make_doc([Section(" ".join(f"Fact {i} is here." for i in range(50)), "page 3")])
    chunks = chunk_document(doc, meta, max_chars=120)
    assert chunks[0].metadata["location"].startswith("page 3 (part 1/")
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_duplicate_sections_are_dropped():
    doc, meta = make_doc([Section("Same text.", "line 1"), Section("Same text.", "line 3")])
    assert len(chunk_document(doc, meta)) == 1


def test_embed_text_carries_context(ingested):
    milestone = ingested["Project_Orion.docx"].chunks[1]
    assert "Milestone 2" in milestone.text
    assert "Orion" not in milestone.text  # the raw text never names the project...
    assert "Project Orion" in milestone.embed_text  # ...but the contextual header does
    assert "Location: paragraph 3" in milestone.embed_text


def test_every_corpus_chunk_has_citation_metadata(ingested):
    for result in ingested.values():
        for c in result.chunks:
            m = c.metadata
            assert m["filename"] and m["location"] and m["status"]
            assert c.chunk_id.startswith(c.doc_id + "::")
