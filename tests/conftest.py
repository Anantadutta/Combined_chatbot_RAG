import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    return RAW


@pytest.fixture(scope="session")
def ingested(raw_dir):
    """Ingest the whole trainee corpus once; return {filename: IngestResult}."""
    from app.ingestion import ingest_directory

    return {r.filename: r for r in ingest_directory(raw_dir)}
