"""Settings, read from environment variables / the .env file.

Only GEMINI_API_KEY is required for real use. Without a key the app runs in *offline mode*
(hash embeddings + extractive answers) so that tests and demos work without network access.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None = None
    llm_provider: str = "offline"  # "gemini" | "offline"
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    embedding_provider: str = "hash"  # "gemini" | "hash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    reranker: str = "local"  # "local" | "gemini" | "none"

    data_dir: str = str(ROOT / "data" / "raw")
    index_dir: str = str(ROOT / "data" / "index")
    qdrant_path: str | None = None  # local folder; ":memory:" for tests; default <index_dir>/qdrant
    qdrant_url: str | None = None  # set to use Qdrant Cloud / a Qdrant server instead
    qdrant_api_key: str | None = None
    collection: str = "nexacorp_chunks"

    candidate_k: int = 20  # candidates from each retriever before fusion
    top_k: int = 6  # chunks given to the LLM after reranking
    temperature: float = 0.1
    request_timeout_s: int = 60
    max_retries: int = 4

    @property
    def resolved_qdrant_path(self) -> str:
        return self.qdrant_path or str(Path(self.index_dir) / "qdrant")

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)

    def describe(self) -> dict:
        key = self.gemini_api_key
        return {
            "llm_provider": self.llm_provider,
            "gemini_model": self.gemini_model if self.llm_provider == "gemini" else None,
            "embedding": f"{self.embedding_provider}:{self.embedding_model if self.embedding_provider == 'gemini' else 'hash'}",
            "reranker": self.reranker,
            "vector_db": self.qdrant_url or self.resolved_qdrant_path,
            "api_key": f"{key[:4]}...{key[-4:]}" if key else "(not set)",
        }


def load_settings(env_file: str | Path | None = None, **overrides) -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file or ROOT / ".env", override=False)
    except ImportError:  # python-dotenv is optional
        pass

    key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY") or _env("LLM_API_KEY")
    default_provider = "gemini" if key else "offline"
    settings = Settings(
        gemini_api_key=key,
        llm_provider=_env("LLM_PROVIDER", default_provider).lower(),
        gemini_model=_env("GEMINI_MODEL", Settings.gemini_model),
        gemini_fallback_model=_env("GEMINI_FALLBACK_MODEL", Settings.gemini_fallback_model),
        embedding_provider=_env("EMBEDDING_PROVIDER", "gemini" if key else "hash").lower(),
        embedding_model=_env("EMBEDDING_MODEL", Settings.embedding_model),
        embedding_dim=int(_env("EMBEDDING_DIM", str(Settings.embedding_dim))),
        reranker=_env("RERANKER", Settings.reranker).lower(),
        data_dir=_env("DATA_DIR", Settings.data_dir),
        index_dir=_env("INDEX_DIR", Settings.index_dir),
        qdrant_path=_env("QDRANT_PATH"),
        qdrant_url=_env("QDRANT_URL"),
        qdrant_api_key=_env("QDRANT_API_KEY"),
        collection=_env("QDRANT_COLLECTION", Settings.collection),
        candidate_k=int(_env("CANDIDATE_K", str(Settings.candidate_k))),
        top_k=int(_env("TOP_K", str(Settings.top_k))),
        temperature=float(_env("LLM_TEMPERATURE", str(Settings.temperature))),
    )
    if overrides:
        settings = settings.with_overrides(**overrides)
    if settings.llm_provider == "gemini" or settings.embedding_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set - add it to .env or use LLM_PROVIDER=offline")
    return settings
