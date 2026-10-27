"""Central configuration, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


SUPPORTED_LANGUAGES = ("en", "fr")


@dataclass
class Settings:
    # Corpus language: drives BM25 stopwords and the generation prompt.
    # Named RAG_LANG (not LANG) to avoid clashing with the system locale variable.
    language: str = _env("RAG_LANG", "en")

    # Paths
    raw_dir: Path = field(default_factory=lambda: Path(_env("RAW_DIR", "data/raw")))
    chunks_path: Path = field(
        default_factory=lambda: Path(_env("CHUNKS_PATH", "data/processed/chunks.jsonl"))
    )
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", "data/cache")))

    # Chunking
    max_words: int = int(_env("MAX_WORDS", "250"))
    overlap_words: int = int(_env("OVERLAP_WORDS", "50"))

    # Retrieval
    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-m3")
    reranker_model: str = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    n_candidates: int = int(_env("N_CANDIDATES", "50"))  # pool size before fusion / reranking
    rrf_k: int = int(_env("RRF_K", "60"))

    # Generation (any OpenAI-compatible endpoint: Ollama, vLLM, Mistral, OpenAI...)
    llm_base_url: str = _env("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key: str = _env("LLM_API_KEY", "ollama")
    llm_model: str = _env("LLM_MODEL", "qwen2.5:7b-instruct")
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.0"))

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"RAG_LANG must be one of {SUPPORTED_LANGUAGES}, got {self.language!r}")


def get_settings() -> Settings:
    return Settings()
