"""Assemble retrievers by name and run the full RAG pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from .chunking import Chunk, load_chunks
from .config import Settings, get_settings
from .generate import Answer, generate_answer
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RerankedRetriever, Retriever

CONFIGS = ("bm25", "dense", "hybrid", "hybrid_rerank")


def build_retriever(name: str, chunks: Sequence[Chunk], s: Settings) -> Retriever:
    if name == "bm25":
        return BM25Retriever(chunks, s.language)
    if name == "dense":
        return DenseRetriever(chunks, s.embedding_model, s.cache_dir)
    if name == "hybrid":
        return HybridRetriever(
            [BM25Retriever(chunks, s.language), DenseRetriever(chunks, s.embedding_model, s.cache_dir)],
            n_candidates=s.n_candidates,
            rrf_k=s.rrf_k,
        )
    if name == "hybrid_rerank":
        return RerankedRetriever(
            build_retriever("hybrid", chunks, s), chunks, s.reranker_model, n_candidates=s.n_candidates
        )
    raise ValueError(f"Unknown config {name!r}, expected one of {CONFIGS}")


class RAGPipeline:
    def __init__(self, config: str = "hybrid_rerank", settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.chunks = load_chunks(self.settings.chunks_path)
        self.retriever = build_retriever(config, self.chunks, self.settings)

    def retrieve(self, question: str, k: int = 5) -> list[Chunk]:
        return [self.chunks[i] for i, _ in self.retriever.search(question, k)]

    def ask(self, question: str, k: int = 5) -> Answer:
        return generate_answer(question, self.retrieve(question, k), self.settings)


@lru_cache(maxsize=4)
def get_pipeline(config: str) -> RAGPipeline:
    return RAGPipeline(config)
