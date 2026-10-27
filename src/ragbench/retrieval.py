"""Retrievers: BM25, dense, hybrid (Reciprocal Rank Fusion) and cross-encoder reranking.

Every retriever exposes ``search(query, k) -> list[(chunk_index, score)]``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .chunking import Chunk

Hits = list[tuple[int, float]]

STOPWORDS = {
    "fr": set(
        """
        a au aux avec ce ces cet cette dans de des du elle en et eux il ils je la le les leur leurs
        lui ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son
        sur ta te tes toi ton tu un une vos votre vous est sont ete etre avoir ont fait faire peut
        doit doivent ainsi lorsque dont selon tout tous toute toutes autre autres plus si sans sous entre
        """.split()
    ),
    "en": set(
        """
        a an and are as at be been being but by can could did do does for from had has have he her his
        how i if in into is it its may me more must my no not of on or our shall should so such than
        that the their them then there these they this those to under upon was we were what when where
        which while who whom why will with within without would you your any all other each also
        """.split()
    ),
}


def tokenize(text: str, language: str = "en") -> list[str]:
    stopwords = STOPWORDS[language]
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 1 and t not in stopwords]


def _top_k(scores: np.ndarray, k: int) -> Hits:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


class Retriever(Protocol):
    def search(self, query: str, k: int) -> Hits: ...


class BM25Retriever:
    """Okapi BM25 implemented from scratch with an inverted index.

    score(d, q) = sum_t idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    with the non-negative idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)) used by Lucene.
    """

    def __init__(self, chunks: Sequence[Chunk], language: str = "en", k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.language = k1, b, language
        docs = [tokenize(c.text, language) for c in chunks]
        self.n_docs = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        postings: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(docs):
            for term in doc:
                postings.setdefault(term, {}).setdefault(i, 0)
                postings[term][i] += 1
        self.postings = {
            t: (np.fromiter(p.keys(), dtype=np.int64), np.fromiter(p.values(), dtype=np.float32))
            for t, p in postings.items()
        }

    def idf(self, term: str) -> float:
        df = len(self.postings[term][0]) if term in self.postings else 0
        return float(np.log1p((self.n_docs - df + 0.5) / (df + 0.5)))

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in set(tokenize(query, self.language)):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            scores[ids] += self.idf(term) * tf * (self.k1 + 1) / (tf + norm[ids])
        return scores

    def search(self, query: str, k: int) -> Hits:
        return [(i, s) for i, s in _top_k(self.scores(query), k) if s > 0]


class DenseRetriever:
    """Bi-encoder retrieval with an on-disk embedding cache keyed by model + corpus."""

    def __init__(self, chunks: Sequence[Chunk], model_name: str, cache_dir: Path | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        texts = [c.text for c in chunks]
        digest = hashlib.sha1((model_name + "\x00".join(texts)).encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"emb-{digest}.npy" if cache_dir else None

        if cache and cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(
                texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
            ).astype(np.float32)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.emb)

    def search(self, query: str, k: int) -> Hits:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return _top_k(self.emb @ q, k)


def reciprocal_rank_fusion(rankings: Sequence[Hits], k: int = 60) -> Hits:
    """RRF (Cormack et al., 2009): score(d) = sum_r 1 / (k + rank_r(d))."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, retrievers: Sequence[Retriever], n_candidates: int = 50, rrf_k: int = 60):
        self.retrievers = retrievers
        self.n_candidates = n_candidates
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> Hits:
        rankings = [r.search(query, self.n_candidates) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, self.rrf_k)[:k]


class RerankedRetriever:
    """Retrieve a candidate pool with a first-stage retriever, then rerank with a cross-encoder."""

    def __init__(self, base: Retriever, chunks: Sequence[Chunk], model_name: str, n_candidates: int = 50):
        from sentence_transformers import CrossEncoder

        self.base = base
        self.chunks = chunks
        self.model = CrossEncoder(model_name)
        self.n_candidates = n_candidates

    def search(self, query: str, k: int) -> Hits:
        candidates = self.base.search(query, self.n_candidates)
        if not candidates:
            return []
        scores = self.model.predict([(query, self.chunks[i].text) for i, _ in candidates])
        order = np.argsort(-np.asarray(scores))[:k]
        return [(candidates[j][0], float(scores[j])) for j in order]
