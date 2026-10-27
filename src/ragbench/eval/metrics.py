"""Retrieval metrics at the *section* level.

Relevance is judged on ``section_key`` (e.g. ``ai_act::Article 5``), not on
chunk ids, so labels survive changes in chunk size. A section retrieved twice
only counts once (duplicates get zero gain).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _dedup(keys: Sequence[str]) -> list[str | None]:
    """Replace repeated keys by None so they bring no gain."""
    seen: set[str] = set()
    out: list[str | None] = []
    for key in keys:
        out.append(None if key in seen else key)
        seen.add(key)
    return out


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return float(bool(set(retrieved[:k]) & relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, key in enumerate(retrieved, start=1):
        if key in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, key in enumerate(_dedup(retrieved[:k]), start=1)
        if key is not None and key in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def all_metrics(retrieved: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    return {
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"hit@{k}": hit_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }
