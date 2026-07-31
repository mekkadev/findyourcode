"""Hybrid retrieval: dense vectors + BM25, fused into one ranking.

Two fusion strategies: normalized score blending (default — keeps semantics
dominant and yields a readable 0..1 score) and reciprocal rank fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .embeddings import Embedder
from .store import Filters, Row, Store


@dataclass
class Hit:
    row: Row
    score: float
    semantic: float | None = None
    lexical: float | None = None
    semantic_rank: int | None = None
    lexical_rank: int | None = None


def search(
    store: Store,
    embedder: Embedder,
    query: str,
    cfg: Config,
    limit: int = 10,
    filters: Filters | None = None,
    mode: str = "hybrid",
    fusion: str = "",
    per_file: int | None = None,
) -> list[Hit]:
    filters = filters or Filters()
    depth = max(limit * cfg.oversample, 50)
    fusion = fusion or cfg.fusion

    query_vector = None
    dense: list[tuple[int, float]] = []
    sparse: list[tuple[int, float]] = []
    if mode in ("hybrid", "semantic"):
        query_vector = embedder.embed_query(query)
        dense = store.search_vector(query_vector, depth, filters)
    if mode in ("hybrid", "lexical"):
        sparse = store.search_lexical(query, depth, filters)

    ids = {cid for cid, _ in dense} | {cid for cid, _ in sparse}
    if not ids:
        return []
    rows = store.rows(list(ids))

    hits: dict[int, Hit] = {
        cid: Hit(row=rows[cid], score=0.0) for cid in ids if cid in rows
    }
    for rank, (cid, score) in enumerate(dense, 1):
        if cid in hits:
            hits[cid].semantic, hits[cid].semantic_rank = score, rank
    for rank, (cid, score) in enumerate(sparse, 1):
        if cid in hits:
            hits[cid].lexical, hits[cid].lexical_rank = score, rank

    if query_vector is not None:
        _fill_missing_semantics(store, hits, query_vector)

    if fusion == "rrf":
        _score_rrf(hits, cfg)
    else:
        _score_blend(hits, cfg, lexical_used=bool(sparse))

    ranked = sorted(hits.values(), key=lambda h: -h.score)
    cap = cfg.per_file if per_file is None else per_file
    return _dedupe(ranked, cap)[:limit]


def similar_to(
    store: Store,
    location: str,
    cfg: Config,
    limit: int = 10,
    filters: Filters | None = None,
    same_file: bool = False,
) -> tuple[Row | None, list[Hit]]:
    """Neighbours of an indexed chunk, addressed as `path` or `path:line`."""
    anchor = store.chunk_at(location)
    if anchor is None:
        return None, []

    vectors = store.vectors_for([anchor.id])
    if anchor.id not in vectors:
        return anchor, []

    depth = max(limit * cfg.oversample, 50)
    dense = store.search_vector(vectors[anchor.id], depth, filters or Filters())
    rows = store.rows([cid for cid, _ in dense])

    hits = [
        Hit(row=rows[cid], score=score, semantic=score, semantic_rank=rank)
        for rank, (cid, score) in enumerate(dense, 1)
        if cid in rows and cid != anchor.id and (same_file or rows[cid].rel != anchor.rel)
    ]
    return anchor, _dedupe(hits, cfg.per_file)[:limit]


def _fill_missing_semantics(store: Store, hits: dict[int, Hit], query_vector) -> None:
    """Lexical-only candidates have no cosine yet — fetch their vectors and score them."""
    missing = [cid for cid, hit in hits.items() if hit.semantic is None]
    for cid, vector in store.vectors_for(missing).items():
        hits[cid].semantic = float(np.dot(vector, query_vector))


def _score_blend(hits: dict[int, Hit], cfg: Config, lexical_used: bool) -> None:
    semantics = _minmax([h.semantic for h in hits.values()])
    lexicals = _minmax([h.lexical for h in hits.values()])
    alpha = cfg.alpha if lexical_used else 1.0
    for hit, sem, lex in zip(hits.values(), semantics, lexicals):
        hit.score = alpha * sem + (1.0 - alpha) * lex


def _score_rrf(hits: dict[int, Hit], cfg: Config) -> None:
    for hit in hits.values():
        score = 0.0
        if hit.semantic_rank:
            score += cfg.semantic_weight / (cfg.rrf_k + hit.semantic_rank)
        if hit.lexical_rank:
            score += cfg.lexical_weight / (cfg.rrf_k + hit.lexical_rank)
        hit.score = score


def _minmax(values: list[float | None]) -> list[float]:
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values)
    low, high = min(present), max(present)
    span = high - low
    if span < 1e-9:
        return [1.0 if v is not None else 0.0 for v in values]
    return [0.0 if v is None else (v - low) / span for v in values]


def _dedupe(hits: list[Hit], per_file: int = 0) -> list[Hit]:
    """Drop overlapping spans, and keep one file from swallowing the whole page."""
    kept: list[Hit] = []
    spans: dict[str, list[tuple[int, int]]] = {}
    for hit in hits:
        row = hit.row
        seen = spans.setdefault(row.rel, [])
        if any(row.start_line <= end and start <= row.end_line for start, end in seen):
            continue
        if per_file and len(seen) >= per_file:
            continue
        seen.append((row.start_line, row.end_line))
        kept.append(hit)
    return kept
