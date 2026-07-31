"""Optional second pass: a cross-encoder reads the query and the chunk together.

The retrievers score a query against a vector that was computed without ever seeing
the query. A cross-encoder sees both at once, which is more accurate and far too
slow to run over an index — so it only ever rescores what fusion already shortlisted.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enrich import path_words
from .search import Hit

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
MAX_CHARS = 1200


@dataclass
class Reranker:
    model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "reranking needs fastembed: pip install 'findyourcode[local]'"
            ) from exc
        self._backend = TextCrossEncoder(self.model)

    def rescore(self, query: str, hits: list[Hit]) -> list[Hit]:
        if len(hits) < 2:
            return hits

        scores = list(self._backend.rerank(query, [_passage(hit) for hit in hits]))
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: -pair[1])

        top, bottom = ranked[0][1], ranked[-1][1]
        span = top - bottom
        for hit, score in ranked:
            hit.score = 1.0 if span < 1e-9 else (score - bottom) / span
        return [hit for hit, _ in ranked]


def _passage(hit: Hit) -> str:
    """What the cross-encoder reads: the same signals the embedding got, minus the noise."""
    row = hit.row
    symbol = ".".join(part for part in (row.parent, row.symbol) if part)
    header = " ".join(part for part in (row.kind, symbol) if part)
    return f"{header}\n{' '.join(path_words(row.rel))}\n{row.code[:MAX_CHARS]}"
