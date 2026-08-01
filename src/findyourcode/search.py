"""Hybrid retrieval: dense vectors + BM25 + the call graph, fused into one ranking.

Two fusion strategies: normalized score blending (default — keeps semantics
dominant and yields a readable 0..1 score) and reciprocal rank fusion. On top of
either, structure: a chunk one call away from a strong match is pulled in even
when neither retriever saw it, which is how a query reaches the implementation
it never names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .embeddings import Embedder
from .store import Edge, Filters, Row, Store


@dataclass
class Hit:
    row: Row
    score: float
    semantic: float | None = None
    lexical: float | None = None
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    graph: float | None = None
    via: str = ""


# What a call edge may argue with when there is no reach window to check it against —
# `--mode lexical` embeds nothing, so nothing can say whether the query points at a
# neighbour at all. It is the weight the graph was measured safe at before the window
# existed; louder than that, ungated, costs the lexical set 0.011 mrr on the stdlib.
UNGATED_WEIGHT = 0.65


@dataclass
class TraceNode:
    """One step of a call path: how we got here, and what this place calls in turn."""

    row: Row
    via: str = ""
    direction: str = ""
    relevance: float = 0.0
    children: list[TraceNode] = field(default_factory=list)


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
    reranker=None,
    graph: bool | None = None,
) -> list[Hit]:
    filters = filters or Filters()
    depth = max(limit * cfg.oversample, 50)
    fusion = fusion or cfg.fusion
    use_graph = cfg.graph if graph is None else graph

    query_vector = None
    near: set[int] | None = None
    dense: list[tuple[int, float]] = []
    sparse: list[tuple[int, float]] = []
    if mode in ("hybrid", "semantic"):
        vector = embedder.embed_query(query)
        # A query the model reduces to nothing — punctuation, whitespace — has no
        # direction to compare against. Asking anyway returns arbitrary neighbours
        # under a brute-force scan and NULL distances under sqlite-vec.
        if np.any(vector):
            query_vector = vector
            # One list, read twice: the head of it is the dense ranking as before, and
            # the tail is only ever consulted to ask whether a call-graph neighbour is
            # somewhere the query points at all. See `_propagate`.
            reach = depth * cfg.graph_reach if use_graph and cfg.graph_reach > 1 else depth
            dense = store.search_vector(query_vector, reach, filters)
            if reach > depth:
                near = {cid for cid, _ in dense}
                dense = dense[:depth]
    if mode in ("hybrid", "lexical"):
        sparse = store.search_lexical(query, depth, filters)

    ids = {cid for cid, _ in dense} | {cid for cid, _ in sparse}
    if not ids:
        return []
    rows = store.rows(list(ids))

    hits: dict[int, Hit] = {cid: Hit(row=rows[cid], score=0.0) for cid in ids if cid in rows}
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
        _score_blend(
            hits, blend_alpha(query, cfg), lexical_used=bool(sparse), semantic_used=bool(dense)
        )

    if use_graph:
        reached = _propagate(store, hits, cfg, filters, near)
        if query_vector is not None and reached:
            _fill_missing_semantics(store, hits, query_vector)

    ranked = sorted(hits.values(), key=lambda h: -h.score)
    cap = cfg.per_file if per_file is None else per_file
    shortlist = _dedupe(ranked, cap)
    if reranker is not None:
        shortlist = reranker.rescore(query, shortlist[: max(cfg.rerank_depth, limit)])
        shortlist = _keep_the_graph_below(shortlist, cfg)
    return shortlist[:limit]


def reached_by_graph(hit: Hit) -> bool:
    """True for a chunk no retriever returned — it is here because something calls it."""
    return hit.graph is not None


def _keep_the_graph_below(hits: list[Hit], cfg: Config) -> list[Hit]:
    """A reranker rewrites every score from scratch, ceiling and all, so the promise
    that structure never outranks a direct answer has to be made again afterwards."""
    direct = [hit.score for hit in hits if not reached_by_graph(hit)]
    if not direct:
        return hits
    ceiling = max(direct) * cfg.graph_ceiling
    for hit in hits:
        if reached_by_graph(hit):
            hit.score = min(hit.score, ceiling)
    # On a tie the direct answer goes first; a reranker that flattens everything to one
    # score must not leave the page led by something no retriever returned.
    return sorted(hits, key=lambda hit: (-hit.score, reached_by_graph(hit)))


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
    if anchor.id not in vectors or not np.any(vectors[anchor.id]):
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


def build_trace(
    store: Store,
    row: Row,
    cfg: Config,
    query_vector=None,
) -> TraceNode:
    """The call path through a result: who reaches it, and what it reaches in turn.

    Which callee to follow is decided by the query, not by the source order — the
    branch that answers the question is the one worth printing. Filters are not
    applied: a call path that stopped at the edge of `--path` would not be a path."""
    root = TraceNode(row=row, relevance=1.0)
    visited = {row.id}

    up = _step(store, [row.id], "up", query_vector, cfg.trace_callers, visited)
    down = (
        _step(store, [row.id], "down", query_vector, cfg.trace_fanout, visited)
        if cfg.trace_depth > 0
        else []
    )
    for node in down:
        node.children = _descend(
            store, node.row, cfg, query_vector, cfg.trace_depth - 1, visited | {node.row.id}
        )
    root.children = up + down
    return root


def _descend(store, row: Row, cfg: Config, query_vector, depth: int, visited: set[int]):
    """`visited` is the path to here, not everything seen anywhere: two branches of a
    call tree may legitimately end at the same helper, and both should say so."""
    if depth <= 0:
        return []
    nodes = _step(store, [row.id], "down", query_vector, cfg.trace_fanout, set(visited))
    for node in nodes:
        node.children = _descend(
            store, node.row, cfg, query_vector, depth - 1, visited | {node.row.id}
        )
    return nodes


def _step(
    store: Store,
    ids: list[int],
    direction: str,
    query_vector,
    fanout: int,
    visited: set[int],
) -> list[TraceNode]:
    if fanout <= 0:
        return []
    edges = store.edges_from(ids) if direction == "down" else store.edges_to(ids)
    reachable: dict[int, Edge] = {}
    for edge in edges:
        target = edge.dst if direction == "down" else edge.src
        if target not in visited:
            reachable.setdefault(target, edge)
    if not reachable:
        return []

    rows = store.rows(list(reachable))
    label = "calls" if direction == "down" else "called by"
    nodes = []
    for cid, relevance in _by_relevance(store, list(rows), query_vector, reachable):
        visited.add(cid)
        nodes.append(
            TraceNode(row=rows[cid], via=reachable[cid].name, direction=label, relevance=relevance)
        )
        if len(nodes) >= fanout:
            break
    return nodes


def _by_relevance(store: Store, ids: list[int], query_vector, edges) -> list[tuple[int, float]]:
    if query_vector is None:
        return sorted(((cid, edges[cid].weight) for cid in ids), key=lambda pair: -pair[1])
    vectors = store.vectors_for(ids)
    # No vector is a reason to rank a branch last, never a reason to drop it.
    scored = [
        (cid, float(np.dot(vectors[cid], query_vector)) if cid in vectors else -1.0) for cid in ids
    ]
    return sorted(scored, key=lambda pair: -pair[1])


def _propagate(
    store: Store, hits: dict[int, Hit], cfg: Config, filters: Filters, near: set[int] | None = None
) -> set[int]:
    """Spread the score of the strongest results along call edges.

    The graph adds candidates; it never re-ranks the ones the text already found.
    Letting a call edge push a text match up the page cost 0.016 MRR on the stdlib
    set for nothing, so a neighbour only ever enters below the best direct answer:
    structure is a reason to read something, not a reason to trust it more than the
    query itself.

    `near` is the reach window: the chunks the query ranks within `graph_reach` times
    the retrieval depth. A call edge is evidence about *which* nearly-relevant chunk to
    surface, not evidence that an irrelevant one is relevant, so a neighbour the query
    ranks nowhere is dropped however loudly the structure argues for it. Measured on the
    cpython index, this is what makes `graph_weight` safe to raise: the neighbours that
    used to displace direct answers when it went up sit at dense rank 700, 1500, 2000 or
    outside the top 3000 entirely, while the ones worth having sit at 120 to 640."""
    seeds = sorted(hits.values(), key=lambda h: -h.score)[: cfg.graph_seeds]
    seeds = [hit for hit in seeds if hit.score > 0]
    if not seeds:
        return set()

    seed_ids = [hit.row.id for hit in seeds]
    ceiling = seeds[0].score * cfg.graph_ceiling
    loudest = cfg.graph_weight if near is not None else min(cfg.graph_weight, UNGATED_WEIGHT)
    boosts: dict[int, tuple[float, str]] = {}

    def collect(target: int, source: Hit, weight: float, label: str) -> None:
        amount = loudest * source.score * weight
        total, first = boosts.get(target, (0.0, label))
        boosts[target] = (total + amount, first)

    for edge in store.edges_from(seed_ids):
        source = hits.get(edge.src)
        if source is not None and edge.dst != edge.src:
            collect(edge.dst, source, edge.weight, f"called by {_name_of(source.row)}")
    for edge in store.edges_to(seed_ids):
        source = hits.get(edge.dst)
        if source is not None and edge.src != edge.dst:
            collect(edge.src, source, edge.weight, f"calls {edge.name}")

    candidates = {
        cid: value
        for cid, value in boosts.items()
        if cid not in hits and (near is None or cid in near)
    }
    if not candidates:
        return set()

    # Filter before taking the best few, not after: a page of `--lang python` should not
    # lose its call-graph slots to five typescript neighbours that are then discarded.
    allowed = store.restrict(sorted(candidates), filters)
    fresh = sorted(
        ((cid, value) for cid, value in candidates.items() if cid in allowed),
        key=lambda pair: -pair[1][0],
    )[: cfg.graph_limit]
    if not fresh:
        return set()
    rows = store.rows(sorted(cid for cid, _ in fresh))
    for cid, (amount, label) in fresh:
        row = rows.get(cid)
        if row is None:
            continue
        strength = min(amount, loudest)
        hits[cid] = Hit(row=row, score=min(strength, ceiling), graph=strength, via=label)
    return set(rows)


def _name_of(row: Row) -> str:
    return row.symbol or row.parent or row.rel.rsplit("/", 1)[-1]


def _fill_missing_semantics(store: Store, hits: dict[int, Hit], query_vector) -> None:
    """Lexical-only candidates have no cosine yet — fetch their vectors and score them."""
    missing = [cid for cid, hit in hits.items() if hit.semantic is None]
    for cid, vector in store.vectors_for(missing).items():
        hits[cid].semantic = float(np.dot(vector, query_vector))


def is_short(query: str, cfg: Config) -> bool:
    """Short is measured in words, not characters. `deserialize_untrusted_payload` is
    long and still one word — and one word is the thing that makes a query hard."""
    if cfg.short_query_words <= 0:
        return False
    return 0 < len(query.split()) < cfg.short_query_words


def blend_alpha(query: str, cfg: Config) -> float:
    """How much of the ranking the embedding gets to decide, for this query.

    A sentence is mostly ordinary English and the model reads it better than BM25
    ever will. `epoll` is not a sentence. It is one rare token that names a thing
    living in exactly one file, the model has never seen it used as a word, and it
    answers `colorsys` — while BM25, which needs nothing but the posting list, has
    had the right file all along. So the two retrievers are weighted by how much
    query there is to read.

    Only the blend fusion has an alpha; under `--fusion rrf` this does nothing and
    the rank-based weights apply unchanged."""
    if cfg.short_query_alpha < 0 or not is_short(query, cfg):
        return cfg.alpha
    return cfg.short_query_alpha


def _score_blend(
    hits: dict[int, Hit], alpha: float, lexical_used: bool, semantic_used: bool = True
) -> None:
    semantics = _minmax([h.semantic for h in hits.values()])
    lexicals = _minmax([h.lexical for h in hits.values()])
    if not lexical_used:
        alpha = 1.0
    elif not semantic_used:
        alpha = 0.0
    for hit, sem, lex in zip(hits.values(), semantics, lexicals, strict=True):
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


def _overlap_ratio(row: Row, start: int, end: int) -> float:
    """Consecutive windows of one long function share an overlap by design — only
    call two chunks duplicates when most of the smaller one is inside the other."""
    shared = min(row.end_line, end) - max(row.start_line, start) + 1
    if shared <= 0:
        return 0.0
    shortest = min(row.end_line - row.start_line, end - start) + 1
    return shared / shortest


def _dedupe(hits: list[Hit], per_file: int = 0) -> list[Hit]:
    """Drop overlapping spans, and keep one file from swallowing the whole page."""
    kept: list[Hit] = []
    spans: dict[str, list[tuple[int, int]]] = {}
    for hit in hits:
        row = hit.row
        seen = spans.setdefault(row.rel, [])
        if any(_overlap_ratio(row, start, end) > 0.5 for start, end in seen):
            continue
        if per_file and len(seen) >= per_file:
            continue
        seen.append((row.start_line, row.end_line))
        kept.append(hit)
    return kept
