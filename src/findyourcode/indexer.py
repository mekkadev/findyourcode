"""Indexing pipeline: discover -> chunk -> enrich -> embed (cached) -> store."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .chunker import chunk_source
from .config import Config
from .embeddings import Embedder
from .enrich import build_embed_text, chunk_sha, lexical_text
from .store import Store
from .walker import SourceFile, discover, read_source

Progress = Callable[[str], None]


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    embedded: int = 0
    reused: int = 0
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)


@dataclass
class _Unit:
    source: SourceFile
    sha: str
    chunks: list
    texts: list[str]


def build_index(
    cfg: Config,
    embedder: Embedder,
    store: Store,
    reindex: bool = False,
    progress: Progress | None = None,
) -> IndexStats:
    say = progress or (lambda _msg: None)
    started = time.time()
    stats = IndexStats()

    if reindex:
        # Archive under the signature the existing vectors were produced with, never the
        # new one — otherwise a model switch relabels the old vector space as the new one.
        store.archive_vectors(store.get_meta("signature") or embedder.signature)
        store.reset_vectors()
        store.commit()

    store.prepare(embedder.signature, embedder.dim, force=reindex)

    files = discover(cfg)
    stats.scanned = len(files)
    known = store.file_states()
    seen: set[str] = set()
    todo: list[tuple[SourceFile, str]] = []

    # An index built before the call graph existed has to be re-read once. Nothing is
    # re-embedded: the chunk texts are unchanged, so every vector comes from the cache.
    stale_graph = store.get_meta("graph") == "rebuild"
    if stale_graph:
        say("index predates the call graph — rebuilding it from the cached vectors")

    for source in files:
        sha = _file_sha(source.path)
        if sha is None:
            continue
        seen.add(source.rel)
        if not reindex and not stale_graph and known.get(source.rel) == sha:
            stats.unchanged += 1
            continue
        todo.append((source, sha))

    stale = [rel for rel in known if rel not in seen]
    if stale:
        store.remove_files(stale)
        stats.removed = len(stale)

    say(f"{stats.scanned} files, {len(todo)} to (re)index, {stats.unchanged} unchanged")

    buffer: list[_Unit] = []
    buffered_chunks = 0
    flush_at = max(cfg.batch_size * 4, 128)

    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        # Every file in a submitted batch is held in memory as text plus chunks, so the
        # batch size is the real memory bound, not the flush threshold.
        for start in range(0, len(todo), 96):
            batch = todo[start : start + 96]
            for unit in pool.map(lambda item: _prepare(item, cfg, stats), batch):
                if unit is None:
                    continue
                buffer.append(unit)
                buffered_chunks += len(unit.chunks)
                if buffered_chunks >= flush_at:
                    _flush(store, embedder, buffer, stats)
                    buffer, buffered_chunks = [], 0
                    say(_progress_line(stats, len(todo), started))

    _flush(store, embedder, buffer, stats)
    if stats.unreadable:
        # Readable when we hashed it, gone by the time we parsed it: drop the stale chunks.
        store.remove_files(stats.unreadable)
        stats.removed += len(stats.unreadable)
    store.prune_cache()
    store.set_meta("graph", "ready")
    store.set_meta("indexed_at", time.time())
    store.commit()

    stats.elapsed = time.time() - started
    return stats


def _progress_line(stats: IndexStats, total: int, started: float) -> str:
    elapsed = max(time.time() - started, 1e-6)
    rate = stats.chunks / elapsed
    left = ""
    if stats.indexed and stats.indexed < total:
        remaining = (total - stats.indexed) * (elapsed / stats.indexed)
        left = f" · ~{_duration(remaining)} left"
    return (
        f"indexing {stats.indexed}/{total} files · {stats.chunks} chunks"
        f" · {rate:.0f} chunks/s{left}"
    )


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _file_sha(path) -> str | None:
    """Hash raw bytes so file contents never have to be held in memory to decide staleness."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()[:32]


def _prepare(item: tuple[SourceFile, str], cfg: Config, stats: IndexStats) -> _Unit | None:
    source, sha = item
    text = read_source(source.path)
    if text is None:
        stats.unreadable.append(source.rel)
        return None
    try:
        chunks = chunk_source(source.rel, source.lang, text, cfg)
    except Exception as exc:  # a broken grammar must not kill the run
        stats.errors.append(f"{source.rel}: {exc}")
        stats.unreadable.append(source.rel)
        return None

    kept, texts = [], []
    for chunk in chunks:
        embed_text = build_embed_text(chunk, cfg.max_embed_chars)
        chunk.sha = chunk_sha(embed_text)
        kept.append(chunk)
        texts.append(embed_text)
    return _Unit(source, sha, kept, texts)


def _flush(store: Store, embedder: Embedder, buffer: list[_Unit], stats: IndexStats) -> None:
    if not buffer:
        return

    wanted: dict[str, str] = {}
    for unit in buffer:
        for chunk, text in zip(unit.chunks, unit.texts, strict=True):
            wanted.setdefault(chunk.sha, text)

    sig = embedder.signature
    cache = store.cached_vectors(sig, list(wanted), embedder.dim)
    missing = [sha for sha in wanted if sha not in cache]
    stats.reused += len(wanted) - len(missing)

    if missing:
        # No copy into emb_cache: the vector is about to be written to the index itself,
        # and that is where `cached_vectors` looks first. Only vectors that leave the
        # index — archived by --reindex — need a second home.
        vectors = embedder.embed_documents([wanted[sha] for sha in missing])
        cache.update({sha: vectors[i] for i, sha in enumerate(missing)})
        stats.embedded += len(missing)

    for unit in buffer:
        matrix = (
            np.vstack([cache[chunk.sha] for chunk in unit.chunks])
            if unit.chunks
            else np.zeros((0, embedder.dim), dtype=np.float32)
        )
        store.add_file(
            unit.source.rel,
            unit.sha,
            unit.source.mtime,
            unit.source.size,
            unit.source.lang,
            unit.chunks,
            [lexical_text(c, t) for c, t in zip(unit.chunks, unit.texts, strict=True)],
            matrix,
        )
        stats.indexed += 1
        stats.chunks += len(unit.chunks)
    store.commit()
