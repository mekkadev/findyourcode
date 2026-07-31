"""Turn a code chunk into the natural-language-ish text that gets embedded.

Identifiers carry most of the meaning in source code, but an embedding model only
sees it when they are split into words: `checkUserCredentials` -> `check user
credentials` is what makes a query like "где происходит авторизация" match.
"""

from __future__ import annotations

import hashlib
import re

from .chunker import Chunk

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

_STOPWORDS = set(
    """
    self this true false none null nil undefined void return if else elif for while do break continue
    import from export default const let var func fun def class struct enum impl trait interface type
    public private protected static final async await yield new delete try catch except finally raise
    throw with as in is not and or int str bool string float double char byte list dict map set array
    len print console log err error ok value item items data args kwargs param params result res req
    the and for you not are was but all can has have will out get set add tmp temp foo bar baz
    """.split()
)


def split_identifiers(text: str, limit: int = 80) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    for ident in _IDENT.findall(text):
        for part in _CAMEL.findall(ident):
            word = part.lower()
            if len(word) < 3 or word in _STOPWORDS or word.isdigit() or word in seen:
                continue
            seen.add(word)
            words.append(word)
            if len(words) >= limit:
                return words
    return words


def path_words(rel: str) -> list[str]:
    return split_identifiers(re.sub(r"[/\\.\-]", " ", rel), limit=24)


def build_embed_text(chunk: Chunk, max_chars: int = 4000) -> str:
    """Most models truncate to a few hundred tokens, so the order matters:
    identity first, then meaning, then the raw code as a bonus for long-context models."""
    full_symbol = ".".join(p for p in (chunk.parent, chunk.symbol) if p)
    parts = [" ".join(p for p in (chunk.lang, chunk.kind, full_symbol) if p)]

    doc = " ".join(chunk.doc.split())
    if doc:
        parts.append(f"about: {doc[:400]}")

    signature = chunk.code.strip().split("\n", 1)[0][:200]
    names = split_identifiers(f"{full_symbol} {signature} {chunk.code}", limit=60)
    if names:
        parts.append(f"names: {' '.join(names)}")

    parts.append(f"file: {chunk.rel} ({' '.join(path_words(chunk.rel))})")
    parts.append("code:\n" + chunk.code[:max_chars])
    return "\n".join(parts)


def chunk_sha(embed_text: str) -> str:
    return hashlib.sha256(embed_text.encode("utf-8")).hexdigest()[:32]


def lexical_text(chunk: Chunk, embed_text: str) -> str:
    """Text fed to FTS5 — code plus split identifiers so BM25 sees both forms."""
    return f"{chunk.rel} {chunk.symbol} {chunk.parent}\n{embed_text}"
