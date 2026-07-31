"""`fyc doctor` — answer "why is it behaving like that" without reading the source."""

from __future__ import annotations

import shutil
import sqlite3
import sys
from dataclasses import dataclass

from .config import Config
from .embeddings import PROVIDERS
from .indexer import _file_sha
from .languages import get_parser
from .store import Store
from .walker import discover

_SAMPLE_LANGS = ("python", "javascript", "typescript", "go", "rust", "java")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def run(cfg: Config) -> list[Check]:
    checks = [
        Check("python", sys.version_info >= (3, 10), sys.version.split()[0]),
        _sqlite(),
        _vector_backend(),
        _fts5(),
        _grammars(),
        _git(),
    ]
    checks.extend(_providers())
    checks.extend(_index(cfg))
    return checks


def render(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    return "\n".join(f"{'ok ' if c.ok else '!! '}{c.name:<{width}}  {c.detail}" for c in checks)


def _sqlite() -> Check:
    return Check("sqlite", sqlite3.sqlite_version_info >= (3, 35), sqlite3.sqlite_version)


def _vector_backend() -> Check:
    try:
        import sqlite_vec
    except ImportError:
        return Check(
            "sqlite-vec",
            True,
            "not installed — vectors fall back to a numpy scan, fine below ~100k chunks",
        )
    return Check("sqlite-vec", True, f"{getattr(sqlite_vec, '__version__', 'installed')}")


def _fts5() -> Check:
    db = sqlite3.connect(":memory:")
    try:
        db.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return Check("fts5", True, "available, lexical branch enabled")
    except sqlite3.OperationalError:
        return Check("fts5", False, "missing from this python's sqlite — search is vectors only")
    finally:
        db.close()


def _grammars() -> Check:
    loaded = [lang for lang in _SAMPLE_LANGS if get_parser(lang) is not None]
    return Check(
        "tree-sitter",
        len(loaded) == len(_SAMPLE_LANGS),
        f"{len(loaded)}/{len(_SAMPLE_LANGS)} sample grammars load ({', '.join(loaded) or 'none'})",
    )


def _git() -> Check:
    found = shutil.which("git")
    return Check(
        "git",
        True,
        "found — file list comes from git ls-files, .gitignore honoured"
        if found
        else "not found — falling back to a directory walk",
    )


def _providers() -> list[Check]:
    import os

    checks = []
    try:
        import fastembed  # noqa: F401

        detail = "fastembed installed, runs offline"
        ok = True
    except ImportError:
        detail = "fastembed missing — pip install 'findyourcode[local]'"
        ok = False
    checks.append(Check("provider local", ok, detail))

    for name, env in (("voyage", "VOYAGE_API_KEY"), ("openai", "OPENAI_API_KEY")):
        has_key = bool(os.environ.get(env))
        checks.append(
            Check(f"provider {name}", True, f"{env} set" if has_key else f"{env} not set, unused")
        )
    assert set(PROVIDERS) >= {"local", "voyage", "openai", "hash"}
    return checks


def _readable(store: Store) -> Check:
    """The vectors are only reachable through the backend that wrote them."""
    try:
        store.check_backend()
    except SystemExit as exc:
        return Check("vectors", False, str(exc).replace("\n", " "))
    return Check("vectors", True, f"readable through {store.vector_backend}")


def _index(cfg: Config) -> list[Check]:
    if not cfg.db_path.exists():
        return [Check("index", False, f"none at {cfg.root} — run `fyc index`")]

    store = Store(cfg.db_path)
    try:
        stats = store.stats()
        known = store.file_states()
        checks = [
            Check(
                "index",
                stats["chunks"] > 0,
                f"{stats['files']} files, {stats['chunks']} chunks, "
                f"{stats['db_bytes'] / 1e6:.1f} mb, vectors via {stats['backend']}",
            ),
            Check("model", stats["signature"] != "-", stats["signature"]),
            _readable(store),
        ]
    finally:
        store.close()

    present = {source.rel: source.path for source in discover(cfg)}
    added = len(set(present) - set(known))
    gone = len(set(known) - set(present))
    changed = sum(
        1
        for rel, path in present.items()
        if rel in known and _file_sha(path) not in (None, known[rel])
    )
    fresh = added == 0 and gone == 0 and changed == 0
    parts = [f"{added} new", f"{changed} edited", f"{gone} deleted"]
    checks.append(
        Check(
            "freshness",
            fresh,
            "index covers every file on disk"
            if fresh
            else ", ".join(parts) + " since the last index — run `fyc index`",
        )
    )
    return checks
