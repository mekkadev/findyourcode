"""Cases found by adversarial review of the indexing and chunking paths."""

import pytest

from findyourcode.chunker import chunk_source
from findyourcode.config import load_config
from findyourcode.embeddings import get_embedder
from findyourcode.indexer import build_index
from findyourcode.search import search
from findyourcode.store import Store
from findyourcode.walker import discover

SOURCES = {
    "python": "class A:\n    def m(self):\n        return 1\n\n\ndef top():\n    return 2\n",
    "typescript": (
        "declare function ambient(x: number): void;\n"
        "export class Widget {\n  render() { return 1; }\n}\n"
    ),
    "go": "package m\n\ntype S struct {\n\tA int\n}\n\nfunc (s S) M() int {\n\treturn s.A\n}\n",
    "rust": "pub struct S;\n\nimpl S {\n    pub fn go(&self) -> u8 {\n        1\n    }\n}\n",
    "php": "<?php\nnamespace App;\n\nenum Suit { case Hearts; }\n\nfunction helper() { return 1; }\n",
    "kotlin": "class A {\n    companion object {\n        fun create(): A = A()\n    }\n}\n",
    "swift": "class A {\n    init(x: Int) { }\n    func run() -> Int { return 1 }\n}\n",
    "ruby": "class A\n  def m\n    1\n  end\nend\n",
    "java": "class A {\n    void m() {\n    }\n}\n",
}


@pytest.mark.parametrize("lang", sorted(SOURCES))
def test_no_source_line_is_lost(lang, cfg):
    source = SOURCES[lang]
    chunks = chunk_source(f"sample.{lang}", lang, source, cfg)
    assert chunks, lang

    covered = set()
    for chunk in chunks:
        assert 1 <= chunk.start_line <= chunk.end_line
        covered |= set(range(chunk.start_line, chunk.end_line + 1))

    lines = source.split("\n")
    missing = [i for i, text in enumerate(lines, 1) if text.strip() and i not in covered]
    assert missing == [], f"{lang} lost lines {missing}"


@pytest.mark.parametrize(
    "lang,source,symbol",
    [
        ("typescript", "declare function ambient(x: number): void;\n", "ambient"),
        ("php", "<?php\nenum Suit { case Hearts; }\n", "Suit"),
        ("kotlin", "class A {\n  companion object {\n    fun make() = 1\n  }\n}\n", "A"),
        ("swift", "class A {\n  init(x: Int) { }\n}\n", "A"),
    ],
)
def test_definitions_the_language_table_used_to_miss(lang, source, symbol, cfg):
    chunks = chunk_source(f"s.{lang}", lang, source, cfg)
    names = {c.symbol for c in chunks} | {c.parent for c in chunks}
    assert symbol in names, [(c.kind, c.symbol, c.parent) for c in chunks]


def test_overlap_larger_than_the_window_does_not_explode(cfg):
    cfg.max_chunk_lines = 10
    cfg.overlap_lines = 50
    source = "def huge():\n" + "\n".join(f"    x{i} = {i}" for i in range(200))
    chunks = chunk_source("huge.py", "python", source, cfg)
    assert len(chunks) < 60  # step must stay meaningful, not collapse to one line


def test_exclude_patterns_with_slashes(repo, cfg):
    cfg.root = repo(
        {"build/app.py": "x = 1\n", "src/app.py": "y = 2\n", "deep/build/app.py": "z = 3\n"}
    )
    cfg.exclude = ["/build/**"]
    assert {f.rel for f in discover(cfg)} == {"src/app.py", "deep/build/app.py"}

    cfg.exclude = ["build/"]
    assert {f.rel for f in discover(cfg)} == {"src/app.py"}


def test_unreadable_file_loses_its_stale_chunks(repo):
    root = repo({"gone.py": "def alpha():\n    return 1\n", "stays.py": "def beta():\n    return 2\n"})
    cfg = load_config(root, provider="hash")
    embedder = get_embedder(cfg.provider)
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)
    assert any(h.row.rel == "gone.py" for h in search(store, embedder, "alpha", cfg))

    # Readable when hashed, unreadable when parsed.
    target = root / "gone.py"
    target.write_text("def alpha():\n    return 1\n# touched\n", encoding="utf-8")
    original = target.read_bytes

    import findyourcode.walker as walker

    real_read = walker.read_source

    def fail_for_gone(path):
        return None if path.name == "gone.py" else real_read(path)

    walker.read_source = fail_for_gone
    try:
        import findyourcode.indexer as indexer

        indexer.read_source = fail_for_gone
        stats = build_index(cfg, embedder, store)
    finally:
        walker.read_source = real_read
        indexer.read_source = real_read

    assert "gone.py" in stats.unreadable
    assert not any(h.row.rel == "gone.py" for h in search(store, embedder, "alpha", cfg))
    assert any(h.row.rel == "stays.py" for h in search(store, embedder, "beta", cfg))
    assert original
    store.close()


def test_backend_mismatch_reports_instead_of_crashing(repo, monkeypatch):
    root = repo({"a.py": "def alpha():\n    return 1\n"})
    cfg = load_config(root, provider="hash")
    embedder = get_embedder(cfg.provider)
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)
    store.set_meta("backend", "vec0" if store.get_meta("backend") == "numpy" else "numpy")
    store.commit()

    with pytest.raises(SystemExit) as excinfo:
        search(store, embedder, "alpha", cfg)
    assert "--reindex" in str(excinfo.value)
    store.close()


def test_git_listing_is_deduplicated(repo, cfg, monkeypatch):
    cfg.root = repo({"a.py": "x = 1\n"})
    import findyourcode.walker as walker

    monkeypatch.setattr(walker, "_git_files", lambda root: ["a.py", "a.py", "a.py"])
    assert [f.rel for f in discover(cfg)] == ["a.py"]


def test_knn_beyond_the_backend_limit(repo):
    root = repo({"a.py": "def alpha():\n    return 1\n"})
    cfg = load_config(root, provider="hash")
    embedder = get_embedder(cfg.provider)
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    from findyourcode.store import Filters

    assert store.search_vector(embedder.embed_query("alpha"), 50_000, Filters()) != []
    store.close()


def test_narrow_filter_still_reaches_its_matches(repo):
    files = {f"noise/mod_{i}.py": f"def unrelated_{i}():\n    return {i}\n" for i in range(60)}
    files["rare/cache.rs"] = "fn evict_expired_entries() -> u8 {\n    1\n}\n"
    root = repo(files)
    cfg = load_config(root, provider="hash")
    embedder = get_embedder(cfg.provider)
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    from findyourcode.store import Filters

    hits = search(store, embedder, "unrelated", cfg, filters=Filters(langs=["rust"]))
    assert hits and all(h.row.lang == "rust" for h in hits)
    store.close()


def test_lexical_mode_uses_the_full_score_range(repo):
    root = repo({"a.py": "def verify_password(login):\n    return login\n"})
    cfg = load_config(root, provider="hash")
    embedder = get_embedder(cfg.provider)
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    hits = search(store, embedder, "verify password", cfg, mode="lexical")
    assert hits and hits[0].score == pytest.approx(1.0)
    store.close()


def test_bm25_sees_the_tail_of_a_long_chunk(repo):
    body = "\n".join(f"    step_{i} = {i}" for i in range(300))
    root = repo({"long.py": f"def pipeline():\n{body}\n    return unmistakable_tail_marker\n"})
    cfg = load_config(root, provider="hash")
    embedder = get_embedder(cfg.provider)
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    hits = search(store, embedder, "unmistakable_tail_marker", cfg, mode="lexical")
    assert hits, "the identifier past the embedding-text cutoff must still be searchable"
    store.close()


def test_consecutive_windows_of_one_function_are_not_deduped(cfg):
    from findyourcode.search import Hit, _dedupe
    from findyourcode.store import Row

    first = Hit(row=Row(1, "a.py", "python", "function", "f [1/2]", "", 1, 110, "x"), score=1.0)
    second = Hit(row=Row(2, "a.py", "python", "function", "f [2/2]", "", 98, 208, "y"), score=0.9)
    duplicate = Hit(row=Row(3, "a.py", "python", "function", "f", "", 1, 108, "x"), score=0.8)

    kept = _dedupe([first, second, duplicate], per_file=0)
    assert [h.row.id for h in kept] == [1, 2]
