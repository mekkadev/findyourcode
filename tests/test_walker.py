from findyourcode.chunker import chunk_source
from findyourcode.walker import discover


def rels(cfg):
    return {f.rel for f in discover(cfg)}


def test_filters_binary_empty_and_oversized(repo, cfg):
    root = repo({"a.py": "x = 1\n", "empty.py": "", "big.py": "y = 2\n"})
    (root / "blob.py").write_bytes(b"\x00\x01binary")
    cfg.root = root
    cfg.max_file_bytes = 5
    found = rels(cfg)
    assert "a.py" not in found  # 6 bytes, over the limit
    assert "empty.py" not in found
    assert "blob.py" not in found

    cfg.max_file_bytes = 1_000_000
    assert rels(cfg) == {"a.py", "big.py"}


def test_unknown_extension_is_skipped(repo, cfg):
    cfg.root = repo({"a.py": "x = 1\n", "notes.bin.xyz": "hello\n"})
    assert rels(cfg) == {"a.py"}


def test_exclude_and_include_globs(repo, cfg):
    cfg.root = repo(
        {
            "src/app.py": "x = 1\n",
            "src/generated/api.py": "y = 2\n",
            "vendor/lib.py": "z = 3\n",
            "web/app.min.js": "var a=1\n",
        }
    )
    cfg.exclude = ["**/generated/**", "**/vendor/**", "*.min.js"]
    assert rels(cfg) == {"src/app.py"}

    cfg.exclude = []
    cfg.include = ["src/**"]
    assert rels(cfg) == {"src/app.py", "src/generated/api.py"}


def test_chunker_survives_broken_and_empty_sources(cfg):
    assert chunk_source("empty.py", "python", "", cfg) == []
    assert chunk_source("blank.py", "python", "\n\n", cfg) == []

    broken = chunk_source("broken.py", "python", "def f(:\n  x = ((((\nclass ???", cfg)
    assert broken and broken[0].code.startswith("def f(:")

    unicode_chunks = chunk_source("u.py", "python", "def привет_мир():\n    return 'ok'\n", cfg)
    assert unicode_chunks[0].symbol == "привет_мир"


def test_language_without_recognised_definitions_falls_back(cfg):
    chunks = chunk_source("main.zig", "zig", "pub fn main() void { }\n", cfg)
    assert len(chunks) == 1
    assert "pub fn main" in chunks[0].code
