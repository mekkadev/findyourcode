from findyourcode.chunker import chunk_source

PY = '''
import os

CONSTANT = 42


def top_level(a, b):
    """Adds two numbers."""
    return a + b


class Service:
    # keeps a connection pool
    def __init__(self, pool):
        self.pool = pool

    def handle(self, request):
        return self.pool.run(request)
'''


def test_python_units(cfg):
    chunks = chunk_source("svc.py", "python", PY, cfg)
    kinds = {(c.kind, c.symbol) for c in chunks}
    assert ("function", "top_level") in kinds
    assert ("class", "Service") in kinds
    assert any(c.kind == "block" for c in chunks)

    fn = next(c for c in chunks if c.symbol == "top_level")
    assert "def top_level" in c_head(fn)
    assert "Adds two numbers" in fn.doc
    assert fn.start_line < fn.end_line


def test_leading_comment_is_attached(cfg):
    source = "# explains the helper\ndef helper():\n    return 1\n"
    chunks = chunk_source("h.py", "python", source, cfg)
    fn = next(c for c in chunks if c.symbol == "helper")
    assert fn.start_line == 1
    assert "explains the helper" in fn.doc


def test_big_class_splits_into_methods(cfg):
    cfg.max_chunk_lines = 12
    body = "\n".join(f"    def m{i}(self):\n        return {i}\n" for i in range(10))
    chunks = chunk_source("big.py", "python", f"class Big:\n{body}", cfg)
    methods = [c for c in chunks if c.kind == "method"]
    assert len(methods) >= 8
    assert all(m.parent == "Big" for m in methods)


def test_oversized_function_windows(cfg):
    cfg.max_chunk_lines = 10
    cfg.overlap_lines = 2
    source = "def huge():\n" + "\n".join(f"    x{i} = {i}" for i in range(60))
    parts = [c for c in chunk_source("huge.py", "python", source, cfg) if "huge" in c.symbol]
    assert len(parts) > 3
    assert all(c.end_line - c.start_line + 1 <= 10 for c in parts)


def test_typescript_and_go(cfg):
    ts = chunk_source("m.ts", "typescript", "export function guard(ctx) { return ctx; }\n", cfg)
    assert any(c.symbol == "guard" for c in ts)

    go = chunk_source("m.go", "go", "package m\n\nfunc Collect(x int) int {\n\treturn x\n}\n", cfg)
    assert any(c.symbol == "Collect" for c in go)


def test_unknown_language_falls_back_to_windows(cfg):
    chunks = chunk_source("notes.md", "markdown", "# Title\n\ntext\n\n## Second\n\nmore\n", cfg)
    assert [c.symbol for c in chunks] == ["", "Title", "Second"] or len(chunks) >= 2


def test_module_summary_is_attached_to_every_chunk(cfg):
    source = '"""Command-line parsing library.\n\nMore prose.\n"""\n\ndef parse():\n    return 1\n'
    chunks = chunk_source("cli.py", "python", source, cfg)
    assert chunks
    assert all(c.file_doc.startswith("Command-line parsing library.") for c in chunks)


def test_function_docstring_is_not_mistaken_for_a_module_summary(cfg):
    source = 'def only(x):\n    """Doc that belongs to the function."""\n    return x\n'
    assert all(c.file_doc == "" for c in chunk_source("f.py", "python", source, cfg))


def test_leading_comments_are_used_when_there_is_no_docstring(cfg):
    source = "// Package billing charges cards.\n\nfunc Collect() {}\n"
    chunks = chunk_source("charge.go", "go", source, cfg)
    assert chunks[0].file_doc == "Package billing charges cards."


def c_head(chunk):
    return chunk.code.split("\n")[0]
