import json

from findyourcode import cli
from findyourcode.config import load_config
from findyourcode.embeddings import get_embedder
from findyourcode.indexer import build_index
from findyourcode.search import similar_to
from findyourcode.store import Store

FILES = {
    "queue/ring.py": (
        "class RingBuffer:\n"
        "    def is_empty(self):\n"
        "        return self.head == self.tail\n"
    ),
    "queue/deque.py": (
        "class Deque:\n"
        "    def is_empty(self):\n"
        "        return len(self.items) == 0\n"
    ),
    "ui/paint.py": "def fill_rect(canvas, box, colour):\n    canvas.rect(box, colour)\n",
}


def _index(root):
    cfg = load_config(root, provider="hash")
    store = Store(cfg.db_path)
    build_index(cfg, get_embedder(cfg.provider), store)
    return cfg, store


def test_similar_resolves_path_and_line(repo):
    root = repo(FILES)
    cfg, store = _index(root)

    anchor, hits = similar_to(store, "queue/ring.py:2", cfg)
    assert anchor.rel == "queue/ring.py"
    assert anchor.start_line <= 2 <= anchor.end_line
    assert hits and hits[0].row.rel == "queue/deque.py"
    assert all(h.row.rel != "queue/ring.py" for h in hits)

    by_path, _ = similar_to(store, "queue/ring.py", cfg)
    assert by_path.rel == "queue/ring.py"
    store.close()


def test_similar_can_include_the_same_file(repo):
    root = repo({"a.py": "def one():\n    return 1\n\n\ndef two():\n    return 2\n"})
    cfg, store = _index(root)
    _, without = similar_to(store, "a.py:1", cfg)
    _, with_self = similar_to(store, "a.py:1", cfg, same_file=True)
    assert without == []
    assert with_self and with_self[0].row.rel == "a.py"
    store.close()


def test_similar_on_unknown_location(repo):
    root = repo(FILES)
    cfg, store = _index(root)
    anchor, hits = similar_to(store, "nowhere/at/all.py:5", cfg)
    assert anchor is None and hits == []
    store.close()


def test_output_formats(repo, capsys):
    root = repo(FILES)
    cli.main(["-C", str(root), "--provider", "hash", "index", "-q"])
    capsys.readouterr()

    cli.main(["-C", str(root), "find", "empty check", "-n", "2", "-f", "paths"])
    paths = capsys.readouterr().out.strip().split("\n")
    assert all(":" in line for line in paths)

    cli.main(["-C", str(root), "find", "empty check", "-n", "2", "-f", "files"])
    files = capsys.readouterr().out.strip().split("\n")
    assert all(":" not in line for line in files)
    assert len(set(files)) == len(files)

    cli.main(["-C", str(root), "find", "empty check", "-n", "2", "-f", "json"])
    assert isinstance(json.loads(capsys.readouterr().out), list)


def test_similar_cli(repo, capsys):
    root = repo(FILES)
    cli.main(["-C", str(root), "--provider", "hash", "index", "-q"])
    capsys.readouterr()

    assert cli.main(["-C", str(root), "similar", "queue/ring.py:2", "-f", "files"]) == 0
    assert capsys.readouterr().out.strip().split("\n")[0] == "queue/deque.py"

    assert cli.main(["-C", str(root), "similar", "missing.py"]) == 2
