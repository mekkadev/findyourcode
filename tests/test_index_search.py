import pytest

from findyourcode.config import load_config
from findyourcode.embeddings import get_embedder
from findyourcode.indexer import build_index
from findyourcode.search import search
from findyourcode.store import Filters, Store

FILES = {
    "api/session.py": (
        "class CredentialChecker:\n"
        '    """Validates the login and password a client presents."""\n'
        "    def check(self, login, password):\n"
        "        record = self.users.get(login)\n"
        "        if not verify_hash(password, record.digest):\n"
        "            raise PermissionError('wrong password')\n"
        "        return self.issue_ticket(login)\n"
    ),
    "ui/chart.js": (
        "export function drawSeries(canvas, points) {\n"
        "  const ctx = canvas.getContext('2d');\n"
        "  points.forEach((p) => ctx.lineTo(p.x, p.y));\n"
        "}\n"
    ),
    "docs/readme.md": "# Project\n\nSome prose about the project.\n",
}


def index(root, **overrides):
    cfg = load_config(root, provider="hash", **overrides)
    embedder = get_embedder(cfg.provider, cfg.model, cfg.batch_size)
    store = Store(cfg.db_path)
    stats = build_index(cfg, embedder, store, reindex=overrides.pop("reindex", False))
    return cfg, embedder, store, stats


def test_index_and_find(repo):
    root = repo(FILES)
    cfg, embedder, store, stats = index(root)
    assert stats.indexed == 3
    assert stats.chunks >= 3

    hits = search(store, embedder, "login password check", cfg, limit=5)
    assert hits[0].row.rel == "api/session.py"
    store.close()


def test_incremental_reuses_and_updates(repo):
    root = repo(FILES)
    cfg, embedder, store, first = index(root)
    store.close()

    _, _, store, second = index(root)
    assert second.indexed == 0
    assert second.unchanged == 3
    store.close()

    (root / "api" / "session.py").write_text("def unrelated():\n    return 1\n", encoding="utf-8")
    (root / "ui" / "chart.js").unlink()
    cfg, embedder, store, third = index(root)
    assert third.indexed == 1
    assert third.removed == 1
    assert not any(h.row.rel == "ui/chart.js" for h in search(store, embedder, "canvas", cfg))
    store.close()


def test_embedding_cache_is_reused_after_reindex(repo):
    root = repo(FILES)
    _, _, store, _ = index(root)
    store.close()
    _, _, store, again = index(root, reindex=True)
    assert again.embedded == 0
    assert again.reused > 0
    store.close()


def test_partial_edit_only_embeds_the_changed_chunk(repo):
    source = (
        "def alpha():\n    return 1\n\n\n"
        "def beta():\n    return 2\n\n\n"
        "def gamma():\n    return 3\n"
    )
    root = repo({"mod.py": source})
    _, _, store, _ = index(root)
    store.close()

    (root / "mod.py").write_text(source.replace("return 2", "return 22"), encoding="utf-8")
    _, _, store, second = index(root)
    assert second.indexed == 1
    assert second.embedded == 1
    assert second.reused == 2
    store.close()


def test_reindex_after_switching_models(repo):
    root = repo(FILES)
    _, _, store, _ = index(root)
    store.close()

    cfg = load_config(root, provider="hash", model="hash-256")
    embedder = get_embedder(cfg.provider, cfg.model, cfg.batch_size)
    store = Store(cfg.db_path)
    stats = build_index(cfg, embedder, store, reindex=True)
    assert stats.indexed == 3
    assert store.get_meta("signature") == embedder.signature
    # Vectors of the previous model must not be handed to the new one.
    assert stats.reused == 0
    assert stats.embedded == stats.chunks
    assert int(store.get_meta("dim")) == embedder.dim

    stored = next(iter(store.vectors_for([r["id"] for r in store.db.execute(
        "SELECT id FROM chunks LIMIT 1")]).values()))
    assert stored.shape == (embedder.dim,)
    assert search(store, embedder, "login password", cfg)[0].row.rel == "api/session.py"
    store.close()


def test_filters(repo):
    root = repo(FILES)
    cfg, embedder, store, _ = index(root)

    only_js = search(store, embedder, "canvas points", cfg, filters=Filters(langs=["javascript"]))
    assert only_js and all(h.row.lang == "javascript" for h in only_js)

    by_path = search(store, embedder, "login", cfg, filters=Filters(paths=["api/"]))
    assert by_path and all(h.row.rel.startswith("api/") for h in by_path)

    assert search(store, embedder, "login", cfg, filters=Filters(langs=["cobol"])) == []
    store.close()


def test_modes(repo):
    root = repo(FILES)
    cfg, embedder, store, _ = index(root)
    for mode in ("hybrid", "semantic", "lexical"):
        hits = search(store, embedder, "password", cfg, mode=mode)
        assert hits, mode
        assert hits[0].row.rel == "api/session.py"
    store.close()


def test_model_switch_is_rejected(repo):
    root = repo(FILES)
    _, _, store, _ = index(root)
    store.close()

    cfg = load_config(root, provider="hash", model="hash-256")
    other = get_embedder(cfg.provider, cfg.model, cfg.batch_size)
    store = Store(cfg.db_path)
    with pytest.raises(SystemExit):
        store.prepare(other.signature, other.dim)
    store.close()
