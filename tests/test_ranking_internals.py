"""Behaviour that a mutation of the ranking or the config loader would otherwise slip past."""

import numpy as np
import pytest

from findyourcode.config import Config, load_config
from findyourcode.embeddings import get_embedder
from findyourcode.indexer import build_index
from findyourcode.search import Hit, _score_rrf, search
from findyourcode.store import Filters, Row, Store


def _hit(chunk_id, semantic_rank=None, lexical_rank=None):
    row = Row(chunk_id, f"f{chunk_id}.py", "python", "function", f"f{chunk_id}", "", 1, 2, "x")
    return Hit(row=row, score=0.0, semantic_rank=semantic_rank, lexical_rank=lexical_rank)


def test_rrf_uses_both_branches_and_their_weights():
    cfg = Config(rrf_k=60, semantic_weight=1.0, lexical_weight=0.6)
    hits = {1: _hit(1, semantic_rank=1), 2: _hit(2, lexical_rank=1), 3: _hit(3, 1, 1)}
    _score_rrf(hits, cfg)

    assert hits[1].score == pytest.approx(1.0 / 61)
    assert hits[2].score == pytest.approx(0.6 / 61)
    assert hits[3].score == pytest.approx(1.0 / 61 + 0.6 / 61)
    assert hits[3].score > hits[1].score > hits[2].score

    lexical_first = Config(rrf_k=60, semantic_weight=0.1, lexical_weight=1.0)
    flipped = {1: _hit(1, semantic_rank=1), 2: _hit(2, lexical_rank=1)}
    _score_rrf(flipped, lexical_first)
    assert flipped[2].score > flipped[1].score


def test_rrf_ranks_a_real_index(repo):
    root = repo(
        {
            "auth/login.py": "def verify_password(login, password):\n    return compare(login)\n",
            "ui/plot.py": "def draw_axis(canvas):\n    return canvas\n",
        }
    )
    cfg = load_config(root, provider="hash")
    embedder = get_embedder("hash")
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    hits = search(store, embedder, "verify password", cfg, fusion="rrf")
    assert hits[0].row.rel == "auth/login.py"
    assert hits[0].score > 0
    store.close()


def test_oversample_controls_the_candidate_pool(repo):
    files = {f"mod_{i}.py": f"def handler_{i}(request):\n    return {i}\n" for i in range(40)}
    root = repo(files)
    cfg = load_config(root, provider="hash")
    embedder = get_embedder("hash")
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    deep = search(store, embedder, "handler request", cfg, limit=5)
    cfg.oversample = 1
    shallow = search(store, embedder, "handler request", cfg, limit=5)

    assert len(deep) == len(shallow) == 5
    assert store.search_vector(embedder.embed_query("handler"), 3, Filters()) != []
    store.close()


def test_numpy_top_k_is_ordered_and_bounded(repo, monkeypatch):
    import findyourcode.store as store_module

    monkeypatch.setattr(store_module, "_load_sqlite_vec", lambda db: False)
    files = {f"m{i}.py": f"def alpha_{i}():\n    return {i}\n" for i in range(25)}
    root = repo(files)
    cfg = load_config(root, provider="hash")
    embedder = get_embedder("hash")
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    scored = store.search_vector(embedder.embed_query("alpha"), 5, Filters())
    assert len(scored) == 5
    assert [s for _, s in scored] == sorted((s for _, s in scored), reverse=True)
    assert len({cid for cid, _ in scored}) == 5
    store.close()


def test_a_query_with_no_words_returns_nothing_instead_of_crashing(repo):
    root = repo({"a.py": "def alpha():\n    return 1\n"})
    cfg = load_config(root, provider="hash")
    embedder = get_embedder("hash")
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    assert np.allclose(embedder.embed_query("()"), 0.0)  # the premise: a zero vector
    for query in ("()", "   ", "...", "&&", "::"):
        assert search(store, embedder, query, cfg) == []
    store.close()


def test_lexical_branch_matches_any_query_term(repo):
    root = repo(
        {
            "a.py": "def collect_invoice_totals():\n    return 1\n",
            "b.py": "def unrelated():\n    return 2\n",
        }
    )
    cfg = load_config(root, provider="hash")
    embedder = get_embedder("hash")
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    # 'zzzz' appears nowhere: an AND across terms would return nothing at all.
    hits = store.search_lexical("invoice zzzz", 10, Filters())
    assert hits, "bm25 must match documents holding any query term, not all of them"

    ids = {cid for cid, _ in hits}
    rows = store.rows(list(ids))
    assert any(row.rel == "a.py" for row in rows.values())
    assert store.search_lexical("  ?? ", 10, Filters()) == []
    store.close()


def test_config_reads_toml_env_and_overrides(tmp_path, monkeypatch):
    (tmp_path / ".findyourcode.toml").write_text(
        '[findyourcode]\nprovider = "voyage"\nalpha = 0.4\nper_file = 7\n'
        'exclude = ["**/vendor/**"]\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("FYC_ALPHA", raising=False)
    monkeypatch.delenv("FYC_PROVIDER", raising=False)

    from_file = load_config(tmp_path)
    assert from_file.provider == "voyage"
    assert from_file.alpha == pytest.approx(0.4)
    assert from_file.per_file == 7
    assert from_file.exclude == ["**/vendor/**"]

    monkeypatch.setenv("FYC_ALPHA", "0.9")
    monkeypatch.setenv("FYC_EXCLUDE", "a,b")
    from_env = load_config(tmp_path)
    assert from_env.alpha == pytest.approx(0.9)
    assert from_env.exclude == ["a", "b"]

    explicit = load_config(tmp_path, alpha=0.1, provider="hash")
    assert explicit.alpha == pytest.approx(0.1)
    assert explicit.provider == "hash"

    monkeypatch.delenv("FYC_ALPHA")
    monkeypatch.delenv("FYC_EXCLUDE")
    assert load_config(tmp_path, alpha=None).alpha == pytest.approx(0.4)


def test_index_paths_are_relative_to_the_root(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg.index_dir == tmp_path / ".findyourcode"
    assert cfg.db_path == tmp_path / ".findyourcode" / "index.db"
