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


@pytest.mark.parametrize("vec_backend", [True, False])
def test_a_query_with_no_words_returns_nothing_instead_of_crashing(repo, monkeypatch, vec_backend):
    """Both backends: sqlite-vec answers NULL, a numpy scan answers every row with 0.0."""
    import findyourcode.store as store_module

    if not vec_backend:
        monkeypatch.setattr(store_module, "_load_sqlite_vec", lambda db: False)
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

    # The short-query blend is swept the same way as everything else, or it never
    # gets swept: both knobs have to survive the string round-trip from the env.
    monkeypatch.setenv("FYC_SHORT_QUERY_ALPHA", "0.35")
    monkeypatch.setenv("FYC_SHORT_QUERY_WORDS", "4")
    swept = load_config(tmp_path)
    assert swept.short_query_alpha == pytest.approx(0.35)
    assert swept.short_query_words == 4
    monkeypatch.setenv("FYC_SHORT_QUERY_ALPHA", "-1")
    assert load_config(tmp_path).short_query_alpha == pytest.approx(-1.0)


def test_index_paths_are_relative_to_the_root(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg.index_dir == tmp_path / ".findyourcode"
    assert cfg.db_path == tmp_path / ".findyourcode" / "index.db"


def test_reranker_reorders_the_shortlist_and_rescales(monkeypatch, repo):
    """A cross-encoder sees query and chunk together; here a stub stands in for the model."""
    import sys
    import types

    class FakeCrossEncoder:
        def __init__(self, model):
            self.model = model

        def rerank(self, query, passages):
            # last passage wins, so the reranker must be able to invert fusion's order
            return [float(i) for i, _ in enumerate(passages)]

    monkeypatch.setitem(
        sys.modules,
        "fastembed.rerank.cross_encoder",
        types.SimpleNamespace(TextCrossEncoder=FakeCrossEncoder),
    )
    from findyourcode.rerank import Reranker

    root = repo(
        {
            "a.py": "def alpha_handler(request):\n    return 1\n",
            "b.py": "def beta_handler(request):\n    return 2\n",
            "c.py": "def gamma_handler(request):\n    return 3\n",
        }
    )
    cfg = load_config(root, provider="hash")
    embedder = get_embedder("hash")
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    plain = search(store, embedder, "handler request", cfg, limit=3)
    reranked = search(store, embedder, "handler request", cfg, limit=3, reranker=Reranker("stub"))

    assert [h.row.rel for h in reranked] == [h.row.rel for h in reversed(plain)]
    assert reranked[0].score == pytest.approx(1.0)
    assert reranked[-1].score == pytest.approx(0.0)

    single = search(store, embedder, "handler request", cfg, limit=1, reranker=Reranker("stub"))
    assert len(single) == 1  # nothing to reorder, and no crash
    store.close()


def test_short_queries_get_their_own_blend_and_long_ones_do_not():
    """The whole promise of the short-query blend: it fires on `checksum` and on
    nothing else. A threshold that leaked into long queries would silently reweight
    every sentence the tool has ever been measured on."""
    from findyourcode.search import blend_alpha, is_short

    cfg = Config(alpha=0.75, short_query_words=3, short_query_alpha=0.55)

    assert is_short("checksum", cfg) and blend_alpha("checksum", cfg) == 0.55
    assert is_short("leap year", cfg) and blend_alpha("leap year", cfg) == 0.55
    # A long identifier is still one word — length in characters is not the point.
    assert blend_alpha("deserialize_untrusted_payload", cfg) == 0.55

    assert not is_short("parsing command line arguments", cfg)
    assert blend_alpha("parsing command line arguments", cfg) == 0.75
    assert blend_alpha("three whole words", cfg) == 0.75  # the threshold is exclusive
    assert blend_alpha("", cfg) == 0.75  # no words is not a short query

    off_by_threshold = Config(alpha=0.75, short_query_words=0, short_query_alpha=0.55)
    assert blend_alpha("checksum", off_by_threshold) == 0.75
    off_by_alpha = Config(alpha=0.75, short_query_words=3, short_query_alpha=-1.0)
    assert blend_alpha("checksum", off_by_alpha) == 0.75


def test_a_one_word_query_leans_on_the_exact_match(repo, monkeypatch):
    """`epoll` names one file and means nothing to a sentence model: on the real
    stdlib index the vector branch answered `colorsys.py` and bm25 had `selectors.py`
    all along. The disagreement is forced here so the test does not depend on which
    way a model happens to lean — what it pins down is who wins when they disagree."""
    root = repo(
        {
            "selectors.py": "def register(self, fileobj):\n    self._epoll.register(fileobj)\n",
            "colorsys.py": "def rgb_to_hls(red, green, blue):\n    return red, green, blue\n",
        }
    )
    cfg = load_config(root, provider="hash")
    embedder = get_embedder("hash")
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    ids = {row.rel: cid for cid, row in store.rows([1, 2]).items()}
    lexical = store.search_lexical("epoll", 10, Filters())
    assert [cid for cid, _ in lexical] == [ids["selectors.py"]], "the premise: bm25 knows"

    # The vector branch is confidently wrong, exactly as it was on `epoll` for real.
    monkeypatch.setattr(
        store,
        "search_vector",
        lambda vector, k, filters: [(ids["colorsys.py"], 0.9), (ids["selectors.py"], 0.1)],
    )

    cfg.short_query_alpha = -1.0  # one blend for every query, the way it used to be
    before = search(store, embedder, "epoll", cfg, limit=2)
    cfg.short_query_alpha = 0.2
    after = search(store, embedder, "epoll", cfg, limit=2)

    assert before[0].row.rel == "colorsys.py"
    assert after[0].row.rel == "selectors.py"
    assert after[0].lexical is not None
    store.close()


def test_the_short_query_blend_leaves_long_queries_byte_identical(repo):
    """Measured on the stdlib index: 46 long queries, not one changed row or score.
    Here is that check in miniature, so a future threshold cannot quietly reweight
    the sentences the project reports its numbers on."""
    root = repo(
        {
            "auth/login.py": "def verify_password(login, password):\n    return compare(login)\n",
            "auth/session.py": "def issue_token(user):\n    return sign(user)\n",
            "ui/plot.py": "def draw_axis(canvas):\n    return canvas\n",
        }
    )
    cfg = load_config(root, provider="hash")
    embedder = get_embedder("hash")
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    def page(query):
        return [(h.row.rel, h.row.start_line, h.score) for h in search(store, embedder, query, cfg)]

    long_queries = [
        "where do we check the user password",
        "issue a signed token for the session",
        "drawing an axis on the canvas",
    ]
    cfg.short_query_alpha = -1.0
    before = {q: page(q) for q in long_queries}
    cfg.short_query_alpha = 0.1  # an extreme the blend would be very visible at
    after = {q: page(q) for q in long_queries}

    assert before == after
    assert page("password") != []  # and the short query still answers
    store.close()


def test_a_sweep_asked_for_one_alpha_does_not_claim_a_default_row(repo, capsys):
    """`--alpha` turns the short-query blend off for the whole run, so there is no
    row left that means "everything as shipped" — printing one would be a lie."""
    from findyourcode import cli

    root = repo({"auth/login.py": "def verify_password(login):\n    return login\n"})
    cli.main(["-C", str(root), "--provider", "hash", "index", "-q"])
    cases = root / "cases.json"
    cases.write_text('[{"query": "verify password", "expect": "auth/login.py"}]', encoding="utf-8")
    capsys.readouterr()

    assert cli.main(["-C", str(root), "eval", str(cases), "--sweep"]) == 0
    assert "default" in capsys.readouterr().out
    assert cli.main(["-C", str(root), "eval", str(cases), "--sweep", "--alpha", "0.9"]) == 0
    assert "default" not in capsys.readouterr().out
