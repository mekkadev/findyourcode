import io
import json
import urllib.error

import numpy as np
import pytest

from findyourcode.config import load_config
from findyourcode.diagnose import render, run
from findyourcode.embeddings import get_embedder
from findyourcode.embeddings.hashing import HashEmbedder
from findyourcode.embeddings.remote import OpenAIEmbedder, VoyageEmbedder
from findyourcode.format import as_json, as_paths
from findyourcode.format import render as render_hits
from findyourcode.indexer import build_index
from findyourcode.search import Hit
from findyourcode.store import Row, Store


def fake_endpoint(monkeypatch, dim=4, capture=None, fail_times=0, error=None):
    calls = {"n": 0}

    def urlopen(request, timeout=0):
        calls["n"] += 1
        if error is not None and calls["n"] <= fail_times:
            raise error
        body = json.loads(request.data)
        if capture is not None:
            capture.append(body)
        vectors = [[float(i + 1)] * dim for i, _ in enumerate(body["input"])]
        payload = json.dumps({"data": [{"embedding": v} for v in vectors]}).encode()
        return _Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    return calls


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_remote_provider_probes_its_dimension(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    fake_endpoint(monkeypatch, dim=6)

    embedder = VoyageEmbedder()
    assert embedder.dim == 6  # known only after a request, and never 0
    assert embedder.signature == "voyage:voyage-code-3:6"


def test_remote_provider_marks_queries_and_documents(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    seen = []
    fake_endpoint(monkeypatch, capture=seen)

    embedder = VoyageEmbedder()
    embedder.embed_documents(["one", "two"])
    embedder.embed_query("three")

    assert seen[-2]["input_type"] == "document"
    assert seen[-1]["input_type"] == "query"


def test_remote_vectors_are_normalised(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_endpoint(monkeypatch, dim=3)

    matrix = OpenAIEmbedder().embed_documents(["a", "b"])
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_remote_retries_then_gives_a_readable_error(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transient = urllib.error.HTTPError("u", 503, "busy", {}, None)
    calls = fake_endpoint(monkeypatch, fail_times=2, error=transient)

    assert VoyageEmbedder().embed_documents(["x"]).shape[0] == 1
    assert calls["n"] == 3  # two failures, then success

    permanent = urllib.error.HTTPError("u", 401, "nope", {}, io.BytesIO(b"bad key"))
    fake_endpoint(monkeypatch, error=permanent, fail_times=99)
    with pytest.raises(RuntimeError, match="401"):
        VoyageEmbedder().embed_documents(["x"])


def test_remote_provider_without_a_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        VoyageEmbedder()


def test_unknown_provider_is_rejected():
    with pytest.raises(SystemExit):
        get_embedder("telepathy")


def test_hash_embedder_is_deterministic_and_sized():
    first, second = HashEmbedder(), HashEmbedder("hash-128")
    assert first.dim == 384 and second.dim == 128
    assert np.allclose(first.embed_query("same text"), first.embed_query("same text"))
    assert first.embed_documents([]).shape == (0, 384)


def test_doctor_reports_environment_and_index(repo):
    root = repo({"a.py": "def alpha():\n    return 1\n"})
    cfg = load_config(root, provider="hash")

    missing = run(cfg)
    assert any(c.name == "index" and not c.ok for c in missing)
    assert "run `fyc index`" in render(missing)

    store = Store(cfg.db_path)
    build_index(cfg, get_embedder("hash"), store)
    store.close()

    checks = {c.name: c for c in run(cfg)}
    assert checks["index"].ok and "1 files" in checks["index"].detail
    assert checks["model"].detail.startswith("hash:")
    assert checks["freshness"].ok

    (root / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    assert not {c.name: c for c in run(cfg)}["freshness"].ok


def test_result_rendering():
    row = Row(
        1,
        "api/session.py",
        "python",
        "method",
        "check",
        "Checker",
        9,
        11,
        "def check():\n    x = 1\n    return x",
    )
    hits = [Hit(row=row, score=0.5, semantic=0.4, semantic_rank=1, lexical=-2.0, lexical_rank=3)]

    plain = render_hits(hits, snippet_lines=2, explain=True, color=False)
    assert "api/session.py:9-11" in plain
    assert "Checker.check" in plain
    assert "semantic #1" in plain and "lexical #3" in plain
    assert "1 more lines" in plain

    coloured = render_hits(hits, color=True)
    assert "\033[" in coloured
    assert render_hits([], color=False) == "nothing found"

    assert as_paths(hits) == "api/session.py:9"
    assert as_paths(hits, with_line=False) == "api/session.py"
    assert json.loads(as_json(hits))[0]["symbol"] == "Checker.check"


def test_local_provider_prefixes_and_dimensions(monkeypatch):
    """The e5 family needs asymmetric prefixes; getting that wrong is silent quality loss."""
    import sys
    import types

    seen = {"documents": [], "queries": []}

    class FakeTextEmbedding:
        def __init__(self, model):
            self.model = model

        @staticmethod
        def list_supported_models():
            return [
                {
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "dim": 384,
                },
                {"model": "intfloat/multilingual-e5-large", "dim": 1024},
            ]

        def embed(self, texts, batch_size=64):
            texts = list(texts)
            seen["documents"].extend(texts)
            return iter([np.full(_dim_of(self.model), float(i + 1)) for i in range(len(texts))])

        def query_embed(self, text):
            seen["queries"].append(text)
            return iter([np.full(_dim_of(self.model), 2.0)])

    def _dim_of(model):
        return next(
            m["dim"] for m in FakeTextEmbedding.list_supported_models() if m["model"] == model
        )

    monkeypatch.setitem(
        sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
    )
    from findyourcode.embeddings.local import DEFAULT_MODEL, LocalEmbedder

    plain = LocalEmbedder()
    assert plain.model == DEFAULT_MODEL and plain.dim == 384
    matrix = plain.embed_documents(["one", "two"])
    assert matrix.shape == (2, 384)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    plain.embed_query("a question")
    assert seen["documents"] == ["one", "two"]
    assert seen["queries"] == ["a question"]

    e5 = LocalEmbedder("intfloat/multilingual-e5-large")
    assert e5.dim == 1024
    e5.embed_documents(["body"])
    e5.embed_query("question")
    assert seen["documents"][-1] == "passage: body"
    assert seen["queries"][-1] == "query: question"
    assert plain.embed_documents([]).shape == (0, 384)
