"""The one test that uses the real embedding model.

Everything else runs on the deterministic `hash` provider so the suite stays offline
and fast — but that leaves the path an actual user takes untested. This downloads
220mb on first run, so it is opt-in:

    FYC_TEST_REAL_MODEL=1 pytest tests/test_real_model.py
"""

import os

import pytest

from findyourcode.config import load_config
from findyourcode.embeddings import get_embedder
from findyourcode.indexer import build_index
from findyourcode.search import search
from findyourcode.store import Store

pytestmark = pytest.mark.skipif(
    not os.environ.get("FYC_TEST_REAL_MODEL"),
    reason="set FYC_TEST_REAL_MODEL=1 to run against the real model (downloads ~220mb)",
)

FILES = {
    "api/session.py": (
        "class CredentialChecker:\n"
        '    """Validates the login/password pair a client presents at sign-in."""\n\n'
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
    "billing/charge.go": (
        "package billing\n\n"
        "// Collect pulls money for an invoice through the card processor.\n"
        "func Collect(inv Invoice, card Card) (Receipt, error) {\n"
        "\treturn processor.Capture(card.Token, inv.AmountEU)\n"
        "}\n"
    ),
}


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("real")
    for rel, body in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    cfg = load_config(root, provider="local")
    embedder = get_embedder(cfg.provider, cfg.model, cfg.batch_size)
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)
    yield cfg, embedder, store
    store.close()


@pytest.mark.parametrize(
    "query,expected",
    [
        ("where do we authenticate users", "api/session.py"),
        ("drawing a graph on a canvas", "ui/chart.js"),
        ("taking money from a bank card", "billing/charge.go"),
        # the point of a multilingual model: ask in another language, find english code
        ("где происходит проверка пароля", "api/session.py"),
    ],
)
def test_meaning_beats_wording(indexed, query, expected):
    cfg, embedder, store = indexed
    hits = search(store, embedder, query, cfg, limit=3)
    assert hits, query
    assert hits[0].row.rel == expected, [h.row.rel for h in hits]


def test_the_model_is_the_one_we_claim(indexed):
    _cfg, embedder, _store = indexed
    assert embedder.name == "local"
    assert embedder.dim == 384
    assert "multilingual" in embedder.model
