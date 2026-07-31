import json

from findyourcode import cli, store as store_module
from findyourcode.config import load_config
from findyourcode.embeddings import get_embedder
from findyourcode.indexer import build_index
from findyourcode.search import search
from findyourcode.store import Store

FILES = {
    "auth/login.py": (
        "def verify_password(login, password):\n"
        "    stored = users.digest_for(login)\n"
        "    return constant_time_compare(stored, hash_password(password))\n"
    ),
    "render/plot.py": "def draw_axis(canvas, ticks):\n    canvas.line(ticks)\n",
}


def _index(root, monkeypatch=None):
    cfg = load_config(root, provider="hash")
    embedder = get_embedder(cfg.provider)
    st = Store(cfg.db_path)
    build_index(cfg, embedder, st)
    return cfg, embedder, st


def test_numpy_backend_matches_vec_backend(repo, monkeypatch):
    root = repo(FILES)
    cfg, embedder, st = _index(root)
    baseline = [h.row.rel for h in search(st, embedder, "password check", cfg)]
    backend = st.get_meta("backend")
    st.close()

    monkeypatch.setattr(store_module, "_load_sqlite_vec", lambda db: False)
    (root / ".findyourcode").rename(root / ".findyourcode.bak")
    cfg, embedder, st = _index(root)
    assert st.get_meta("backend") == "numpy"
    fallback = [h.row.rel for h in search(st, embedder, "password check", cfg)]
    st.close()

    assert fallback == baseline
    assert backend in ("vec0", "numpy")


def test_cli_end_to_end(repo, capsys):
    root = repo(FILES)
    assert cli.main(["-C", str(root), "--provider", "hash", "index", "-q"]) == 0
    capsys.readouterr()

    assert cli.main(["-C", str(root), "find", "verify password", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["path"] == "auth/login.py"
    assert payload[0]["symbol"] == "verify_password"

    assert cli.main(["-C", str(root), "status"]) == 0
    assert "chunks" in capsys.readouterr().out

    assert cli.main(["-C", str(root), "clear"]) == 0
    assert cli.main(["-C", str(root), "find", "anything"]) == 2


def test_cli_reports_no_results(repo, capsys):
    root = repo(FILES)
    cli.main(["-C", str(root), "--provider", "hash", "index", "-q"])
    capsys.readouterr()
    assert cli.main(["-C", str(root), "find", "zzz", "--lang", "erlang"]) == 1
    assert "nothing found" in capsys.readouterr().out


def test_watch_loop_picks_up_changes_and_stops(repo, monkeypatch, capsys):
    root = repo(FILES)
    calls = {"n": 0}

    def fake_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            (root / "extra.py").write_text("def gamma():\n    return 3\n", encoding="utf-8")
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", fake_sleep)
    assert cli.main(["-C", str(root), "--provider", "hash", "index", "--watch"]) == 0

    output = capsys.readouterr().out
    assert output.count("indexed") >= 2
    assert cli.main(["-C", str(root), "find", "gamma", "-f", "files"]) == 0
    assert "extra.py" in capsys.readouterr().out


def test_piping_into_head_does_not_traceback(repo, monkeypatch, capsys):
    root = repo(FILES)
    cli.main(["-C", str(root), "--provider", "hash", "index", "-q"])
    capsys.readouterr()

    real_print = __builtins__["print"] if isinstance(__builtins__, dict) else print

    def closed_pipe(*args, **kwargs):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr("builtins.print", closed_pipe)
    try:
        assert cli.main(["-C", str(root), "status"]) == 0
    finally:
        monkeypatch.setattr("builtins.print", real_print)
