import json

from findyourcode import cli
from findyourcode.config import load_config
from findyourcode.embeddings import get_embedder
from findyourcode.indexer import build_index
from findyourcode.search import build_trace, search
from findyourcode.store import Store

FILES = {
    "web/routes.py": (
        "from auth.session import issue_ticket\n"
        "\n"
        "\n"
        "def handle_login(request):\n"
        '    """Entry point behind the sign-in form."""\n'
        "    user = request.form['user']\n"
        "    return issue_ticket(user)\n"
    ),
    "auth/session.py": (
        "def issue_ticket(user):\n    payload = user.encode()\n    return sign_payload(payload)\n"
    ),
    "crypto/signing.py": "def sign_payload(payload):\n    return payload.hex()\n",
    "ui/table.py": "def draw_table(rows):\n    for row in rows:\n        print(row)\n",
}


def _index(root):
    cfg = load_config(root, provider="hash")
    embedder = get_embedder(cfg.provider)
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)
    return cfg, embedder, store


def _id(store, location: str) -> int:
    row = store.chunk_at(location)
    assert row is not None, location
    return row.id


def test_a_call_becomes_an_edge_in_both_directions(repo):
    _cfg, _embedder, store = _index(repo(FILES))
    login = _id(store, "web/routes.py")
    ticket = _id(store, "auth/session.py")

    assert [(e.src, e.dst, e.name) for e in store.edges_from([login])] == [
        (login, ticket, "issue_ticket")
    ]
    assert [(e.src, e.dst) for e in store.edges_to([ticket])] == [(login, ticket)]
    store.close()


QUERY = "entry point behind the login form"  # every word of it lives in web/routes.py alone


def test_a_neighbour_reaches_the_page_without_matching_the_query(repo):
    """auth/session.py holds none of those words — it is one call away, and that is all."""
    root = repo(FILES)
    cfg, embedder, store = _index(root)

    found = [h.row.rel for h in search(store, embedder, QUERY, cfg, mode="lexical")]
    without = [h.row.rel for h in search(store, embedder, QUERY, cfg, mode="lexical", graph=False)]
    store.close()

    assert found == ["web/routes.py", "auth/session.py"]
    assert without == ["web/routes.py"]


def test_the_graph_never_takes_the_first_place(repo):
    root = repo(FILES)
    cfg, embedder, store = _index(root)
    cfg.graph_weight = 10.0  # however loud the structure shouts, the text answered first
    hits = search(store, embedder, QUERY, cfg, mode="lexical")
    store.close()

    assert hits[0].row.rel == "web/routes.py"
    reached = [h for h in hits if h.semantic_rank is None and h.lexical_rank is None]
    assert reached and all(h.score < hits[0].score for h in reached)
    assert all(h.via for h in reached)


def test_a_name_defined_everywhere_is_not_an_edge(repo):
    files = {f"mod{i}/run.py": "def run(job):\n    return job\n" for i in range(12)}
    files["caller.py"] = "def start(job):\n    return run(job)\n"
    _cfg, _embedder, store = _index(repo(files))

    assert store.edges_from([_id(store, "caller.py")]) == []
    store.close()


def test_an_edge_stays_inside_one_language(repo):
    files = {
        "a/handler.py": "def dispatch(event):\n    return notify(event)\n",
        "b/notify.js": "function notify(event) {\n  return event;\n}\n",
        "c/notify.py": "def notify(event):\n    return event\n",
    }
    _cfg, _embedder, store = _index(repo(files))
    targets = {e.dst for e in store.edges_from([_id(store, "a/handler.py")])}
    assert targets == {_id(store, "c/notify.py")}
    store.close()


def test_a_definition_in_the_same_file_wins(repo):
    files = {
        "local.py": "def send(msg):\n    return deliver(msg)\n\n\ndef deliver(msg):\n    return msg\n",
        "other.py": "def deliver(msg):\n    return msg\n",
    }
    _cfg, _embedder, store = _index(repo(files))
    edges = store.edges_from([_id(store, "local.py:2")])
    assert [e.weight for e in edges] == [1.0]
    assert all(e.dst == _id(store, "local.py:5") for e in edges)
    store.close()


def test_a_method_inside_a_class_is_a_call_target(repo):
    files = {
        "svc/mailer.py": ("class Mailer:\n    def deliver(self, msg):\n        return msg\n"),
        "svc/queue.py": "def flush(mailer, msg):\n    return mailer.deliver(msg)\n",
    }
    _cfg, _embedder, store = _index(repo(files))
    edges = store.edges_from([_id(store, "svc/queue.py")])
    store.close()

    assert [e.name for e in edges] == ["deliver"]


def test_a_trace_follows_the_calls_and_the_callers(repo):
    root = repo(FILES)
    cfg, embedder, store = _index(root)
    row = store.chunk_at("auth/session.py")
    trace = build_trace(store, row, cfg, embedder.embed_query("sign a ticket"))
    store.close()

    directions = {child.direction: child.row.rel for child in trace.children}
    assert directions == {"called by": "web/routes.py", "calls": "crypto/signing.py"}


def test_a_trace_stops_at_the_configured_depth(repo):
    files = {f"step{i}.py": f"def step{i}(x):\n    return step{i + 1}(x)\n" for i in range(6)}
    cfg, _embedder, store = _index(repo(files))
    cfg.trace_depth, cfg.trace_fanout, cfg.trace_callers = 2, 1, 0
    trace = build_trace(store, store.chunk_at("step0.py"), cfg)
    store.close()

    depth = 0
    node = trace
    while node.children:
        node = node.children[0]
        depth += 1
    assert depth == 2


def test_editing_a_file_replaces_its_edges(repo):
    root = repo(FILES)
    cfg, embedder, store = _index(root)
    (root / "web" / "routes.py").write_text(
        "def handle_login(request):\n    return draw_table(request)\n", encoding="utf-8"
    )
    build_index(cfg, embedder, store)

    login = _id(store, "web/routes.py")
    assert [e.name for e in store.edges_from([login])] == ["draw_table"]
    assert store.edges_to([_id(store, "auth/session.py")]) == []
    store.close()


def test_an_index_built_before_the_graph_is_refilled_without_re_embedding(repo):
    root = repo(FILES)
    cfg, embedder, store = _index(root)
    store.db.executescript("DROP TABLE defs; DROP TABLE refs;")
    store.commit()
    store.close()

    store = Store(cfg.db_path)
    stats = build_index(cfg, embedder, store)
    assert stats.embedded == 0
    assert stats.indexed == len(FILES)
    assert store.edges_from([_id(store, "web/routes.py")])
    store.close()


def test_a_filtered_search_does_not_smuggle_neighbours_in(repo):
    files = dict(FILES)
    files["auth/session.js"] = "function issueTicket(user) {\n  return user;\n}\n"
    root = repo(files)
    cfg, embedder, store = _index(root)

    from findyourcode.store import Filters

    hits = search(store, embedder, "sign-in form", cfg, limit=10, filters=Filters(paths=["web/"]))
    store.close()
    assert {h.row.rel for h in hits} == {"web/routes.py"}


def test_the_cli_prints_a_call_path(repo, capsys):
    root = repo(FILES)
    assert cli.main(["-C", str(root), "--provider", "hash", "index", "-q"]) == 0
    capsys.readouterr()

    assert cli.main(["-C", str(root), "find", "sign-in form", "-n", "1", "--trace", "-L", "0"]) == 0
    out = capsys.readouterr().out
    assert "→ auth/session.py" in out

    assert cli.main(["-C", str(root), "find", "sign-in form", "-n", "1", "--trace", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["trace"][0]["path"] == "auth/session.py"


def test_status_reports_the_graph(repo, capsys):
    root = repo(FILES)
    cli.main(["-C", str(root), "--provider", "hash", "index", "-q"])
    capsys.readouterr()
    cli.main(["-C", str(root), "status"])
    assert "call sites" in capsys.readouterr().out


def test_the_qualifier_decides_which_definition_is_meant(repo):
    """`linecache.getline` names its module; without that the edge is a coin toss."""
    files = {
        "linecache.py": "def getline(path):\n    return path\n",
        "ftplib.py": "def getline(sock):\n    return sock\n",
        "traceback.py": "import linecache\n\n\ndef frame(path):\n    return linecache.getline(path)\n",
    }
    _cfg, _embedder, store = _index(repo(files))
    edges = store.edges_from([_id(store, "traceback.py")])

    assert [(e.dst, e.weight) for e in edges] == [(_id(store, "linecache.py"), 1.0)]
    store.close()


def test_an_unqualified_ambiguous_call_keeps_every_candidate_at_a_discount(repo):
    files = {
        "linecache.py": "def getline(path):\n    return path\n",
        "ftplib.py": "def getline(sock):\n    return sock\n",
        "caller.py": "def frame(path):\n    return getline(path)\n",
    }
    _cfg, _embedder, store = _index(repo(files))
    edges = store.edges_from([_id(store, "caller.py")])

    assert sorted(e.weight for e in edges) == [0.5, 0.5]
    store.close()


def test_the_graph_may_not_flood_the_page(repo):
    callees = {f"lib/step{i}.py": f"def step{i}(x):\n    return x\n" for i in range(9)}
    body = "".join(f"    step{i}(x)\n" for i in range(9))
    files = {
        **callees,
        "hub.py": f'def dispatch(x):\n    """the login form entry point."""\n{body}',
    }
    root = repo(files)
    cfg, embedder, store = _index(root)
    cfg.graph_limit = 3

    hits = search(store, embedder, "login form entry point", cfg, limit=20, mode="lexical")
    store.close()
    assert len([h for h in hits if h.graph is not None]) == 3


def test_eval_can_be_run_without_the_graph(repo, capsys, tmp_path):
    root = repo(FILES)
    cli.main(["-C", str(root), "--provider", "hash", "index", "-q"])
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([{"query": "entry point behind the login form", "expect": "web/routes.py"}]),
        encoding="utf-8",
    )
    capsys.readouterr()

    assert cli.main(["-C", str(root), "eval", str(cases), "--no-graph"]) == 0
    assert "recall@1 1.00" in capsys.readouterr().out
