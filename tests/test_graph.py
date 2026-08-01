import json

from findyourcode import cli
from findyourcode.config import Config, load_config
from findyourcode.embeddings import get_embedder
from findyourcode.indexer import build_index
from findyourcode.search import Hit, _propagate, build_trace, search
from findyourcode.store import Filters, Store

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


def test_a_common_name_does_not_crowd_out_the_real_callers(repo, monkeypatch):
    """A chunk defining both `get` and something rare must still find the rare one's
    callers: `get` resolves nowhere, but it would eat the row budget first."""
    from findyourcode import store as store_module

    files = {f"noise/mod{i}.py": "def get(x):\n    return x\n" for i in range(12)}
    files.update({f"noise/use{i}.py": "def use(x):\n    return get(x)\n" for i in range(6)})
    files["svc/api.py"] = (
        "class Api:\n"
        "    def get(self, x):\n"
        "        return x\n"
        "\n"
        "    def rotate_secret(self, x):\n"
        "        return x\n"
    )
    files["svc/boot.py"] = "def start(api, x):\n    return api.rotate_secret(x)\n"
    _cfg, _embedder, store = _index(repo(files))

    monkeypatch.setattr(store_module, "MAX_EDGES", 3)
    callers = {e.src for e in store.edges_to([_id(store, "svc/api.py")])}
    assert _id(store, "svc/boot.py") in callers
    store.close()


def test_a_filter_does_not_cost_the_graph_its_slots(repo):
    """The slots must go to neighbours that survive the filter, not be spent on six
    that are about to be discarded — the excluded ones are listed first on purpose."""
    body = "".join(f"    far{i}(x)\n" for i in range(6)) + "".join(
        f"    near{i}(x)\n" for i in range(2)
    )
    files = {"hub.py": f'def dispatch(x):\n    """the login form entry point."""\n{body}'}
    files.update({f"lib/near{i}.py": f"def near{i}(x):\n    return x\n" for i in range(2)})
    files.update({f"other/far{i}.py": f"def far{i}(x):\n    return x\n" for i in range(6)})
    cfg, embedder, store = _index(repo(files))

    from findyourcode.store import Filters

    hits = search(
        store,
        embedder,
        "login form entry point",
        cfg,
        limit=20,
        mode="lexical",
        filters=Filters(paths=["hub.py", "lib/"]),
    )
    store.close()
    assert {h.row.rel for h in hits if h.graph is not None} == {"lib/near0.py", "lib/near1.py"}


def test_a_reranker_cannot_lift_a_graph_hit_to_the_top(repo):
    class Inverter:
        """The contract a reranker has: rewrite every score, ceiling included."""

        def rescore(self, query, hits):
            ordered = list(reversed(hits))
            for i, hit in enumerate(ordered):
                hit.score = 1.0 - 0.1 * i
            return ordered

    root = repo(FILES)
    cfg, embedder, store = _index(root)
    hits = search(store, embedder, QUERY, cfg, mode="lexical", reranker=Inverter())
    store.close()

    assert hits[0].row.rel == "web/routes.py"
    assert all(h.score < hits[0].score for h in hits if h.graph is not None)


def test_trace_depth_zero_means_no_hops(repo):
    files = {f"step{i}.py": f"def step{i}(x):\n    return step{i + 1}(x)\n" for i in range(3)}
    cfg, _embedder, store = _index(repo(files))
    cfg.trace_depth, cfg.trace_callers = 0, 0
    trace = build_trace(store, store.chunk_at("step0.py"), cfg)
    store.close()

    assert trace.children == []


def test_two_branches_may_end_at_the_same_helper(repo):
    files = {
        "hub.py": "def dispatch(x):\n    alpha_one(x)\n    beta_two(x)\n",
        "a.py": "def alpha_one(x):\n    return shared_helper(x)\n",
        "b.py": "def beta_two(x):\n    return shared_helper(x)\n",
        "h.py": "def shared_helper(x):\n    return x\n",
    }
    cfg, _embedder, store = _index(repo(files))
    cfg.trace_callers = 0
    trace = build_trace(store, store.chunk_at("hub.py"), cfg)
    store.close()

    reached = [child.children[0].row.rel for child in trace.children if child.children]
    assert reached == ["h.py", "h.py"]


def test_a_chunk_without_a_vector_still_shows_up_in_a_trace(repo):
    root = repo(FILES)
    cfg, embedder, store = _index(root)
    signing = _id(store, "crypto/signing.py")
    table = "vec_chunks" if store.vector_backend == "sqlite-vec" else "vectors"
    column = "rowid" if store.vector_backend == "sqlite-vec" else "chunk_id"
    store.db.execute(f"DELETE FROM {table} WHERE {column} = ?", (signing,))
    store.commit()

    cfg.trace_callers = 0
    trace = build_trace(store, store.chunk_at("auth/session.py"), cfg, embedder.embed_query("sign"))
    store.close()
    assert [child.row.rel for child in trace.children] == ["crypto/signing.py"]


def test_another_language_cannot_bury_a_name(repo):
    """`handler` five times in python and five in javascript is two ordinary names,
    not one hopeless one — the halves were never going to be joined anyway."""
    files = {f"py{i}.py": "def handler(x):\n    return x\n" for i in range(5)}
    files["caller.py"] = "def go(x):\n    return handler(x)\n"
    files.update({f"js{i}.js": "function handler(x) {\n  return x;\n}\n" for i in range(5)})
    _cfg, _embedder, store = _index(repo(files))

    edges = store.edges_from([_id(store, "caller.py")])
    assert len(edges) == 5
    assert all(store.rows([e.dst])[e.dst].lang == "python" for e in edges)
    store.close()


def test_a_grammar_that_calls_its_definitions_still_records_the_calls(repo):
    """In elixir `def deliver do` is a call node. Reading every call as a definition
    left every module claiming to define what it merely called, and no refs at all."""
    files = {
        "lib/worker.ex": "defmodule Worker do\n  def run(x) do\n    deliver(x)\n  end\nend\n",
    }
    root = repo(files)
    cfg = load_config(root, provider="hash")
    embedder = get_embedder(cfg.provider)
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    refs = {r["name"] for r in store.db.execute("SELECT name FROM refs")}
    defs = {r["name"] for r in store.db.execute("SELECT name FROM defs")}
    store.close()
    assert "deliver" in refs
    assert "deliver" not in defs


def test_the_qualifier_survives_a_grammar_that_splits_the_receiver(repo):
    """java, ruby and php hang the receiver on the call node, not on the callee."""
    files = {
        "Linecache.java": "class Linecache {\n  static String getline(String p) { return p; }\n}\n",
        "Ftplib.java": "class Ftplib {\n  static String getline(String s) { return s; }\n}\n",
        "Traceback.java": (
            "class Traceback {\n  String frame(String p) { return Linecache.getline(p); }\n}\n"
        ),
    }
    _cfg, _embedder, store = _index(repo(files))
    edges = store.edges_from([_id(store, "Traceback.java")])

    assert [(e.name, e.weight) for e in edges] == [("getline", 1.0)]
    assert store.rows([edges[0].dst])[edges[0].dst].rel == "Linecache.java"
    store.close()


def test_a_dropped_defs_table_is_refilled_like_a_dropped_refs_table(repo):
    root = repo(FILES)
    cfg, embedder, store = _index(root)
    store.db.executescript("DROP TABLE defs;")
    store.commit()
    store.close()

    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)
    assert store.edges_from([_id(store, "web/routes.py")])
    store.close()


def test_one_popular_name_does_not_spend_the_whole_row_budget(repo, monkeypatch):
    from findyourcode import store as store_module

    files = {
        "hub.py": (
            "class Hub:\n"
            "    def append(self, x):\n"
            "        return x\n"
            "\n"
            "    def zrare(self, x):\n"
            "        return x\n"
        )
    }
    files.update({f"crowd{i}.py": "def use(h, x):\n    return h.append(x)\n" for i in range(20)})
    files["rare.py"] = "def only(h, x):\n    return h.zrare(x)\n"
    _cfg, _embedder, store = _index(repo(files))

    monkeypatch.setattr(store_module, "MAX_EDGES", 4)
    callers = {e.src for e in store.edges_to([_id(store, "hub.py")])}
    assert _id(store, "rare.py") in callers
    store.close()


def test_a_neighbour_the_query_points_nowhere_near_is_dropped(repo):
    """The reach window: a call edge says which nearly-relevant chunk to surface, never
    that an irrelevant one is relevant."""
    _cfg, _embedder, store = _index(repo(FILES))
    login = _id(store, "web/routes.py")
    ticket = _id(store, "auth/session.py")
    rows = store.rows([login])
    cfg = Config(graph_weight=0.85)

    def propagate(near):
        hits = {login: Hit(row=rows[login], score=1.0)}
        _propagate(store, hits, cfg, Filters(), near)
        return set(hits) - {login}

    assert propagate(None) == {ticket}  # no window: every neighbour, as before
    assert propagate({login, ticket}) == {ticket}
    assert propagate({login}) == set()  # the query ranks the callee nowhere
    store.close()


def test_the_reach_window_widens_one_query_rather_than_adding_a_second(repo):
    """`graph_reach` reads further down the ranking the retriever was already producing,
    and only when the graph is on — the head of that list is unchanged either way."""
    files = {f"m{i}.py": f"def alpha_{i}(x):\n    return alpha_{i + 1}(x)\n" for i in range(30)}
    cfg = load_config(repo(files), provider="hash")
    embedder = get_embedder("hash")
    store = Store(cfg.db_path)
    build_index(cfg, embedder, store)

    asked: list[int] = []
    deep = store.search_vector

    def spy(query, k, filters):
        asked.append(k)
        return deep(query, k, filters)

    store.search_vector = spy  # type: ignore[method-assign]

    def depths(**kw):
        asked.clear()
        search(store, embedder, "alpha", cfg, limit=10, **kw)
        return list(asked)

    cfg.graph_reach = 5
    assert depths() == [400]  # one query, five times the retrieval depth of 80
    assert depths(graph=False) == [80]
    cfg.graph_reach = 1
    assert depths() == [80]
    store.close()


def test_without_a_reach_window_the_graph_stays_quiet(repo):
    """`--mode lexical` embeds nothing, so nothing can say whether the query points at
    a neighbour at all. The edge still counts — it just does not get to shout."""
    from findyourcode.search import UNGATED_WEIGHT

    root = repo(FILES)
    cfg, embedder, store = _index(root)
    cfg.graph_weight = 0.95

    hits = {h.row.rel: h for h in search(store, embedder, QUERY, cfg, mode="lexical")}
    store.close()
    assert hits["auth/session.py"].graph is not None
    assert hits["auth/session.py"].graph <= UNGATED_WEIGHT


def test_a_large_page_does_not_silently_mute_the_graph(repo, monkeypatch):
    """`-n 600` asks the vector index for 4800 and it will not answer past 4096, so
    the window would have been no wider than the page — and every candidate outside
    a page that wide is, by definition, already in it."""
    from findyourcode.store import VEC_MAX_K

    root = repo(FILES)
    cfg, embedder, store = _index(root)
    asked = []
    real = store.search_vector
    monkeypatch.setattr(store, "search_vector", lambda v, k, f: (asked.append(k), real(v, k, f))[1])

    hits = search(store, embedder, QUERY, cfg, limit=600, mode="lexical")
    store.close()
    assert all(k <= VEC_MAX_K for k in asked)
    assert [h.row.rel for h in hits if h.graph is not None] == ["auth/session.py"]


def test_a_corpus_smaller_than_the_window_is_not_a_gated_one(repo):
    """Every chunk being 'near the query' is not evidence about any of them, so the
    graph goes back to the weight it was measured safe at without a gate."""
    from findyourcode.search import UNGATED_WEIGHT

    files = {f"filler/mod{i}.py": f"def helper{i}(x):\n    return {i}\n" for i in range(120)}
    files["hub.py"] = 'def dispatch(x):\n    """the login form entry point."""\n    helper7(x)\n'
    root = repo(files)
    cfg, embedder, store = _index(root)
    cfg.graph_weight = 0.95

    hits = search(store, embedder, "login form entry point", cfg, mode="lexical")
    store.close()
    reached = [h for h in hits if h.graph is not None]
    assert reached and all(h.graph <= UNGATED_WEIGHT for h in reached)
