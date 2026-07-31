import io
import json

from findyourcode.config import load_config
from findyourcode.embeddings import get_embedder
from findyourcode.indexer import build_index
from findyourcode.mcp_server import PROTOCOL_VERSION, Server
from findyourcode.store import Store

FILES = {
    "auth/login.py": (
        "def verify_password(login, password):\n"
        "    stored = users.digest_for(login)\n"
        "    return constant_time_compare(stored, hash_password(password))\n"
    ),
    "ui/plot.py": "def draw_axis(canvas, ticks):\n    canvas.line(ticks)\n",
}


def drive(root, *messages):
    cfg = load_config(root, provider="hash")
    stdin = io.StringIO("".join(json.dumps(m) + "\n" for m in messages))
    stdout = io.StringIO()
    Server(cfg, stdin, stdout).serve()
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def indexed(repo):
    root = repo(FILES)
    cfg = load_config(root, provider="hash")
    store = Store(cfg.db_path)
    build_index(cfg, get_embedder(cfg.provider), store)
    store.close()
    return root


def call(name, **arguments):
    return {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def test_handshake_and_tool_list(repo):
    root = indexed(repo)
    replies = drive(
        root,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )

    assert [r["id"] for r in replies] == [1, 2]  # the notification gets no reply
    assert replies[0]["result"]["protocolVersion"] == "2024-11-05"
    assert replies[0]["result"]["serverInfo"]["name"] == "findyourcode"
    assert {t["name"] for t in replies[1]["result"]["tools"]} == {
        "search_code",
        "find_similar",
        "index_status",
    }
    for tool in replies[1]["result"]["tools"]:
        assert tool["inputSchema"]["type"] == "object"


def test_unknown_protocol_falls_back_to_ours(repo):
    replies = drive(
        indexed(repo),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "1999-01-01"},
        },
    )
    assert replies[0]["result"]["protocolVersion"] == PROTOCOL_VERSION


def test_search_returns_text_and_structured_results(repo):
    replies = drive(indexed(repo), call("search_code", query="verify password", limit=3))
    result = replies[0]["result"]

    assert result["isError"] is False
    assert "auth/login.py" in result["content"][0]["text"]
    top = result["structuredContent"]["results"][0]
    assert top["path"] == "auth/login.py"
    assert top["symbol"] == "verify_password"
    assert top["start_line"] >= 1 and "score" in top


def test_search_filters_and_limit_clamp(repo):
    root = indexed(repo)
    replies = drive(
        root,
        call("search_code", query="draw", lang=["python"], path=["ui/"]),
        call("search_code", query="draw", limit=10_000),
        call("search_code", query="draw", lang="python"),
    )
    assert all(r["result"]["isError"] is False for r in replies)
    assert all(
        hit["path"].startswith("ui/")
        for hit in replies[0]["result"]["structuredContent"]["results"]
    )
    assert len(replies[1]["result"]["structuredContent"]["results"]) <= 50


def test_find_similar_and_status(repo):
    root = indexed(repo)
    replies = drive(
        root,
        call("find_similar", location="auth/login.py:2", limit=2),
        call("index_status"),
    )

    similar = replies[0]["result"]
    assert similar["content"][0]["text"].startswith("like auth/login.py:")
    assert similar["structuredContent"]["anchor"]["path"] == "auth/login.py"

    status = replies[1]["result"]["structuredContent"]
    assert status["files"] == 2 and status["chunks"] >= 2


def test_errors_are_reported_not_raised(repo):
    root = indexed(repo)
    replies = drive(
        root,
        call("search_code", query="   "),
        call("find_similar", location="nowhere.py:1"),
        call("nope"),
        {"jsonrpc": "2.0", "id": 5, "method": "no/such/method"},
    )

    assert [r["result"]["isError"] for r in replies[:3]] == [True, True, True]
    assert replies[3]["error"]["code"] == -32601


def test_missing_index_is_a_message_not_a_crash(tmp_path):
    replies = drive(tmp_path, call("search_code", query="anything"))
    assert replies[0]["result"]["isError"] is True
    assert "fyc index" in replies[0]["result"]["content"][0]["text"]


def test_malformed_json_line(repo):
    root = indexed(repo)
    cfg = load_config(root, provider="hash")
    stdout = io.StringIO()
    Server(cfg, io.StringIO("not json\n\n"), stdout).serve()
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert replies[0]["error"]["code"] == -32700


def test_a_non_object_message_does_not_kill_the_server(repo):
    root = indexed(repo)
    cfg = load_config(root, provider="hash")
    stdout = io.StringIO()
    stdin = io.StringIO('[]\n"bare"\n{"jsonrpc":"2.0","id":7,"method":"tools/list"}\n')
    Server(cfg, stdin, stdout).serve()

    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [r.get("error", {}).get("code") for r in replies[:2]] == [-32600, -32600]
    assert replies[2]["id"] == 7  # still serving after two bad frames


def test_results_are_trimmed_for_an_agent(repo):
    body = "\n".join(f"    step_{i} = {i}" for i in range(200))
    root = repo({"long.py": f"def pipeline():\n{body}\n"})
    cfg = load_config(root, provider="hash")
    store = Store(cfg.db_path)
    build_index(cfg, get_embedder(cfg.provider), store)
    store.close()

    reply = drive(root, call("search_code", query="pipeline step"))[0]["result"]
    text = reply["content"][0]["text"]
    blocks = text.split("\n\n")
    assert blocks, text
    for block in blocks:  # every hit is trimmed, however many come back
        assert len(block.split("\n")) <= 27, block[:200]
    assert "read long.py from line" in text
    assert all(len(hit["code"]) < 2000 for hit in reply["structuredContent"]["results"])


def test_a_rebuilt_index_is_picked_up(repo):
    root = indexed(repo)
    cfg = load_config(root, provider="hash")
    server = Server(cfg, io.StringIO(), io.StringIO())

    first = server._call_tool({"name": "index_status", "arguments": {}})
    assert first["structuredContent"]["files"] == 2

    (root / "third.py").write_text("def gamma():\n    return 3\n", encoding="utf-8")
    store = Store(cfg.db_path)
    build_index(cfg, get_embedder(cfg.provider), store)
    store.close()

    second = server._call_tool({"name": "index_status", "arguments": {}})
    assert second["structuredContent"]["files"] == 3
    server.close()


def test_bad_arguments_are_reported(repo):
    root = indexed(repo)
    replies = drive(
        root,
        call("search_code", query="verify", mode="telepathy"),
        call("search_code", query="verify", lang=[1, 2]),
    )
    assert all(r["result"]["isError"] for r in replies)
    assert "mode must be" in replies[0]["result"]["content"][0]["text"]
    assert "lang must be" in replies[1]["result"]["content"][0]["text"]


def test_search_returns_the_call_path_when_asked(repo):
    root = repo(
        {
            "web/routes.py": (
                "def handle_login(request):\n"
                '    """Entry point behind the login form."""\n'
                "    return issue_ticket(request)\n"
            ),
            "auth/session.py": "def issue_ticket(request):\n    return request\n",
        }
    )
    cfg = load_config(root, provider="hash")
    store = Store(cfg.db_path)
    build_index(cfg, get_embedder(cfg.provider), store)
    store.close()

    plain = drive(root, call("search_code", query="entry point behind the login form", limit=1))
    traced = drive(
        root, call("search_code", query="entry point behind the login form", limit=1, trace=True)
    )

    assert "→ auth/session.py" not in plain[0]["result"]["content"][0]["text"]
    text = traced[0]["result"]["content"][0]["text"]
    assert "→ auth/session.py:1  issue_ticket" in text
    first = traced[0]["result"]["structuredContent"]["results"][0]
    assert first["trace"][0]["path"] == "auth/session.py"
