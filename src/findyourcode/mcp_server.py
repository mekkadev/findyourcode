"""An MCP server over stdio, so an agent searches the same index you do.

Hand-rolled JSON-RPC rather than a dependency: the protocol surface a search tool
needs is four methods, and the point of this project is that it installs with one
`pip install` and no server.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from .config import Config, load_config
from .embeddings import Embedder, get_embedder
from .format import as_trace, symbol_of
from .search import build_trace, reached_by_graph, search, similar_to
from .store import Filters, Store

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOLS = {"2024-11-05", "2025-03-26", "2025-06-18"}

# An agent pays for every byte of a tool result, so a hit is trimmed to something
# worth reading and the caller is told where to look for the rest.
MAX_SNIPPET_LINES = 24
MAX_SNIPPET_CHARS = 1400

_FILTERS = {
    "lang": {"type": "array", "items": {"type": "string"}, "description": "restrict to languages"},
    "path": {
        "type": "array",
        "items": {"type": "string"},
        "description": "restrict to path substrings",
    },
    "kind": {
        "type": "array",
        "items": {"type": "string"},
        "description": "restrict to function, class, method, interface, block, ...",
    },
    "limit": {"type": "integer", "description": "how many results (default 10)"},
}

TOOLS = [
    {
        "name": "search_code",
        "description": (
            "Search this codebase by meaning. The query is a natural-language description of "
            "what the code does — 'where do we authenticate users', 'retry failed payments' — "
            "not a grep pattern. Returns file paths with line ranges and the matching code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what the code you want does"},
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "semantic", "lexical"],
                    "description": "hybrid (default) blends vectors and bm25",
                },
                "trace": {
                    "type": "boolean",
                    "description": (
                        "also return the call path around each hit — what reaches it and "
                        "what it reaches. Use it to follow a flow instead of reading files."
                    ),
                },
                **_FILTERS,
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_similar",
        "description": (
            "Given a place in the codebase as 'path' or 'path:line', return the chunks most "
            "similar to it elsewhere. Use it to find every implementation of a pattern you "
            "have one example of."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "path or path:line"},
                "same_file": {
                    "type": "boolean",
                    "description": "also return chunks from that file",
                },
                **_FILTERS,
            },
            "required": ["location"],
        },
    },
    {
        "name": "index_status",
        "description": (
            "What is in the index: file and chunk counts, the embedding model, languages."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class Server:
    def __init__(self, cfg: Config, stdin: TextIO, stdout: TextIO):
        self.cfg = cfg
        self.stdin = stdin
        self.stdout = stdout
        self._store: Store | None = None
        self._embedder: Embedder | None = None
        self._reranker = None
        self._stamp = 0.0
        self.handlers: dict[str, Callable[[dict], Any]] = {
            "initialize": self._initialize,
            "ping": lambda params: {},
            "tools/list": lambda params: {"tools": TOOLS},
            "tools/call": self._call_tool,
        }

    # ---- transport ----------------------------------------------------

    def serve(self) -> int:
        for line in self.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._send({"jsonrpc": "2.0", "id": None, "error": _error(-32700, "invalid json")})
                continue
            if not isinstance(message, dict):
                # a batch (array) or a bare value: answerable, not fatal
                self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": _error(-32600, "expected a single json-rpc object per line"),
                    }
                )
                continue
            response = self._dispatch(message)
            if response is not None:
                self._send(response)
        self.close()
        return 0

    def _dispatch(self, message: dict) -> dict | None:
        method = message.get("method", "")
        message_id = message.get("id")
        if message_id is None:  # a notification expects no reply
            return None

        handler = self.handlers.get(method)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": _error(-32601, f"unknown method '{method}'"),
            }
        try:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": handler(message.get("params") or {}),
            }
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": message_id, "error": _error(-32603, str(exc))}

    def _send(self, payload: dict) -> None:
        self.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.stdout.flush()

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    # ---- methods ------------------------------------------------------

    def _initialize(self, params: dict) -> dict:
        asked = params.get("protocolVersion")
        return {
            "protocolVersion": asked if asked in SUPPORTED_PROTOCOLS else PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "findyourcode", "version": _version()},
        }

    def _call_tool(self, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        runner = {
            "search_code": self._search,
            "find_similar": self._similar,
            "index_status": self._status,
        }.get(name)
        if runner is None:
            return _text(f"unknown tool '{name}'", is_error=True)
        try:
            return runner(arguments)
        except (SystemExit, ValueError) as exc:  # guards and argument validation
            return _text(str(exc), is_error=True)

    def _search(self, arguments: dict) -> dict:
        query = (arguments.get("query") or "").strip()
        if not query:
            return _text("query is empty", is_error=True)

        mode = arguments.get("mode") or "hybrid"
        if mode not in ("hybrid", "semantic", "lexical"):
            return _text(f"mode must be hybrid, semantic or lexical, not {mode!r}", is_error=True)

        store, embedder = self._open()
        hits = search(
            store,
            embedder,
            query,
            self.cfg,
            limit=_limit(arguments),
            filters=_filters(arguments),
            mode=mode,
            reranker=self._rerank(),
        )
        if not hits:
            return _text(f"nothing in the index matches {query!r}")

        traces = {}
        if arguments.get("trace"):
            vector = embedder.embed_query(query)
            traces = {
                hit.row.id: build_trace(store, hit.row, self.cfg, vector if vector.any() else None)
                for hit in hits
            }
        return _text(
            _render(hits, traces),
            data={"query": query, "results": [_as_dict(h, traces) for h in hits]},
        )

    def _similar(self, arguments: dict) -> dict:
        location = (arguments.get("location") or "").strip()
        if not location:
            return _text("location is empty", is_error=True)

        store, _ = self._open()
        anchor, hits = similar_to(
            store,
            location,
            self.cfg,
            limit=_limit(arguments),
            filters=_filters(arguments),
            same_file=bool(arguments.get("same_file")),
        )
        if anchor is None:
            return _text(f"nothing indexed at {location!r}", is_error=True)
        if not hits:
            return _text(f"no neighbours found for {location!r}")

        header = (
            f"like {anchor.rel}:{anchor.start_line}-{anchor.end_line} {symbol_of(anchor)}".rstrip()
        )
        return _text(
            f"{header}\n\n{_render(hits)}",
            data={"anchor": _row_dict(anchor), "results": [_as_dict(h) for h in hits]},
        )

    def _status(self, arguments: dict) -> dict:
        store, _ = self._open()
        stats = store.stats()
        lines = [
            f"root      {self.cfg.root}",
            f"model     {stats['signature']}",
            f"files     {stats['files']}",
            f"chunks    {stats['chunks']}",
        ]
        if stats["langs"]:
            lines.append("languages " + ", ".join(f"{lang} {n}" for lang, n in stats["langs"]))
        return _text("\n".join(lines), data=stats)

    def _rerank(self):
        """The cli honours cfg.rerank; an agent must get the same ranking a human gets."""
        if self._reranker is None and self.cfg.rerank:
            from .rerank import Reranker

            self._reranker = Reranker(self.cfg.rerank)
        return self._reranker

    def _open(self) -> tuple[Store, Embedder]:
        stamp = self.cfg.db_path.stat().st_mtime if self.cfg.db_path.exists() else 0.0
        if self._store is not None and stamp != self._stamp:
            # the index was rebuilt while we were serving: rows we hold are gone
            self.close()
        self._stamp = stamp
        if self._store is None:
            if not self.cfg.db_path.exists():
                raise SystemExit(f"no index at {self.cfg.root} — run `fyc index` there first")
            self._store = Store(self.cfg.db_path)
            signature = self._store.get_meta("signature") or ""
            provider, _, model_and_dim = signature.partition(":")
            model = model_and_dim.rsplit(":", 1)[0] if model_and_dim else self.cfg.model
            self._embedder = get_embedder(provider or self.cfg.provider, model, self.cfg.batch_size)
        assert self._embedder is not None
        return self._store, self._embedder


def serve(root: str = ".", stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    cfg = load_config(Path(root))
    return Server(cfg, stdin or sys.stdin, stdout or sys.stdout).serve()


def _filters(arguments: dict) -> Filters:
    def listed(key: str) -> list[str] | None:
        value = arguments.get(key)
        if not value:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)
        raise ValueError(f"{key} must be a string or a list of strings")

    return Filters(langs=listed("lang"), paths=listed("path"), kinds=listed("kind"))


def _limit(arguments: dict) -> int:
    try:
        return max(1, min(int(arguments.get("limit") or 10), 50))
    except (TypeError, ValueError):
        return 10


def _render(hits, traces: dict | None = None) -> str:
    blocks = []
    for hit in hits:
        row = hit.row
        label = " ".join(p for p in (row.kind, symbol_of(row)) if p)
        where = f"{row.rel}:{row.start_line}-{row.end_line}"
        head = f"{where}  {label}  [{hit.score:.3f}]"
        if reached_by_graph(hit):
            head += f"  (reached through the call graph — {hit.via})"
        node = (traces or {}).get(row.id)
        if node is not None:
            head += "".join("\n" + line for line in _trace_lines(node))
        blocks.append(f"{head}\n{_snippet(row)}")
    return "\n\n".join(blocks)


def _trace_lines(node, depth: int = 0) -> list[str]:
    out = []
    for child in node.children:
        arrow = "↑" if child.direction == "called by" else "→"
        pad = "  " * (depth + 1)
        out.append(
            f"{pad}{arrow} {child.row.rel}:{child.row.start_line}  {symbol_of(child.row)}".rstrip()
        )
        out.extend(_trace_lines(child, depth + 1))
    return out


def _snippet(row) -> str:
    lines = row.code.split("\n")
    shown = lines[:MAX_SNIPPET_LINES]
    text = "\n".join(shown)[:MAX_SNIPPET_CHARS]
    if len(shown) < len(lines) or len(text) < len("\n".join(shown)):
        remaining = row.end_line - row.start_line + 1 - len(shown)
        text += f"\n    ... read {row.rel} from line {row.start_line}"
        if remaining > 0:
            text += f" for {remaining} more lines"
    return text


def _as_dict(hit, traces: dict | None = None) -> dict:
    entry = {**_row_dict(hit.row), "score": round(hit.score, 6)}
    if hit.via:
        entry["via"] = hit.via
    node = (traces or {}).get(hit.row.id)
    if node is not None:
        entry["trace"] = as_trace(node)["calls"]
    return entry


def _row_dict(row) -> dict:
    return {
        "path": row.rel,
        "start_line": row.start_line,
        "end_line": row.end_line,
        "lang": row.lang,
        "kind": row.kind,
        "symbol": symbol_of(row),
        "code": _snippet(row),
    }


def _text(message: str, data: dict | None = None, is_error: bool = False) -> dict:
    result: dict[str, Any] = {"content": [{"type": "text", "text": message}], "isError": is_error}
    if data is not None:
        result["structuredContent"] = data
    return result


def _error(code: int, message: str) -> dict:
    return {"code": code, "message": message}


def _version() -> str:
    from . import __version__

    return __version__
