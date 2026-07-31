"""Terminal and JSON rendering of search results."""

from __future__ import annotations

import json
import os
import sys

from .search import Hit

_C = {
    "path": "\033[1;36m",
    "meta": "\033[2m",
    "score": "\033[33m",
    "line": "\033[2;37m",
    "reset": "\033[0m",
}


def _colors(enabled: bool) -> dict:
    return _C if enabled else dict.fromkeys(_C, "")


def use_color(stream=sys.stdout) -> bool:
    return stream.isatty() and not os.environ.get("NO_COLOR")


def render(
    hits: list[Hit], snippet_lines: int = 8, explain: bool = False, color: bool = True
) -> str:
    c = _colors(color)
    if not hits:
        return "nothing found"

    out: list[str] = []
    for i, hit in enumerate(hits, 1):
        row = hit.row
        where = f"{row.rel}:{row.start_line}-{row.end_line}"
        label = " ".join(p for p in (row.kind, symbol_of(row)) if p)
        head = f"{c['score']}{i:>2}.{c['reset']} {c['path']}{where}{c['reset']}"
        if label:
            head += f"  {c['meta']}{label}{c['reset']}"
        head += f"  {c['meta']}[{hit.score:.3f}]{c['reset']}"
        out.append(head)

        if explain:
            parts = []
            if hit.semantic is not None:
                rank = f"#{hit.semantic_rank} " if hit.semantic_rank else "cosine "
                parts.append(f"semantic {rank}({hit.semantic:.3f})")
            if hit.lexical is not None:
                parts.append(f"lexical #{hit.lexical_rank} (bm25 {hit.lexical:.2f})")
            out.append(f"    {c['meta']}{' | '.join(parts) or 'no sub-scores'}{c['reset']}")

        body = row.code.split("\n")
        shown = body if snippet_lines <= 0 else body[:snippet_lines]
        width = len(str(row.start_line + len(shown)))
        for offset, line in enumerate(shown):
            number = str(row.start_line + offset).rjust(width)
            out.append(f"    {c['line']}{number}{c['reset']} {line.rstrip()}")
        if len(body) > len(shown):
            out.append(f"    {c['meta']}... {len(body) - len(shown)} more lines{c['reset']}")
        out.append("")
    return "\n".join(out).rstrip()


def as_paths(hits: list[Hit], with_line: bool = True) -> str:
    """One location per line — meant for `| xargs`, `$EDITOR` and fzf."""
    seen: list[str] = []
    for hit in hits:
        entry = f"{hit.row.rel}:{hit.row.start_line}" if with_line else hit.row.rel
        if entry not in seen:
            seen.append(entry)
    return "\n".join(seen)


def as_json(hits: list[Hit]) -> str:
    payload = [
        {
            "path": h.row.rel,
            "start_line": h.row.start_line,
            "end_line": h.row.end_line,
            "lang": h.row.lang,
            "kind": h.row.kind,
            "symbol": symbol_of(h.row),
            "score": round(h.score, 6),
            "semantic": None if h.semantic is None else round(h.semantic, 6),
            "lexical": None if h.lexical is None else round(h.lexical, 6),
            "code": h.row.code,
        }
        for h in hits
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def symbol_of(row) -> str:
    if row.parent and row.symbol:
        return f"{row.parent}.{row.symbol}"
    return row.symbol or row.parent
