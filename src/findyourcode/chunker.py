"""Split source files into semantic units using tree-sitter, with a textual fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Config
from .languages import (
    CONTAINER_NODES,
    TEXTUAL_LANGS,
    get_parser,
    is_definition,
    node_name,
)

_COMMENT_LINE = re.compile(r"^\s*(#|//|/\*|\*|--|;;|<!--)")
_NOISE_LINE = re.compile(r"^[\s\}\)\]\;,`\"']*$")
_DOCSTRING = re.compile(r'(?:[rubf]{0,2})("""|\'\'\')(.*?)\1', re.DOTALL | re.IGNORECASE)
_WRAPPERS = {"decorated_definition", "export_statement", "expression_statement", "template_declaration"}
_DECL_TYPES = {
    "lexical_declaration",
    "variable_declaration",
    "const_declaration",
    "type_declaration",
    "type_alias_declaration",
    "property_declaration",
    "field_declaration",
}
_CALLABLE_HINTS = ("function", "arrow", "lambda", "closure", "class", "struct", "interface")
_FIRST_DEFINITION = re.compile(
    r"^\s*(?:async\s+)?(?:def|class|func|function|impl|trait|struct|type|public|private|export)\b",
    re.MULTILINE,
)
_HEADING = re.compile(r"^#{1,6}\s+\S")

_KIND_BY_TYPE = (
    ("method", "method"),
    ("constructor", "method"),
    ("class", "class"),
    ("struct", "class"),
    ("record", "class"),
    ("interface", "interface"),
    ("trait", "interface"),
    ("protocol", "interface"),
    ("impl", "impl"),
    ("enum", "enum"),
    ("mod", "module"),
    ("namespace", "module"),
    ("function", "function"),
    ("subroutine", "function"),
    ("type", "type"),
    ("property", "property"),
    ("const", "const"),
    ("variable", "const"),
    ("lexical", "const"),
    ("export", "export"),
)


@dataclass
class Chunk:
    rel: str
    lang: str
    kind: str
    symbol: str
    parent: str
    start_line: int
    end_line: int
    code: str
    doc: str = ""
    file_doc: str = ""
    sha: str = ""


def chunk_source(rel: str, lang: str, text: str, cfg: Config) -> list[Chunk]:
    parser = None if lang in TEXTUAL_LANGS else get_parser(lang)
    lines = text.split("\n")
    chunks = None

    if parser is not None:
        try:
            tree = parser.parse(text.encode("utf-8"))
        except Exception:
            tree = None
        if tree is not None:
            builder = _AstChunker(rel, lang, lines, text.encode("utf-8"), cfg)
            builder.walk(tree.root_node, [])
            builder.collect_gaps()
            if builder.chunks:
                chunks = sorted(builder.chunks, key=lambda c: (c.start_line, c.end_line))

    if chunks is None:
        chunks = _chunk_textual(rel, lang, lines, cfg)

    file_doc = _file_summary(lines)
    for chunk in chunks:
        chunk.file_doc = file_doc
    return chunks


class _AstChunker:
    def __init__(self, rel: str, lang: str, lines: list[str], source: bytes, cfg: Config):
        self.rel = rel
        self.lang = lang
        self.lines = lines
        self.source = source
        self.cfg = cfg
        self.chunks: list[Chunk] = []
        self.covered = [False] * len(lines)
        self.containers: list[tuple[int, int, str]] = []

    def walk(self, node, trail: list[str]) -> None:
        for child in node.named_children:
            if is_definition(self.lang, child.type):
                if self._worth_own_chunk(child):
                    self.emit(child, trail)
            elif child.type in ("expression_statement", "export_statement"):
                for inner in child.named_children:
                    if is_definition(self.lang, inner.type):
                        self.emit(inner, trail)

    def emit(self, node, trail: list[str]) -> None:
        start = self._with_leading_comments(node.start_point[0])
        end = node.end_point[0]
        if end < start:
            return
        inner = self._unwrap(node)
        name = node_name(inner, self.source) or ""
        kind = _classify(inner.type)
        parent = ".".join(trail)
        if kind == "function" and parent:
            kind = "method"
        size = end - start + 1

        if size <= self.cfg.max_chunk_lines:
            self._add(kind, name, parent, start, end)
            return

        body = _body_of(inner)
        if inner.type in CONTAINER_NODES and body is not None:
            header_end = min(body.start_point[0], start + self.cfg.max_chunk_lines - 1)
            self._add(kind, name, parent, start, header_end, mark=range(start, header_end + 1))
            self.containers.append((start, end, ".".join(trail + [name]) if name else parent))
            self.walk(body, trail + [name] if name else trail)
            return

        windows = []
        step = max(1, self.cfg.max_chunk_lines - self.cfg.overlap_lines)
        for wstart in range(start, end + 1, step):
            wend = min(wstart + self.cfg.max_chunk_lines - 1, end)
            windows.append((wstart, wend))
            if wend >= end:
                break
        for i, (wstart, wend) in enumerate(windows, 1):
            self._add(kind, f"{name} [{i}/{len(windows)}]".strip(), parent, wstart, wend)

    def _worth_own_chunk(self, node) -> bool:
        """A one-line `const` is noise on its own; it belongs to the surrounding block."""
        if node.type not in _DECL_TYPES:
            return True
        if node.end_point[0] - node.start_point[0] >= self.cfg.min_chunk_lines:
            return True
        return _holds_callable(node)

    def _unwrap(self, node):
        while node.type in _WRAPPERS:
            nested = [c for c in node.named_children if is_definition(self.lang, c.type) and c.type != node.type]
            if not nested:
                return node
            node = nested[-1]
        return node

    def collect_gaps(self) -> None:
        run: list[int] = []
        for i, done in enumerate(self.covered):
            if not done:
                run.append(i)
                continue
            self._flush_gap(run)
            run = []
        self._flush_gap(run)

    def _flush_gap(self, run: list[int]) -> None:
        if not run:
            return
        for start in range(0, len(run), self.cfg.max_chunk_lines):
            block = run[start : start + self.cfg.max_chunk_lines]
            first, last = block[0], block[-1]
            while first <= last and not self.lines[first].strip():
                first += 1
            while last >= first and not self.lines[last].strip():
                last -= 1
            if last < first:
                continue
            body = self.lines[first : last + 1]
            meaningful = [ln for ln in body if ln.strip() and not _NOISE_LINE.match(ln)]
            if len(meaningful) < self.cfg.min_chunk_lines:
                continue
            self._add("block", "", self._container_at(first), first, last)

    def _container_at(self, line: int) -> str:
        best = ""
        span = None
        for start, end, trail in self.containers:
            if start <= line <= end and (span is None or end - start < span):
                span, best = end - start, trail
        return best

    def _add(self, kind: str, symbol: str, parent: str, start: int, end: int, mark=None) -> None:
        code = "\n".join(self.lines[start : end + 1]).rstrip()
        if not code.strip():
            return
        for i in mark if mark is not None else range(start, end + 1):
            if 0 <= i < len(self.covered):
                self.covered[i] = True
        self.chunks.append(
            Chunk(
                rel=self.rel,
                lang=self.lang,
                kind=kind,
                symbol=symbol,
                parent=parent,
                start_line=start + 1,
                end_line=end + 1,
                code=code,
                doc=_extract_doc(self.lines, start, end),
            )
        )

    def _with_leading_comments(self, start: int) -> int:
        i = start - 1
        seen = 0
        while i >= 0 and seen < 20:
            line = self.lines[i]
            if not line.strip():
                break
            if not _COMMENT_LINE.match(line):
                break
            if self.covered[i]:
                break
            i -= 1
            seen += 1
        return i + 1


def _chunk_textual(rel: str, lang: str, lines: list[str], cfg: Config) -> list[Chunk]:
    sections: list[tuple[str, int, int]] = []
    if lang == "markdown":
        current_title, start = "", 0
        for i, line in enumerate(lines):
            if _HEADING.match(line):
                if i > start:
                    sections.append((current_title, start, i - 1))
                current_title, start = line.lstrip("# ").strip(), i
        sections.append((current_title, start, len(lines) - 1))
    else:
        sections.append(("", 0, len(lines) - 1))

    chunks: list[Chunk] = []
    step = max(1, cfg.max_chunk_lines - cfg.overlap_lines)
    for title, start, end in sections:
        if end < start:
            continue
        for wstart in range(start, end + 1, step):
            wend = min(wstart + cfg.max_chunk_lines - 1, end)
            code = "\n".join(lines[wstart : wend + 1]).strip("\n")
            if not code.strip():
                continue
            chunks.append(
                Chunk(
                    rel=rel,
                    lang=lang,
                    kind="text",
                    symbol=title,
                    parent="",
                    start_line=wstart + 1,
                    end_line=wend + 1,
                    code=code,
                )
            )
            if wend >= end:
                break
    return chunks


def _holds_callable(node, depth: int = 3) -> bool:
    for child in node.named_children:
        if any(hint in child.type for hint in _CALLABLE_HINTS):
            return True
        if depth > 0 and _holds_callable(child, depth - 1):
            return True
    return False


def _classify(node_type: str) -> str:
    for needle, kind in _KIND_BY_TYPE:
        if needle in node_type:
            return kind
    return "block"


def _body_of(node):
    for field in ("body", "declaration_list"):
        found = node.child_by_field_name(field)
        if found is not None:
            return found
    for child in node.named_children:
        if child.type.endswith(("_body", "block", "declaration_list", "statement_block")):
            return child
    return None


def _file_summary(lines: list[str], limit: int = 200) -> str:
    """What the file as a whole is about — a strong topical signal for every chunk in it."""
    head = "\n".join(lines[:120])
    body_starts = _FIRST_DEFINITION.search(head)
    opening = re.search(r'("""|\'\'\')', head)
    if opening and not (body_starts and body_starts.start() < opening.start()):
        rest = head[opening.end() :]
        closing = rest.find(opening.group(1))
        summary = " ".join((rest if closing < 0 else rest[:closing]).split())[:limit]
        if summary:
            return summary

    comments = []
    for line in lines[:20]:
        if _COMMENT_LINE.match(line):
            comments.append(re.sub(r"^\s*(#+|//+|/\*+|\*+/?|--|;;|<!--)\s?", "", line).strip())
        elif line.strip():
            break
    return " ".join(" ".join(comments).split())[:limit]


def _extract_doc(lines: list[str], start: int, end: int) -> str:
    doc: list[str] = []
    for line in lines[start : min(end, start + 24) + 1]:
        if _COMMENT_LINE.match(line):
            doc.append(re.sub(r"^\s*(#+|//+|/\*+|\*+/?|--|;;|<!--)\s?", "", line).rstrip("*/ ").strip())
        elif doc and not line.strip():
            break

    window = "\n".join(lines[start : min(end, start + 30) + 1])
    for match in _DOCSTRING.finditer(window):
        doc.append(match.group(2).strip())
        break
    return "\n".join(d for d in doc if d)[:800]
