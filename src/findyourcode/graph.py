"""What each chunk calls, and what name each chunk defines.

Embeddings answer "what looks like this". They cannot answer "what is one call
away" — the callee of the right function is often a file that shares no word
with the query and no meaning with it either. Extracting the call sites while
tree-sitter already holds the parse costs nothing and gives retrieval a second,
structural signal.

Calls only, deliberately: imports and type annotations resolve far less
reliably, and a wrong edge is worse than a missing one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypeVar

from .languages import is_definition, node_name

if TYPE_CHECKING:
    from .chunker import Chunk

_T = TypeVar("_T")

MAX_NAMES_PER_CHUNK = 40
MAX_NAMES_PER_FILE = 4000

_CALL_HINTS = ("call", "invocation")
_CALL_TYPES = {"new_expression", "object_creation_expression", "macro_invocation"}
_NOT_CALLS = ("signature", "declaration", "definition", "parameter")
_NAME_FIELDS = ("name", "function", "method", "constructor", "callee", "macro")
_IDENT_TYPES = {
    "identifier",
    "field_identifier",
    "property_identifier",
    "type_identifier",
    "simple_identifier",
    "constant",
    "name",
    "word",
}

_WINDOW_SUFFIX = re.compile(r"\s*\[\d+/\d+\]$")
_TEXT_CALL = re.compile(r"(?:([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*)?\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(")
_IDENT_TAIL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Control flow and casts look exactly like calls to a regex, and to some grammars.
_KEYWORDS = (
    "if elif else for while do switch case catch except finally with match when "
    "return yield await throw raise defer go func function def fn fun lambda "
    "print println printf sprintf format echo len sizeof typeof instanceof "
    "int str bool float double char byte long short void var let const new delete "
    "require import include using assert panic recover super this self "
    "defmodule defp defmacro defstruct defimpl defprotocol defdelegate"
)
_NOT_A_CALLEE = frozenset(_KEYWORDS.split(" "))


def references_from_tree(root, source: bytes) -> list[tuple[int, str, str]]:
    """Every call site under a parse tree as (1-based line, callee, qualifier).

    The qualifier is what stood before the dot. `linecache.getline(...)` says which
    of the four `getline` definitions in a large codebase is meant, and that turns
    out to be the difference between an edge and a guess."""
    found: list[tuple[int, str, str]] = []
    stack = [root]
    while stack and len(found) < MAX_NAMES_PER_FILE:
        node = stack.pop()
        stack.extend(node.named_children)
        if not _is_call(node.type):
            continue
        name, scope = _callee_name(node, source)
        if name and _worth_an_edge(name):
            found.append((node.start_point[0] + 1, name, scope))
    return found


def definitions_from_tree(root, source: bytes, lang: str) -> list[tuple[int, str]]:
    """Every name this file declares, at any depth — a method inside a class is a call
    target too, and the chunk holding the class is where a caller wants to be sent."""
    found: list[tuple[int, str]] = []
    stack = [root]
    while stack and len(found) < MAX_NAMES_PER_FILE:
        node = stack.pop()
        stack.extend(node.named_children)
        # In elixir a definition *is* a call — `def deliver do`. A grammar that cannot
        # tell the two apart would have every call site claim to define its callee, so
        # such a language keeps only the definition its chunk is named after.
        if _is_call(node.type) or not is_definition(lang, node.type):
            continue
        name = node_name(node, source)
        if name and _worth_an_edge(name):
            found.append((node.start_point[0] + 1, name))
    return found


def references_from_text(code: str, first_line: int) -> list[tuple[int, str, str]]:
    """Fallback for languages with no grammar: what looks like a call, is one."""
    found: list[tuple[int, str, str]] = []
    for offset, line in enumerate(code.split("\n")):
        for scope, name in _TEXT_CALL.findall(line):
            if _worth_an_edge(name):
                found.append((first_line + offset, name, scope if _usable_scope(scope) else ""))
    return found


def attach_names(
    chunks: list[Chunk], refs: list[tuple[int, str, str]], defs: list[tuple[int, str]]
) -> None:
    ref_lines = _by_line([(line, (name, scope)) for line, name, scope in refs])
    def_lines = _by_line(list(defs))
    for chunk in chunks:
        chunk.defs = _collect(def_lines, chunk.start_line, chunk.end_line)
        own = set(chunk.defs)
        chunk.refs = [
            ref
            for ref in _collect(ref_lines, chunk.start_line, chunk.end_line)
            if ref[0] not in own
        ]


def definition_names(chunk: Chunk) -> list[str]:
    """The names other code would use to reach this chunk."""
    names = list(chunk.defs)
    symbol = _WINDOW_SUFFIX.sub("", chunk.symbol or "").strip()
    if symbol and symbol not in names and _worth_an_edge(symbol):
        names.append(symbol)
    return names


def _by_line(pairs: list[tuple[int, _T]]) -> dict[int, list[_T]]:
    found: dict[int, list[_T]] = {}
    for line, value in pairs:
        found.setdefault(line, []).append(value)
    return found


def _collect(by_line: dict[int, list[_T]], start: int, end: int) -> list[_T]:
    if not by_line:
        return []
    names: list[_T] = []
    seen: set[_T] = set()
    for line in range(start, end + 1):
        for name in by_line.get(line, ()):
            if name not in seen:
                seen.add(name)
                names.append(name)
                if len(names) >= MAX_NAMES_PER_CHUNK:
                    return names
    return names


def _is_call(node_type: str) -> bool:
    if node_type in _CALL_TYPES:
        return True
    if any(bad in node_type for bad in _NOT_CALLS):
        return False
    return any(hint in node_type for hint in _CALL_HINTS)


def _callee_name(node, source: bytes) -> tuple[str | None, str]:
    target = None
    for field in _NAME_FIELDS:
        target = node.child_by_field_name(field)
        if target is not None:
            break
    if target is None:
        target = node.named_children[0] if node.named_children else None
    if target is None:
        return None, ""

    scope = _qualifier(node, target, source)
    return _last_identifier(target, source), scope if _usable_scope(scope) else ""


def _qualifier(node, target, source: bytes) -> str:
    """What stood before the dot, read off the tree rather than off the text.

    Java, ruby and php hang the receiver on the call node instead of folding it into
    the callee, and reading the text of a callee is quadratic on `a().b().c()`."""
    for field in ("object", "receiver"):
        holder = node.child_by_field_name(field)
        if holder is not None:
            return _last_identifier(holder, source) or ""
    children = target.named_children
    return (_last_identifier(children[-2], source) or "") if len(children) >= 2 else ""


def _usable_scope(scope: str) -> bool:
    return len(scope) > 1 and scope.lower() not in ("self", "this", "cls", "super", "obj")


def _last_identifier(node, source: bytes) -> str | None:
    """`store.search_vector(...)` is an edge to search_vector, not to store."""
    if node.type in _IDENT_TYPES:
        return _text(node, source)
    for child in reversed(node.named_children):
        found = _last_identifier(child, source)
        if found:
            return found
    tail = _IDENT_TAIL.findall(_text(node, source))
    return tail[-1] if tail else None


def _text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace").strip()


def _worth_an_edge(name: str) -> bool:
    if len(name) <= 2 or name.isdigit() or name.lower() in _NOT_A_CALLEE:
        return False
    return not (name.startswith("__") and name.endswith("__"))
