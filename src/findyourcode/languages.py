"""Language detection and the tree-sitter node types that form a searchable unit."""

from __future__ import annotations

import re
from functools import lru_cache

EXT_TO_LANG = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".m": "objc",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".dart": "dart",
    ".vue": "vue",
    ".svelte": "svelte",
    ".proto": "proto",
    ".tf": "hcl",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
    ".mdx": "markdown",
    ".rst": "text",
    ".txt": "text",
}

FILENAME_TO_LANG = {
    "Dockerfile": "dockerfile",
    "Makefile": "make",
    "CMakeLists.txt": "cmake",
    "go.mod": "gomod",
}

# Languages we chunk with a plain line/paragraph splitter instead of an AST.
TEXTUAL_LANGS = {"markdown", "text", "json", "yaml", "toml"}

_DEFINITIONS: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "method_definition",
        "lexical_declaration",
        "variable_declaration",
        "export_statement",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "const_declaration",
    },
    "rust": {
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "trait_item",
        "mod_item",
        "macro_definition",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "method_declaration",
        "constructor_declaration",
    },
    "ruby": {"method", "singleton_method", "class", "module"},
    "php": {
        "function_definition",
        "class_declaration",
        "method_declaration",
        "interface_declaration",
        "trait_declaration",
    },
    "c": {"function_definition", "struct_specifier", "enum_specifier", "type_definition"},
    "lua": {"function_declaration", "local_function", "function_definition"},
    "bash": {"function_definition"},
    "elixir": {"call"},
}
_DEFINITIONS["typescript"] = _DEFINITIONS["javascript"] | {
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    "abstract_class_declaration",
    "module",
}
_DEFINITIONS["tsx"] = _DEFINITIONS["typescript"]
_DEFINITIONS["cpp"] = _DEFINITIONS["c"] | {
    "class_specifier",
    "namespace_definition",
    "template_declaration",
}
_DEFINITIONS["csharp"] = {
    "class_declaration",
    "interface_declaration",
    "struct_declaration",
    "record_declaration",
    "enum_declaration",
    "method_declaration",
    "constructor_declaration",
    "property_declaration",
    "namespace_declaration",
}
_DEFINITIONS["kotlin"] = {"function_declaration", "class_declaration", "object_declaration"}
_DEFINITIONS["swift"] = {
    "function_declaration",
    "class_declaration",
    "protocol_declaration",
    "property_declaration",
}
_DEFINITIONS["scala"] = {"function_definition", "class_definition", "object_definition", "trait_definition"}

# Nodes we descend into when they are too big to keep as one chunk.
CONTAINER_NODES = {
    "class_definition",
    "class_declaration",
    "abstract_class_declaration",
    "class_specifier",
    "class",
    "impl_item",
    "trait_item",
    "mod_item",
    "module",
    "namespace_definition",
    "namespace_declaration",
    "interface_declaration",
    "object_declaration",
    "object_definition",
    "trait_definition",
    "record_declaration",
}

_GENERIC_DEFINITION = re.compile(
    r"(function|method|class|struct|impl|trait|interface|enum|module|namespace|"
    r"constructor|property|declaration|definition|subroutine|procedure)"
)

_NAME_NODE_TYPES = (
    "identifier",
    "type_identifier",
    "field_identifier",
    "property_identifier",
    "constant",
    "name",
    "word",
    "simple_identifier",
    "type_spec",
    "variable_declarator",
)


def lang_for_path(path) -> str | None:
    name = getattr(path, "name", str(path))
    if name in FILENAME_TO_LANG:
        return FILENAME_TO_LANG[name]
    suffix = getattr(path, "suffix", "")
    return EXT_TO_LANG.get(suffix.lower())


def is_definition(lang: str, node_type: str) -> bool:
    table = _DEFINITIONS.get(lang)
    if table is not None:
        return node_type in table
    return looks_like_definition(node_type)


def looks_like_definition(node_type: str) -> bool:
    """Grammar-agnostic guess, used for languages without a curated node list."""
    return bool(_GENERIC_DEFINITION.search(node_type))


@lru_cache(maxsize=64)
def get_parser(lang: str):
    """Return a tree-sitter parser, or None when the grammar is unavailable."""
    if lang in TEXTUAL_LANGS:
        return None
    try:
        from tree_sitter_language_pack import get_parser as _get

        return _get(lang)  # type: ignore[arg-type]
    except Exception:
        return None


def node_name(node, source: bytes) -> str | None:
    named = node.child_by_field_name("name")
    if named is not None:
        return _text(named, source)

    declarator = node.child_by_field_name("declarator")
    while declarator is not None:
        inner = declarator.child_by_field_name("declarator")
        if inner is None:
            return _text(declarator, source).split("(")[0].strip() or None
        declarator = inner

    for child in node.named_children:
        if child.type in _NAME_NODE_TYPES:
            if child.type in ("variable_declarator", "type_spec"):
                return node_name(child, source) or _text(child, source).split("=")[0].strip()
            return _text(child, source)
        if child.type in CONTAINER_NODES or looks_like_definition(child.type):
            found = node_name(child, source)
            if found:
                return found
    return None


def _text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace").strip()
