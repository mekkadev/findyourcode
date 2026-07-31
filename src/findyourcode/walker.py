"""File discovery: git-aware listing, glob filters, binary and size guards."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import Config
from .languages import lang_for_path

_SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".findyourcode"}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    rel: str
    lang: str
    size: int
    mtime: float


def discover(cfg: Config) -> list[SourceFile]:
    paths = dict.fromkeys(_git_files(cfg.root) or _walk(cfg.root))
    include = [_compile(p) for p in cfg.include]
    exclude = [_compile(p) for p in cfg.exclude]

    found: list[SourceFile] = []
    for rel in sorted(paths):
        if include and not any(rx.match(rel) for rx in include):
            continue
        if any(rx.match(rel) for rx in exclude):
            continue

        path = cfg.root / rel
        lang = lang_for_path(path)
        if lang is None:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file() or stat.st_size == 0 or stat.st_size > cfg.max_file_bytes:
            continue
        if _is_binary(path):
            continue
        found.append(SourceFile(path, rel, lang, stat.st_size, stat.st_mtime))
    return found


def read_source(path: Path) -> str | None:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _git_files(root: Path) -> list[str]:
    if not (root / ".git").exists():
        return []
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [p for p in out.stdout.decode("utf-8", "replace").split("\0") if p]


def _walk(root: Path) -> list[str]:
    import os

    rels: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        base = Path(dirpath)
        for name in filenames:
            rels.append((base / name).relative_to(root).as_posix())
    return rels


@lru_cache(maxsize=512)
def _compile(pattern: str) -> re.Pattern[str]:
    pattern = pattern.strip()
    rooted = pattern.startswith("/")
    pattern = pattern.lstrip("/")
    # gitignore semantics: `build/` matches that directory at any depth, `a/b` only where written.
    anchored = rooted or "/" in pattern.rstrip("/")
    if pattern.endswith("/"):
        pattern += "**"
    body = _glob_to_regex(pattern)
    return re.compile(body if anchored else rf"(?:.*/)?{body}")


def _glob_to_regex(pattern: str) -> str:
    out = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if pattern[i : i + 3] == "**/":
                out.append("(?:.*/)?")
                i += 3
                continue
            if pattern[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out) + r"\Z"


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(8192)
    except OSError:
        return True
