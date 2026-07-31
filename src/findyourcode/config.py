"""Runtime configuration: defaults, .findyourcode.toml, environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - python 3.10
    import tomli as tomllib

CONFIG_FILENAME = ".findyourcode.toml"
INDEX_DIRNAME = ".findyourcode"

DEFAULT_EXCLUDE = [
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "*.svg",
    "*.snap",
    "**/node_modules/**",
    "**/vendor/**",
    "**/dist/**",
    "**/build/**",
    "**/target/**",
    "**/.venv/**",
    "**/venv/**",
    "**/__pycache__/**",
    "**/.git/**",
    "**/migrations/**",
    "**/testdata/**",
    "**/*.generated.*",
    "**/*_pb2.py",
]


@dataclass
class Config:
    root: Path = field(default_factory=Path.cwd)
    provider: str = "local"
    model: str = ""
    batch_size: int = 64

    max_chunk_lines: int = 110
    min_chunk_lines: int = 2
    overlap_lines: int = 12
    max_file_bytes: int = 1_500_000
    max_embed_chars: int = 4000

    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))

    workers: int = 8
    oversample: int = 8
    per_file: int = 2
    fusion: str = "blend"
    alpha: float = 0.75
    rrf_k: int = 60
    semantic_weight: float = 1.0
    lexical_weight: float = 0.6

    @property
    def index_dir(self) -> Path:
        return self.root / INDEX_DIRNAME

    @property
    def db_path(self) -> Path:
        return self.index_dir / "index.db"


_ENV_PREFIX = "FYC_"


def load_config(root: Path | None = None, **overrides) -> Config:
    root = Path(root or Path.cwd()).resolve()
    cfg = Config(root=root)

    path = root / CONFIG_FILENAME
    if path.is_file():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        _apply(cfg, data.get("findyourcode", data))

    known = {f.name for f in fields(Config)} - {"root"}
    env = {}
    for name in known:
        raw = os.environ.get(_ENV_PREFIX + name.upper())
        if raw is not None:
            env[name] = raw
    _apply(cfg, env)

    _apply(cfg, {k: v for k, v in overrides.items() if v is not None})
    return cfg


def _apply(cfg: Config, data: dict) -> None:
    types = {f.name: f.type for f in fields(Config)}
    for key, value in data.items():
        if key not in types or key == "root":
            continue
        current = getattr(cfg, key)
        if isinstance(current, bool):
            value = str(value).lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int) and not isinstance(value, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, list) and isinstance(value, str):
            value = [p for p in value.split(",") if p]
        setattr(cfg, key, value)
