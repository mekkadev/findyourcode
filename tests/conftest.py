import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from findyourcode.config import Config


@pytest.fixture
def cfg(tmp_path):
    return Config(root=tmp_path, provider="hash")


@pytest.fixture
def repo(tmp_path):
    def build(files: dict[str, str]) -> Path:
        for rel, body in files.items():
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        return tmp_path

    return build
