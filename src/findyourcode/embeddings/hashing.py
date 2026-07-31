"""Dependency-free deterministic embedder.

No real semantics — it exists so the pipeline runs offline (tests, CI, air-gapped
machines) and still returns sane lexical neighbours.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from .base import Embedder, normalize

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashEmbedder(Embedder):
    name = "hash"

    def __init__(self, model: str = "", batch_size: int = 64):
        self.model = model or "hash-384"
        self.dim = int(self.model.rsplit("-", 1)[-1]) if self.model[-1].isdigit() else 384

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = [w.lower() for w in _TOKEN.findall(text)]
        counts = Counter(words)
        for word, count in counts.items():
            weight = 1.0 + math.log(count)
            for feature in (word, word[:4], word[-4:]):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 else -1.0
                vec[index] += sign * weight
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(self._vector(text))[0]
