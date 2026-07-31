"""Offline embeddings via fastembed (ONNX, no torch)."""

from __future__ import annotations

import warnings

import numpy as np

from .base import Embedder, normalize

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LocalEmbedder(Embedder):
    name = "local"

    def __init__(self, model: str = "", batch_size: int = 64):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "provider 'local' needs fastembed: pip install 'findyourcode[local]'"
            ) from exc

        self.model = model or DEFAULT_MODEL
        self.batch_size = batch_size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._backend = TextEmbedding(self.model)
        self.dim = next(
            m["dim"] for m in TextEmbedding.list_supported_models() if m["model"] == self.model
        )
        # e5-family models were trained with asymmetric prefixes.
        e5 = "e5" in self.model.lower()
        self._query_prefix = "query: " if e5 else ""
        self._doc_prefix = "passage: " if e5 else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefixed = [self._doc_prefix + t for t in texts]
        vectors = list(self._backend.embed(prefixed, batch_size=self.batch_size))
        return normalize(np.vstack(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        vector = next(iter(self._backend.query_embed(self._query_prefix + text)))
        return normalize(np.asarray(vector))[0]
