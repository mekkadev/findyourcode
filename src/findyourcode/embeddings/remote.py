"""HTTP embedding providers (Voyage, OpenAI and any OpenAI-compatible endpoint)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

from .base import Embedder, normalize


class _HttpEmbedder(Embedder):
    url = ""
    env_key = ""
    default_model = ""
    max_batch = 96

    def __init__(self, model: str = "", batch_size: int = 64):
        self.model = model or self.default_model
        self.batch_size = min(batch_size, self.max_batch)
        self.api_key = os.environ.get(self.env_key, "")
        if not self.api_key:
            raise RuntimeError(f"provider '{self.name}' needs {self.env_key} in the environment")
        self._dim = 0

    @property
    def dim(self) -> int:
        """Only the endpoint knows the width; probe once so callers never see 0."""
        if not self._dim:
            self._request(["dimension probe"], True)
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value

    def _payload(self, texts: list[str], is_query: bool) -> dict:
        raise NotImplementedError

    def _request(self, texts: list[str], is_query: bool) -> np.ndarray:
        body = json.dumps(self._payload(texts, is_query)).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        last: Exception | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.loads(response.read())
                vectors = [item["embedding"] for item in data["data"]]
                matrix = normalize(np.asarray(vectors, dtype=np.float32))
                self.dim = matrix.shape[1]
                return matrix
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise
            except urllib.error.URLError as exc:
                last = exc
            time.sleep(2**attempt)
        raise RuntimeError(f"{self.name}: embedding request failed: {last}")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 1), dtype=np.float32)
        out = [
            self._request(texts[i : i + self.batch_size], False)
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.vstack(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self._request([text], True)[0]


class VoyageEmbedder(_HttpEmbedder):
    name = "voyage"
    url = "https://api.voyageai.com/v1/embeddings"
    env_key = "VOYAGE_API_KEY"
    default_model = "voyage-code-3"
    max_batch = 96

    def _payload(self, texts: list[str], is_query: bool) -> dict:
        return {
            "model": self.model,
            "input": texts,
            "input_type": "query" if is_query else "document",
            "truncation": True,
        }


class OpenAIEmbedder(_HttpEmbedder):
    name = "openai"
    url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/embeddings"
    env_key = "OPENAI_API_KEY"
    default_model = "text-embedding-3-small"
    max_batch = 128

    def _payload(self, texts: list[str], is_query: bool) -> dict:
        return {"model": self.model, "input": texts}
