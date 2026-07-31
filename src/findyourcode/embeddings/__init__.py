from __future__ import annotations

from .base import Embedder, normalize

PROVIDERS = {
    "local": (
        "findyourcode.embeddings.local",
        "LocalEmbedder",
        "offline ONNX model (multilingual, default)",
    ),
    "voyage": (
        "findyourcode.embeddings.remote",
        "VoyageEmbedder",
        "Voyage AI API, code-tuned (VOYAGE_API_KEY)",
    ),
    "openai": (
        "findyourcode.embeddings.remote",
        "OpenAIEmbedder",
        "OpenAI-compatible API (OPENAI_API_KEY)",
    ),
    "hash": (
        "findyourcode.embeddings.hashing",
        "HashEmbedder",
        "deterministic lexical fallback, no deps",
    ),
}


def get_embedder(provider: str, model: str = "", batch_size: int = 64) -> Embedder:
    if provider not in PROVIDERS:
        raise SystemExit(f"unknown provider '{provider}'; available: {', '.join(PROVIDERS)}")
    module_name, class_name, _ = PROVIDERS[provider]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)(model=model, batch_size=batch_size)


__all__ = ["PROVIDERS", "Embedder", "get_embedder", "normalize"]
