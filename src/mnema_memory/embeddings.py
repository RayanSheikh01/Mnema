from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math
import os
from typing import Any


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed_bytes = (digest * ((self.dimensions // len(digest)) + 1))[: self.dimensions]
            raw = [((byte / 255.0) * 2.0) - 1.0 for byte in seed_bytes]
            norm = math.sqrt(sum(v * v for v in raw)) or 1.0
            vectors.append([v / norm for v in raw])
        return vectors


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str) -> None:
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for openai embedding provider")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed; install with `pip install .[openai]`"
            ) from exc
        self._client: Any = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


def build_embedding_provider(provider: str, model: str) -> EmbeddingProvider:
    normalized = provider.strip().lower()
    if normalized in {"openai", "openai-api"}:
        return OpenAIEmbeddingProvider(model=model)
    if normalized in {"local-hash", "hash"}:
        return HashEmbeddingProvider()
    raise ValueError(f"unsupported embedding provider: {provider}")
