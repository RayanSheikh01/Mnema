from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
import logging
import math
import os
from typing import Any


LOGGER = logging.getLogger("mnema_memory")


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


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Real local semantic embeddings via ``sentence-transformers`` (CPU-first).

    The heavy ML dependency is imported lazily so MCP users on OpenAI or
    ``local-hash`` never pay for it. The model is constructed once per provider
    instance; no network call happens at module import time (a first-run model
    download may happen when the model is constructed, unless ``local_files_only``
    is set). Vectors are normalized so cosine semantics match the existing
    indexes, and are validated (finite, non-empty, rectangular) before they can
    reach SQLite or the ANN graph.

    The provider never logs input text, tokens, or full vectors — only
    provider/model/device and batch counts at diagnostic level.
    """

    def __init__(
        self,
        model_name: str,
        *,
        cache_folder: Path | str | None = None,
        local_files_only: bool = False,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("local batch size must be positive")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; install with `pip install .[local]`"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        kwargs: dict[str, Any] = {}
        if cache_folder is not None:
            kwargs["cache_folder"] = str(cache_folder)
        if local_files_only:
            kwargs["local_files_only"] = True
        if device is not None:
            kwargs["device"] = device
        self._model: Any = SentenceTransformer(model_name, **kwargs)
        LOGGER.debug(
            "local embedding provider ready model=%s device=%s batch_size=%s",
            model_name,
            device or "default",
            batch_size,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import numpy as np

        raw = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
        )
        matrix = np.asarray(raw, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError(
                "local provider returned an unexpected embedding shape "
                f"{getattr(matrix, 'shape', None)} for {len(texts)} inputs"
            )
        vectors: list[list[float]] = []
        for row in matrix:
            vec = [float(x) for x in row.tolist()]
            if not vec or any(not math.isfinite(x) for x in vec):
                raise ValueError("local provider produced an empty or non-finite embedding")
            vectors.append(vec)
        dim = len(vectors[0])
        if any(len(v) != dim for v in vectors):
            raise ValueError("local provider produced ragged embeddings")
        LOGGER.debug(
            "local embedding batch encoded count=%s dim=%s model=%s",
            len(vectors),
            dim,
            self.model_name,
        )
        return vectors


_LOCAL_ALIASES = {"local", "sentence-transformers", "sentence_transformers"}
_OPENAI_ALIASES = {"openai", "openai-api"}
_HASH_ALIASES = {"local-hash", "hash"}


def canonical_provider_name(provider: str) -> str:
    """Collapse provider spelling variants to a single stored identity value.

    Keeps the embedding provenance in ``embeddings.provider`` stable regardless
    of how the operator spelled the config (``sentence-transformers`` -> ``local``)."""
    normalized = provider.strip().lower()
    if normalized in _LOCAL_ALIASES:
        return "local"
    if normalized in _OPENAI_ALIASES:
        return "openai"
    if normalized in _HASH_ALIASES:
        return "local-hash"
    return normalized


def build_embedding_provider(
    provider: str,
    model: str,
    *,
    local_model_cache: Path | str | None = None,
    local_files_only: bool = False,
    local_device: str | None = None,
    local_batch_size: int = 32,
) -> EmbeddingProvider:
    normalized = provider.strip().lower()
    if normalized in _OPENAI_ALIASES:
        return OpenAIEmbeddingProvider(model=model)
    if normalized in _HASH_ALIASES:
        return HashEmbeddingProvider()
    if normalized in _LOCAL_ALIASES:
        return SentenceTransformerEmbeddingProvider(
            model,
            cache_folder=local_model_cache,
            local_files_only=local_files_only,
            device=local_device,
            batch_size=local_batch_size,
        )
    raise ValueError(f"unsupported embedding provider: {provider}")
