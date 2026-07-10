from __future__ import annotations

import hashlib
import sys
import types

import numpy as np
import pytest


def _token_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic bag-of-tokens vector: shared words -> higher cosine.

    This gives a *semantic-ish* signal for tests (paraphrases that share words
    rank together, unrelated text does not) without a real model download. The
    token hash uses hashlib so it is stable across processes (unlike builtin
    ``hash``, which is salted per interpreter)."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).lower().split():
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0
    return vec


def make_fake_sentence_transformers(dim: int = 16, mode: str = "tokens") -> types.ModuleType:
    module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        last_encode_kwargs: dict | None = None

        def __init__(self, model_name: str, **kwargs) -> None:
            self.model_name = model_name
            self.init_kwargs = kwargs
            self.dim = dim

        def encode(
            self,
            texts,
            normalize_embeddings: bool = False,
            convert_to_numpy: bool = True,
            batch_size: int = 32,
        ):
            SentenceTransformer.last_encode_kwargs = {
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
                "batch_size": batch_size,
            }
            rows = []
            for text in texts:
                if mode == "nan":
                    rows.append(np.full(dim, np.nan, dtype=np.float32))
                    continue
                vec = _token_vector(text, dim)
                if normalize_embeddings:
                    norm = float(np.linalg.norm(vec)) or 1.0
                    vec = vec / norm
                rows.append(vec)
            if not rows:
                return np.empty((0, dim), dtype=np.float32)
            return np.vstack(rows)

    module.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    return module


@pytest.fixture
def install_fake_st(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``sentence_transformers`` module into sys.modules.

    Returns an installer so a test can pick the vector dimension / failure mode.
    """

    def _install(dim: int = 16, mode: str = "tokens") -> types.ModuleType:
        module = make_fake_sentence_transformers(dim=dim, mode=mode)
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        return module

    return _install
