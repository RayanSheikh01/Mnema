from __future__ import annotations

import sys

import pytest

from mnema_memory.embeddings import (
    HashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    build_embedding_provider,
    canonical_provider_name,
)


def test_missing_dependency_reports_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the lazy import to fail as if sentence-transformers is absent.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match=r"pip install \.\[local\]"):
        build_embedding_provider("local", "any-model")


def test_construction_passes_cache_offline_device(install_fake_st) -> None:
    module = install_fake_st(dim=16)
    provider = SentenceTransformerEmbeddingProvider(
        "some/model",
        cache_folder="/tmp/models",
        local_files_only=True,
        device="cpu",
        batch_size=8,
    )
    kwargs = provider._model.init_kwargs
    assert kwargs["cache_folder"] == "/tmp/models"
    assert kwargs["local_files_only"] is True
    assert kwargs["device"] == "cpu"
    assert provider.batch_size == 8
    assert module.SentenceTransformer is not None


def test_encode_flags_batching_order_and_empty(install_fake_st) -> None:
    install_fake_st(dim=16)
    provider = SentenceTransformerEmbeddingProvider("m", batch_size=2)
    assert provider.embed_texts([]) == []

    vectors = provider.embed_texts(["alpha beta", "gamma", "alpha beta"])
    assert len(vectors) == 3
    # Order preserved: identical inputs -> identical vectors.
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]
    # Normalization requested and applied (unit norm).
    from math import isclose

    norm = sum(x * x for x in vectors[0]) ** 0.5
    assert isclose(norm, 1.0, rel_tol=1e-5)
    kwargs = type(provider._model).last_encode_kwargs
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["batch_size"] == 2


def test_non_finite_output_is_rejected(install_fake_st) -> None:
    install_fake_st(dim=8, mode="nan")
    provider = SentenceTransformerEmbeddingProvider("m")
    with pytest.raises(ValueError, match="non-finite"):
        provider.embed_texts(["anything"])


def test_zero_batch_size_rejected(install_fake_st) -> None:
    install_fake_st(dim=8)
    with pytest.raises(ValueError, match="batch size"):
        SentenceTransformerEmbeddingProvider("m", batch_size=0)


@pytest.mark.parametrize("alias", ["local", "sentence-transformers", "sentence_transformers"])
def test_aliases_resolve_to_local_provider(alias: str, install_fake_st) -> None:
    install_fake_st(dim=8)
    provider = build_embedding_provider(alias, "m")
    assert isinstance(provider, SentenceTransformerEmbeddingProvider)


@pytest.mark.parametrize("alias", ["local-hash", "hash"])
def test_hash_aliases_still_resolve(alias: str) -> None:
    assert isinstance(build_embedding_provider(alias, "m"), HashEmbeddingProvider)


def test_canonical_provider_name() -> None:
    assert canonical_provider_name("sentence-transformers") == "local"
    assert canonical_provider_name("SENTENCE_TRANSFORMERS") == "local"
    assert canonical_provider_name("openai-api") == "openai"
    assert canonical_provider_name("hash") == "local-hash"
    assert canonical_provider_name("weird") == "weird"
