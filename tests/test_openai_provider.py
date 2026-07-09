from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types

import pytest

from mnema_memory.config import AppConfig
from mnema_memory.embeddings import (
    OpenAIEmbeddingProvider,
    build_embedding_provider,
)
from mnema_memory.service import MemoryService


def install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake `openai` module so the provider path runs without a key/network."""

    class FakeEmbeddings:
        def create(self, model: str, input: list[str]):  # noqa: A002 - mirror openai API
            data = [
                types.SimpleNamespace(embedding=[float(len(text)), 1.0, 2.0]) for text in input
            ]
            return types.SimpleNamespace(data=data)

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.embeddings = FakeEmbeddings()

    module = types.ModuleType("openai")
    module.OpenAI = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_openai_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIEmbeddingProvider(model="text-embedding-3-small")


def test_openai_provider_embeds_via_client(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_openai(monkeypatch)
    provider = build_embedding_provider("openai", "text-embedding-3-small")
    assert isinstance(provider, OpenAIEmbeddingProvider)
    vectors = provider.embed_texts(["ab", "cde"])
    assert vectors == [[2.0, 1.0, 2.0], [3.0, 1.0, 2.0]]


def test_service_end_to_end_with_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_openai(monkeypatch)
    root = Path(tempfile.mkdtemp())
    svc = MemoryService(
        AppConfig(
            vault_root=root / "vault",
            sqlite_path=root / "mnema.sqlite3",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
    )
    created = svc.router.call(
        "memory.remember",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "content": "openai path memory"},
    )
    assert created["embedding_status"] == "completed"
    recalled = svc.router.call(
        "memory.recall",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "query": "openai path memory"},
    )
    assert len(recalled["items"]) == 1
    svc.close()
