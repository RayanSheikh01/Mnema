from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from mnema_memory import fileio
from mnema_memory.config import AppConfig
from mnema_memory.embeddings import EmbeddingProvider, HashEmbeddingProvider
from mnema_memory.service import MemoryService


class FailingProvider(EmbeddingProvider):
    """Embedding provider that always fails, simulating a timeout/outage."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated embedding provider outage")


def build_service() -> MemoryService:
    root = Path(tempfile.mkdtemp())
    return MemoryService(
        AppConfig(
            vault_root=root / "vault",
            sqlite_path=root / "mnema.sqlite3",
            embedding_provider="local-hash",
        )
    )


def test_embedding_provider_failure_is_surfaced_not_silent() -> None:
    svc = build_service()
    svc.embedding_provider = FailingProvider()
    created = svc.router.call(
        "memory.remember",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "content": "persist me"},
    )
    # Memory note + row must still exist even though embedding failed.
    assert created["embedding_status"] == "failed"
    listed = svc.router.call("memory.list", {"namespace": "org/proj/dev", "agent_id": "agent-x"})
    assert len(listed["items"]) == 1
    # Failure recorded explicitly, not swallowed.
    row = svc.conn.execute(
        "SELECT status, error FROM embeddings WHERE memory_id=?", (created["memory_id"],)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"]
    svc.close()


def test_recall_provider_failure_raises_not_empty() -> None:
    svc = build_service()
    svc.router.call(
        "memory.remember",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "content": "recall target"},
    )
    svc.embedding_provider = FailingProvider()
    with pytest.raises(RuntimeError):
        svc.router.call(
            "memory.recall",
            {"namespace": "org/proj/dev", "agent_id": "agent-x", "query": "target"},
        )
    svc.close()


def test_pending_embeddings_recovered_after_outage() -> None:
    # A transient provider outage leaves the embedding 'failed'; once the
    # provider recovers, the retry drain must re-embed and make it recallable.
    svc = build_service()
    svc.embedding_provider = FailingProvider()
    created = svc.router.call(
        "memory.remember",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "content": "delta epsilon"},
    )
    assert created["embedding_status"] == "failed"

    # Provider recovers; before draining, the memory has no vector so recall misses it.
    svc.embedding_provider = HashEmbeddingProvider()
    assert svc.router.call(
        "memory.recall",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "query": "delta epsilon"},
    )["items"] == []

    drained = svc.process_pending_embeddings()
    assert drained["recovered"] == 1
    assert drained["failed"] == 0

    recalled = svc.router.call(
        "memory.recall",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "query": "delta epsilon"},
    )
    assert len(recalled["items"]) == 1
    svc.close()


def test_recall_works_after_index_rebuild() -> None:
    # Rebuilding the index from the vault must restore embeddings/vectors,
    # otherwise recall silently returns nothing after a rebuild.
    svc = build_service()
    svc.router.call(
        "memory.remember",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "content": "alpha beta gamma"},
    )
    before = svc.router.call(
        "memory.recall",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "query": "alpha beta gamma"},
    )
    assert len(before["items"]) == 1

    result = svc.rebuild_index_from_vault()
    assert result["rebuilt_memories"] == 1

    after = svc.router.call(
        "memory.recall",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "query": "alpha beta gamma"},
    )
    assert len(after["items"]) == 1
    svc.close()


def test_partial_write_leaves_original_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(tempfile.mkdtemp())
    note = root / "note.md"
    fileio.write_atomic(note, "original content\n")

    def boom(*_args, **_kwargs):
        raise OSError("simulated crash during rename")

    monkeypatch.setattr(fileio.os, "replace", boom)
    with pytest.raises(OSError):
        fileio.write_atomic(note, "half-written garbage")

    # Canonical file untouched; no temp debris left behind.
    assert note.read_text(encoding="utf-8") == "original content\n"
    assert list(root.glob(".tmp-*")) == []
