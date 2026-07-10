from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mnema_memory.config import AppConfig
from mnema_memory.embeddings import EmbeddingProvider
from mnema_memory.service import MemoryService


BACKENDS = ["numpy", "hnsw"]


def _config(root: Path, *, provider: str, model: str, backend: str) -> AppConfig:
    return AppConfig(
        vault_root=root / "vault",
        sqlite_path=root / "mnema.sqlite3",
        embedding_provider=provider,
        embedding_model=model,
        vector_backend=backend,
    )


def _vector_count(service: MemoryService, namespace: str) -> int:
    return service.conn.execute(
        "SELECT COUNT(*) AS n FROM embedding_vectors WHERE namespace=? AND vector IS NOT NULL",
        (namespace,),
    ).fetchone()["n"]


class FailingProvider(EmbeddingProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated provider outage")


@pytest.mark.parametrize("backend", BACKENDS)
def test_local_recall_ranks_related_content(backend: str, install_fake_st) -> None:
    install_fake_st(dim=32)
    root = Path(tempfile.mkdtemp())
    svc = MemoryService(_config(root, provider="local", model="m", backend=backend))
    try:
        svc.router.call(
            "memory.remember",
            {"namespace": "ns/a", "agent_id": "ag", "content": "alpha beta gamma delta"},
        )
        svc.router.call(
            "memory.remember",
            {"namespace": "ns/a", "agent_id": "ag", "content": "zeta eta theta iota"},
        )
        recalled = svc.router.call(
            "memory.recall",
            {"namespace": "ns/a", "agent_id": "ag", "query": "alpha beta gamma", "top_k": 2},
        )
        assert len(recalled["items"]) == 2
        # The token-overlapping memory ranks first.
        assert "alpha" in recalled["items"][0]["excerpt"]
    finally:
        svc.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_local_dedup_collapses_same_tokens(backend: str, install_fake_st) -> None:
    install_fake_st(dim=32)
    root = Path(tempfile.mkdtemp())
    svc = MemoryService(_config(root, provider="local", model="m", backend=backend))
    try:
        first = svc.router.call(
            "memory.remember",
            {"namespace": "ns/a", "agent_id": "ag", "content": "alpha beta gamma"},
        )
        # Same tokens, different byte order -> identical vector, distinct hash.
        second = svc.router.call(
            "memory.remember",
            {"namespace": "ns/a", "agent_id": "ag", "content": "gamma beta alpha"},
        )
        assert second["embedding_status"] == "duplicate"
        assert second["memory_id"] == first["memory_id"]
    finally:
        svc.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_identity_mismatch_blocks_recall_and_remember(backend: str, install_fake_st) -> None:
    root = Path(tempfile.mkdtemp())
    # Namespace first written by local-hash (dim 64).
    svc_a = MemoryService(_config(root, provider="local-hash", model="text-embedding-3-small", backend=backend))
    svc_a.router.call(
        "memory.remember",
        {"namespace": "ns/a", "agent_id": "ag", "content": "alpha beta gamma"},
    )
    svc_a.close()

    # Reopen with a real local model (fake, dim 32) against the same store.
    install_fake_st(dim=32)
    svc_b = MemoryService(_config(root, provider="local", model="m", backend=backend))
    try:
        with pytest.raises(ValueError, match="identity mismatch"):
            svc_b.router.call(
                "memory.recall",
                {"namespace": "ns/a", "agent_id": "ag", "query": "alpha beta"},
            )
        with pytest.raises(ValueError, match="identity mismatch"):
            svc_b.router.call(
                "memory.remember",
                {"namespace": "ns/a", "agent_id": "ag", "content": "new content"},
            )
    finally:
        svc_b.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_reembed_migrates_namespace(backend: str, install_fake_st) -> None:
    root = Path(tempfile.mkdtemp())
    svc_a = MemoryService(_config(root, provider="local-hash", model="text-embedding-3-small", backend=backend))
    svc_a.router.call(
        "memory.remember",
        {"namespace": "ns/a", "agent_id": "ag", "content": "alpha beta gamma"},
    )
    svc_a.router.call(
        "memory.remember",
        {"namespace": "ns/a", "agent_id": "ag", "content": "delta epsilon zeta"},
    )
    svc_a.close()

    install_fake_st(dim=32)
    svc_b = MemoryService(_config(root, provider="local", model="m", backend=backend))
    try:
        result = svc_b.reembed("ns/a")
        assert result["changed"] is True
        assert result["reembedded"] == 2
        assert result["failed"] == 0
        assert result["provider"] == "local"
        assert result["dim"] == 32

        # Recall now works under the new identity.
        recalled = svc_b.router.call(
            "memory.recall",
            {"namespace": "ns/a", "agent_id": "ag", "query": "alpha beta gamma", "top_k": 2},
        )
        assert len(recalled["items"]) == 2

        status = svc_b.embedding_status()
        ns_rows = [n for n in status["namespaces"] if n["namespace"] == "ns/a"]
        assert ns_rows == [
            {"namespace": "ns/a", "provider": "local", "model": "m", "dim": 32, "count": 2}
        ]

        # Re-running is idempotent.
        again = svc_b.reembed("ns/a")
        assert again["changed"] is False
        assert again["reembedded"] == 0
    finally:
        svc_b.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_reembed_failure_leaves_namespace_intact(backend: str, install_fake_st) -> None:
    install_fake_st(dim=32)
    root = Path(tempfile.mkdtemp())
    svc = MemoryService(_config(root, provider="local", model="m", backend=backend))
    try:
        svc.router.call(
            "memory.remember",
            {"namespace": "ns/a", "agent_id": "ag", "content": "alpha beta gamma"},
        )
        before = _vector_count(svc, "ns/a")
        assert before == 1

        svc.embedding_provider = FailingProvider()
        result = svc.reembed("ns/a")
        assert result["changed"] is False
        assert result["failed"] == 1
        # No mutation on failure: old vectors still present.
        assert _vector_count(svc, "ns/a") == before

        # Recall still works once a working provider is restored.
        install_fake_st(dim=32)
        from mnema_memory.embeddings import SentenceTransformerEmbeddingProvider

        svc.embedding_provider = SentenceTransformerEmbeddingProvider("m")
        recalled = svc.router.call(
            "memory.recall",
            {"namespace": "ns/a", "agent_id": "ag", "query": "alpha beta gamma"},
        )
        assert len(recalled["items"]) == 1
    finally:
        svc.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_reembed_excludes_forgotten_and_survives_rebuild(backend: str, install_fake_st) -> None:
    install_fake_st(dim=32)
    root = Path(tempfile.mkdtemp())
    svc = MemoryService(_config(root, provider="local", model="m", backend=backend))
    try:
        keep = svc.router.call(
            "memory.remember",
            {"namespace": "ns/a", "agent_id": "ag", "content": "alpha beta gamma"},
        )
        drop = svc.router.call(
            "memory.remember",
            {"namespace": "ns/a", "agent_id": "ag", "content": "delta epsilon zeta"},
        )
        svc.router.call("memory.forget", {"namespace": "ns/a", "memory_id": drop["memory_id"]})

        result = svc.reembed("ns/a")
        # Only the one live memory is scanned; the forgotten one is skipped.
        assert result["scanned"] == 1
        assert result["skipped_deleted"] == 1
        # Identity already owned by the live memory -> nothing to rewrite.
        assert result["reembedded"] == 0

        rebuilt = svc.rebuild_index_from_vault()
        assert rebuilt["rebuilt_memories"] >= 1

        recalled = svc.router.call(
            "memory.recall",
            {"namespace": "ns/a", "agent_id": "ag", "query": "alpha beta gamma", "top_k": 5},
        )
        ids = {item["memory_id"] for item in recalled["items"]}
        assert keep["memory_id"] in ids
        assert drop["memory_id"] not in ids
    finally:
        svc.close()
