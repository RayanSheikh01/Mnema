from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("hnswlib")

from mnema_memory.config import AppConfig
from mnema_memory.db import bootstrap
from mnema_memory.hnsw_index import HnswVectorIndex
from mnema_memory.service import MemoryService


def make_index() -> tuple[HnswVectorIndex, sqlite3.Connection, Path]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    bootstrap(conn)
    tmp = Path(tempfile.mkdtemp())
    config = SimpleNamespace(
        sqlite_path=tmp / "mnema.db",
        hnsw_m=16,
        hnsw_ef_construction=200,
        hnsw_ef=64,
    )
    return HnswVectorIndex(conn, config), conn, tmp


def test_hnsw_top_k_and_namespace_isolation() -> None:
    index, _, _ = make_index()
    index.upsert("a", [1.0, 0.0, 0.0], "ns/one")
    index.upsert("b", [0.9, 0.1, 0.0], "ns/one")
    index.upsert("c", [0.0, 1.0, 0.0], "ns/one")
    index.upsert("z", [1.0, 0.0, 0.0], "ns/two")

    results = index.search([1.0, 0.05, 0.0], top_k=2, namespace="ns/one")
    ids = [emb_id for emb_id, _ in results]

    assert ids == ["a", "b"]
    assert "z" not in ids  # ns/two is a separate graph
    assert results[0][1] == pytest.approx(0.99875, abs=5e-3)  # cosine(query, a)


def test_hnsw_empty_namespace_returns_empty() -> None:
    index, _, _ = make_index()
    assert index.search([1.0, 0.0], top_k=5, namespace="ns/none") == []


def test_hnsw_persistence_roundtrip() -> None:
    index, conn, tmp = make_index()
    index.upsert("a", [1.0, 0.0], "ns/one")
    index.upsert("b", [0.0, 1.0], "ns/one")

    # A fresh instance over the same DB + sidecar dir must recover the graph.
    config = SimpleNamespace(
        sqlite_path=tmp / "mnema.db", hnsw_m=16, hnsw_ef_construction=200, hnsw_ef=64
    )
    reopened = HnswVectorIndex(conn, config)
    results = reopened.search([1.0, 0.0], top_k=1, namespace="ns/one")
    assert results[0][0] == "a"


def test_hnsw_cold_rebuild_from_blobs() -> None:
    import shutil

    index, conn, tmp = make_index()
    index.upsert("a", [1.0, 0.0], "ns/one")
    index.upsert("b", [0.0, 1.0], "ns/one")

    # Delete the sidecar entirely: the graph must be rebuilt from stored BLOBs.
    shutil.rmtree(tmp / "mnema.db.hnsw", ignore_errors=True)
    config = SimpleNamespace(
        sqlite_path=tmp / "mnema.db", hnsw_m=16, hnsw_ef_construction=200, hnsw_ef=64
    )
    rebuilt = HnswVectorIndex(conn, config)
    results = rebuilt.search([0.0, 1.0], top_k=1, namespace="ns/one")
    assert results[0][0] == "b"


def test_hnsw_delete_survives_persistence_and_cold_rebuild() -> None:
    import shutil

    index, conn, tmp = make_index()
    index.upsert("a", [1.0, 0.0], "ns/one")
    index.upsert("b", [0.0, 1.0], "ns/one")

    index.delete("a", "ns/one")
    assert [e for e, _ in index.search([1.0, 0.0], top_k=2, namespace="ns/one")] == ["b"]

    # Reload from the sidecar: the mark_deleted tombstone must persist.
    config = SimpleNamespace(
        sqlite_path=tmp / "mnema.db", hnsw_m=16, hnsw_ef_construction=200, hnsw_ef=64
    )
    reopened = HnswVectorIndex(conn, config)
    assert "a" not in [e for e, _ in reopened.search([1.0, 0.0], top_k=2, namespace="ns/one")]

    # Cold rebuild from BLOBs (sidecar gone) must also exclude it — the BLOB was purged.
    shutil.rmtree(tmp / "mnema.db.hnsw", ignore_errors=True)
    rebuilt = HnswVectorIndex(conn, config)
    assert "a" not in [e for e, _ in rebuilt.search([1.0, 0.0], top_k=2, namespace="ns/one")]


def _hnsw_service() -> MemoryService:
    root = Path(tempfile.mkdtemp())
    return MemoryService(
        AppConfig(
            vault_root=root / "vault",
            sqlite_path=root / "mnema.sqlite3",
            embedding_provider="local-hash",
            vector_backend="hnsw",
            dedup_enabled=False,
        )
    )


def test_service_hnsw_remember_recall_roundtrip() -> None:
    service = _hnsw_service()
    try:
        service.router.call(
            "memory.remember",
            {"namespace": "ns/a", "agent_id": "ag", "content": "alpha beta gamma"},
        )
        out = service.router.call(
            "memory.recall", {"namespace": "ns/a", "agent_id": "ag", "query": "alpha beta gamma"}
        )
        assert out["items"]
        assert out["items"][0]["excerpt"]
    finally:
        service.close()


def test_service_hnsw_recall_after_rebuild() -> None:
    service = _hnsw_service()
    try:
        service.router.call(
            "memory.remember",
            {"namespace": "ns/a", "agent_id": "ag", "content": "alpha beta gamma"},
        )
        service.rebuild_index_from_vault()
        out = service.router.call(
            "memory.recall", {"namespace": "ns/a", "agent_id": "ag", "query": "alpha beta gamma"}
        )
        assert out["items"]
    finally:
        service.close()
