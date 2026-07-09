from __future__ import annotations

import tempfile
from pathlib import Path

from mnema_memory.config import AppConfig
from mnema_memory.service import MemoryService


class ConstantProvider:
    """Every text embeds to the same vector -> maximal semantic similarity."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class CounterProvider:
    """Each call yields an orthogonal vector -> no two texts are similar."""

    def __init__(self) -> None:
        self._n = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for _ in texts:
            vec = [0.0, 0.0, 0.0, 0.0]
            vec[self._n % 4] = 1.0
            self._n += 1
            out.append(vec)
        return out


def build_service(dedup_enabled: bool = True) -> MemoryService:
    root = Path(tempfile.mkdtemp())
    return MemoryService(
        AppConfig(
            vault_root=root / "vault",
            sqlite_path=root / "mnema.sqlite3",
            embedding_provider="local-hash",
            dedup_enabled=dedup_enabled,
        )
    )


def _memory_count(service: MemoryService, namespace: str) -> int:
    return service.conn.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE namespace=? AND deleted_at IS NULL",
        (namespace,),
    ).fetchone()["n"]


def test_semantic_near_duplicate_is_collapsed() -> None:
    service = build_service(dedup_enabled=True)
    service.embedding_provider = ConstantProvider()
    try:
        first = service.router.call(
            "memory.remember",
            {"namespace": "ns/a", "agent_id": "ag", "content": "the cat sat on the mat"},
        )
        # Distinct bytes (so the exact-hash gate misses) but identical vector.
        second = service.router.call(
            "memory.remember",
            {"namespace": "ns/a", "agent_id": "ag", "content": "a feline rested upon a rug"},
        )
        assert second["embedding_status"] == "duplicate"
        assert second["memory_id"] == first["memory_id"]
        assert _memory_count(service, "ns/a") == 1
    finally:
        service.close()


def test_distinct_content_is_not_deduped() -> None:
    service = build_service(dedup_enabled=True)
    service.embedding_provider = CounterProvider()
    try:
        service.router.call(
            "memory.remember", {"namespace": "ns/a", "agent_id": "ag", "content": "first thing"}
        )
        second = service.router.call(
            "memory.remember", {"namespace": "ns/a", "agent_id": "ag", "content": "second thing"}
        )
        assert second["embedding_status"] != "duplicate"
        assert _memory_count(service, "ns/a") == 2
    finally:
        service.close()


def test_dedup_disabled_keeps_both() -> None:
    service = build_service(dedup_enabled=False)
    service.embedding_provider = ConstantProvider()
    try:
        service.router.call(
            "memory.remember", {"namespace": "ns/a", "agent_id": "ag", "content": "alpha text"}
        )
        second = service.router.call(
            "memory.remember", {"namespace": "ns/a", "agent_id": "ag", "content": "bravo text"}
        )
        assert second["embedding_status"] != "duplicate"
        assert _memory_count(service, "ns/a") == 2
    finally:
        service.close()


def test_dedup_is_scoped_per_agent() -> None:
    service = build_service(dedup_enabled=True)
    service.embedding_provider = ConstantProvider()
    try:
        service.router.call(
            "memory.remember", {"namespace": "ns/a", "agent_id": "ag1", "content": "shared idea one"}
        )
        # Same namespace + identical vector but a different agent -> not a dup.
        second = service.router.call(
            "memory.remember", {"namespace": "ns/a", "agent_id": "ag2", "content": "shared idea two"}
        )
        assert second["embedding_status"] != "duplicate"
        assert _memory_count(service, "ns/a") == 2
    finally:
        service.close()
