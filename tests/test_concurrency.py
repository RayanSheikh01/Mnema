from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile

from mnema_memory.config import AppConfig
from mnema_memory.service import MemoryService


def make_service(root: Path) -> MemoryService:
    return MemoryService(
        AppConfig(
            vault_root=root / "vault",
            sqlite_path=root / "mnema.sqlite3",
            embedding_provider="local-hash",
        )
    )


def count_episode_notes(root: Path) -> int:
    return sum(1 for _ in (root / "vault").rglob("*.md"))


def test_parallel_distinct_remembers_all_persist() -> None:
    # Each thread is a separate agent connection writing the same vault/db.
    root = Path(tempfile.mkdtemp())
    make_service(root).close()  # bootstrap schema once before the race
    n = 8

    def worker(i: int) -> str:
        svc = make_service(root)
        try:
            return svc.router.call(
                "memory.remember",
                {"namespace": "org/proj/dev", "agent_id": "a", "content": f"memory number {i}"},
            )["memory_id"]
        finally:
            svc.close()

    with ThreadPoolExecutor(max_workers=n) as pool:
        ids = list(pool.map(worker, range(n)))

    assert len(set(ids)) == n
    svc = make_service(root)
    listed = svc.router.call(
        "memory.list", {"namespace": "org/proj/dev", "agent_id": "a", "limit": 100}
    )
    assert len(listed["items"]) == n
    assert count_episode_notes(root) == n
    svc.close()


def test_parallel_identical_content_is_idempotent() -> None:
    # Many concurrent writers with identical content must yield exactly one
    # live memory and one note on disk (no corruption, no duplicates).
    root = Path(tempfile.mkdtemp())
    make_service(root).close()
    n = 8

    def worker(_: int) -> None:
        svc = make_service(root)
        try:
            svc.router.call(
                "memory.remember",
                {"namespace": "org/proj/dev", "agent_id": "a", "content": "the one true memory"},
            )
        finally:
            svc.close()

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(worker, range(n)))

    svc = make_service(root)
    listed = svc.router.call(
        "memory.list", {"namespace": "org/proj/dev", "agent_id": "a", "limit": 100}
    )
    assert len(listed["items"]) == 1
    assert count_episode_notes(root) == 1
    svc.close()
