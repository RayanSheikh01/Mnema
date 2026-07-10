from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from mnema_memory.config import AppConfig
from mnema_memory.service import MemoryService


def build_service(**overrides: Any) -> MemoryService:
    root = Path(tempfile.mkdtemp())
    cfg: dict[str, Any] = {
        "vault_root": root / "vault",
        "sqlite_path": root / "mnema.sqlite3",
        "embedding_provider": "local-hash",
    }
    cfg.update(overrides)
    return MemoryService(AppConfig(**cfg))


def _remember(svc: MemoryService, content: str, ns: str = "ns/a", agent: str = "ag") -> dict:
    return svc.router.call(
        "memory.remember", {"namespace": ns, "agent_id": agent, "content": content}
    )


def _forget(svc: MemoryService, memory_id: str, ns: str = "ns/a") -> dict:
    return svc.router.call("memory.forget", {"namespace": ns, "memory_id": memory_id})


def _recall_ids(
    svc: MemoryService, query: str, ns: str = "ns/a", agent: str = "ag", top_k: int = 10
) -> list[str]:
    out = svc.router.call(
        "memory.recall",
        {"namespace": ns, "agent_id": agent, "query": query, "top_k": top_k},
    )
    return [item["memory_id"] for item in out["items"]]


def test_forget_removes_memory_from_recall() -> None:
    svc = build_service(dedup_enabled=False)
    try:
        a = _remember(svc, "alpha unique content")
        b = _remember(svc, "beta separate content")
        assert a["memory_id"] in _recall_ids(svc, "alpha unique content")

        _forget(svc, a["memory_id"])

        after = _recall_ids(svc, "alpha unique content")
        assert a["memory_id"] not in after  # vector purged, not just SQL-filtered
        assert b["memory_id"] in after  # survivor still recallable
    finally:
        svc.close()


def test_forget_frees_dedup_slot() -> None:
    # A forgotten memory must not block re-remembering the same content.
    svc = build_service(dedup_enabled=True)
    try:
        a = _remember(svc, "same text to dedup")
        _forget(svc, a["memory_id"])

        again = _remember(svc, "same text to dedup")
        assert again["embedding_status"] != "duplicate"
        assert again["memory_id"] != a["memory_id"]
    finally:
        svc.close()


def test_rebuild_keeps_memory_forgotten() -> None:
    svc = build_service(dedup_enabled=False)
    try:
        a = _remember(svc, "content to forget then rebuild")
        _forget(svc, a["memory_id"])
        svc.rebuild_index_from_vault()

        assert a["memory_id"] not in _recall_ids(svc, "content to forget then rebuild")
        row = svc.conn.execute(
            "SELECT deleted_at FROM memories WHERE id=?", (a["memory_id"],)
        ).fetchone()
        assert row is not None and row["deleted_at"]  # tombstone survived rebuild
    finally:
        svc.close()


def test_unforget_restores_recall() -> None:
    svc = build_service(dedup_enabled=False)
    try:
        a = _remember(svc, "recoverable memory content")
        _forget(svc, a["memory_id"])
        assert a["memory_id"] not in _recall_ids(svc, "recoverable memory content")

        svc.router.call(
            "memory.unforget", {"namespace": "ns/a", "memory_id": a["memory_id"]}
        )
        assert a["memory_id"] in _recall_ids(svc, "recoverable memory content")
    finally:
        svc.close()


def test_restore_from_backup_round_trip() -> None:
    svc = build_service(dedup_enabled=False)
    try:
        a = _remember(svc, "memory present at backup time")
        backup = svc.backup_to(Path(tempfile.mkdtemp()))
        b = _remember(svc, "memory added after the backup")
        assert b["memory_id"] in _recall_ids(svc, "memory added after the backup")

        svc.restore_from(Path(backup["backup_root"]))

        assert a["memory_id"] in _recall_ids(svc, "memory present at backup time")
        assert b["memory_id"] not in _recall_ids(svc, "memory added after the backup")
    finally:
        svc.close()
