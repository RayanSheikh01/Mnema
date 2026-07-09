from __future__ import annotations

from pathlib import Path
import tempfile

from mnema_memory.config import AppConfig
from mnema_memory.service import MemoryService


def build_service() -> MemoryService:
    root = Path(tempfile.mkdtemp())
    return MemoryService(
        AppConfig(
            vault_root=root / "vault",
            sqlite_path=root / "mnema.sqlite3",
            embedding_provider="local-hash",
        )
    )


def test_remember_and_list_roundtrip() -> None:
    svc = build_service()
    created = svc.router.call(
        "memory.remember",
        {
            "namespace": "org/proj/dev",
            "agent_id": "agent-x",
            "content": "The user prefers concise answers.",
            "tags": ["style"],
        },
    )
    listed = svc.router.call(
        "memory.list",
        {"namespace": "org/proj/dev", "agent_id": "agent-x"},
    )
    assert created["memory_id"] == listed["items"][0]["memory_id"]
    svc.close()


def test_recall_returns_ranked_items() -> None:
    svc = build_service()
    svc.router.call(
        "memory.remember",
        {
            "namespace": "org/proj/dev",
            "agent_id": "agent-x",
            "content": "User enjoys mountain travel and skiing trips.",
            "tags": ["travel"],
        },
    )
    svc.router.call(
        "memory.remember",
        {
            "namespace": "org/proj/dev",
            "agent_id": "agent-x",
            "content": "User likes dark editor themes.",
            "tags": ["ux"],
        },
    )
    recalled = svc.router.call(
        "memory.recall",
        {
            "namespace": "org/proj/dev",
            "agent_id": "agent-x",
            "query": "travel mountains",
            "top_k": 2,
        },
    )
    assert len(recalled["items"]) == 2
    assert recalled["items"][0]["score"] >= recalled["items"][1]["score"]
    svc.close()


def test_namespace_isolation() -> None:
    svc = build_service()
    svc.router.call(
        "memory.remember",
        {"namespace": "a/b/c", "agent_id": "agent-x", "content": "Namespace A memory"},
    )
    svc.router.call(
        "memory.remember",
        {"namespace": "x/y/z", "agent_id": "agent-x", "content": "Namespace B memory"},
    )
    listed = svc.router.call("memory.list", {"namespace": "a/b/c", "agent_id": "agent-x"})
    assert len(listed["items"]) == 1
    assert listed["items"][0]["namespace"] == "a/b/c"
    svc.close()


def test_summarize_link_and_forget() -> None:
    svc = build_service()
    a = svc.router.call(
        "memory.remember",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "content": "Use JWT access tokens."},
    )
    b = svc.router.call(
        "memory.remember",
        {
            "namespace": "org/proj/dev",
            "agent_id": "agent-x",
            "content": "Rotate refresh tokens after use.",
        },
    )
    linked = svc.router.call(
        "memory.link",
        {
            "namespace": "org/proj/dev",
            "memory_id_a": a["memory_id"],
            "memory_id_b": b["memory_id"],
            "relation": "related_to",
        },
    )
    summary = svc.router.call(
        "memory.summarize",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "topic": "auth"},
    )
    forgotten = svc.router.call(
        "memory.forget",
        {"namespace": "org/proj/dev", "memory_id": b["memory_id"]},
    )
    listed = svc.router.call("memory.list", {"namespace": "org/proj/dev", "agent_id": "agent-x"})
    assert linked["status"] == "linked"
    assert summary["memory_id"]
    assert forgotten["status"] == "forgotten"
    assert all(item["memory_id"] != b["memory_id"] for item in listed["items"])
    svc.close()


def test_backup_and_rebuild() -> None:
    svc = build_service()
    svc.router.call(
        "memory.remember",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "content": "Backup me."},
    )
    backup = svc.backup_to(svc.config.vault_root.parent / "backups")
    rebuilt = svc.rebuild_index_from_vault()
    assert Path(backup["backup_root"]).exists()
    assert rebuilt["rebuilt_memories"] >= 1
    svc.close()
