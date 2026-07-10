from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mnema_memory.config import AppConfig
from mnema_memory.service import MemoryService


def build_service(**overrides: Any) -> MemoryService:
    root = Path(tempfile.mkdtemp())
    config: dict[str, Any] = {
        "vault_root": root / "vault",
        "sqlite_path": root / "mnema.sqlite3",
        "embedding_provider": "local-hash",
        "dedup_enabled": False,
    }
    config.update(overrides)
    return MemoryService(AppConfig(**config))


def remember(svc: MemoryService, content: str, **payload: Any) -> dict[str, Any]:
    return svc.router.call(
        "memory.remember",
        {"namespace": "ns/a", "agent_id": "agent", "content": content, **payload},
    )


def importance(svc: MemoryService, memory_id: str) -> float:
    row = svc.conn.execute("SELECT importance FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return float(row[0])


def test_auto_importance_is_opt_in_and_explicit_value_wins() -> None:
    off = build_service()
    on = build_service(auto_importance=True)
    try:
        assert importance(off, remember(off, "short")["memory_id"]) == 0.5
        short = remember(on, "short")
        rich = remember(on, "long " * 500, tags=["one", "two"], type="summary")
        explicit = remember(on, "long " * 500, importance=0.12)
        assert importance(on, rich["memory_id"]) > importance(on, short["memory_id"])
        assert importance(on, explicit["memory_id"]) == 0.12
    finally:
        off.close()
        on.close()


def test_recency_weight_and_half_life_change_ranking() -> None:
    svc = build_service(
        rank_weight_vector=0,
        rank_weight_importance=0,
        rank_weight_tag=0,
        rank_weight_recency=1,
    )
    try:
        old = remember(svc, "same query old", timestamp=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat())
        fresh = remember(svc, "same query fresh")
        payload = {"namespace": "ns/a", "agent_id": "agent", "query": "same query", "top_k": 2}
        result = svc.router.call("memory.recall", payload)
        assert [item["memory_id"] for item in result["items"]] == [fresh["memory_id"], old["memory_id"]]
        svc.config = AppConfig(**{**svc.config.__dict__, "recency_half_life_days": 100.0})
        flatter = svc.router.call("memory.recall", payload)
        assert flatter["items"][1]["score"] > result["items"][1]["score"]
    finally:
        svc.close()


def test_retention_dry_run_scope_and_reversibility() -> None:
    svc = build_service(retention_enabled=True, retention_max_age_days=30, retention_min_importance=0.4)
    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    try:
        sweep = remember(svc, "sweep me", timestamp=old, importance=0.2)
        summary = remember(svc, "keep summary", timestamp=old, importance=0.1, type="summary")
        keep = remember(svc, "keep important", timestamp=old, importance=0.9)
        other = svc.router.call("memory.remember", {"namespace": "ns/b", "agent_id": "agent", "content": "other namespace", "timestamp": old, "importance": 0.1})
        dry = svc.apply_retention(namespace="ns/a", dry_run=True)
        assert dry["forgotten"] == 0
        assert [item["memory_id"] for item in dry["candidates"]] == [sweep["memory_id"]]
        assert svc.conn.execute("SELECT deleted_at FROM memories WHERE id = ?", (sweep["memory_id"],)).fetchone()[0] is None
        assert svc.apply_retention(namespace="ns/a")["forgotten"] == 1
        for memory in (summary, keep, other):
            assert svc.conn.execute("SELECT deleted_at FROM memories WHERE id = ?", (memory["memory_id"],)).fetchone()[0] is None
        svc.router.call("memory.unforget", {"namespace": "ns/a", "memory_id": sweep["memory_id"]})
        svc.rebuild_index_from_vault()
        assert svc.conn.execute("SELECT deleted_at FROM memories WHERE id = ?", (sweep["memory_id"],)).fetchone()[0] is None
    finally:
        svc.close()


def test_retention_is_disabled_by_default() -> None:
    svc = build_service(retention_max_age_days=0, retention_min_importance=1)
    try:
        memory = remember(svc, "not swept", importance=0.1)
        assert svc.apply_retention()["forgotten"] == 0
        assert svc.conn.execute("SELECT deleted_at FROM memories WHERE id = ?", (memory["memory_id"],)).fetchone()[0] is None
    finally:
        svc.close()
