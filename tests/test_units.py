from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile

import pytest

from mnema_memory.config import AppConfig
from mnema_memory.ids import generate_memory_id, slugify
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


def test_generate_memory_id_is_time_sortable_and_unique() -> None:
    early = generate_memory_id(datetime(2026, 1, 1, tzinfo=timezone.utc))
    late = generate_memory_id(datetime(2026, 12, 31, tzinfo=timezone.utc))
    assert early < late
    ids = {generate_memory_id(datetime(2026, 1, 1, tzinfo=timezone.utc)) for _ in range(100)}
    assert len(ids) == 100  # uuid suffix guarantees uniqueness within same second


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Hello World", "hello-world"),
        ("  Trailing/Slashes!!  ", "trailing-slashes"),
        ("***", "memory"),
        ("a" * 80, "a" * 48),
    ],
)
def test_slugify(value: str, expected: str) -> None:
    assert slugify(value) == expected


def test_build_note_path_layout() -> None:
    svc = build_service()
    path = svc._build_note_path(
        "agent-x", "episode", datetime(2026, 7, 9, 13, 5, 1, tzinfo=timezone.utc), "my-slug", "mid123"
    )
    parts = path.parts
    assert "agents" in parts and "agent-x" in parts and "episodes" in parts
    assert "2026" in parts and "07" in parts
    assert path.name.endswith("--my-slug--mid123.md")
    summary_path = svc._build_note_path(
        "agent-x", "summary", datetime(2026, 7, 9, tzinfo=timezone.utc), "s", "sid"
    )
    assert "summaries" in summary_path.parts
    svc.close()


def test_namespace_traversal_rejected() -> None:
    svc = build_service()
    with pytest.raises(ValueError):
        svc.router.call(
            "memory.remember",
            {"namespace": "../etc/passwd", "agent_id": "a", "content": "x"},
        )
    svc.close()


def test_recall_importance_breaks_ties() -> None:
    # Identical content + identical timestamp => identical vector and recency scores,
    # so the higher-importance memory must rank first via the importance boost.
    svc = build_service()
    ts = "2026-01-01T00:00:00+00:00"
    svc.router.call(
        "memory.remember",
        {
            "namespace": "org/proj/dev",
            "agent_id": "high",
            "content": "shared content",
            "importance": 0.9,
            "timestamp": ts,
        },
    )
    svc.router.call(
        "memory.remember",
        {
            "namespace": "org/proj/dev",
            "agent_id": "low",
            "content": "shared content",
            "importance": 0.1,
            "timestamp": ts,
        },
    )
    recalled = svc.router.call(
        "memory.recall",
        {"namespace": "org/proj/dev", "query": "shared content", "top_k": 2},
    )
    assert len(recalled["items"]) == 2
    assert recalled["items"][0]["agent_id"] == "high"
    svc.close()
