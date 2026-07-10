from __future__ import annotations

import tempfile
from pathlib import Path

from mnema_memory.config import AppConfig
from mnema_memory.service import MemoryService
from mnema_memory.summarizer import SummaryGenerator


def _build_service(**overrides) -> MemoryService:
    root = Path(tempfile.mkdtemp())
    return MemoryService(
        AppConfig(
            vault_root=root / "vault",
            sqlite_path=root / "mnema.sqlite3",
            embedding_provider="local-hash",
            **overrides,
        )
    )


class FakeLLMGenerator(SummaryGenerator):
    """Records the sources it received and returns synthesized prose."""

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def summarize(self, topic, sources):
        self.seen = sources
        return "### Key Points\n- synthesized decision about " + topic


def _read_note_body(service: MemoryService, memory_id: str) -> str:
    row = service.conn.execute(
        "SELECT path FROM memories WHERE id=?", (memory_id,)
    ).fetchone()
    return Path(row["path"]).read_text(encoding="utf-8")


def test_extractive_default_summarize_links_and_recalls() -> None:
    svc = _build_service()
    try:
        a = svc.router.call(
            "memory.remember",
            {"namespace": "org/p/dev", "agent_id": "ag", "content": "Use JWT access tokens."},
        )
        b = svc.router.call(
            "memory.remember",
            {"namespace": "org/p/dev", "agent_id": "ag", "content": "Rotate refresh tokens."},
        )
        summary = svc.router.call(
            "memory.summarize",
            {"namespace": "org/p/dev", "agent_id": "ag", "topic": "auth"},
        )
        note = _read_note_body(svc, summary["memory_id"])
        assert "## Summary Topic: auth" in note
        assert "### Key Points" in note
        assert "### Derived From" in note
        # Deterministic derived_from links reference both source ids.
        assert a["memory_id"] in note
        assert b["memory_id"] in note
        # Summary is a first-class, recallable memory.
        listed = svc.router.call(
            "memory.list", {"namespace": "org/p/dev", "agent_id": "ag", "type": "summary"}
        )
        assert any(item["memory_id"] == summary["memory_id"] for item in listed["items"])
    finally:
        svc.close()


def test_llm_generator_body_used_links_still_deterministic() -> None:
    svc = _build_service()
    fake = FakeLLMGenerator()
    svc.summary_generator = fake
    try:
        a = svc.router.call(
            "memory.remember",
            {"namespace": "org/p/dev", "agent_id": "ag", "content": "Prefer async IO in the worker."},
        )
        summary = svc.router.call(
            "memory.summarize",
            {"namespace": "org/p/dev", "agent_id": "ag", "topic": "perf"},
        )
        note = _read_note_body(svc, summary["memory_id"])
        # Synthesized LLM body landed in the note...
        assert "synthesized decision about perf" in note
        # ...and the generator received the full source content.
        assert fake.seen and "async IO" in fake.seen[0]["content"]
        # ...while Mnema still wrote the deterministic Derived From link.
        assert "### Derived From" in note
        assert a["memory_id"] in note
    finally:
        svc.close()


def test_config_selects_summary_provider() -> None:
    svc = _build_service(summary_provider="extractive")
    try:
        from mnema_memory.summarizer import ExtractiveSummaryGenerator

        assert isinstance(svc.summary_generator, ExtractiveSummaryGenerator)
    finally:
        svc.close()
