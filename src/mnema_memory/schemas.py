from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

MemoryType = Literal["episode", "summary"]


@dataclass(frozen=True)
class MemoryInput:
    namespace: str
    agent_id: str
    content: str
    title: str | None = None
    session_id: str | None = None
    source: str = "chat"
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    memory_type: MemoryType = "episode"
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def validate(self) -> None:
        if not self.namespace.strip():
            raise ValueError("namespace is required")
        if not self.agent_id.strip():
            raise ValueError("agent_id is required")
        if not self.content.strip():
            raise ValueError("content is required")
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError("importance must be between 0 and 1")
        if self.memory_type not in {"episode", "summary"}:
            raise ValueError("memory_type must be episode or summary")
