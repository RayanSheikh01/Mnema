from __future__ import annotations

from pathlib import Path
import logging
import sqlite3
from datetime import datetime, timezone
import hashlib
from typing import Any

from .config import AppConfig
from .db import bootstrap, connect
from .fileio import write_atomic
from .ids import generate_memory_id, slugify
from .mcp import ToolRouter
from .renderer import render_note
from .schemas import MemoryInput


LOGGER = logging.getLogger("mnema_memory")


class MemoryService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.vault_root.mkdir(parents=True, exist_ok=True)
        self.conn: sqlite3.Connection = connect(config.sqlite_path)
        bootstrap(self.conn)
        self.router = ToolRouter()
        self._register_tools()

    def _register_tools(self) -> None:
        self.router.register("memory.remember", self._remember_tool)
        self.router.register("memory.list", self._list_tool)
        self.router.register("memory.recall", self._recall_tool)
        self.router.register("memory.summarize", self._summarize_tool)
        self.router.register("memory.link", self._link_tool)
        self.router.register("memory.forget", self._forget_tool)

    def _remember_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        memory_input = self._memory_input_from_payload(payload)
        memory_input.validate()
        memory_id = generate_memory_id(memory_input.timestamp)
        title = memory_input.title or memory_input.content.splitlines()[0][:80]
        slug = slugify(title)
        note_path = self._build_note_path(
            memory_input.agent_id,
            memory_input.memory_type,
            memory_input.timestamp,
            slug,
            memory_id,
        )
        frontmatter = {
            "type": memory_input.memory_type,
            "memory_id": memory_id,
            "agent_id": memory_input.agent_id,
            "namespace": memory_input.namespace,
            "session_id": memory_input.session_id,
            "timestamp": memory_input.timestamp,
            "source": memory_input.source,
            "tags": sorted(set(memory_input.tags)),
            "importance": memory_input.importance,
            "embedding_id": None,
            "links": [],
        }
        rendered = render_note(frontmatter, memory_input.content)
        write_atomic(note_path, rendered)
        content_hash = hashlib.sha256(memory_input.content.encode("utf-8")).hexdigest()
        self.conn.execute(
            """
            INSERT INTO memories (id, namespace, agent_id, type, timestamp, title, path, hash, importance, deleted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                memory_id,
                memory_input.namespace,
                memory_input.agent_id,
                memory_input.memory_type,
                memory_input.timestamp.isoformat(),
                title,
                str(note_path),
                content_hash,
                memory_input.importance,
            ),
        )
        for tag in sorted(set(memory_input.tags)):
            self.conn.execute(
                "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                (memory_id, tag),
            )
        self.conn.commit()
        return {
            "memory_id": memory_id,
            "file_path": str(note_path),
            "embedding_status": "pending",
        }

    def _list_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        namespace = str(payload.get("namespace", self.config.default_namespace))
        self._validate_namespace(namespace)
        agent_id = payload.get("agent_id")
        memory_type = payload.get("type")
        include_deleted = bool(payload.get("include_deleted", False))
        params: list[Any] = [namespace]
        where_clauses = ["namespace = ?"]
        if agent_id:
            where_clauses.append("agent_id = ?")
            params.append(str(agent_id))
        if memory_type:
            where_clauses.append("type = ?")
            params.append(str(memory_type))
        if not include_deleted:
            where_clauses.append("deleted_at IS NULL")

        query = (
            "SELECT id, namespace, agent_id, type, timestamp, title, path, importance, deleted_at "
            "FROM memories "
            f"WHERE {' AND '.join(where_clauses)} "
            "ORDER BY timestamp DESC "
            "LIMIT ?"
        )
        params.append(int(payload.get("limit", 50)))
        rows = self.conn.execute(query, params).fetchall()
        return {
            "items": [
                {
                    "memory_id": row["id"],
                    "namespace": row["namespace"],
                    "agent_id": row["agent_id"],
                    "type": row["type"],
                    "timestamp": row["timestamp"],
                    "title": row["title"],
                    "path": row["path"],
                    "importance": row["importance"],
                    "deleted_at": row["deleted_at"],
                }
                for row in rows
            ]
        }

    def _recall_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("memory.recall is not implemented yet")

    def _summarize_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("memory.summarize is not implemented yet")

    def _link_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("memory.link is not implemented yet")

    def _forget_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("memory.forget is not implemented yet")

    def close(self) -> None:
        self.conn.close()

    def _memory_input_from_payload(self, payload: dict[str, Any]) -> MemoryInput:
        timestamp_raw = payload.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_raw)
            if timestamp_raw
            else datetime.now(tz=timezone.utc)
        )
        namespace = str(payload.get("namespace", self.config.default_namespace))
        self._validate_namespace(namespace)
        return MemoryInput(
            namespace=namespace,
            agent_id=str(payload["agent_id"]),
            content=str(payload["content"]),
            title=payload.get("title"),
            session_id=payload.get("session_id"),
            source=str(payload.get("source", "chat")),
            tags=[str(tag) for tag in payload.get("tags", [])],
            importance=float(payload.get("importance", 0.5)),
            memory_type=str(payload.get("type", "episode")),  # type: ignore[arg-type]
            timestamp=timestamp,
        )

    def _build_note_path(
        self,
        agent_id: str,
        memory_type: str,
        timestamp: datetime,
        slug: str,
        memory_id: str,
    ) -> Path:
        year = timestamp.strftime("%Y")
        month = timestamp.strftime("%m")
        base = self.config.vault_root / "agents" / agent_id
        subdir = "episodes" if memory_type == "episode" else "summaries"
        filename = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}--{slug}--{memory_id}.md"
        return base / subdir / year / month / filename

    def _validate_namespace(self, namespace: str) -> None:
        if not namespace.strip():
            raise ValueError("namespace is required")
        if ".." in namespace:
            raise ValueError("namespace contains invalid traversal sequence")
