from __future__ import annotations

from pathlib import Path
import logging
import sqlite3
from datetime import datetime, timezone
import hashlib
import uuid
from typing import Any

from .config import AppConfig
from .db import bootstrap, connect
from .embeddings import EmbeddingProvider, build_embedding_provider
from .fileio import file_lock, write_atomic
from .ids import generate_memory_id, slugify
from .mcp import ToolRouter
from .renderer import render_note
from .schemas import MemoryInput
from .vector_index import VectorIndex, build_vector_index


LOGGER = logging.getLogger("mnema_memory")


class MemoryService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.vault_root.mkdir(parents=True, exist_ok=True)
        self.conn: sqlite3.Connection = connect(config.sqlite_path)
        bootstrap(self.conn)
        self.embedding_provider: EmbeddingProvider = build_embedding_provider(
            config.embedding_provider, config.embedding_model
        )
        self.vector_index: VectorIndex = build_vector_index(
            config.vector_backend, self.conn, config
        )
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
        request_id = str(payload.get("request_id", uuid.uuid4().hex))
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
            "title": title,
            # Alias to the memory_id so Obsidian resolves [[memory_id]] wikilinks
            # (filenames carry a timestamp/slug prefix and won't match on their own).
            "aliases": [memory_id],
            "agent_id": memory_input.agent_id,
            "namespace": memory_input.namespace,
            "session_id": memory_input.session_id,
            "timestamp": memory_input.timestamp,
            "source": memory_input.source,
            "tags": sorted(set(memory_input.tags)),
            "importance": memory_input.importance,
            "embedding_id": None,
            # Store as [[memory_id]] wikilinks so Obsidian's graph draws edges.
            "links": [f"[[{item}]]" for item in payload.get("links", [])],
        }
        extra_frontmatter = payload.get("extra_frontmatter", {})
        if not isinstance(extra_frontmatter, dict):
            raise ValueError("extra_frontmatter must be an object")
        if "derived_from" in payload:
            extra_frontmatter["derived_from"] = [str(item) for item in payload["derived_from"]]
        if "topics" in payload:
            extra_frontmatter["topics"] = [str(item) for item in payload["topics"]]
        frontmatter.update(extra_frontmatter)
        content_hash = hashlib.sha256(memory_input.content.encode("utf-8")).hexdigest()
        existing = self.conn.execute(
            """
            SELECT id, path FROM memories
            WHERE namespace=? AND agent_id=? AND type=? AND hash=? AND deleted_at IS NULL
            LIMIT 1
            """,
            (
                memory_input.namespace,
                memory_input.agent_id,
                memory_input.memory_type,
                content_hash,
            ),
        ).fetchone()
        if existing is not None:
            LOGGER.info("remember idempotent hit request_id=%s memory_id=%s", request_id, existing["id"])
            return {
                "memory_id": existing["id"],
                "file_path": existing["path"],
                "embedding_status": "existing",
            }
        rendered = render_note(frontmatter, memory_input.content)
        lock_path = note_path.with_suffix(note_path.suffix + ".lock")
        with file_lock(lock_path):
            write_atomic(note_path, rendered)
        try:
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
        except sqlite3.IntegrityError:
            # A concurrent writer inserted the same content first. Roll back,
            # discard our orphan note, and return the winning memory.
            self.conn.rollback()
            note_path.unlink(missing_ok=True)
            winner = self.conn.execute(
                """
                SELECT id, path FROM memories
                WHERE namespace=? AND agent_id=? AND type=? AND hash=? AND deleted_at IS NULL
                LIMIT 1
                """,
                (
                    memory_input.namespace,
                    memory_input.agent_id,
                    memory_input.memory_type,
                    content_hash,
                ),
            ).fetchone()
            LOGGER.info(
                "remember idempotent race request_id=%s memory_id=%s",
                request_id,
                winner["id"] if winner else None,
            )
            return {
                "memory_id": winner["id"] if winner else memory_id,
                "file_path": winner["path"] if winner else str(note_path),
                "embedding_status": "existing",
            }
        for tag in sorted(set(memory_input.tags)):
            self.conn.execute(
                "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                (memory_id, tag),
            )
        self.conn.commit()
        embedding_id, embedding_status = self._create_embedding_record(
            memory_id, memory_input.content
        )
        LOGGER.info(
            "remember completed request_id=%s memory_id=%s status=%s",
            request_id,
            memory_id,
            embedding_status,
        )
        return {
            "memory_id": memory_id,
            "file_path": str(note_path),
            "embedding_status": embedding_status,
            "embedding_id": embedding_id,
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
        namespace = str(payload.get("namespace", self.config.default_namespace))
        self._validate_namespace(namespace)
        query = str(payload["query"])
        top_k = int(payload.get("top_k", 10))
        agent_id = payload.get("agent_id")
        query_vector = self.embedding_provider.embed_texts([query])[0]
        candidate_scores = self.vector_index.search(query_vector, top_k * 5, namespace)
        if not candidate_scores:
            return {"items": []}
        score_by_embedding = {embedding_id: score for embedding_id, score in candidate_scores}
        placeholders = ",".join("?" for _ in score_by_embedding.keys())
        params: list[Any] = [*score_by_embedding.keys(), namespace]
        query_sql = (
            "SELECT e.embedding_id, m.id, m.namespace, m.agent_id, m.type, m.timestamp, "
            "m.title, m.path, m.importance "
            "FROM embeddings e JOIN memories m ON m.id = e.memory_id "
            f"WHERE e.embedding_id IN ({placeholders}) AND m.namespace = ? AND m.deleted_at IS NULL"
        )
        if agent_id:
            query_sql += " AND m.agent_id = ?"
            params.append(str(agent_id))
        rows = self.conn.execute(query_sql, params).fetchall()
        query_tags = {str(tag) for tag in payload.get("tags", [])}
        tags_by_memory = self._tags_for_memories([row["id"] for row in rows])
        results: list[dict[str, Any]] = []
        now = datetime.now(tz=timezone.utc)
        for row in rows:
            created_at = datetime.fromisoformat(row["timestamp"])
            age_days = max((now - created_at).total_seconds() / 86400.0, 0.0)
            recency_score = 1.0 / (1.0 + age_days)
            vector_score = score_by_embedding.get(row["embedding_id"], 0.0)
            tag_score = 0.0
            if query_tags:
                overlap = query_tags & tags_by_memory.get(row["id"], set())
                tag_score = len(overlap) / len(query_tags)
            final_score = (
                (0.6 * vector_score)
                + (0.2 * recency_score)
                + (0.1 * row["importance"])
                + (0.1 * tag_score)
            )
            excerpt = self._read_excerpt(Path(row["path"]))
            results.append(
                {
                    "memory_id": row["id"],
                    "embedding_id": row["embedding_id"],
                    "namespace": row["namespace"],
                    "agent_id": row["agent_id"],
                    "type": row["type"],
                    "timestamp": row["timestamp"],
                    "title": row["title"],
                    "path": row["path"],
                    "score": final_score,
                    "excerpt": excerpt,
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return {"items": results[:top_k]}

    def _summarize_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        namespace = str(payload.get("namespace", self.config.default_namespace))
        self._validate_namespace(namespace)
        agent_id = str(payload["agent_id"])
        memory_ids = [str(value) for value in payload.get("memory_ids", [])]
        if not memory_ids:
            rows = self.conn.execute(
                """
                SELECT id FROM memories
                WHERE namespace = ? AND agent_id = ? AND type = 'episode' AND deleted_at IS NULL
                ORDER BY timestamp DESC LIMIT ?
                """,
                (namespace, agent_id, int(payload.get("limit", 10))),
            ).fetchall()
            memory_ids = [row["id"] for row in rows]
        if not memory_ids:
            raise ValueError("no source memories available for summary")
        source_rows = self.conn.execute(
            f"""
            SELECT id, title, path FROM memories
            WHERE id IN ({",".join("?" for _ in memory_ids)}) AND namespace = ? AND agent_id = ? AND deleted_at IS NULL
            """,
            [*memory_ids, namespace, agent_id],
        ).fetchall()
        if not source_rows:
            raise ValueError("no matching memories found for summary")
        bullets: list[str] = []
        for row in source_rows:
            excerpt = self._read_excerpt(Path(row["path"]), max_chars=160)
            bullets.append(f"- {row['title']}: {excerpt}")
        topic = str(payload.get("topic", "session-summary"))
        summary_title = payload.get("title") or f"Summary: {topic}"
        summary_content = "\n".join(
            [
                f"## Summary Topic: {topic}",
                "",
                "### Key Points",
                *bullets,
                "",
                "### Derived From",
                *[f"- [[{row['id']}]]" for row in source_rows],
            ]
        )
        result = self._remember_tool(
            {
                "namespace": namespace,
                "agent_id": agent_id,
                "content": summary_content,
                "title": summary_title,
                "type": "summary",
                "source": "system",
                "tags": ["summary", topic],
                "links": [row["id"] for row in source_rows],
                "derived_from": [row["id"] for row in source_rows],
                "topics": [topic],
                "importance": float(payload.get("importance", 0.7)),
            }
        )
        return result

    def _link_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        namespace = str(payload.get("namespace", self.config.default_namespace))
        self._validate_namespace(namespace)
        src_id = str(payload["memory_id_a"])
        dst_id = str(payload["memory_id_b"])
        relation = str(payload.get("relation", "related_to"))
        src_row = self.conn.execute(
            "SELECT path FROM memories WHERE id=? AND namespace=?",
            (src_id, namespace),
        ).fetchone()
        dst_row = self.conn.execute(
            "SELECT path FROM memories WHERE id=? AND namespace=?",
            (dst_id, namespace),
        ).fetchone()
        if src_row is None or dst_row is None:
            raise ValueError("both memories must exist in namespace before linking")
        self.conn.execute(
            "INSERT INTO memory_links (src_id, dst_id, relation) VALUES (?, ?, ?)",
            (src_id, dst_id, relation),
        )
        self.conn.commit()
        self._upsert_frontmatter_link(Path(src_row["path"]), dst_id)
        self._upsert_frontmatter_link(Path(dst_row["path"]), src_id)
        return {"status": "linked", "memory_id_a": src_id, "memory_id_b": dst_id, "relation": relation}

    def _forget_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        namespace = str(payload.get("namespace", self.config.default_namespace))
        self._validate_namespace(namespace)
        memory_id = str(payload["memory_id"])
        deleted_at = datetime.now(tz=timezone.utc).isoformat()
        cursor = self.conn.execute(
            "UPDATE memories SET deleted_at=? WHERE id=? AND namespace=? AND deleted_at IS NULL",
            (deleted_at, memory_id, namespace),
        )
        self.conn.commit()
        if cursor.rowcount == 0:
            raise ValueError("memory not found or already forgotten")
        return {"status": "forgotten", "memory_id": memory_id, "deleted_at": deleted_at}

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

    def _create_embedding_record(self, memory_id: str, content: str) -> tuple[str, str]:
        """Insert a pending embedding row and process it, returning (id, status)."""
        embedding_id = f"emb-{uuid.uuid4().hex}"
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        created_at = datetime.now(tz=timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO embeddings (embedding_id, memory_id, provider, model, dim, checksum, created_at, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL)
            """,
            (
                embedding_id,
                memory_id,
                self.config.embedding_provider,
                self.config.embedding_model,
                0,
                checksum,
                created_at,
            ),
        )
        self.conn.commit()
        status = self._process_embedding(memory_id, embedding_id, content)
        return embedding_id, status

    def process_pending_embeddings(self, limit: int = 100) -> dict[str, int]:
        """Retry embeddings left pending/failed (e.g. after a provider outage).

        Reads note content from the vault and re-embeds, so a transient
        provider failure never permanently loses a memory's recall vector.
        """
        rows = self.conn.execute(
            """
            SELECT e.embedding_id, e.memory_id, m.path
            FROM embeddings e JOIN memories m ON m.id = e.memory_id
            WHERE e.status != 'completed' AND m.deleted_at IS NULL
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        recovered = 0
        failed = 0
        for row in rows:
            content = self._note_body(Path(row["path"]))
            status = self._process_embedding(row["memory_id"], row["embedding_id"], content)
            if status == "completed":
                recovered += 1
            else:
                failed += 1
        LOGGER.info("embedding drain recovered=%s failed=%s", recovered, failed)
        return {"recovered": recovered, "failed": failed}

    def _note_body(self, note_path: Path) -> str:
        text = note_path.read_text(encoding="utf-8")
        if text.startswith("---\n"):
            parts = text.split("---\n", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return text.strip()

    def _process_embedding(self, memory_id: str, embedding_id: str, content: str) -> str:
        try:
            vector = self.embedding_provider.embed_texts([content])[0]
        except Exception as exc:
            self.conn.execute(
                "UPDATE embeddings SET status='failed', error=? WHERE embedding_id=?",
                (str(exc), embedding_id),
            )
            self.conn.commit()
            LOGGER.exception("Embedding generation failed for memory_id=%s", memory_id)
            return "failed"
        namespace = self._namespace_for_memory(memory_id)
        self.vector_index.upsert(embedding_id, vector, namespace)
        self.conn.execute(
            "UPDATE embeddings SET status='completed', dim=?, error=NULL WHERE embedding_id=?",
            (len(vector), embedding_id),
        )
        self.conn.commit()
        return "completed"

    def _namespace_for_memory(self, memory_id: str) -> str:
        row = self.conn.execute(
            "SELECT namespace FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"memory not found for embedding upsert: {memory_id}")
        return row["namespace"]

    def _tags_for_memories(self, memory_ids: list[str]) -> dict[str, set[str]]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self.conn.execute(
            f"SELECT memory_id, tag FROM memory_tags WHERE memory_id IN ({placeholders})",
            memory_ids,
        ).fetchall()
        tags: dict[str, set[str]] = {}
        for row in rows:
            tags.setdefault(row["memory_id"], set()).add(row["tag"])
        return tags

    def _read_excerpt(self, note_path: Path, max_chars: int = 240) -> str:
        text = note_path.read_text(encoding="utf-8")
        if "---\n" in text:
            parts = text.split("---\n")
            body = "---\n".join(parts[2:]).strip()
        else:
            body = text
        return body[:max_chars]

    def _upsert_frontmatter_link(self, note_path: Path, linked_memory_id: str) -> None:
        text = note_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or lines[0] != "---":
            raise ValueError(f"note missing frontmatter: {note_path}")
        end_index = None
        for idx in range(1, len(lines)):
            if lines[idx] == "---":
                end_index = idx
                break
        if end_index is None:
            raise ValueError(f"note missing frontmatter closing marker: {note_path}")
        link_line_index = None
        for idx in range(1, end_index):
            if lines[idx].startswith("links:"):
                link_line_index = idx
                break
        if link_line_index is None:
            lines.insert(end_index, f'links: ["[[{linked_memory_id}]]"]')
        else:
            current = lines[link_line_index].split(":", 1)[1].strip()
            existing: list[str] = []
            if current.startswith("[") and current.endswith("]"):
                raw_values = current[1:-1].strip()
                if raw_values:
                    for item in raw_values.split(","):
                        # Normalize to the bare id whether stored as "id" or "[[id]]".
                        inner = item.strip().strip('"')
                        if inner.startswith("[[") and inner.endswith("]]"):
                            inner = inner[2:-2]
                        existing.append(inner)
            if linked_memory_id not in existing:
                existing.append(linked_memory_id)
            # Re-serialize as [[id]] wikilinks so Obsidian's graph draws edges.
            serialized = ", ".join(f'"[[{item}]]"' for item in existing)
            lines[link_line_index] = f"links: [{serialized}]"
        write_atomic(note_path, "\n".join(lines) + "\n")

    def backup_to(self, destination_root: Path) -> dict[str, str]:
        # Fold any WAL contents back into the main db file so the copied
        # SQLite snapshot is complete.
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_root = destination_root / f"mnema-backup-{timestamp}"
        vault_backup = backup_root / "vault"
        db_backup = backup_root / "sqlite"
        vault_backup.mkdir(parents=True, exist_ok=True)
        db_backup.mkdir(parents=True, exist_ok=True)
        for source in self.config.vault_root.rglob("*"):
            if source.is_file():
                relative = source.relative_to(self.config.vault_root)
                target = vault_backup / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        db_target = db_backup / self.config.sqlite_path.name
        db_target.write_bytes(self.config.sqlite_path.read_bytes())
        return {
            "backup_root": str(backup_root),
            "vault_backup": str(vault_backup),
            "sqlite_backup": str(db_target),
        }

    def rebuild_index_from_vault(self) -> dict[str, int]:
        self.conn.execute("DELETE FROM memory_tags")
        self.conn.execute("DELETE FROM memory_links")
        self.conn.execute("DELETE FROM embedding_vectors")
        self.conn.execute("DELETE FROM embedding_labels")
        self.conn.execute("DELETE FROM embeddings")
        self.conn.execute("DELETE FROM memories")
        # Drop derived ANN caches/sidecars; they get rebuilt from fresh vectors.
        self.vector_index.reset()
        rebuilt = 0
        for note_path in self.config.vault_root.rglob("*.md"):
            record = self._parse_note(note_path)
            if record is None:
                continue
            rebuilt += 1
            content_hash = hashlib.sha256(record["content"].encode("utf-8")).hexdigest()
            self.conn.execute(
                """
                INSERT INTO memories (id, namespace, agent_id, type, timestamp, title, path, hash, importance, deleted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    record["memory_id"],
                    record["namespace"],
                    record["agent_id"],
                    record["type"],
                    record["timestamp"],
                    record["title"],
                    str(note_path),
                    content_hash,
                    float(record.get("importance", 0.5)),
                ),
            )
            for tag in record.get("tags", []):
                self.conn.execute(
                    "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    (record["memory_id"], str(tag)),
                )
            # Regenerate embeddings + vectors so recall works after a rebuild.
            self._create_embedding_record(record["memory_id"], record["content"])
        self.conn.commit()
        return {"rebuilt_memories": rebuilt}

    def _parse_note(self, note_path: Path) -> dict[str, Any] | None:
        text = note_path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            return None
        parts = text.split("---\n", 2)
        if len(parts) < 3:
            return None
        raw_frontmatter = parts[1].splitlines()
        body = parts[2].strip()
        parsed: dict[str, Any] = {"content": body}
        for line in raw_frontmatter:
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            value = raw_value.strip().strip('"')
            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1].strip()
                parsed[key.strip()] = (
                    [item.strip().strip('"') for item in inner.split(",")] if inner else []
                )
            else:
                parsed[key.strip()] = value
        required = {"memory_id", "namespace", "agent_id", "type", "timestamp"}
        if not required.issubset(parsed.keys()):
            return None
        parsed["title"] = parsed.get("title") or note_path.stem
        return parsed
