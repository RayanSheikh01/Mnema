from __future__ import annotations

from pathlib import Path
import sqlite3


def connect(sqlite_path: Path) -> sqlite3.Connection:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # WAL + busy_timeout let multiple agent connections write the same DB
    # concurrently without "database is locked" errors.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def bootstrap(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            title TEXT NOT NULL,
            path TEXT NOT NULL,
            hash TEXT NOT NULL,
            importance REAL NOT NULL,
            deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS memory_tags (
            memory_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES memories(id)
        );

        CREATE TABLE IF NOT EXISTS memory_links (
            src_id TEXT NOT NULL,
            dst_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            FOREIGN KEY(src_id) REFERENCES memories(id),
            FOREIGN KEY(dst_id) REFERENCES memories(id)
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            embedding_id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            checksum TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            FOREIGN KEY(memory_id) REFERENCES memories(id)
        );

        CREATE TABLE IF NOT EXISTS embedding_vectors (
            embedding_id TEXT PRIMARY KEY,
            vector_json TEXT NOT NULL,
            FOREIGN KEY(embedding_id) REFERENCES embeddings(embedding_id)
        );

        CREATE TABLE IF NOT EXISTS ingest_jobs (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            status TEXT NOT NULL,
            retries INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS summary_jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            retries INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        -- Concurrency-safe idempotency: at most one live memory per
        -- (namespace, agent, type, content-hash). Enforced at the DB level so
        -- racing writers cannot both insert the same content.
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_dedupe
            ON memories(namespace, agent_id, type, hash)
            WHERE deleted_at IS NULL;
        """
    )
    conn.commit()
