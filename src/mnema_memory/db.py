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
            namespace TEXT NOT NULL DEFAULT '',
            dim INTEGER NOT NULL DEFAULT 0,
            vector BLOB,
            vector_json TEXT,
            FOREIGN KEY(embedding_id) REFERENCES embeddings(embedding_id)
        );

        -- NOTE: idx_embedding_vectors_namespace is created in
        -- _migrate_embedding_vectors, AFTER the namespace column is ensured.
        -- Creating it here would fail bootstrap on a pre-v2 embedding_vectors
        -- table (no namespace column) before the migration can add it.

        -- Stable integer labels for ANN backends (hnswlib requires int ids).
        CREATE TABLE IF NOT EXISTS embedding_labels (
            embedding_id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            label INTEGER NOT NULL,
            UNIQUE(namespace, label),
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
    _migrate_embedding_vectors(conn)
    conn.commit()


def _migrate_embedding_vectors(conn: sqlite3.Connection) -> None:
    """Bring a pre-existing embedding_vectors table up to the v2 schema.

    Older databases created the table as (embedding_id, vector_json TEXT NOT
    NULL). CREATE TABLE IF NOT EXISTS never alters an existing table, so:
      * backfill the namespace/dim/vector columns, and
      * if the legacy ``vector_json NOT NULL`` constraint is present, rebuild the
        table so BLOB-only inserts (persist_vector never writes vector_json)
        don't hit a NOT NULL violation.
    Legacy vector_json data is unused by the v2 backends (they read the ``vector``
    BLOB); rebuild_index_from_vault repopulates the BLOBs.
    """
    info = conn.execute("PRAGMA table_info(embedding_vectors)").fetchall()
    if not info:
        return
    columns = {row["name"] for row in info}
    if "namespace" not in columns:
        conn.execute("ALTER TABLE embedding_vectors ADD COLUMN namespace TEXT NOT NULL DEFAULT ''")
    if "dim" not in columns:
        conn.execute("ALTER TABLE embedding_vectors ADD COLUMN dim INTEGER NOT NULL DEFAULT 0")
    if "vector" not in columns:
        conn.execute("ALTER TABLE embedding_vectors ADD COLUMN vector BLOB")

    vector_json_notnull = any(
        row["name"] == "vector_json" and row["notnull"] for row in info
    )
    if vector_json_notnull:
        # SQLite can't drop a NOT NULL in place — recreate the table.
        conn.executescript(
            """
            CREATE TABLE embedding_vectors_v2 (
                embedding_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL DEFAULT '',
                dim INTEGER NOT NULL DEFAULT 0,
                vector BLOB,
                vector_json TEXT,
                FOREIGN KEY(embedding_id) REFERENCES embeddings(embedding_id)
            );
            INSERT INTO embedding_vectors_v2 (embedding_id, namespace, dim, vector, vector_json)
                SELECT embedding_id, namespace, dim, vector, vector_json FROM embedding_vectors;
            DROP TABLE embedding_vectors;
            ALTER TABLE embedding_vectors_v2 RENAME TO embedding_vectors;
            """
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_embedding_vectors_namespace "
        "ON embedding_vectors(namespace)"
    )
