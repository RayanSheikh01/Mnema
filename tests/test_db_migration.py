from __future__ import annotations

import sqlite3

from mnema_memory.db import bootstrap


def _pre_v2_conn() -> sqlite3.Connection:
    """A database whose embedding_vectors predates the v2 namespace/BLOB columns,
    matching the v1 schema: vector_json is NOT NULL."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE embedding_vectors (
            embedding_id TEXT PRIMARY KEY,
            vector_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO embedding_vectors (embedding_id, vector_json) VALUES ('legacy', '[0.1]')"
    )
    conn.commit()
    return conn


def test_bootstrap_migrates_pre_v2_embedding_vectors() -> None:
    conn = _pre_v2_conn()
    # Must not raise "no such column: namespace" — the namespace index has to be
    # created only after the migration adds the column.
    bootstrap(conn)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(embedding_vectors)")}
    assert {"namespace", "dim", "vector"} <= columns
    indexes = {row["name"] for row in conn.execute("PRAGMA index_list(embedding_vectors)")}
    assert "idx_embedding_vectors_namespace" in indexes
    # Legacy rows survive the table rebuild.
    assert conn.execute(
        "SELECT vector_json FROM embedding_vectors WHERE embedding_id='legacy'"
    ).fetchone()["vector_json"] == "[0.1]"


def test_migrated_table_accepts_blob_only_insert() -> None:
    # The v1 vector_json NOT NULL constraint must be gone, or persist_vector's
    # BLOB-only insert (used by rebuild) fails.
    conn = _pre_v2_conn()
    bootstrap(conn)
    conn.execute(
        "INSERT INTO embedding_vectors (embedding_id, namespace, dim, vector) "
        "VALUES ('new', 'ns/a', 2, ?)",
        (b"\x00\x01",),
    )
    conn.commit()  # must not raise NOT NULL constraint failed: vector_json


def test_bootstrap_is_idempotent_on_pre_v2_db() -> None:
    conn = _pre_v2_conn()
    bootstrap(conn)
    bootstrap(conn)  # second run must be a no-op, not an error
