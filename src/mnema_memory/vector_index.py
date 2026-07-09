from __future__ import annotations

from abc import ABC, abstractmethod
import sqlite3

import numpy as np


class VectorIndex(ABC):
    @abstractmethod
    def upsert(self, embedding_id: str, vector: list[float], namespace: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int, namespace: str
    ) -> list[tuple[str, float]]:
        raise NotImplementedError

    def reset(self) -> None:
        """Drop any derived in-memory/on-disk state. Durable BLOBs are the
        source of truth; called after a full index rebuild."""


def _as_float32(vector: list[float]) -> np.ndarray:
    return np.asarray(vector, dtype=np.float32)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def guard_dim(conn: sqlite3.Connection, namespace: str, dim: int) -> None:
    """Raise if ``dim`` disagrees with vectors already stored for ``namespace``."""
    row = conn.execute(
        "SELECT dim FROM embedding_vectors WHERE namespace=? AND vector IS NOT NULL LIMIT 1",
        (namespace,),
    ).fetchone()
    if row is not None and row["dim"] != dim:
        raise ValueError(
            f"embedding dim {dim} does not match existing dim {row['dim']} "
            f"for namespace '{namespace}'"
        )


def persist_vector(
    conn: sqlite3.Connection, embedding_id: str, vector: list[float], namespace: str
) -> None:
    """Write a vector as a float32 BLOB — the durable source of truth for every
    backend. ANN graphs are rebuildable caches derived from these rows."""
    guard_dim(conn, namespace, len(vector))
    blob = _as_float32(vector).tobytes()
    conn.execute(
        """
        INSERT INTO embedding_vectors (embedding_id, namespace, dim, vector)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(embedding_id) DO UPDATE SET
            namespace=excluded.namespace, dim=excluded.dim, vector=excluded.vector
        """,
        (embedding_id, namespace, len(vector), blob),
    )
    conn.commit()


class NumpyVectorIndex(VectorIndex):
    """Exact cosine search backed by float32 BLOBs in SQLite.

    Vectors are the source of truth in ``embedding_vectors``; searches load only
    the requested namespace's rows and score them with a single vectorized
    matmul. A per-namespace matrix cache, keyed on row count, avoids re-reading
    BLOBs on repeated queries while still picking up new inserts.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._cache: dict[str, tuple[int, list[str], np.ndarray]] = {}

    def upsert(self, embedding_id: str, vector: list[float], namespace: str) -> None:
        persist_vector(self.conn, embedding_id, vector, namespace)
        self._cache.pop(namespace, None)

    def reset(self) -> None:
        self._cache.clear()

    def search(
        self, query_vector: list[float], top_k: int, namespace: str
    ) -> list[tuple[str, float]]:
        ids, matrix = self._namespace_matrix(namespace)
        if not ids:
            return []
        query = _as_float32(query_vector)
        if query.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"query dim {query.shape[0]} does not match namespace '{namespace}' "
                f"index dim {matrix.shape[1]}"
            )
        norm = float(np.linalg.norm(query)) or 1.0
        scores = matrix @ (query / norm)
        k = min(top_k, len(ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(ids[i], float(scores[i])) for i in top]

    def _namespace_matrix(self, namespace: str) -> tuple[list[str], np.ndarray]:
        count = self.conn.execute(
            "SELECT COUNT(*) AS n FROM embedding_vectors "
            "WHERE namespace=? AND vector IS NOT NULL",
            (namespace,),
        ).fetchone()["n"]
        cached = self._cache.get(namespace)
        if cached is not None and cached[0] == count:
            return cached[1], cached[2]
        rows = self.conn.execute(
            "SELECT embedding_id, dim, vector FROM embedding_vectors "
            "WHERE namespace=? AND vector IS NOT NULL",
            (namespace,),
        ).fetchall()
        ids = [row["embedding_id"] for row in rows]
        if not rows:
            matrix = np.empty((0, 0), dtype=np.float32)
        else:
            matrix = np.vstack(
                [np.frombuffer(row["vector"], dtype=np.float32) for row in rows]
            )
            matrix = _normalize_rows(matrix)
        self._cache[namespace] = (count, ids, matrix)
        return ids, matrix


def build_vector_index(backend: str, conn: sqlite3.Connection, config=None) -> VectorIndex:
    normalized = backend.strip().lower()
    if normalized in {"numpy", "inmemory"}:
        return NumpyVectorIndex(conn)
    if normalized in {"hnsw", "ann"}:
        from .hnsw_index import HnswVectorIndex  # local import: optional hnswlib dep

        return HnswVectorIndex(conn, config)
    raise ValueError(f"unsupported vector backend: {backend}")
