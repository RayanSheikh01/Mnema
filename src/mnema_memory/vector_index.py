from __future__ import annotations

from abc import ABC, abstractmethod
import json
import math
import sqlite3


class VectorIndex(ABC):
    @abstractmethod
    def upsert(self, embedding_id: str, vector: list[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        raise NotImplementedError


class SQLiteVectorIndex(VectorIndex):
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def upsert(self, embedding_id: str, vector: list[float]) -> None:
        self.conn.execute(
            """
            INSERT INTO embedding_vectors (embedding_id, vector_json)
            VALUES (?, ?)
            ON CONFLICT(embedding_id) DO UPDATE SET vector_json=excluded.vector_json
            """,
            (embedding_id, json.dumps(vector)),
        )
        self.conn.commit()

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        rows = self.conn.execute(
            "SELECT embedding_id, vector_json FROM embedding_vectors"
        ).fetchall()
        scored: list[tuple[str, float]] = []
        for row in rows:
            vector = json.loads(row["vector_json"])
            score = _cosine_similarity(query_vector, vector)
            scored.append((row["embedding_id"], score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return dot / (left_norm * right_norm)
