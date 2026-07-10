from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .vector_index import VectorIndex, delete_vector, persist_vector


class HnswVectorIndex(VectorIndex):
    """Approximate cosine search via hnswlib, one HNSW graph per namespace.

    Durable vectors live in ``embedding_vectors`` (written by persist_vector);
    the graphs are rebuildable caches persisted as per-namespace sidecar files
    next to the SQLite database. A cold start or a missing/corrupt sidecar
    triggers a rebuild from the stored BLOBs. Integer labels required by
    hnswlib are mapped to embedding ids in the ``embedding_labels`` table so
    they stay stable across rebuilds.
    """

    def __init__(self, conn: sqlite3.Connection, config: Any) -> None:
        try:
            import hnswlib  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised via extras
            raise RuntimeError(
                "hnswlib is not installed; install with `pip install .[ann]`"
            ) from exc
        self._hnswlib = hnswlib
        self.conn = conn
        self.m = int(getattr(config, "hnsw_m", 16))
        self.ef_construction = int(getattr(config, "hnsw_ef_construction", 200))
        self.ef = int(getattr(config, "hnsw_ef", 64))
        sqlite_path = getattr(config, "sqlite_path", Path("mnema.db"))
        self._sidecar_dir = Path(str(sqlite_path) + ".hnsw")
        self._indexes: dict[str, Any] = {}

    # ---- public API -------------------------------------------------------

    def upsert(self, embedding_id: str, vector: list[float], namespace: str) -> None:
        persist_vector(self.conn, embedding_id, vector, namespace)
        index = self._ensure_index(namespace, dim=len(vector))
        label = self._label_for(namespace, embedding_id)
        self._grow_if_needed(index, extra=1)
        index.add_items(
            np.asarray([vector], dtype=np.float32), np.asarray([label], dtype=np.int64)
        )
        self._save(namespace, index)

    def search(
        self, query_vector: list[float], top_k: int, namespace: str
    ) -> list[tuple[str, float]]:
        index = self._ensure_index(namespace)
        # Live count comes from the BLOB table, not the graph: mark_deleted
        # leaves tombstones in get_current_count(), and asking knn_query for more
        # neighbours than live elements raises "Cannot return ... 2D array".
        live = self._stored_count(namespace)
        if index is None or live == 0:
            return []
        k = min(top_k, live)
        index.set_ef(max(self.ef, k))
        labels, distances = index.knn_query(
            np.asarray([query_vector], dtype=np.float32), k=k
        )
        id_by_label = self._ids_for_labels(namespace, labels[0].tolist())
        results: list[tuple[str, float]] = []
        for label, distance in zip(labels[0], distances[0]):
            embedding_id = id_by_label.get(int(label))
            if embedding_id is None:
                continue
            # hnswlib 'cosine' space returns distance = 1 - cosine_similarity.
            results.append((embedding_id, 1.0 - float(distance)))
        return results

    def delete(self, embedding_id: str, namespace: str) -> None:
        # Tombstone the element in the live graph so knn_query stops returning
        # it, and persist the tombstone into the sidecar (hnswlib preserves
        # mark_deleted across save/load). Purge the BLOB + label rows last so a
        # cold rebuild from BLOBs also excludes it.
        row = self.conn.execute(
            "SELECT label FROM embedding_labels WHERE embedding_id=?",
            (embedding_id,),
        ).fetchone()
        if row is not None:
            index = self._ensure_index(namespace)
            if index is not None:
                try:
                    index.mark_deleted(int(row["label"]))
                    self._save(namespace, index)
                except RuntimeError:
                    # Label absent from a cold graph or already deleted — the
                    # BLOB purge below still removes it from any future rebuild.
                    pass
        delete_vector(self.conn, embedding_id)

    def reset(self) -> None:
        self._indexes.clear()
        if self._sidecar_dir.exists():
            shutil.rmtree(self._sidecar_dir, ignore_errors=True)

    # ---- index lifecycle --------------------------------------------------

    def _ensure_index(self, namespace: str, dim: int | None = None) -> Any | None:
        if namespace in self._indexes:
            return self._indexes[namespace]
        resolved_dim = dim or self._stored_dim(namespace)
        if resolved_dim is None:
            return None  # empty namespace, nothing to search yet
        index = self._load_or_build(namespace, resolved_dim)
        self._indexes[namespace] = index
        return index

    def _load_or_build(self, namespace: str, dim: int) -> Any:
        count = self._stored_count(namespace)
        capacity = max(count * 2, 1024)
        index = self._hnswlib.Index(space="cosine", dim=dim)
        sidecar = self._sidecar_path(namespace)
        if sidecar.exists():
            try:
                index.load_index(str(sidecar), max_elements=capacity)
                index.set_ef(self.ef)
                return index
            except Exception:  # pragma: no cover - corrupt sidecar -> rebuild
                pass
        index.init_index(
            max_elements=capacity, M=self.m, ef_construction=self.ef_construction
        )
        index.set_ef(self.ef)
        self._add_stored_vectors(namespace, index)
        return index

    def _add_stored_vectors(self, namespace: str, index: Any) -> None:
        rows = self.conn.execute(
            "SELECT embedding_id, vector FROM embedding_vectors "
            "WHERE namespace=? AND vector IS NOT NULL",
            (namespace,),
        ).fetchall()
        if not rows:
            return
        vectors = np.vstack(
            [np.frombuffer(row["vector"], dtype=np.float32) for row in rows]
        )
        labels = np.asarray(
            [self._label_for(namespace, row["embedding_id"]) for row in rows],
            dtype=np.int64,
        )
        self._grow_if_needed(index, extra=len(rows))
        index.add_items(vectors, labels)

    def _grow_if_needed(self, index: Any, extra: int) -> None:
        needed = index.get_current_count() + extra
        if needed > index.get_max_elements():
            index.resize_index(max(needed, index.get_max_elements() * 2))

    def _save(self, namespace: str, index: Any) -> None:
        self._sidecar_dir.mkdir(parents=True, exist_ok=True)
        index.save_index(str(self._sidecar_path(namespace)))

    def _sidecar_path(self, namespace: str) -> Path:
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
        return self._sidecar_dir / f"{digest}.bin"

    # ---- label mapping ----------------------------------------------------

    def _label_for(self, namespace: str, embedding_id: str) -> int:
        row = self.conn.execute(
            "SELECT label FROM embedding_labels WHERE embedding_id=?",
            (embedding_id,),
        ).fetchone()
        if row is not None:
            return int(row["label"])
        nxt = self.conn.execute(
            "SELECT COALESCE(MAX(label), -1) + 1 AS nxt FROM embedding_labels WHERE namespace=?",
            (namespace,),
        ).fetchone()["nxt"]
        self.conn.execute(
            "INSERT INTO embedding_labels (embedding_id, namespace, label) VALUES (?, ?, ?)",
            (embedding_id, namespace, int(nxt)),
        )
        self.conn.commit()
        return int(nxt)

    def _ids_for_labels(self, namespace: str, labels: list[int]) -> dict[int, str]:
        if not labels:
            return {}
        placeholders = ",".join("?" for _ in labels)
        rows = self.conn.execute(
            f"SELECT label, embedding_id FROM embedding_labels "
            f"WHERE namespace=? AND label IN ({placeholders})",
            [namespace, *[int(label) for label in labels]],
        ).fetchall()
        return {int(row["label"]): row["embedding_id"] for row in rows}

    # ---- helpers ----------------------------------------------------------

    def _stored_dim(self, namespace: str) -> int | None:
        row = self.conn.execute(
            "SELECT dim FROM embedding_vectors WHERE namespace=? AND vector IS NOT NULL LIMIT 1",
            (namespace,),
        ).fetchone()
        return int(row["dim"]) if row is not None else None

    def _stored_count(self, namespace: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS n FROM embedding_vectors WHERE namespace=? AND vector IS NOT NULL",
            (namespace,),
        ).fetchone()["n"]
