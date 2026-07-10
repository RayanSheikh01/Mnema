from __future__ import annotations

import math
import sqlite3

import pytest

from mnema_memory.db import bootstrap
from mnema_memory.vector_index import NumpyVectorIndex, build_vector_index


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    bootstrap(conn)
    return conn


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def test_search_returns_exact_top_k_in_score_order() -> None:
    index = NumpyVectorIndex(make_conn())
    vectors = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.9, 0.1, 0.0],
        "c": [0.0, 1.0, 0.0],
        "d": [0.0, 0.0, 1.0],
    }
    for emb_id, vec in vectors.items():
        index.upsert(emb_id, vec, "ns/one")

    query = [1.0, 0.05, 0.0]
    results = index.search(query, top_k=2, namespace="ns/one")

    assert [emb_id for emb_id, _ in results] == ["a", "b"]
    # Scores match a brute-force cosine within float tolerance.
    assert results[0][1] == pytest.approx(_cosine(query, vectors["a"]), abs=1e-5)


def test_search_is_namespace_isolated() -> None:
    index = NumpyVectorIndex(make_conn())
    index.upsert("a", [1.0, 0.0], "ns/one")
    index.upsert("b", [1.0, 0.0], "ns/two")

    results = index.search([1.0, 0.0], top_k=10, namespace="ns/one")

    assert [emb_id for emb_id, _ in results] == ["a"]


def test_search_empty_namespace_returns_empty() -> None:
    index = NumpyVectorIndex(make_conn())
    assert index.search([1.0, 0.0], top_k=5, namespace="ns/empty") == []


def test_upsert_dim_mismatch_raises() -> None:
    index = NumpyVectorIndex(make_conn())
    index.upsert("a", [1.0, 0.0, 0.0], "ns/one")
    with pytest.raises(ValueError, match="does not match existing dim"):
        index.upsert("b", [1.0, 0.0], "ns/one")


def test_upsert_overwrites_same_id_and_refreshes_cache() -> None:
    index = NumpyVectorIndex(make_conn())
    index.upsert("a", [1.0, 0.0], "ns/one")
    index.search([1.0, 0.0], top_k=1, namespace="ns/one")  # warm cache
    index.upsert("a", [0.0, 1.0], "ns/one")

    results = index.search([0.0, 1.0], top_k=1, namespace="ns/one")
    assert results[0][0] == "a"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_delete_removes_from_search_and_blobs() -> None:
    index = NumpyVectorIndex(make_conn())
    for emb_id, vec in {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.9, 0.1]}.items():
        index.upsert(emb_id, vec, "ns/one")
    index.search([1.0, 0.0], top_k=3, namespace="ns/one")  # warm cache

    index.delete("a", "ns/one")

    ids = [emb_id for emb_id, _ in index.search([1.0, 0.0], top_k=3, namespace="ns/one")]
    assert "a" not in ids
    assert ids == ["c", "b"]  # survivors still ranked
    # BLOB is gone, so a cold index rebuilt from the same DB also excludes it.
    cold = NumpyVectorIndex(index.conn)
    assert "a" not in [e for e, _ in cold.search([1.0, 0.0], top_k=3, namespace="ns/one")]


def test_build_vector_index_defaults_to_numpy() -> None:
    index = build_vector_index("numpy", make_conn())
    assert isinstance(index, NumpyVectorIndex)
    index_default = build_vector_index("inmemory", make_conn())
    assert isinstance(index_default, NumpyVectorIndex)


def test_build_vector_index_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported vector backend"):
        build_vector_index("bogus", make_conn())
