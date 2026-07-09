from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from mnema_memory.config import AppConfig
from mnema_memory.service import MemoryService


def bench_backend(backend: str, total: int, namespaces: int) -> dict[str, float]:
    root = Path(tempfile.mkdtemp())
    service = MemoryService(
        AppConfig(
            vault_root=root / "vault",
            sqlite_path=root / "bench.sqlite3",
            embedding_provider="local-hash",
            vector_backend=backend,
            dedup_enabled=False,  # isolate index cost from the dedup embed/search
        )
    )
    try:
        start = time.perf_counter()
        for idx in range(total):
            service.router.call(
                "memory.remember",
                {
                    "namespace": f"bench/proj/ns{idx % namespaces}",
                    "agent_id": "agent-perf",
                    "content": f"Memory item {idx}: benchmark sample text for indexing.",
                },
            )
        ingest_elapsed = time.perf_counter() - start

        # Warm one recall, then time a batch to average out noise.
        service.router.call(
            "memory.recall",
            {"namespace": "bench/proj/ns0", "query": "benchmark sample text", "top_k": 10},
        )
        runs = 20
        start = time.perf_counter()
        for r in range(runs):
            service.router.call(
                "memory.recall",
                {
                    "namespace": f"bench/proj/ns{r % namespaces}",
                    "query": "benchmark sample text",
                    "top_k": 10,
                },
            )
        recall_ms = (time.perf_counter() - start) / runs * 1000.0
        return {"ingest_s": ingest_elapsed, "recall_ms": recall_ms}
    finally:
        service.close()


def main() -> None:
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    namespaces = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    backends = ["numpy"]
    try:
        import hnswlib  # noqa: F401

        backends.append("hnsw")
    except ImportError:
        print("hnswlib not installed; skipping hnsw backend (pip install .[ann])")

    print(f"total_memories={total} namespaces={namespaces} per_namespace~={total // namespaces}")
    for backend in backends:
        stats = bench_backend(backend, total, namespaces)
        print(
            f"backend={backend:6s} "
            f"ingest_{total}_s={stats['ingest_s']:.2f} "
            f"recall_avg_ms={stats['recall_ms']:.2f}"
        )


if __name__ == "__main__":
    main()
