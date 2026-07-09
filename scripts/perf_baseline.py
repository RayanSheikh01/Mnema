from __future__ import annotations

from pathlib import Path
import tempfile
import time

from mnema_memory.config import AppConfig
from mnema_memory.service import MemoryService


def main() -> None:
    root = Path(tempfile.mkdtemp())
    service = MemoryService(
        AppConfig(
            vault_root=root / "vault",
            sqlite_path=root / "bench.sqlite3",
            embedding_provider="local-hash",
        )
    )
    start = time.perf_counter()
    for idx in range(250):
        service.router.call(
            "memory.remember",
            {
                "namespace": "bench/proj/dev",
                "agent_id": "agent-perf",
                "content": f"Memory item {idx}: benchmark sample text for indexing.",
            },
        )
    ingest_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    service.router.call(
        "memory.recall",
        {
            "namespace": "bench/proj/dev",
            "agent_id": "agent-perf",
            "query": "benchmark sample text",
            "top_k": 10,
        },
    )
    recall_elapsed = time.perf_counter() - start
    service.close()
    print(f"ingest_250_seconds={ingest_elapsed:.3f}")
    print(f"recall_10_seconds={recall_elapsed:.3f}")


if __name__ == "__main__":
    main()
