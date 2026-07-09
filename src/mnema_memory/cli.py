from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import AppConfig
from .service import MemoryService


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Mnema memory backend")
    parser.add_argument("--config", type=Path, default=None, help="Path to TOML config")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Create a vault/sqlite backup under this directory",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild SQLite index from vault markdown notes",
    )
    parser.add_argument(
        "--drain-embeddings",
        action="store_true",
        help="Retry pending/failed embeddings (e.g. after a provider outage)",
    )
    args = parser.parse_args()

    service = MemoryService(AppConfig.load(args.config))
    if args.backup_dir is not None:
        result = service.backup_to(args.backup_dir)
        print("backup completed:", result["backup_root"])
    elif args.rebuild_index:
        result = service.rebuild_index_from_vault()
        print("rebuild completed:", result["rebuilt_memories"])
    elif args.drain_embeddings:
        result = service.process_pending_embeddings()
        print("embedding drain:", f"recovered={result['recovered']} failed={result['failed']}")
    else:
        print("mnema-memory server initialized")
        print("registered tools:", ", ".join(service.router.tool_names))
    service.close()


if __name__ == "__main__":
    main()
