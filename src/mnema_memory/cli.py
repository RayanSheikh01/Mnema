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
        "--restore-dir",
        type=Path,
        default=None,
        help="Restore vault/sqlite from a backup created by --backup-dir (destructive overwrite)",
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
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as an MCP stdio server so external apps can call the memory tools",
    )
    args = parser.parse_args()

    service = MemoryService(AppConfig.load(args.config))
    if args.serve:
        from .server import serve_stdio

        serve_stdio(service)
        service.close()
        return
    if args.backup_dir is not None:
        result = service.backup_to(args.backup_dir)
        print("backup completed:", result["backup_root"])
    elif args.restore_dir is not None:
        result = service.restore_from(args.restore_dir)
        print("restore completed:")
        print("  vault replaced:", result["vault_root"])
        print("  sqlite replaced:", result["sqlite_path"])
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
