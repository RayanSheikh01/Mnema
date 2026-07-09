from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig
from .service import MemoryService


def main() -> None:
    parser = argparse.ArgumentParser(description="Mnema memory backend")
    parser.add_argument("--config", type=Path, default=None, help="Path to TOML config")
    args = parser.parse_args()

    service = MemoryService(AppConfig.load(args.config))
    print("mnema-memory server initialized")
    print("registered tools:", ", ".join(service.router.tool_names))
    service.close()


if __name__ == "__main__":
    main()
