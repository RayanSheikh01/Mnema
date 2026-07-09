from __future__ import annotations

from pathlib import Path
import logging
import sqlite3
from typing import Any

from .config import AppConfig
from .db import bootstrap, connect
from .mcp import ToolRouter


LOGGER = logging.getLogger("mnema_memory")


class MemoryService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.vault_root.mkdir(parents=True, exist_ok=True)
        self.conn: sqlite3.Connection = connect(config.sqlite_path)
        bootstrap(self.conn)
        self.router = ToolRouter()
        self._register_tools()

    def _register_tools(self) -> None:
        self.router.register("memory.remember", self._remember_tool)
        self.router.register("memory.list", self._list_tool)
        self.router.register("memory.recall", self._recall_tool)
        self.router.register("memory.summarize", self._summarize_tool)
        self.router.register("memory.link", self._link_tool)
        self.router.register("memory.forget", self._forget_tool)

    def _remember_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("memory.remember is not implemented yet")

    def _list_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("memory.list is not implemented yet")

    def _recall_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("memory.recall is not implemented yet")

    def _summarize_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("memory.summarize is not implemented yet")

    def _link_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("memory.link is not implemented yet")

    def _forget_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("memory.forget is not implemented yet")

    def close(self) -> None:
        self.conn.close()
