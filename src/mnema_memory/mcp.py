from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ToolRouter:
    _handlers: dict[str, ToolHandler] = field(default_factory=dict)

    def register(self, tool_name: str, handler: ToolHandler) -> None:
        if tool_name in self._handlers:
            raise ValueError(f"tool already registered: {tool_name}")
        self._handlers[tool_name] = handler

    def call(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._handlers:
            raise KeyError(f"unknown tool: {tool_name}")
        return self._handlers[tool_name](payload)

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._handlers.keys())
