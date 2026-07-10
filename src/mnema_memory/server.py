"""Minimal MCP stdio server exposing the memory tools to external apps.

Implements the Model Context Protocol over newline-delimited JSON-RPC 2.0 on
stdin/stdout with zero third-party dependencies, so any MCP client (Claude
Desktop, IDE extensions, etc.) can drive the same tools the in-process
``ToolRouter`` exposes.

MCP tool names must match ``^[a-zA-Z0-9_-]{1,64}$`` (no dots), so the router's
dotted names (``memory.remember``) are surfaced with underscores
(``memory_remember``) and translated on dispatch.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, TextIO

from .config import AppConfig
from .service import MemoryService

LOGGER = logging.getLogger("mnema_memory.server")

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "mnema-memory"
SERVER_VERSION = "0.1.0"

_STRING = {"type": "string"}
_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}

# Each entry: MCP tool name -> (router name, description, JSON Schema properties, required).
TOOLS: dict[str, dict[str, Any]] = {
    "memory_remember": {
        "router": "memory.remember",
        "description": "Store a memory (episode or summary) in the vault and index it for recall.",
        "properties": {
            "agent_id": _STRING,
            "content": _STRING,
            "namespace": _STRING,
            "title": _STRING,
            "type": {"type": "string", "enum": ["episode", "summary"]},
            "tags": _STRING_ARRAY,
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
            "source": _STRING,
            "session_id": _STRING,
            "timestamp": _STRING,
            "links": _STRING_ARRAY,
            "request_id": _STRING,
        },
        "required": ["agent_id", "content"],
    },
    "memory_list": {
        "router": "memory.list",
        "description": "List stored memories in a namespace, newest first.",
        "properties": {
            "namespace": _STRING,
            "agent_id": _STRING,
            "type": {"type": "string", "enum": ["episode", "summary"]},
            "include_deleted": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": [],
    },
    "memory_recall": {
        "router": "memory.recall",
        "description": "Semantic recall of memories ranked by similarity, recency, importance and tags.",
        "properties": {
            "query": _STRING,
            "namespace": _STRING,
            "agent_id": _STRING,
            "top_k": {"type": "integer", "minimum": 1},
            "tags": _STRING_ARRAY,
        },
        "required": ["query"],
    },
    "memory_summarize": {
        "router": "memory.summarize",
        "description": "Summarize recent or specified memories into a new summary memory.",
        "properties": {
            "agent_id": _STRING,
            "namespace": _STRING,
            "memory_ids": _STRING_ARRAY,
            "limit": {"type": "integer", "minimum": 1},
            "topic": _STRING,
            "title": _STRING,
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["agent_id"],
    },
    "memory_link": {
        "router": "memory.link",
        "description": "Create a bidirectional link between two memories.",
        "properties": {
            "memory_id_a": _STRING,
            "memory_id_b": _STRING,
            "relation": _STRING,
            "namespace": _STRING,
        },
        "required": ["memory_id_a", "memory_id_b"],
    },
    "memory_forget": {
        "router": "memory.forget",
        "description": "Soft-delete a memory by id (purges its vectors; survives a rebuild).",
        "properties": {
            "memory_id": _STRING,
            "namespace": _STRING,
        },
        "required": ["memory_id"],
    },
    "memory_unforget": {
        "router": "memory.unforget",
        "description": "Reverse a forget: clear the tombstone and re-embed so the memory returns to recall.",
        "properties": {
            "memory_id": _STRING,
            "namespace": _STRING,
        },
        "required": ["memory_id"],
    },
}


class MCPServer:
    """Dispatches JSON-RPC messages to the underlying MemoryService router."""

    def __init__(self, service: MemoryService) -> None:
        self.service = service

    def tools_list(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": spec["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": spec["properties"],
                    "required": spec["required"],
                },
            }
            for name, spec in TOOLS.items()
        ]

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Return a JSON-RPC response, or None for notifications (no id)."""
        method = message.get("method")
        msg_id = message.get("id")
        # Notifications carry no id and expect no response.
        if msg_id is None and method is not None and method.startswith("notifications/"):
            return None
        try:
            result = self._dispatch(method, message.get("params") or {})
        except _RpcError as exc:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": exc.code, "message": str(exc)}}
        except Exception as exc:  # noqa: BLE001 - surface as JSON-RPC internal error
            LOGGER.exception("internal error handling %s", method)
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32603, "message": str(exc)}}
        if msg_id is None:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _dispatch(self, method: str | None, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self.tools_list()}
        if method == "tools/call":
            return self._call_tool(params)
        raise _RpcError(-32601, f"method not found: {method}")

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        spec = TOOLS.get(name)
        if spec is None:
            raise _RpcError(-32602, f"unknown tool: {name}")
        arguments = params.get("arguments") or {}
        try:
            result = self.service.router.call(spec["router"], arguments)
        except Exception as exc:  # noqa: BLE001 - report tool failure to the client
            LOGGER.exception("tool %s failed", name)
            return {
                "content": [{"type": "text", "text": f"error: {exc}"}],
                "isError": True,
            }
        return {
            "content": [{"type": "text", "text": json.dumps(result, default=str)}],
            "isError": False,
        }


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def serve_stdio(service: MemoryService, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """Run the newline-delimited JSON-RPC loop until stdin closes."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = MCPServer(service)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            LOGGER.warning("dropping malformed JSON line")
            error = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            _write(stdout, error)
            continue
        response = server.handle_message(message)
        if response is not None:
            _write(stdout, response)


def _write(stdout: TextIO, payload: dict[str, Any]) -> None:
    stdout.write(json.dumps(payload) + "\n")
    stdout.flush()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,  # keep stdout clean for the JSON-RPC channel
    )
    # Force UTF-8 on the JSON-RPC channel: on Windows sys.stdin/stdout default
    # to the locale code page (e.g. cp1252), which mangles multi-byte UTF-8 the
    # MCP client sends (an em-dash "—" is decoded as "â€”" and persisted to the vault).
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    config_env = None
    service = MemoryService(AppConfig.load(config_env))
    LOGGER.info("mnema-memory MCP server ready (tools: %s)", ", ".join(TOOLS.keys()))
    try:
        serve_stdio(service)
    finally:
        service.close()


if __name__ == "__main__":
    main()
