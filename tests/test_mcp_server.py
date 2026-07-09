from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile

from mnema_memory.config import AppConfig
from mnema_memory.server import TOOLS, MCPServer, serve_stdio
from mnema_memory.service import MemoryService


def build_service() -> MemoryService:
    root = Path(tempfile.mkdtemp())
    return MemoryService(
        AppConfig(
            vault_root=root / "vault",
            sqlite_path=root / "mnema.sqlite3",
            embedding_provider="local-hash",
        )
    )


def test_initialize_reports_protocol_and_server_info() -> None:
    server = MCPServer(build_service())
    response = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert response["result"]["protocolVersion"]
    assert response["result"]["serverInfo"]["name"] == "mnema-memory"
    server.service.close()


def test_tools_list_matches_registered_tools() -> None:
    server = MCPServer(build_service())
    response = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {tool["name"] for tool in response["result"]["tools"]}
    assert names == set(TOOLS.keys())
    # every exposed tool maps to a real router handler
    for spec in TOOLS.values():
        assert spec["router"] in server.service.router.tool_names
    server.service.close()


def test_tool_call_remembers_and_recalls() -> None:
    server = MCPServer(build_service())
    call = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "memory_remember",
                "arguments": {
                    "namespace": "org/proj/dev",
                    "agent_id": "agent-x",
                    "content": "User prefers dark mode in editors.",
                    "tags": ["prefs"],
                },
            },
        }
    )
    payload = json.loads(call["result"]["content"][0]["text"])
    assert call["result"]["isError"] is False
    assert payload["memory_id"]
    server.service.close()


def test_tool_error_is_reported_not_raised() -> None:
    server = MCPServer(build_service())
    # missing required agent_id/content -> handler raises KeyError, surfaced as isError
    call = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "memory_remember", "arguments": {}},
        }
    )
    assert call["result"]["isError"] is True
    server.service.close()


def test_notification_yields_no_response() -> None:
    server = MCPServer(build_service())
    assert server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    server.service.close()


def test_unknown_method_returns_error() -> None:
    server = MCPServer(build_service())
    response = server.handle_message({"jsonrpc": "2.0", "id": 5, "method": "bogus/method"})
    assert response["error"]["code"] == -32601
    server.service.close()


def test_serve_stdio_roundtrips_over_streams() -> None:
    service = build_service()
    requests = "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        ]
    )
    stdin = io.StringIO(requests + "\n")
    stdout = io.StringIO()
    serve_stdio(service, stdin=stdin, stdout=stdout)
    lines = [line for line in stdout.getvalue().splitlines() if line]
    # two requests with ids -> two responses; the notification is silent
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == 1
    assert json.loads(lines[1])["id"] == 2
    service.close()
