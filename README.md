# Mnema Memory

Python MCP-oriented memory backend that stores canonical memory records in an Obsidian vault and uses local indexing for fast retrieval.

## Quick start

```powershell
pip install -e .[test]
$env:PYTHONPATH = "src"
python -m mnema_memory.cli
```

## Connect to other apps (MCP)

Mnema speaks the Model Context Protocol over stdio, so any MCP client (Claude
Desktop, IDE extensions, etc.) can call the memory tools. Zero extra deps.

Run the server manually:

```powershell
$env:PYTHONPATH = "src"
python -m mnema_memory.server
# or, after `pip install -e .`
mnema-memory-mcp
# or via the CLI flag
mnema-memory --serve
```

It reads JSON-RPC 2.0 requests on stdin and writes responses to stdout; logs go
to stderr. Exposed tools: `memory_remember`, `memory_list`, `memory_recall`,
`memory_summarize`, `memory_link`, `memory_forget`.

Register it in an MCP client config (e.g. Claude Desktop
`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "mnema-memory": {
      "command": "mnema-memory-mcp",
      "env": {
        "MNEMA_VAULT_ROOT": "C:/path/to/vault",
        "MNEMA_SQLITE_PATH": "C:/path/to/mnema.db",
        "MNEMA_EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Use `mnema-memory-mcp` only if the package is installed on PATH; otherwise set
`"command": "python"` with `"args": ["-m", "mnema_memory.server"]` and a `PYTHONPATH`
entry in `env`. Set `MNEMA_EMBEDDING_PROVIDER=local-hash` to run without an API key.

## Test

```powershell
$env:PYTHONPATH = "src"
pytest
```

## Performance baseline

```powershell
$env:PYTHONPATH = "src"
python scripts/perf_baseline.py
```
