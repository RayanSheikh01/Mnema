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

## Vector search backends

Recall is powered by a pluggable vector index, selected with
`MNEMA_VECTOR_BACKEND` (or `vector_backend` in TOML):

- `numpy` (default) — exact cosine over float32 vectors, vectorized per
  namespace. No extra native deps; scales comfortably to ~100k memories per
  namespace.
- `hnsw` — approximate nearest-neighbour via hnswlib, one HNSW graph per
  namespace, persisted as sidecar files beside the SQLite DB and rebuilt from
  stored vectors on a cold start. Sub-linear; for large namespaces. Requires
  the `[ann]` extra:

```powershell
pip install -e .[ann]
$env:MNEMA_VECTOR_BACKEND = "hnsw"
```

Both backends partition search by namespace, so a small namespace never scans
the global index. Tunables (hnsw): `MNEMA_HNSW_M`, `MNEMA_HNSW_EF_CONSTRUCTION`,
`MNEMA_HNSW_EF`.

## Deduplication

`remember` collapses near-duplicates: after the exact content-hash check, it
compares the new memory's embedding against the same agent's existing memories
in the namespace and, above `MNEMA_DEDUP_THRESHOLD` (default `0.95`), returns
the existing memory with `embedding_status="duplicate"` instead of writing a
new note. Disable with `MNEMA_DEDUP_ENABLED=false`.

## Test

```powershell
$env:PYTHONPATH = "src"
pytest
```

## Performance baseline

Benchmarks ingest + average recall latency across namespaces for each
available backend. Optional args: total memories, namespace count.

```powershell
$env:PYTHONPATH = "src"
python scripts/perf_baseline.py 4000 4
```

Sample run (4000 memories, 4 namespaces, ~1000 each, local-hash provider):

```text
backend=numpy  ingest_4000_s=65.06 recall_avg_ms=14.58
backend=hnsw   ingest_4000_s=103.18 recall_avg_ms=9.01
```

`hnsw` trades higher ingest cost (graph construction) for lower, flatter recall
latency as namespaces grow; `numpy` recall stays exact and cheap up to ~100k
per namespace. Numbers vary by machine.
