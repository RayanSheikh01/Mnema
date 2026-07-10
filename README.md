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
`memory_summarize`, `memory_link`, `memory_forget`, `memory_unforget`.

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

## Embedding providers

Recall and deduplication run on a pluggable `EmbeddingProvider`, selected with
`MNEMA_EMBEDDING_PROVIDER` (or `embedding_provider` in TOML):

- `openai` (default) — real remote semantic embeddings (`text-embedding-3-small`).
  Needs `OPENAI_API_KEY` and the `[openai]` extra.
- `local` — real **local** semantic embeddings via `sentence-transformers`. No
  API key, and memory contents never leave the machine. Aliases:
  `sentence-transformers`, `sentence_transformers`.
- `local-hash` — deterministic, dependency-free hash vectors. Fast and offline,
  but **not semantically meaningful** — use it only for tests/offline smoke runs.

### Local embeddings

```powershell
pip install -e .[local]
$env:MNEMA_EMBEDDING_PROVIDER = "local"
$env:MNEMA_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, CPU-friendly
```

The model is downloaded once to the standard Hugging Face cache (override with
`MNEMA_LOCAL_MODEL_CACHE`; the cache is never written inside the vault). Extra
knobs: `MNEMA_LOCAL_FILES_ONLY=true` fails fast unless the model is already
cached (fully offline), `MNEMA_LOCAL_DEVICE` forces a device (CPU by default),
`MNEMA_LOCAL_BATCH_SIZE` tunes encode batches.

### Switching a namespace's model (migration)

A namespace stores vectors from exactly one model — cosine scores across models
are meaningless. Mnema records each namespace's `(provider, model, dim)` and
refuses ordinary `remember`/`recall` when the configured identity disagrees with
the stored one, naming the exact fix. Migrate a populated namespace with an
explicit, all-or-nothing re-embed:

```powershell
mnema-memory --embedding-status                       # configured vs stored identity per namespace
mnema-memory --reembed --namespace org/project/dev    # re-embed live memories with the configured model
```

`--reembed` computes and validates every new vector before touching the index,
so a provider failure leaves the old namespace untouched; forgotten
(tombstoned) memories are never re-embedded; re-running with an
already-owned identity reports zero changes.

## Summaries

`memory.summarize` turns episodes into a first-class `summary` note. The prose is
produced by a pluggable `SummaryGenerator`, selected with `MNEMA_SUMMARY_PROVIDER`
(or `summary_provider` in TOML):

- `extractive` (default) — offline, dependency-free bullet excerpts.
- `openai` — chat-completion synthesis (`gpt-4o-mini` default). Needs
  `OPENAI_API_KEY` and the `[openai]` extra.
- `anthropic` / `claude` — Claude synthesis (`claude-haiku-4-5` default — cheap
  for a bounded task). Needs `ANTHROPIC_API_KEY` and the `[anthropic]` extra.

```powershell
pip install -e .[anthropic]
$env:MNEMA_SUMMARY_PROVIDER = "anthropic"   # or "openai"
```

The LLM writes only the summary body; Mnema always composes the topic header and
the `Derived From` wikilinks deterministically from SQLite, so a summary can
never invent a source link. Remote providers send note contents to the API;
`extractive` keeps everything local.

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

## Forgetting & recovery

`forget` is a durable soft delete. It sets `deleted_at` on the SQLite row, purges
the memory's vectors from the active index (numpy cache and hnsw graph, plus the
float32 BLOBs that back a rebuild), and tombstones the canonical note's
frontmatter with `deleted_at:`. Because the vault is canonical, the tombstone is
what makes a forget survive `--rebuild-index`: the forgotten memory is never
re-embedded and never re-enters recall or the dedup gate. `unforget` reverses it
— clears the tombstone and re-embeds so the memory returns to recall.

Back up and restore the whole store from the CLI:

```powershell
mnema-memory --backup-dir C:/backups          # snapshot vault + SQLite
mnema-memory --restore-dir C:/backups/mnema-backup-<ts>   # destructive overwrite
```

`--restore-dir` replaces the live vault and database with the snapshot (any
memory created after the backup is discarded) and drops stale hnsw sidecars,
which rebuild lazily from the restored vectors.

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

## Ranking

Recall combines vector similarity, recency, caller importance, and requested-tag
overlap. Its defaults are `0.6`, `0.2`, `0.1`, and `0.1`; change their relative
influence with `MNEMA_RANK_WEIGHT_VECTOR`, `MNEMA_RANK_WEIGHT_RECENCY`,
`MNEMA_RANK_WEIGHT_IMPORTANCE`, and `MNEMA_RANK_WEIGHT_TAG`. Recency decays
exponentially and halves every `MNEMA_RECENCY_HALF_LIFE_DAYS` (default `1.0`).

Set `MNEMA_AUTO_IMPORTANCE=true` to give memories without an explicit
`importance` a local bounded heuristic based on note length, tags, and whether
the memory is a summary. An explicit `importance` always takes precedence.

## Retention

Retention is disabled by default and never hard-deletes. Once
`MNEMA_RETENTION_ENABLED=true`, run a reversible sweep with:

```powershell
mnema-memory --retention --dry-run
mnema-memory --retention --namespace org/project/dev
```

It forgets only live memories at least `MNEMA_RETENTION_MAX_AGE_DAYS` old
(default `365`) whose importance is below `MNEMA_RETENTION_MIN_IMPORTANCE`
(default `0.25`). Summaries are exempt unless
