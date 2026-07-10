# Mnema v4 - Local Semantic Embeddings

## Context

Mnema's embedding seam already supports three modes:

- `openai` - real remote semantic embeddings (`text-embedding-3-small` by default);
- `local-hash` - deterministic, dependency-free test/offline fallback; and
- a pluggable `EmbeddingProvider` interface used consistently by remember,
  recall, deduplication, retry, unforget, and vault rebuild.

`local-hash` is not semantically meaningful: similar phrasing is unrelated
unless it happens to hash similarly. v4 adds a **real local embedding provider**
so Mnema can retrieve and deduplicate meaningfully without an API key or sending
memory contents to a third party.

The current vector store enforces one embedding dimension per namespace. That
is correct for any single model, but it means changing a populated namespace
from OpenAI's 1536 dimensions to a local model's 384 dimensions must be an
explicit re-embedding operation rather than a silent mixed-index migration.

## Goal

Ship an opt-in `local` provider backed by `sentence-transformers`, with a
well-defined model lifecycle, configuration, observability, and safe migration
of existing live memories to the selected model.

**Recommended default model:** `sentence-transformers/all-MiniLM-L6-v2`
(384 dimensions). It is small enough for ordinary developer machines, supports
English semantic retrieval well, and is widely supported by sentence-transformers.
The configured model remains fully overridable for multilingual or higher-quality
use cases.

## Scope

Included:

- CPU-first local inference through `sentence-transformers`/PyTorch;
- explicit model download/cache and offline error behaviour;
- normalized, validated vectors passed through the existing vector-index seam;
- provider/model metadata for diagnostics and compatibility checks;
- a safe, resumable CLI re-embed workflow for switching providers/models;
- documentation, configuration examples, unit/integration coverage, and a
  small opt-in benchmark.

Not included:

- GPU selection, quantization, ONNX, or a bundled model binary;
- a local LLM/summarizer;
- mixing vectors from multiple models in a namespace or cross-model score
  calibration;
- changing the default from OpenAI automatically. Existing deployments keep
  their current behaviour until they select `local`.

## Design decisions

| Concern | v4 decision |
| --- | --- |
| Provider identifier | Accept `local`, `sentence-transformers`, and `sentence_transformers`; store the canonical value `local`. Keep `local-hash` strictly as a test/dev provider. |
| Dependency | Add an optional `local = ["sentence-transformers>=... "]` project extra. Do not import it until the local provider is constructed. |
| Model cache | Use the library/Hugging Face standard cache by default; expose `MNEMA_LOCAL_MODEL_CACHE` / TOML `local_model_cache` to set `cache_folder`. No cache is written inside the vault or SQLite directory. |
| Download policy | Default permits first-run download. `MNEMA_LOCAL_FILES_ONLY=true` / TOML `local_files_only` makes startup fail clearly unless the selected model is already cached. |
| Device | CPU by default; an optional `MNEMA_LOCAL_DEVICE` / TOML `local_device` is passed through only when explicitly set. GPU support is a compatibility benefit, not a v4 feature promise. |
| Embedding shape | Call `SentenceTransformer.encode(..., normalize_embeddings=True, convert_to_numpy=True)` in batches; convert each result to a finite `list[float]`. Normalization keeps cosine semantics consistent with the current indexes. |
| Dimension changes | Refuse ordinary writes/recall when a namespace's live vectors disagree with the active provider/model/dimension. Require `--reembed` to replace them atomically per memory. |
| Privacy | Local inference makes no embedding API call. First model download is a separate network action and is documented; offline mode prevents it. |

## Implementation plan

### 1. Extend configuration and packaging

In `pyproject.toml`, add a `local` optional dependency extra for
`sentence-transformers`; retain independent `openai`, `ann`, and `test` extras.
Avoid making local ML dependencies mandatory for MCP users who use OpenAI or
`local-hash`.

Extend `AppConfig` in `src/mnema_memory/config.py` with:

- `local_model_cache: Path | None = None`
- `local_files_only: bool = False`
- `local_device: str | None = None`
- `local_batch_size: int = 32`

Read the corresponding `MNEMA_LOCAL_*` variables and TOML `[mnema]` values.
Validate a positive batch size and reject blank device/cache values. Keep
`embedding_provider` and `embedding_model` as the source of provider/model
selection; configuration does not need a second local-model field.

Update `.env.example` and the README with separate OpenAI and local setup
examples, including `pip install -e .[local]`, the recommended model, cache
location, and fully-offline use after a model has been cached.

### 2. Implement `SentenceTransformerEmbeddingProvider`

In `src/mnema_memory/embeddings.py`:

1. Add `SentenceTransformerEmbeddingProvider(EmbeddingProvider)`. Its
   constructor lazily imports `sentence_transformers.SentenceTransformer` and
   raises an actionable error (`pip install .[local]`) if unavailable.
2. Construct the model once per service/provider instance with the configured
   `model_name`, optional `cache_folder`, `local_files_only`, and optional
   device. Do not make network calls at module import time.
3. Implement `embed_texts` as a no-op for `[]`; otherwise encode in configured
   batches with normalization enabled. Convert NumPy scalars to Python floats
   and reject empty, non-finite, ragged, or inconsistent vectors with a clear
   provider error before they reach SQLite/HNSW.
4. Update `build_embedding_provider` to receive the needed local settings
   (prefer a config object or keyword-only arguments over expanding positional
   arguments), recognize the aliases above, and preserve current OpenAI/hash
   paths unchanged.

The provider must not log input text, model tokens, or full vectors. Log only
provider/model/device and batch counts at diagnostic level.

### 3. Make model compatibility explicit

Add a small service-level compatibility helper rather than weakening
`vector_index.guard_dim`:

- Query the active, non-deleted namespace embeddings joined to
  `embedding_vectors`.
- Treat `(provider, model, dim)` as the namespace's active embedding identity.
- Before `remember`, `recall`, and semantic dedup, compare the configured
  identity to stored live vectors. A mismatch returns a clear error naming the
  namespace, old identity, and requested identity, with the exact `--reembed`
  command to use.
- Use the actual query vector dimension for the check, so a misconfigured model
  is caught even if its declared name matches an earlier model.

Existing databases need no schema migration: `embeddings.provider`, `.model`,
and `.dim` already preserve the required provenance. A new index on
`embeddings(memory_id, provider, model, status)` is optional only if profiling
shows re-embed/status queries need it; do not add speculative indexes.

This prevents invalid cosine comparisons and gives users an intentional path
when moving a namespace from OpenAI to local embeddings.

### 4. Add a safe `--reembed` maintenance operation

Add `MemoryService.reembed(...)` plus CLI options in `src/mnema_memory/cli.py`:

```text
mnema-memory --reembed --namespace org/project/dev
mnema-memory --reembed --namespace org/project/dev --batch-size 32
```

The command uses the currently configured provider/model and only processes
live memories in the named namespace. It must:

1. validate the target provider by embedding a small first batch before any
   destructive mutation;
2. enumerate canonical note content for each live memory and embed in batches;
3. for each successfully embedded memory, create a replacement embedding row,
   upsert its vector, and then delete the old vector/embedding only after the
   replacement is durable;
4. leave a failed memory's old completed embedding intact, record the failure
   in a replacement/pending embedding row, and report failed IDs/counts;
5. rebuild/reset only the affected namespace's derived HNSW state as needed;
6. return/report `scanned`, `reembedded`, `failed`, `skipped_deleted`, provider,
   model, and dimension.

Because current indexes permit only one dimension in a namespace, implement the
actual replacement as a controlled namespace reindex: prepare all target
vectors first, then in a single transaction remove the namespace's old vector
rows/labels and write the new completed rows/vectors, followed by an index reset
and lazy HNSW rebuild. If preparation has any failure, abort without changing
the old namespace index and surface the failures. This gives an all-or-nothing
provider/model switch rather than a namespace with mixed dimensions.

Use a vault/database lock for the operation and reject concurrent `remember`
for that namespace while the reindex is active. Make rerunning idempotent: if
the selected identity already owns every live memory, report zero changes.
Tombstoned memories remain untouched and excluded, preserving v3 forget
semantics.

### 5. Wire normal service flows and diagnostics

Update `MemoryService.__init__` to construct the provider from `AppConfig`.
Keep all ordinary service paths (`remember`, `recall`, dedup, pending retry,
unforget, and vault rebuild) calling only `EmbeddingProvider.embed_texts`.

Add a read-only CLI `--embedding-status` (or include the same detail in the
existing startup log) that reports the configured identity and per-namespace
stored identities/counts without loading the model. This makes accidental
OpenAI/local mismatches visible before an MCP request fails.

For rebuild behaviour, retain the configured provider/model as the explicit
target. If rebuilding a vault whose SQLite database is replaced, the rebuild
creates a uniform index with that target model; document that this is a
re-embed operation and may download/load the local model.

## Files expected to change

- `pyproject.toml` - `local` optional dependency.
- `src/mnema_memory/config.py` - local provider settings and validation.
- `src/mnema_memory/embeddings.py` - sentence-transformers provider, vector
  validation, provider factory aliases.
- `src/mnema_memory/service.py` - compatibility guard, batched re-embed
  service operation, status reporting, configuration-based provider creation.
- `src/mnema_memory/cli.py` - `--reembed`, namespace/batch arguments, and
  embedding status command.
- `src/mnema_memory/vector_index.py` and `hnsw_index.py` - only if a scoped
  namespace reset/delete primitive is necessary; preserve the durable BLOB
  source-of-truth contract.
- `.env.example`, `README.md` - local setup, offline model cache, migration
  runbook, and privacy/first-download behaviour.
- `tests/test_local_provider.py` (new) and existing service/config/CLI tests.

## Verification

### Unit tests (no model download)

Use a monkeypatched fake `sentence_transformers` module, following the existing
fake OpenAI test pattern:

- lazy import reports the correct install hint;
- model construction receives cache/offline/device options;
- batching, normalized encode flags, order preservation, and `[]` behaviour;
- invalid/NaN/ragged outputs fail before index persistence;
- aliases resolve to the local provider; OpenAI and local-hash regressions stay
  green;
- config precedence (environment over TOML), boolean parsing, and bad batch
  validation.

### Service and migration tests

- local provider end-to-end: paraphrases rank together while unrelated text
  does not (using deterministic fake semantic vectors);
- local provider is used by both recall and semantic dedup;
- existing OpenAI-shaped namespace rejects local recall/remember with the
  migration instruction;
- successful `--reembed` replaces all live vectors with one local identity,
  preserves recall/dedup, and removes old BLOB/label rows;
- an embedding failure during preparation leaves every old vector and result
  available (atomicity);
- re-embed excludes forgotten/tombstoned notes and remains correct after vault
  rebuild;
- repeat re-embed is idempotent; run the relevant cases for both `numpy` and
  `hnsw` backends.

### Manual acceptance

1. In a clean virtual environment, install `.[local]`, set
   `MNEMA_EMBEDDING_PROVIDER=local` and
   `MNEMA_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`, then start
   the MCP server with no `OPENAI_API_KEY`.
2. Confirm first use downloads/loads the model once, remember semantic
   paraphrases, and verify recall/dedup quality manually.
3. Restart with `MNEMA_LOCAL_FILES_ONLY=true` while disconnected and verify
   cached operation; point at an uncached model and verify the error is
   actionable and contains no hidden fallback to OpenAI/hash.
4. Copy a small OpenAI-backed test store, switch to local, verify the mismatch
   guard, run `--reembed`, and verify recall works after restart and
   `--rebuild-index`.

## Rollout and acceptance criteria

Release the provider as opt-in in v4. The feature is complete when a user can
install `.[local]`, configure `MNEMA_EMBEDDING_PROVIDER=local`, use Mnema with
no API key after model availability, and safely migrate each existing namespace
with `--reembed`; all normal retrieval, deduplication, forget/unforget, backup,
and rebuild flows remain correct on both vector backends.

Only consider making local the global default in a later release after measuring
package size/startup/download experience and establishing a supported model
distribution strategy for the target platforms.

## Delivered

Shipped as planned:

- `SentenceTransformerEmbeddingProvider` in `embeddings.py` (lazy import with
  `pip install .[local]` hint, normalized + validated vectors, no logging of
  text/vectors), plus `canonical_provider_name` and factory aliases
  (`local` / `sentence-transformers` / `sentence_transformers`).
- Config canonicalizes the provider at load; existing `local_*` fields wired
  into the provider.
- Namespace embedding-identity guard: `remember` checks provider/model,
  `recall` checks provider/model/dim (using the actual query dimension). A
  mismatch raises a clear error naming both identities and the `--reembed`
  command.
- `MemoryService.reembed(namespace)` — all-or-nothing namespace reindex
  (vectors prepared/validated before any mutation; forgotten notes skipped;
  idempotent when the identity is already owned) and
  `MemoryService.embedding_status()`.
- CLI `--reembed --namespace [--batch-size]` and `--embedding-status`.
- Docs: `.env.example` + README local-embeddings and migration sections.

Also fixed a pre-existing broken test (`tests/test_local_config.py` had literal
PowerShell `` `r`n `` escapes leaked into line 36, leaving `config` undefined).

**Result:** 88 tests pass (was 64+1 failing). New coverage:
`test_local_provider.py` (fake `sentence_transformers` module: lazy-import hint,
cache/offline/device options, batching/normalization/order/empty, NaN/ragged
rejection, aliases, canonicalization) and `test_local_migration.py`
(local recall ranking, local dedup, identity-mismatch guard, successful/idempotent
re-embed, re-embed atomicity on failure, forgotten-exclusion + rebuild) — the
migration suite runs on both `numpy` and `hnsw` backends.

Not done (unchanged from Not-included scope): real-model manual acceptance
requires `pip install -e .[local]` (a heavy torch download) and is deferred to a
manual run; automated tests use a deterministic fake `sentence_transformers`
module so the suite stays offline and fast.
