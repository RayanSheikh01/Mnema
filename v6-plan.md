# Mnema v6 — Retention & Ranking

## Context

Two things `plan.md` promised are still missing, and both are pure-local
(no API key, no new cost):

1. **Retention / forgetting policy.** `plan.md` §8 lists a "retention /
   forgetting policy" and §9 a retention config knob, but nothing enforces one.
   A namespace grows without bound; stale, low-value episodes dilute the recall
   candidate pool forever. v3 gave us a durable, reversible `forget`
   (tombstone + vector purge, survives rebuild, reversible via `unforget`) — the
   right primitive to build an automated policy on.

2. **Ranking is hardcoded and untuned.** Recall blends
   `0.6*vector + 0.2*recency + 0.1*importance + 0.1*tag`
   ([service.py](src/mnema_memory/service.py) `_recall_tool`, ~line 300). The
   weights and the `1/(1+age_days)` recency curve cannot be changed without
   editing source. `importance` is whatever the caller passed (default `0.5` in
   `_memory_input_from_payload`), so the importance term is usually a constant
   and contributes nothing.

v6 makes ranking configurable, gives `importance` a real default signal, and
adds an opt-in retention sweep — the recall-quality foundation every later
feature (entity graph, reranking) rides on.

## Goal

- Configurable recall ranking (weights + recency half-life), read from
  `AppConfig`, defaulting to today's exact behavior.
- Optional heuristic auto-importance at write time (caller value still wins).
- An opt-in, reversible, dry-runnable retention sweep + CLI, built on v3
  `forget`/`unforget` so it survives rebuild and is fully recoverable.

## Scope

Included: ranking config + wiring; a small local importance heuristic; a
`MemoryService.apply_retention` operation + `--retention` CLI; config, docs,
and offline tests.

Not included: hybrid/BM25 keyword search, an LLM reranker, an LLM importance
scorer, and hard-delete retention. All are deferred (candidates for v7+). v6
stays local, reversible, and default-off for anything destructive.

## Design decisions

| Concern | v6 decision |
| --- | --- |
| Ranking config | Add `rank_weight_vector/recency/importance/tag` (defaults `0.6/0.2/0.1/0.1`) and `recency_half_life_days` (default `1.0`). Recall reads these; defaults reproduce current output. Weights are not required to sum to 1 (they are relative). |
| Recency curve | Keep the shape but parameterize: `recency = 0.5 ** (age_days / half_life)`. With `half_life=1.0` this is close to today's `1/(1+age)` and is monotonic/interpretable ("score halves every N days"). |
| Auto-importance | `auto_importance` (default `false`, opt-in). When on **and** the caller did not pass `importance`, compute a bounded heuristic from local signals (content length, tag count, `type=summary` boost). An explicit caller `importance` always wins. No network, no LLM. |
| Retention primitive | Reuse v3 `forget` (reversible tombstone + vector purge, survives rebuild). Retention never hard-deletes; every swept memory is recoverable with `unforget`. |
| Retention policy | Sweep live memories where `age_days >= retention_max_age_days` **and** `importance < retention_min_importance`. Both thresholds configurable; `retention_enabled` default `false`. Summaries are exempt by default (they are the distilled keepers). |
| Safety | `--retention` supports `--dry-run` (list candidates, mutate nothing) and an optional `--namespace` scope. A real run prints how many were forgotten and reminds that `unforget` reverses it. |

## Implementation

1. **`config.py`** — add `rank_weight_vector/recency/importance/tag`,
   `recency_half_life_days`, `auto_importance`, `retention_enabled`,
   `retention_max_age_days`, `retention_min_importance`,
   `retention_exempt_summaries` (default `true`). Read `MNEMA_*` / TOML; validate
   positive half-life and non-negative weights.
2. **`service.py`**
   - `_recall_tool`: replace the hardcoded blend with the configured weights and
     the half-life recency formula. Behavior is byte-identical under defaults.
   - `_memory_input_from_payload` / a new `_resolve_importance`: when
     `auto_importance` is on and the payload omits `importance`, use the
     heuristic; otherwise keep the `0.5` default and always honor an explicit
     value.
   - `apply_retention(namespace: str | None = None, dry_run: bool = False)`:
     select live, non-exempt memories past both thresholds; in a real run call
     the existing forget path per memory (tombstone + `_purge_memory_vectors`);
     return `{scanned, forgotten, candidates: [...], dry_run, namespace}`.
3. **`cli.py`** — `--retention [--namespace] [--dry-run]`, mirroring the
   `--reembed` / `--drain-embeddings` handlers.
4. **Docs** — README "Ranking" + "Retention" sections; `.env.example` entries.

## Verification

Offline unit + service tests:

- **Ranking:** default weights reproduce current ordering; raising
  `rank_weight_recency` reorders a fresh-vs-stale pair as expected; a larger
  `recency_half_life_days` flattens the recency penalty (deterministic fake
  embeddings, monkeypatched timestamps for age).
- **Auto-importance:** off by default → `importance == 0.5`; on → longer/tagged/
  summary content scores higher; an explicit caller `importance` always wins.
- **Retention dry-run:** lists the right candidates (past both thresholds,
  summaries exempt) and mutates nothing (`forgotten == 0`, memories still live).
- **Retention real run:** forgets exactly the policy matches; recall/list no
  longer return them; `unforget` restores one (v3 reversibility intact); the
  sweep survives `rebuild_index_from_vault` (tombstones persist).
- **Namespace scope + regression:** `--namespace` limits the sweep; full suite
  stays green on both `numpy` and `hnsw`.

## Delivered

Implemented in commits `944275f`, `c4004cb`, `2ee7982`, and `049439e`:

- Configurable ranking, optional auto-importance, and default-off reversible retention.
- Retention CLI, docs, environment examples, and offline coverage.
