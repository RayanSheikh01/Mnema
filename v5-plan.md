# Mnema v5 — Pluggable LLM Summaries

## Context

`memory.summarize` ([service.py](src/mnema_memory/service.py) `_summarize_tool`)
is the weakest first-class feature: it concatenates a 160-char excerpt of each
source note into a bullet list. There is no synthesis — no salient points,
decisions, unresolved items, or entities/topics that `plan.md` §7 promised for
the episodic→semantic layer. v3 and v4 both explicitly deferred a real LLM
summarizer.

v5 adds a **pluggable `SummaryGenerator` seam** mirroring the existing
`EmbeddingProvider` design (v1) and local-embedding provider (v4): a stable
interface with an offline default and opt-in remote backends. It is provider-
agnostic (OpenAI **and** Anthropic/Claude), never mandatory, and preserves the
canonical vault-and-links model — the LLM writes only the prose; Mnema still
generates the `derived_from` wikilinks deterministically.

## Goal

Ship `MNEMA_SUMMARY_PROVIDER` with:

- `extractive` (default) — the current dependency-free behavior, unchanged;
- `openai` — chat-completion synthesis (`gpt-4o-mini` default);
- `anthropic` / `claude` — Claude synthesis (`claude-haiku-4-5` default — cheap,
  bounded task; see the v4 cost analysis).

## Scope

Included: the `SummaryGenerator` interface + three implementations + factory;
config (`summary_provider`, `summary_model`) with env/TOML; service wiring so
`_summarize_tool` delegates the "Key Points" body while still writing the topic
header and `Derived From` links; unit + service tests with a fake generator and
fake SDK modules; docs.

Not included: streaming, batching multiple summaries, an LLM re-ranker, a local
LLM (transformers text-generation), or auto-summarize triggers. Existing
deployments keep the extractive default until they opt in.

## Design decisions

| Concern | v5 decision |
| --- | --- |
| Interface | `SummaryGenerator.summarize(topic, sources) -> str`, where `sources` is `list[{title, content}]` and the return value is the markdown body starting at `### Key Points`. |
| Default | `extractive` — moves today's bullet logic into `ExtractiveSummaryGenerator`; output is byte-identical to v4 for the default path. |
| Links stay deterministic | The generator returns only prose. `_summarize_tool` always prepends `## Summary Topic:` and appends `### Derived From` wikilinks from SQLite — the LLM never invents links. |
| Source content | Pass the full note body to the generator (LLMs need it); the extractive generator truncates to 160 chars to preserve current output. |
| Dependency | Reuse the existing `[openai]` extra; add an `[anthropic]` extra. Lazy-import inside each provider; never import at module load. |
| Failure | No silent fallback: a remote generator raises and the error surfaces. The default provider is offline and cannot fail. |
| Privacy | Remote providers send note contents to the API; documented. `extractive` keeps everything local. |

## Implementation

1. **`src/mnema_memory/summarizer.py`** (new): `SummaryGenerator` ABC;
   `ExtractiveSummaryGenerator`; `OpenAISummaryGenerator`;
   `AnthropicSummaryGenerator`; `build_summary_generator(provider, model)` with
   aliases (`extractive`/`""`, `openai`, `anthropic`/`claude`). Remote providers
   build a system+user prompt asking for concise `### Key Points` bullets
   capturing decisions, entities, and unresolved items, with no preamble.
2. **`config.py`**: `summary_provider: str = "extractive"`,
   `summary_model: str = ""`; read `MNEMA_SUMMARY_PROVIDER` / `MNEMA_SUMMARY_MODEL`
   and TOML equivalents.
3. **`service.py`**: build `self.summary_generator` in `__init__`; rework
   `_summarize_tool` to gather `(title, full body)` sources and delegate the body
   to the generator, keeping the header + `Derived From` composition and the
   `_remember_tool` write path unchanged.
4. **`pyproject.toml`**: add `anthropic = ["anthropic>=0.40"]` extra.
5. **Docs**: README "Summaries" section + `.env.example` entries.

## Verification

- `test_summary_generators.py`: extractive output shape; factory aliases resolve
  to the right classes; OpenAI/Anthropic providers, with fake SDK modules
  installed in `sys.modules`, produce the stubbed completion text and send the
  source content in the request; missing-key/missing-package errors are
  actionable.
- `test_summarize.py`: default extractive summarize writes a `summary` note with
  correct `derived_from` links and is recallable (current behavior preserved);
  swapping in a fake LLM generator puts synthesized text in the note body while
  Mnema still writes the deterministic links.
- Full suite stays offline and green on both vector backends.

## Delivered

Shipped as planned:

- `src/mnema_memory/summarizer.py`: `SummaryGenerator` ABC;
  `ExtractiveSummaryGenerator` (default, offline, byte-compatible with the pre-v5
  bullet output); `OpenAISummaryGenerator` (`gpt-4o-mini` default);
  `AnthropicSummaryGenerator` (`claude-haiku-4-5` default); and
  `build_summary_generator` with aliases (`extractive`/`""`/`none`, `openai`,
  `anthropic`/`claude`). Remote providers lazy-import their SDK and give an
  actionable error on a missing key or package.
- `config.py`: `summary_provider` (default `extractive`) + `summary_model`, read
  from `MNEMA_SUMMARY_PROVIDER` / `MNEMA_SUMMARY_MODEL` and TOML.
- `service.py`: builds `self.summary_generator`; `_summarize_tool` now hands the
  full source note bodies to the generator and uses its `### Key Points` body,
  while still composing the `## Summary Topic:` header and the deterministic
  `### Derived From` wikilinks from SQLite (the LLM never invents a link).
- `pyproject.toml`: new `[anthropic]` extra.
- Docs: README "Summaries" section + `.env.example` entries.

**Result:** 100 tests pass (up from 88). New coverage:
`test_summary_generators.py` (extractive shape, factory aliases, unknown-provider
error, OpenAI + Anthropic providers driven through fake SDK modules that assert
the source content reaches the request and the default models are used, and
missing-key errors) and `test_summarize.py` (extractive default writes a linked,
recallable summary note unchanged; a fake LLM generator's body lands in the note
while Mnema still writes the deterministic Derived From links; config selects the
provider). Suite stays fully offline.

Not done (unchanged from Not-included scope): streaming, batched multi-summary,
LLM re-ranking, a local generation model, and auto-summarize triggers. Real-API
acceptance requires a key and the `[openai]`/`[anthropic]` extra; automated tests
use fake SDK modules so the suite needs no network or key.
