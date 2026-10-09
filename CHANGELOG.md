# Changelog

## Unreleased

- `Runtime.Run`: durable runs with claimed input, preemption at safe points, process records with revisions, checkpoints and recovery, handed-over and submitted calls, the completion protocol, guards and extensions. `MemoryJournal` is the default journal.
- Context management (`ContextPolicy`): large results offloaded with previews and read back, older tool calls cleared, long contexts summarized with pinned calls kept.
- `AttachMedia`: tool results with host media, passed to the model within the run's media budget, inside the result or as a following user message.
- `stream`: display snapshots and deltas of a run.
- `llm`: model contract, `Generate` and `GenerateObject`.
- Run records, the `Feed` and `Journal` contracts and `MergeCall`.
- `inmem` implementations and `journaltest` contract suites.
- `provider`: chat models for the supported vendors (`vendor` presets), `embedding`, `rerank`, `discovery`, `probe` and `apierr`.
- `toolname`: valid, stable model-visible names for external tools, with optional transliteration.
- `tools/web`: `web_search` and `web_fetch` tools over host-provided `Searcher` and `Fetcher`, with Chinese and English text.
