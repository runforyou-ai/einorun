# Changelog

## Unreleased

- Calls can pause for a decision: `CallPolicy.Confirm` records the call as `awaiting_decision` and suspends the run; the host writes `ToolCall.Decision` (approve, approve with changed arguments, or reject with a reason), and on resume the runtime carries it out within the same turn. New `StatusRejected`; `MergeCall` keeps the host's decision and `OverlayExternal` writes it; `journaltest` checks it; new texts `CannotConfirm`, `CallRejected` and `CallEdited`.

## 0.1.0 - 2026-10-09

First release.

- `Runtime.Run`: durable runs with claimed input, preemption at safe points, process records with revisions, checkpoints and recovery, handed-over and submitted calls, the completion protocol, guards and extensions. `MemoryJournal` is the default journal.
- Context management (`ContextPolicy`): large results offloaded with previews and read back, older tool calls cleared, long contexts summarized with pinned calls kept.
- `AttachMedia`: tool results with host media, passed to the model within the run's media budget, inside the result or as a following user message.
- Built-in extensions: `Planning` (task list published as the run's plan), `Subagent` (delegation with sub-agent calls recorded under the delegation call) and `Skills` (Eino skills, forked skills run in a sub-agent).
- `memory`: `Recall` extension that shows the memories relevant to each turn, and `Extract` for memory changes from a conversation. Extensions get the run's model through `RunScope.Model`, report its usage with `UsageReporter` and reserve context for text they add to model calls with `ContextReserver`; `IsInput` tells claimed input apart from messages the runtime adds.
- `examples/chat`: a command-line agent with the in-memory feed, a provider model, a tool, the task list and sub-agents.
- `AgentScope.ID` tells agents apart, parallel sub-agents included; `Request.BuiltinTools` gives notes and policies to the tools the runtime and built-in extensions add; `Config.Text` overrides the runtime's model-facing text (`Text`, `DefaultText`), and `Runtime.InterruptedOutcome` settles interrupted calls outside a run with the same text.
- `journaltest` writes UUIDv7 IDs, compares payloads as JSON and lets feeds that derive messages from their own records report what they store: `FeedHarness.Append` now returns the stored message as well as its sequence number.
- `stream`: display snapshots and deltas of a run.
- `llm`: model contract, `Generate` and `GenerateObject`.
- Run records, the `Feed` and `Journal` contracts and `MergeCall`.
- `inmem` implementations and `journaltest` contract suites.
- `provider`: chat models for the supported vendors (`vendor` presets), `embedding`, `rerank`, `discovery`, `probe` and `apierr`.
- `toolname`: valid, deterministic model-visible names for external tools, kept apart by a digest, with optional transliteration.
- `tools/web`: `web_search` and `web_fetch` tools over host-provided `Searcher` and `Fetcher`, with Chinese and English text.
