# einorun design

einorun is a runtime for durable, resumable agent runs on
[Eino](https://github.com/cloudwego/eino). Eino's ADK provides the agent loop,
middlewares and model components; einorun adds what a production host needs on
top: durable input, checkpoints and recovery, a process record and a live
stream, tool calls handed over to other systems, human decisions, structured
completion, output guards, context management and provider adapters.

## Goals

- Business-agnostic. einorun knows nothing about workspaces, customers,
  computers or approval workflows. Hosts plug those in through tool specs,
  guards and extensions.
- Eino is not hidden. Public APIs use `schema.AgenticMessage`,
  `tool.BaseTool`, `model.AgenticModel` and ADK middlewares directly.
- No generic storage abstraction. Persistence and input are two semantic
  interfaces (`Feed`, `Journal`); hosts keep their own schema and transactions.
  In-memory implementations are included.
- The runtime owns the run's state machine; hosts own business decisions,
  database transactions and external executors.
- Model-facing text the library writes is available in Chinese and English.

## Packages

| Package | Contents |
|---|---|
| `einorun` | Runtime, run records, `Feed` and `Journal` contracts, tool specs, extensions, completion and guard protocols |
| `einorun/llm` | Model contract and single calls: `ModelFactory`, `ModelOptions`, `OutputSchema`, `SchemaFor`, `StopReason`, `Usage`, `ModelCallID`, `Generate`, `GenerateObject`, `EstimateTokens` |
| `einorun/provider` | Chat model components; sub-packages `embedding`, `rerank`, `discovery`, `probe` |
| `einorun/stream` | Display projection of a run: snapshots, deltas, merging, applying (standard library only) |
| `einorun/memory` | Memory recall extension and memory extraction loop |
| `einorun/toolname` | Valid model-visible names for external tools |
| `einorun/tools/web` | Optional web search and fetch tools |
| `einorun/inmem` | In-memory `Feed` and `Journal` |
| `einorun/journaltest` | Contract suites for `Feed` and `Journal` implementations |
| `einorun/internal/...` | Checkpoint codec, turn history, context management, prompt catalogs |

Dependencies point one way: `stream` → standard library; `llm` → Eino schema
and model; `provider` → `llm`; `einorun` → `llm`, `stream`. The root package
does not depend on `provider`, so hosts with their own model factory do not
compile every provider SDK. `Usage` is defined once, in `llm`.

## Running

```go
rt := einorun.New(einorun.Config{Language: llm.Chinese})
result, err := rt.Run(ctx, einorun.Request{
    RunID, Instruction string
    Model:      einorun.Model{New: factory, ContextWindow, MaxOutputTokens, Inputs, ToolResultMedia},
    Tools:      []einorun.ToolSpec{...},
    Feed, Journal, Resume, ReadMedia,
    Stream:     func(stream.Delta) {...},
    Limits:     einorun.Limits{MaxIterations, MaxTurns, Corrections},
    Completion: &einorun.CompletionPolicy{...},
    Guard:      guard,
    Extensions: []einorun.Extension{...},
    Context:    einorun.ContextPolicy{...},
    DiscardUndelivered: true,
})
```

`Config.Language` selects the library's own text. The language of ADK's
built-in prompts is a process-wide setting; hosts set it once at start-up with
`adk.SetLanguage`.

Zero values: `MaxIterations <= 0` means 20 per turn (reset when new input is
claimed; a correction rerun keeps the remaining budget); `MaxTurns == 0` means
no limit; `Corrections <= 0` means 1; a nil `Journal` means an in-memory
journal (`MemoryJournal`, also offered as `inmem.Journal`); `ContextWindow <= 0`
means 32000.

### Outcome

- Completed: `err == nil` and `Result{Text, Completion, EndSeq, Usage, Blocks,
  Calls, Plan}`. A run that ends with a completion may have empty `Text`; the
  host interprets `Completion.Value`.
- Suspended: `err == nil` and `Result.Suspended`. With the in-memory journal
  `Result.Resume` can be handed back directly.
- Failed: `err != nil`; `Result` still has the usage, partial blocks (an
  unfinished candidate reply becomes a block), sub-agent calls and plan.

Before `Run` returns, pending stream operations are flushed; afterwards the
stream callback is not called again. Extensions are closed when `Run`
returns, in all three cases.

### Errors

| Source | Handling |
|---|---|
| A tool's ordinary error, unparsable arguments, an error from a spec's `Policy` | The call fails, the model sees `{"error": ...}`, the run continues |
| Any error from a completion tool | Same, and one correction is used (see Completion) |
| `Await`, `Detached`, `Complete` | Control results, see below |
| `SaveToolCall` returns an error wrapping `ErrCallRejected` | The call fails with the reason, the run continues |
| Any other `Feed` or `Journal` error, context cancellation or deadline (of the run or returned by a tool), framework interrupts | The run aborts; the error chain is kept with `%w` |
| The model still produces only reasoning after its retry | The run fails with `ErrEmptyResponse` and partial results |

## Input: Feed

```go
type Feed interface {
    Watch(ctx context.Context) (<-chan struct{}, func(), error)
    Pending(ctx context.Context, after int64) (int64, error)
    Claim(ctx context.Context, through int64) (Claim, error)
}
type Claim struct { Messages []Message; EndSeq int64 }
type Message struct {
    ID, Revision string
    Role         Role
    Content      string
    Media        *MediaRef
    Meta         map[string]string
}
```

- The runtime calls `Pending` only after `Watch` is in effect and also polls
  every 5 seconds. Signals only wake it.
- `Pending` returns the highest sequence that can be claimed now.
- `Claim` returns the conversation snapshot up to the boundary it actually
  claimed, which may be lower than requested; input above it stays pending.
- The runtime adds messages to the model history once per `ID@Revision`; a new
  revision enters again. Claimed history is kept within 50% of the context
  window, newest first, always keeping the newest message.
- Replay: after a restart the runtime claims its checkpointed boundary again
  and expects the same `EndSeq`. Advance: `Pending` only reports input that
  can be claimed, so a normal claim must move the boundary forward; a claim
  that does not breaks the contract and fails the run. A `Feed` fails a claim
  whose snapshot is empty or whose boundary is at or below the input the host
  already consumed.
- Claiming binds input to the run; it is not consumption. Hosts consume input
  in the transaction that delivers the run's result.

## Persistence: Journal

```go
type Journal interface {
    SaveStep(ctx context.Context, step Step) error
    SaveToolCall(ctx context.Context, call ToolCall) error
}
type Step struct {
    Changes    Changes
    Usage      llm.Usage
    Plan       []PlanTask
    Completion *Completion
    State      []byte
}
```

- Every call record carries a revision `Rev` the runtime increments on each
  change, and every call the runtime writes is a complete snapshot of the
  record at that revision, notes included. Journals ignore a write whose `Rev`
  is not above the stored one, so a stale snapshot inside a `Step` never
  overwrites a newer `SaveToolCall`, even across a crash.
- A newer snapshot replaces the descriptive fields and the notes. It replaces
  the outcome only while the call is neither settled nor handed over: settled
  outcomes are final, whatever status a newer write carries, and results
  written by the host outside the runtime are authoritative
  (`OverlayExternal` describes such writes).
- If the host already recorded a handover (for example when its dispatch
  transaction committed before the tool returned), whether or not the call is
  settled by then, a newer runtime snapshot
  that hands the call over too only fills in the payload if there is none and
  the receipt while the call has neither a result nor an error; it never moves
  the host's `queued`/`running` or final status.
- Calls need an ID and a status. `Step.Changes` are incremental; `Usage`,
  `Plan`, `Completion` and `State` are the run's current values and replace
  the stored ones. Blocks only link to their call.
- `SaveStep` calls are sequential; `SaveToolCall` may run concurrently with
  each other and with `SaveStep`.
- The runtime saves a step after every finalized model output, after every
  batch of tool calls and when a completion becomes active.
- Calls the runtime settles during recovery are written one by one with
  `SaveToolCall`, so hosts can notify reviewers.
- Hosts that notify reviewers do so when the merged record newly needs
  review.
- `MergeCall` implements these rules for journals that keep records as
  documents; SQL journals implement them in their upsert. `journaltest` checks
  persisted results: out-of-order and repeated writes, stale snapshots, final
  outcomes, host-first handovers, external results arriving before the
  handover write, stale or non-handover writes that must not fill a receipt,
  notes, steps and removals, block links and concurrent writes to one call.

### Ownership

```go
type Handover string
const (
    HandoverNone      Handover = ""          // the runtime advances the call
    HandoverAwait     Handover = "await"     // advanced elsewhere; the run waits and suspends after the batch
    HandoverDetached  Handover = "detached"  // advanced elsewhere; the run carries on with a receipt
    HandoverSubmitted Handover = "submitted" // awaiting a host decision; the run carries on with a receipt
)
```

- A tool returns `Await(payload)`: `HandoverAwait`; the status becomes
  `waiting` unless the host already recorded a state.
- A tool returns `Detached(receipt, payload)`: `HandoverDetached`, `Result`
  is the receipt and the model sees it immediately.
- A spec's `Policy` returns a `Submission{Receipt, Payload}`: the call is not
  executed; `HandoverSubmitted`, status `awaiting_decision`, `Result` is the
  receipt. The host creates the submission inside `SaveToolCall`.
- Sub-agents cannot `Await` or `Detached` (the call fails and the model sees
  the error); submissions are allowed.
- A crash after the host's dispatch committed but before the tool returned is
  covered by the host: it projects its own state onto `Handover` (and the
  receipt) when building `Resume`.

### Recovery

```go
type Resume struct { State []byte; Blocks []Block; Calls []ToolCall; Plan []PlanTask; Completion *Completion }
```

`State` is opaque, versioned, and only the current version is accepted. The
runtime recovers in this order:

1. Check the version and the registered extensions (a missing stateful
   extension is an error); restore extension state.
2. Merge the latest call records according to the table below; calls it
   settles are written one by one.
3. Notify `RestoreObserver`s with the latest records and results.
4. Patch the model context: calls of the last model output get the
   model-visible result from their records (looked up by call identifier and
   model-visible name); calls without one get a cancellation note. A trailing
   output with text and no tool calls was never delivered and is dropped.
5. Decide completion (see Completion), otherwise continue the loop.

| Call | Not settled | Settled |
|---|---|---|
| Main agent, runtime-owned | `interrupted` (replayable or no side effects) or `needs_review` (side effects, not replayable) | Patch the result |
| Main agent, `HandoverAwait` | The run stays suspended | Patch the result |
| Main or sub-agent, `HandoverDetached` / `HandoverSubmitted` | Not settled; patch the receipt | Patch the receipt |
| Sub-agent, runtime-owned | Settled like the main agent (sub-agents do not suspend) | — |
| Sub-agent, `HandoverAwait` (crash while waiting synchronously, projected by the host) | Not suspended, not settled; the record stays with the external executor and the delegation call is settled by its own traits | Kept |

Tools that are not registered are treated as not replayable with side
effects. A delegation call has side effects when any of the other tools does.

## Output control

### Completion

A completion tool (`ToolSpec.Completion`) returns
`Complete(value json.RawMessage, fixed bool)`.

```go
type Completion struct {
    Source CompletionSource // tool, guard, fallback
    CallID string
    Value  json.RawMessage
    Fixed  bool
    Reason string           // why a fallback completion was produced
}
type CompletionPolicy struct { Fallback func(reason string) json.RawMessage }
```

- Persistence and validity: the active completion lives only in the
  checkpoint (`Step.State`, mirrored in `Step.Completion`). A successful
  completion call is saved with `SaveToolCall` (`ToolCall.Completion` is a copy
  and never ends a run by itself), then immediately registered with
  `SaveStep`. Guard and fallback completions are registered with `SaveStep`
  before `Run` returns. Superseding a completion is committed with the
  checkpoint that enters the new input boundary, whose model context no longer
  ends with the superseded call. On recovery, the active completion is the
  checkpoint's; if the checkpoint has none but its model context ends with an
  output that only calls a completion tool whose record succeeded, that call's
  value becomes active (the crash window between the two saves). Completion
  values on older calls never become active again.
- Batches: checked once in the model phase, after the final-iteration guard
  and before any tool starts. A batch may contain at most one completion tool
  and no other tools. A violating batch is not executed; every call fails with
  the same explanation and one correction is used.
- Ending: a successful completion returns directly. Without pending input the
  whole run ends with `Result.Completion`.
- Superseding: a non-fixed completion is superseded when new input is
  pending: it is dropped from the model history (the process record keeps it)
  and the next turn starts. A fixed completion is never superseded; no further
  input is claimed and the run ends. Fallback completions are fixed.
- Corrections: batch violations, errors of completion tools and
  guard corrections share `Limits.Corrections`, which is checkpointed. A
  remaining correction is used first. When the output is invalid again with no
  correction left, or still invalid at the end of the iteration budget, the
  run ends with a fixed `Fallback(reason)` completion if a `CompletionPolicy`
  is configured; otherwise the error goes to the model and the run is not
  forced to end. Reaching the iteration budget by itself never triggers the
  fallback; valid text or a successful completion at the end of the budget
  ends the run normally.
- End of the iteration budget: only completion tools are kept when there are
  any; otherwise the tool list is set to a non-nil empty list. A closing note is
  added in both cases (naming the completion tools when there are any).
- Recovery: a fixed active completion is delivered without claiming input or
  calling the model; a non-fixed one is superseded when new input is pending
  and delivered otherwise.

### Guard

```go
type Guard interface {
    Review(ctx context.Context, turn TurnView) (Verdict, error)
}
type TurnView struct {
    Claim           Claim
    ModelInput      []*schema.AgenticMessage // what the model saw for the candidate text, after all rewrites and retries
    Calls           []CallOutcome            // this turn's calls: record, raw result, model-visible result, origin
    Text            string
    Superseded      bool
    BudgetSpent     bool
    CorrectionsLeft int
}
type Verdict struct {
    Kind   VerdictKind                  // Accept, Correct, Finish
    Prompt string                       // Correct: enters only the rerun
    Value  json.RawMessage; Fixed bool  // Finish
    Notes  map[string]map[string]string // notes by call record ID
}
```

- Guards review direct text only, never completion results. Tool observers
  see every outcome, including failures and calls waiting for an external
  result (with an empty `Raw`).
- `Review` runs for every candidate text, including one that turns out to be
  superseded (`Superseded`: the verdict is ignored, notes still apply). Notes
  are written to their calls, with a new revision, before `Review` returns,
  and appear in `Result.Blocks`.
- A correction reruns the turn with the rejected text and the prompt; neither
  enters later history. The rerun keeps the remaining iteration budget and
  does not count as a turn. Without corrections left the completion fallback
  applies; without a fallback the run fails with `ErrGuardRejected`.
- `Request.Guard` may be one of the request's extensions; for an
  `Instantiable` extension the run's instance reviews.

## Tools

```go
type ToolSpec struct {
    Tool         tool.BaseTool
    New          func(ctx context.Context, scope AgentScope) (tool.BaseTool, func(), error)
    MainOnly     bool
    RecordName   func(arguments string) string
    Replayable   bool
    SideEffects  bool
    Policy       func(ctx context.Context, call CallView) (CallPolicy, error)
    Completion   bool
    Retain       Retain // Default, Keep (never cleared), Intact (never offloaded or cleared)
    PinInSummary bool
    Notes        map[string]string
}
type CallPolicy struct { Submit *Submission; Replayable, SideEffects *bool; Notes map[string]string }
type Submission struct { Receipt string; Payload json.RawMessage }
```

`Tool` and `New` are mutually exclusive; `New` builds one instance per agent
and its release function runs when that agent ends. At execution time tools
use `CallFrom(ctx)` (record ID, model call ID, provider call ID),
`CanSuspend(ctx)` / `WithoutSuspend(ctx)`, and return `Await`, `Detached`,
or `Complete`, or return media results (see Media).

### Execution order

Model phase, for every model call:

1. Input rewriting: offloading and clearing → empty arguments become `{}` →
   missing results are patched → summarization (the summary replaces the model
   context and the turn history in the same event) → final-iteration guard →
   extension model middlewares in registration order.
2. The model call, wrapped by the runtime: assign `ModelCallID`, record the
   stream, retry once when only reasoning came back, turn multimodal input off
   and retry without media when a request carrying media fails, count discarded
   output in the usage. The input of the call that produced the candidate text
   becomes `TurnView.ModelInput`.
3. Output checks: finalize the process record → completion batch check →
   extension output observers.

Tool phase, for every call (plain and multimodal endpoints alike):

extension `Outer` (batch scheduling; every call passes through it) → batch
violation interception (no side effects) → record, submission and ownership
(`SaveToolCall`) → completion protocol → extension `Inner` → raw result
observation (`ToolObserver.AfterTool` sees the result before offloading) →
the tool. On the way back the result is offloaded into its model-visible form
and the record is saved.

Extension ADK middlewares must not wrap tools; tools are wrapped only through
staged tool middlewares. The raw result observation is always innermost.
Submissions and results patched during recovery are reported to tool observers
with origin `Submitted` and `Restored`.

## Records

```go
type Block struct { ID string; Position int64; ModelCallID string; Kind BlockKind; Text string; Call *ToolCall }
type ToolCall struct {
    ID, ParentID, ModelCallID, CallID, Name, Arguments string
    Rev         uint64
    Result, Error *string
    Media       []MediaRef
    Status      CallStatus // queued running waiting awaiting_decision succeeded failed interrupted needs_review
    StartedAt, CompletedAt *time.Time
    Replayable, SideEffects bool
    Handover    Handover
    Payload     json.RawMessage
    Completion  *CallCompletion
    Notes       map[string]string
}
```

NUL characters are removed from results and errors. Lifecycles beyond the run
(rejected, expired, cancelled, reviewed) belong to the host. Usage covers the
main model, summaries, memory selection, sub-agents and discarded output.

## Extensions

```go
type Extension interface{ Name() string }
type Instantiable    interface { Instance(ctx context.Context, run RunScope) (Extension, error) }
type ModelMiddleware interface { ModelMiddlewares(scope AgentScope) []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage] }
type ToolMiddleware  interface { ToolMiddlewares(scope AgentScope) []StagedToolMiddleware }
type OutputObserver  interface { AfterModelOutput(ctx context.Context, scope AgentScope, output *schema.AgenticMessage) }
type ToolObserver    interface { AfterTool(ctx context.Context, outcome CallOutcome) }
type ClaimObserver   interface { OnClaim(ctx context.Context, claim Claim) }
type RestoreObserver interface { AfterRestore(ctx context.Context, view RestoreView) error }
type Stateful        interface { Save() (json.RawMessage, error); Restore(json.RawMessage) error }
type InstructionProvider interface { Instruction(scope AgentScope) string }
type PinProvider     interface { Pin(messages []*schema.AgenticMessage) map[string]bool }
type Closer          interface { Close() error }
```

- Names are unique; a duplicate is an error.
- `Instantiable` extensions get one instance per run, shared by the main agent
  and its sub-agents; instances must be safe for concurrent use. A guard may be
  the same instance as an extension (`Request.Guard` accepts an extension
  instance implementing `Guard`).
- Built-in capabilities use the same mechanism: `Planning()`,
  `Subagent(SubagentSpec)`, `Skills(SkillsSpec)` (passes Eino's skill backend,
  rendering, resource listing and fork options through; skill tools default to
  `Intact` and `PinInSummary`), `memory.Recall(...)`.
- Sub-agents use the main agent's tools (without delegation and planning;
  `New` tools get their own instances), the same model factory and their own
  middleware instances. They share extension instances, the media budget and
  the multimodal switch with the main agent. Their calls are reported in
  `Result.Calls` with `ParentID`.

## Media

```go
type MediaRef struct { Key, MIME, SHA256 string; Size int64 }
ReadMedia func(ctx context.Context, ref MediaRef) ([]byte, error)
```

- Hosts store media bytes and namespace their keys; tools return references
  with `AttachMedia(ctx, refs...)` and return their text as usual, so large text is still offloaded. References are saved with the call when it settles.
- Media is charged by the bytes actually read (at most 10 MiB per item).
  Media that cannot be passed becomes a note saying why: the model cannot
  view it, it changed since it was read, or the budget is spent.
- Inline formats are an allow-list (png, jpeg, webp, gif, wav variants, mp4,
  webm, quicktime) and only for modalities the model declares. Media is chosen
  newest first within 20% of the context window (1280 tokens per item, at
  least one) and 20 MiB per run, shared by main and sub-agents.
- A modality the model lacks becomes a text note. A modality the model accepts
  but cannot receive inside tool results (`ToolResultMedia` false) is attached
  as a user message right after the result.
- Checkpoints keep the bytes of media already in the model context and only
  the references of media not yet attached. On recovery references are read
  back and checked against SHA256; mismatches are not attached.
- When a request carrying media fails, multimodal input is turned off for the
  rest of the run (checkpointed). This is a deliberately conservative reaction
  to any failure of such a request.

## Context management

`ContextPolicy` holds the thresholds; `CountTokens` is the default
estimate, `OffloadReadTool` the read-back tool and `OffloadedPath` where a
call's result is offloaded. Defaults: a single tool result above 10% of the
window (at least 4000 bytes)
is offloaded with a head-and-tail preview and can be read back with a tool that
is always registered and whose results are never offloaded, cleared or
summarized; at 75% of the window older tool calls are cleared, keeping the
last two rounds; summarization triggers at window − summary output (10% of the
window, at most the model's output limit) − 2000 tokens for the instruction.
A summary keeps the newest input of the turn; when that does not fit it keeps
the newest input, pinned calls and the latest round, and drops the analysis
section. Token estimates count one token per CJK character and a quarter per
other character, include tool definitions as JSON, and count 1280 per media
item. Thresholds, the token counter and the context-management instruction are
configurable.

## Stream

- `Delta{Stream string; Base, Sequence int64; Operations []Operation}` and
  `Snapshot{Stream, Sequence, Blocks, Candidate, Plan}`. `Stream` is chosen by
  the host; every execution attempt uses a new one starting at sequence 1.
- The stream has its own display types: blocks, `CallView{CallID, Name,
  Status, StartedAt, CompletedAt, Description, Activity}` and plan tasks,
  without arguments or results.
- `Apply` returns `ErrMismatch` for another stream, a duplicate without error
  when `Sequence` is not above the snapshot's, and `ErrGap` when `Base` is not
  the snapshot's sequence; a failing operation leaves the snapshot unchanged.
  `MergeDeltas`, `MergeOperations` and `Delta.TextBytes` support relays.
- Display behaviour: when the finalized output differs from its streamed
  chunks the block is replaced; new input discards the candidate text and tool
  blocks still queued; a retry removes the unfinished blocks of the previous
  attempt; operations are merged every 50ms; the callback is called serially
  and must not block.

## llm, provider, memory, prompts

- `llm`: `ModelFactory` (called for the main model, summaries, memory
  selection and each sub-agent; every `Generate` and `Stream` context carries
  `ModelCallID`), `ModelOptions{MaxOutputTokens, DisableThinking, Output}`,
  `SchemaFor[T]`, `StopReasonOf` and `SetStopReason`, `Generate`,
  `GenerateObject[T]` (retries once with the decode error, returns the usage
  of all attempts also on error, leaves field validation to the caller),
  `ErrRefused`, `ErrTruncated`, `Usage`.
- `provider`: configuration separates protocol, vendor preset (brand and its
  differences) and model capabilities (including `ToolResultMedia`).
  `NewChatModel` covers thinking switches, forced-tool structured output,
  Gemini response schemas, falling back to unconstrained output when a
  provider rejects the constraint with HTTP 400, and refusal reasons. The
  structured-output strategy defaults to the preset and can be overridden;
  response metadata is kept intact; the default transport has a two-minute
  response-header timeout and does not follow redirects, and can be replaced.
  `embedding`, `rerank`, `discovery` and `probe` (with its own stable error
  type) complete the package.
- `memory`: `Recall(Source, RecallOptions{Instruction, Limits})` and
  `Extract(ctx, factory, ExtractRequest{Instruction, Entries, Earlier, Recent,
  Language})`, which returns a change set. Default prompts use neutral wording.
- `toolname`: `Namer.Name(namespace, identity, server, tool)` builds
  `<namespace>__<server>__<tool>_<digest>`, at most 64 characters of ASCII
  letters, digits and underscores; the digest of the server's stable identity
  and the original tool name always stays, so names are stable and distinct.
  Transliteration (such as pinyin) is an optional function the host provides.
- `tools/web`: `web_search` and `web_fetch` over host-provided `Searcher` and
  `Fetcher`; the package validates arguments (count, time range, http/https
  addresses) and fails calls as unavailable when no service is provided.
  `SearchSpec` and `FetchSpec` register them as replayable tools without side
  effects.
- The library's own model-facing text (cancellation, interruption, needs
  review, end of budget, batch violations, summary preamble, offload read-back,
  sub-agent description, structured retry, media that cannot be viewed, tool
  unavailable, memory defaults, web tools) exists in Chinese and English and
  can be overridden. Receipts, completion tools and guard corrections are
  written by the host.
