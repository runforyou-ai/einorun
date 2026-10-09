package einorun

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/stream"
)

// Config configures a Runtime.
type Config struct {
	// Language selects the model-facing text the runtime writes. The language
	// of ADK's own prompts is a process-wide setting the host makes with
	// adk.SetLanguage.
	Language llm.Language
}

// Runtime executes runs. It is safe for concurrent use.
type Runtime struct {
	language llm.Language
}

// New returns a runtime.
func New(config Config) *Runtime {
	return &Runtime{language: config.Language}
}

// Model describes the chat model of a run.
type Model struct {
	// New creates model components; see llm.ModelFactory.
	New llm.ModelFactory
	// ContextWindow is the model's context window in tokens; zero means
	// llm.DefaultContextWindow.
	ContextWindow int
	// MaxOutputTokens caps a single output; zero means no limit.
	MaxOutputTokens int
	// Inputs are the input modalities the model accepts besides text.
	Inputs []llm.Modality
	// ToolResultMedia is true when the model accepts media inside tool
	// results; otherwise media of tool results follows as a user message.
	ToolResultMedia bool
}

// Limits bound a run.
type Limits struct {
	// MaxIterations caps model calls per turn; zero or less means 20. A new
	// claim of input resets the count; a correction rerun keeps it.
	MaxIterations int
	// MaxTurns caps the turns of a run; zero means no limit.
	MaxTurns int
	// Corrections is how many corrections a run may use (batch violations,
	// completion tool errors, guard corrections); zero or less means 1.
	Corrections int
}

// CompletionPolicy configures how runs with completion tools end.
type CompletionPolicy struct {
	// Fallback produces the value of a fixed completion when the output is
	// still invalid and cannot be corrected any more. reason is one of
	// ReasonCorrectionsExhausted, ReasonBudgetExhausted and
	// ReasonGuardRejected.
	Fallback func(reason string) json.RawMessage
	// FinalNotice, when set, replaces the note the runtime adds when the
	// iteration budget is spent.
	FinalNotice string
}

// Reasons for fallback completions.
const (
	ReasonCorrectionsExhausted = "corrections_exhausted"
	ReasonBudgetExhausted      = "budget_exhausted"
	ReasonGuardRejected        = "guard_rejected"
)

// MediaReader reads media the host stores.
type MediaReader func(ctx context.Context, ref MediaRef) ([]byte, error)

// Request is one run.
type Request struct {
	RunID string
	// Instruction is the system instruction of the main agent.
	Instruction string
	Model       Model
	Tools       []ToolSpec
	Feed        Feed
	// Journal persists the run; nil means an in-memory journal, and a
	// suspended run then returns Result.Resume.
	Journal Journal
	// Resume continues a run from its last checkpoint.
	Resume *Resume
	// ReadMedia reads the media of messages and tool results; nil means
	// media is only referred to in text.
	ReadMedia MediaReader
	// Stream receives the run's display deltas, serially; it must not block.
	Stream func(stream.Delta)
	// StreamID identifies the stream of this execution attempt; empty means a
	// new random identifier.
	StreamID   string
	Limits     Limits
	Completion *CompletionPolicy
	// Guard reviews candidate text. It may be an extension instance
	// registered in Extensions, or an Instantiable extension's prototype, in
	// which case the run's instance is used.
	Guard      Guard
	Extensions []Extension
	// DiscardUndelivered drops direct text superseded by new input from the
	// model history; the process record keeps it.
	DiscardUndelivered bool
}

// Result is the outcome of a run.
type Result struct {
	// Text is the delivered text; it may be empty when Completion is set.
	Text       string
	Completion *Completion
	// EndSeq is the input boundary the result answers.
	EndSeq    int64
	Usage     Usage
	Blocks    []Block
	Calls     []ToolCall // sub-agent calls
	Plan      []PlanTask
	Suspended bool
	// Resume continues a suspended run that used the in-memory journal.
	Resume *Resume
}

// errSuspended ends the turn loop when the run suspends.
var errSuspended = errors.New("einorun: run suspended")

// Run executes a run until it completes, suspends or fails. See Result for
// what each outcome returns.
func (r *Runtime) Run(ctx context.Context, request Request) (Result, error) {
	if request.Feed == nil {
		return Result{}, errors.New("einorun: request has no feed")
	}
	if request.Model.New == nil {
		return Result{}, errors.New("einorun: request has no model factory")
	}
	for i, spec := range request.Tools {
		if err := spec.validate(i); err != nil {
			return Result{}, err
		}
	}
	execution, err := r.assemble(ctx, request)
	if execution != nil {
		defer execution.close()
	}
	if err != nil {
		return Result{}, err
	}
	result, err := execution.run(ctx)
	if result.Suspended && execution.memory != nil {
		resume := execution.memory.Resume()
		result.Resume = &resume
	}
	if err != nil {
		return result, fmt.Errorf("einorun: run %s: %w", request.RunID, err)
	}
	return result, nil
}
