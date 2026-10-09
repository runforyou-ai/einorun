package einorun

import (
	"context"
	"encoding/json"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/llm"
)

// Extension adds behaviour to runs. An extension implements Name and any of
// the optional interfaces below; names must be unique within a run.
type Extension interface {
	Name() string
}

// RunScope describes the run an extension instance serves.
type RunScope struct {
	RunID    string
	Language llm.Language
	// Model is the run's model factory, for extensions that call the model
	// themselves; they report the usage through UsageReporter.
	Model llm.ModelFactory
}

// Instantiable extensions create one instance per run. The instance is
// shared by the main agent and its sub-agents and must be safe for
// concurrent use. Other extensions are used as they are.
type Instantiable interface {
	Instance(ctx context.Context, run RunScope) (Extension, error)
}

// ModelMiddleware adds ADK middlewares to an agent, after the runtime's own
// (see the execution order in the design). They must not wrap tools; use
// ToolMiddleware for that.
type ModelMiddleware interface {
	ModelMiddlewares(scope AgentScope) []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]
}

// Stage is where a tool middleware runs.
type Stage int

const (
	// StageOuter runs outside the runtime's recording, before a call is
	// recorded or submitted; every call of a batch passes through it. Use it
	// to schedule calls.
	StageOuter Stage = iota
	// StageInner runs around the actual execution of the tool.
	StageInner
)

// StagedToolMiddleware is a tool middleware and where it runs.
type StagedToolMiddleware struct {
	Stage      Stage
	Middleware compose.ToolMiddleware
}

// ToolMiddleware adds tool middlewares to an agent.
type ToolMiddleware interface {
	ToolMiddlewares(scope AgentScope) []StagedToolMiddleware
}

// OutputObserver sees every finalized model output before its tools run.
type OutputObserver interface {
	AfterModelOutput(ctx context.Context, scope AgentScope, output *schema.AgenticMessage)
}

// Origin is where a call outcome reported to observers comes from.
type Origin string

const (
	// OriginExecuted is a call the tool executed.
	OriginExecuted Origin = "executed"
	// OriginSubmitted is a call submitted for a decision; Result is the
	// receipt.
	OriginSubmitted Origin = "submitted"
	// OriginRestored is a result patched into the model context on recovery.
	OriginRestored Origin = "restored"
)

// CallOutcome is the outcome of one call as observers see it.
type CallOutcome struct {
	Agent AgentScope
	// Name is the model-visible tool name.
	Name string
	// Call is the call record after the outcome was recorded.
	Call ToolCall
	// Raw is the result as the tool returned it, before context management
	// turned it into what the model sees; empty for failed calls and calls
	// waiting for an external result.
	Raw    string
	Origin Origin
}

// ToolObserver sees every call outcome, with the raw result.
type ToolObserver interface {
	AfterTool(ctx context.Context, outcome CallOutcome)
}

// ClaimObserver sees every claim of new input.
type ClaimObserver interface {
	OnClaim(ctx context.Context, claim Claim)
}

// RestoreView is what a run resumes from: the latest call records, after the
// runtime settled interrupted calls.
type RestoreView struct {
	Blocks []Block
	Calls  []ToolCall
}

// RestoreObserver sees the records a run resumes from, after extension state
// was restored and before the model context is patched.
type RestoreObserver interface {
	AfterRestore(ctx context.Context, view RestoreView) error
}

// Stateful extensions keep state in the run's checkpoint.
type Stateful interface {
	Save() (json.RawMessage, error)
	Restore(json.RawMessage) error
}

// InstructionProvider adds text to an agent's instruction.
type InstructionProvider interface {
	Instruction(scope AgentScope) string
}

// PinProvider names tool calls to keep verbatim when the context is
// summarized, by the call identifiers the model gave.
type PinProvider interface {
	Pin(messages []*schema.AgenticMessage) map[string]bool
}

// ContextReserver extensions add text to an agent's model calls outside the
// agent's state, such as a reminder added when the model is wrapped. Before
// each model call the runtime lowers the summary threshold of the agent by the
// tokens they report, so the call still fits the window.
type ContextReserver interface {
	ReservedTokens(scope AgentScope) int
}

// UsageReporter extensions report the total usage of the model calls they
// made in this run, which the runtime adds to the run's usage. Only the usage
// since the run's latest start counts; usage from before a restart is already
// in the checkpoint.
type UsageReporter interface {
	Usage() llm.Usage
}

// Closer extensions are closed when Run returns.
type Closer interface {
	Close() error
}
