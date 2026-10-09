package einorun

import (
	"encoding/json"
	"maps"
	"slices"
	"time"

	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/stream"
)

// BlockKind is the kind of a content block.
type BlockKind string

const (
	// KindThinking is model reasoning.
	KindThinking BlockKind = "thinking"
	// KindContent is model text.
	KindContent BlockKind = "content"
	// KindToolCall is a tool call made by the main agent.
	KindToolCall BlockKind = "tool_call"
)

// Block is one piece of the run's process in the order the model produced
// it: reasoning, text or a tool call.
type Block struct {
	ID          string
	Position    int64
	ModelCallID string
	Kind        BlockKind
	Text        string
	// Call is the tool call of a KindToolCall block. Journals persist the call
	// itself through the Calls of a Step or SaveToolCall and keep only the link
	// from the block; Resume hands blocks back with their calls attached.
	Call *ToolCall
}

// CallStatus is the status of a tool call. The runtime only produces the
// statuses below; hosts may persist further statuses for calls they own, such
// as expired or cancelled submissions, and the runtime treats any status it
// does not know as settled. Hosts may also record a submission they reject as
// StatusRejected.
type CallStatus string

const (
	// StatusQueued is a call the model made that has not started.
	StatusQueued CallStatus = "queued"
	// StatusRunning is a call being executed.
	StatusRunning CallStatus = "running"
	// StatusWaiting is a call waiting for an external result.
	StatusWaiting CallStatus = "waiting"
	// StatusAwaitingDecision is a call submitted for a decision by the host,
	// or paused for one (see CallPolicy.Confirm).
	StatusAwaitingDecision CallStatus = "awaiting_decision"
	// StatusSucceeded is a call that returned a result.
	StatusSucceeded CallStatus = "succeeded"
	// StatusFailed is a call that returned an error, which the model saw.
	StatusFailed CallStatus = "failed"
	// StatusInterrupted is a call cut off before it finished whose outcome is
	// harmless to repeat or had no side effects.
	StatusInterrupted CallStatus = "interrupted"
	// StatusNeedsReview is a call cut off before it finished whose side effects
	// may or may not have happened.
	StatusNeedsReview CallStatus = "needs_review"
	// StatusRejected is a paused call the host rejected; the model saw the
	// reason.
	StatusRejected CallStatus = "rejected"
)

// Settled reports whether the call has a final outcome. Statuses the runtime
// does not know are the host's own final statuses; the empty status is not a
// status at all and is not settled.
func (s CallStatus) Settled() bool {
	switch s {
	case "", StatusQueued, StatusRunning, StatusWaiting, StatusAwaitingDecision:
		return false
	}
	return true
}

// Handover records who advances a call other than the runtime.
type Handover string

const (
	// HandoverNone is a call the runtime advances.
	HandoverNone Handover = ""
	// HandoverAwait is a call advanced outside the runtime while the run waits
	// for it: the run suspends after the current batch of calls.
	HandoverAwait Handover = "await"
	// HandoverDetached is a call advanced outside the runtime while the run
	// carries on with a receipt.
	HandoverDetached Handover = "detached"
	// HandoverSubmitted is a call submitted for a decision by the host; the run
	// carries on with a receipt.
	HandoverSubmitted Handover = "submitted"
)

// MediaRef refers to media the host stores. The runtime never stores media
// bytes in records; it reads them back through the host's media reader and
// checks SHA256 before handing them to a model.
type MediaRef struct {
	// Key identifies the media for the host. Hosts namespace keys so that
	// message attachments and tool media cannot collide.
	Key    string `json:"key"`
	MIME   string `json:"mime"`
	SHA256 string `json:"sha256,omitempty"`
	Size   int64  `json:"size"`
}

// CallCompletion is the completion value a completion tool returned. It is a
// copy kept with the call; the active completion of a run lives only in the
// run's checkpoint.
type CallCompletion struct {
	Value json.RawMessage `json:"value"`
	Fixed bool            `json:"fixed"`
}

// CallDecision is the host's decision on a call paused for one (see
// CallPolicy.Confirm). The host writes it; the runtime carries it out when
// the run resumes.
type CallDecision struct {
	// Approved runs the call; otherwise, including for the zero value, it is
	// rejected and the model sees Reason.
	Approved bool   `json:"approved"`
	Reason   string `json:"reason,omitempty"`
	// Arguments, when set on an approval, replace the arguments the model
	// gave; the spec's Policy does not check them again.
	Arguments string `json:"arguments,omitempty"`
}

// ToolCall is the record of one tool call.
type ToolCall struct {
	// ID is the record ID the runtime assigns.
	ID string
	// ParentID is the record ID of the delegation call a sub-agent call
	// belongs to.
	ParentID string
	// ModelCallID is the model call that requested the call; sub-agent calls
	// carry the model call of their delegation call.
	ModelCallID string
	// CallID is the call identifier the model gave.
	CallID string
	// Name is the record name: the tool's original name for tools whose
	// model-visible name is derived, otherwise the model-visible name.
	Name      string
	Arguments string
	// Rev increases with every change the runtime makes to the record.
	Rev uint64
	// Result is the model-visible result, or the receipt of a call handed over.
	Result *string
	Error  *string
	// Media lists the media of the result in order.
	Media       []MediaRef
	Status      CallStatus
	StartedAt   *time.Time
	CompletedAt *time.Time
	// Replayable allows the model to call again when the outcome is unknown.
	Replayable bool
	// SideEffects marks calls that change state outside the run.
	SideEffects bool
	Handover    Handover
	// Payload is opaque data given when the call was handed over or
	// submitted; journals store it as is.
	Payload    json.RawMessage
	Completion *CallCompletion
	// Decision is the host's decision on a call paused for one; the host owns
	// it, and runtime snapshots never change or clear it.
	Decision *CallDecision
	// Notes are annotations by tool specs, guards and extensions. The runtime
	// only adds or changes notes, so every snapshot carries all of them; a
	// newer snapshot replaces the stored notes as a whole, so hosts keep data
	// of their own elsewhere.
	Notes map[string]string
}

// Clone returns a deep copy of c.
func (c ToolCall) Clone() ToolCall {
	if c.Result != nil {
		c.Result = new(*c.Result)
	}
	if c.Error != nil {
		c.Error = new(*c.Error)
	}
	if c.StartedAt != nil {
		c.StartedAt = new(*c.StartedAt)
	}
	if c.CompletedAt != nil {
		c.CompletedAt = new(*c.CompletedAt)
	}
	c.Media = slices.Clone(c.Media)
	c.Payload = slices.Clone(c.Payload)
	if c.Completion != nil {
		completion := *c.Completion
		completion.Value = slices.Clone(completion.Value)
		c.Completion = &completion
	}
	c.Notes = maps.Clone(c.Notes)
	if c.Decision != nil {
		c.Decision = new(*c.Decision)
	}
	return c
}

// PlanTask is one entry of the run's task list.
type PlanTask = stream.PlanTask

// CompletionSource is what produced a completion.
type CompletionSource string

const (
	// FromTool is a completion returned by a completion tool.
	FromTool CompletionSource = "tool"
	// FromGuard is a completion a guard decided.
	FromGuard CompletionSource = "guard"
	// FromFallback is a completion the host's fallback produced.
	FromFallback CompletionSource = "fallback"
)

// Completion is a structured result that ends a run.
type Completion struct {
	Source CompletionSource `json:"source"`
	// CallID is the record ID of the completion call when Source is FromTool.
	CallID string          `json:"callId,omitempty"`
	Value  json.RawMessage `json:"value"`
	// Fixed completions are not superseded by new input.
	Fixed bool `json:"fixed"`
	// Reason is why a fallback completion was produced.
	Reason string `json:"reason,omitempty"`
}

// Changes are the changes to a run's process since the last successful save.
// Journals apply removals first, then calls, then blocks.
type Changes struct {
	// Blocks are new blocks and blocks whose position or text changed, in
	// position order.
	Blocks []Block
	// Calls are new and changed calls, including sub-agent calls.
	Calls []ToolCall
	// RemovedBlocks are the IDs of removed blocks.
	RemovedBlocks []string
	// RemovedCalls are the IDs of the calls of removed blocks.
	RemovedCalls []string
}

// Empty reports whether there are no changes.
func (c Changes) Empty() bool {
	return len(c.Blocks) == 0 && len(c.Calls) == 0 && len(c.RemovedBlocks) == 0 && len(c.RemovedCalls) == 0
}

// Usage is the token usage of a run.
type Usage = llm.Usage
