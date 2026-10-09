package einorun

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/cloudwego/eino/components/tool"
)

// Retain controls how context management treats a tool's results.
type Retain int

const (
	// RetainDefault results are offloaded when large and cleared when old.
	RetainDefault Retain = iota
	// RetainKeep results are offloaded when large but never cleared.
	RetainKeep
	// RetainIntact results are never offloaded or cleared.
	RetainIntact
)

// AgentScope identifies the agent a tool instance or middleware belongs to.
type AgentScope struct {
	// Name is the agent's name.
	Name string
	// Main is true for the main agent and false for sub-agents.
	Main bool
	// ID identifies the agent within the run: MainAgentID for the main agent,
	// and for a sub-agent the record ID of the call it serves, which is the
	// ParentID of its calls. Parallel sub-agents of the same type have
	// different IDs.
	ID string
}

// MainAgentID is the AgentScope.ID of the main agent.
const MainAgentID = "main"

// ToolSpec registers a tool for a run.
type ToolSpec struct {
	// Tool is an instance shared by the main agent and sub-agents. Exactly one
	// of Tool and New is set.
	Tool tool.BaseTool
	// New builds one instance per agent; the returned function, when not nil,
	// is called when that agent ends.
	New func(ctx context.Context, scope AgentScope) (tool.BaseTool, func(), error)
	// MainOnly registers the tool for the main agent only.
	MainOnly bool
	// RecordName returns the name recorded for a call, such as the original
	// name of a tool whose model-visible name is derived. Nil records the
	// model-visible name.
	RecordName func(arguments string) string
	// Replayable allows the model to call again when the outcome is unknown.
	// It does not make the runtime repeat calls.
	Replayable bool
	// SideEffects marks tools that change state outside the run.
	SideEffects bool
	// Policy decides per call, with the complete arguments, whether the call
	// is submitted for a decision instead of executed, and may adjust traits
	// and notes. An error fails the call and the model sees it.
	Policy func(ctx context.Context, call CallView) (CallPolicy, error)
	// Completion makes the tool a completion tool (see Complete).
	Completion bool
	// Retain controls context management of the tool's results.
	Retain Retain
	// PinInSummary keeps the tool's calls and results verbatim when the
	// context is summarized.
	PinInSummary bool
	// Notes are written to every call record of the tool.
	Notes map[string]string
}

// CallView describes a call to a policy.
type CallView struct {
	// Name is the model-visible tool name.
	Name      string
	Arguments string
	// CallID is the identifier the model gave.
	CallID string
	Agent  AgentScope
}

// CallPolicy is a policy's decision for one call.
type CallPolicy struct {
	// Submit, when set, submits the call for a decision by the host instead
	// of executing it.
	Submit *Submission
	// Confirm, when set, pauses the run for a decision on the call before it
	// runs. Only main-agent calls that can suspend pause; elsewhere the call
	// fails. Submit and Confirm are exclusive.
	Confirm *Confirmation
	// Replayable and SideEffects, when set, override the spec's traits.
	Replayable  *bool
	SideEffects *bool
	// Notes are added to the call record.
	Notes map[string]string
}

// Submission is a call submitted for a decision. The host creates the
// submission when the runtime saves the call (see Journal); the model receives
// Receipt as the call's result.
type Submission struct {
	Receipt string
	Payload json.RawMessage
}

// Confirmation pauses a call for a decision by the host. The call is
// recorded with StatusAwaitingDecision and Payload, the run suspends after the
// current batch, and the host decides by writing the call's Decision (see
// Journal). When the run resumes, the runtime carries the decision out within
// the same turn: an approved call runs, with the decision's arguments when
// set, and a rejected call gives the model the reason.
type Confirmation struct {
	Payload json.RawMessage
}

// CallContext identifies the call a tool is executing.
type CallContext struct {
	// RecordID is the ID of the call record.
	RecordID string
	// ModelCallID is the model call that requested the call.
	ModelCallID string
	// ProviderCallID is the call identifier the model gave.
	ProviderCallID string
}

type callContextKey struct{}

// callMeta is what the runtime knows about the executing call.
type callMeta struct {
	CallContext
	suspendable bool
	media       *callMedia
}

// callMedia collects the media a call attaches to its result.
type callMedia struct {
	mu   sync.Mutex
	refs []MediaRef
}

// AttachMedia attaches media the host stores to the result of the executing
// call, in order. The references are saved with the call; the runtime reads
// the media back before the next model call and passes it to the model within
// the run's media budget. It fails outside a call.
func AttachMedia(ctx context.Context, refs ...MediaRef) error {
	meta, ok := ctx.Value(callContextKey{}).(callMeta)
	if !ok || meta.media == nil {
		return errors.New("einorun: AttachMedia outside a tool call")
	}
	meta.media.mu.Lock()
	defer meta.media.mu.Unlock()
	meta.media.refs = append(meta.media.refs, refs...)
	return nil
}

// attached returns the attached media.
func (m *callMedia) attached() []MediaRef {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.refs)
}

// CallFrom returns the call a tool is executing, and false outside a call.
func CallFrom(ctx context.Context) (CallContext, bool) {
	meta, ok := ctx.Value(callContextKey{}).(callMeta)
	return meta.CallContext, ok
}

// CanSuspend reports whether the executing call may return Await. Calls of
// sub-agents and calls under WithoutSuspend cannot.
func CanSuspend(ctx context.Context) bool {
	meta, _ := ctx.Value(callContextKey{}).(callMeta)
	return meta.suspendable
}

// WithoutSuspend returns a context in which the executing call cannot suspend
// the run, for tools that need to process a result themselves before handing
// it to the model.
func WithoutSuspend(ctx context.Context) context.Context {
	meta, ok := ctx.Value(callContextKey{}).(callMeta)
	if !ok {
		return ctx
	}
	meta.suspendable = false
	return context.WithValue(ctx, callContextKey{}, meta)
}

// control is a control result a tool returns in place of an error.
type control struct {
	handover   Handover
	receipt    string
	payload    json.RawMessage
	completion *CallCompletion
}

// Error describes the control result.
func (c *control) Error() string {
	switch {
	case c.completion != nil:
		return "einorun: call completes the run"
	case c.handover == HandoverAwait:
		return "einorun: call awaits an external result"
	default:
		return "einorun: call was handed over"
	}
}

// Await hands the call to an external executor and makes the run wait for
// its result: the call becomes HandoverAwait and the run suspends after the
// current batch of calls, to be resumed once the host has recorded the
// result. Only calls for which CanSuspend is true may await; elsewhere the
// call fails.
func Await(payload json.RawMessage) error {
	return &control{handover: HandoverAwait, payload: payload}
}

// Detached hands the call to an external executor and lets the run carry on:
// the model receives receipt as the result and the external executor advances
// the record. Sub-agents cannot detach calls.
func Detached(receipt string, payload json.RawMessage) error {
	return &control{handover: HandoverDetached, receipt: receipt, payload: payload}
}

// Complete is returned by completion tools to end the run with value. A fixed
// completion ends the run even when new input is pending; a non-fixed one is
// superseded by new input. Any other error from a completion tool goes to the
// model and uses one of the run's corrections.
func Complete(value json.RawMessage, fixed bool) error {
	return &control{completion: &CallCompletion{Value: value, Fixed: fixed}}
}

// asControl returns the control result in err.
func asControl(err error) (*control, bool) {
	return errors.AsType[*control](err)
}

// validate checks that exactly one of Tool and New is set.
func (s ToolSpec) validate(index int) error {
	if (s.Tool == nil) == (s.New == nil) {
		return fmt.Errorf("einorun: tool spec %d must set exactly one of Tool and New", index)
	}
	return nil
}
