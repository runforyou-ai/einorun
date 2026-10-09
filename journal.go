package einorun

import (
	"context"
	"errors"
)

// Step is what the runtime saves at a safe point: the process changes since
// the last successful save, the run's totals and its checkpoint.
type Step struct {
	Changes Changes
	Usage   Usage
	Plan    []PlanTask
	// Completion is the run's active completion, or nil when there is none.
	Completion *Completion
	// State is the opaque checkpoint the runtime resumes from.
	State []byte
}

// Resume is what a host hands back to continue a run: the last saved
// checkpoint and the latest records.
//
// Blocks and Calls must reflect the host's latest records, including writes
// made outside the runtime. A host that dispatched a call to an external
// executor before the runtime recorded the handover projects that state onto
// the call: Handover and, for detached calls, the receipt in Result.
type Resume struct {
	State      []byte
	Blocks     []Block
	Calls      []ToolCall // sub-agent calls
	Plan       []PlanTask
	Completion *Completion
}

// ErrCallRejected is returned (wrapped) by SaveToolCall when the host refuses a
// submission for a business reason, such as nobody being able to decide on it.
// The runtime records the call as failed, gives the reason to the model and
// continues. Use RejectCall to build such an error.
var ErrCallRejected = errors.New("einorun: call rejected")

// CallRejection is a rejected submission with a reason for the model.
type CallRejection struct {
	Reason string
}

// Error returns the reason.
func (e *CallRejection) Error() string { return e.Reason }

// Is reports whether target is ErrCallRejected.
func (e *CallRejection) Is(target error) bool { return target == ErrCallRejected }

// RejectCall returns an error that makes the runtime fail the call with reason.
func RejectCall(reason string) error { return &CallRejection{Reason: reason} }

// Journal persists a run as it executes.
//
// Contract:
//   - SaveStep writes a step atomically. Steps are written one at a time;
//     SaveToolCall may be called concurrently with each other and with
//     SaveStep.
//   - Calls are merged with MergeCall semantics: a write never replaces a
//     record with a higher Rev, settled outcomes and the outcomes of handed
//     over calls are never changed by the runtime, and notes are merged by key.
//     A Step may carry a call snapshot older than a SaveToolCall that already
//     happened; it must not win.
//   - Writes the host makes outside the runtime (dispatching to an external
//     executor, settling an external call, deciding a submission) are
//     authoritative.
//   - SaveToolCall that submits a call for a decision (Handover
//     HandoverSubmitted) is where the host creates the submission, in the same
//     transaction. It returns an error wrapping ErrCallRejected to refuse it.
//   - Any other error aborts the run and is returned from Run.
//
// The journaltest package checks these rules against an implementation.
type Journal interface {
	SaveStep(ctx context.Context, step Step) error
	SaveToolCall(ctx context.Context, call ToolCall) error
}

// MergeCall returns the record that results from writing incoming over stored,
// following the Journal contract. Journals that keep records in memory or in a
// document store use it directly; SQL journals implement the same rules in
// their upsert.
//
//   - Descriptive fields (name, arguments, model call, replay and side-effect
//     traits) and, for calls the runtime advances, the outcome (status, result,
//     error, media, times, completion) are taken from incoming only when its Rev
//     is higher.
//   - Once a call is settled or handed over, its outcome is frozen. A handover
//     recorded by the runtime after the host already recorded one only fills
//     in the payload and, while the call has no result yet, the receipt.
//   - The first write that hands a call over records Handover and Payload with
//     the incoming outcome.
//   - Notes are merged by key; for a key both carry, the higher Rev wins.
func MergeCall(stored *ToolCall, incoming ToolCall) ToolCall {
	if stored == nil {
		return incoming.Clone()
	}
	merged := stored.Clone()
	newer := incoming.Rev > stored.Rev
	if newer {
		merged.ParentID, merged.ModelCallID, merged.CallID = incoming.ParentID, incoming.ModelCallID, incoming.CallID
		merged.Name, merged.Arguments = incoming.Name, incoming.Arguments
		merged.Replayable, merged.SideEffects = incoming.Replayable, incoming.SideEffects
		merged.Rev = incoming.Rev
	}
	frozen := stored.Status.Settled() || stored.Handover != HandoverNone
	switch {
	case frozen:
		if len(merged.Payload) == 0 && len(incoming.Payload) > 0 {
			merged.Payload = append(merged.Payload[:0:0], incoming.Payload...)
		}
		if merged.Result == nil && merged.Error == nil && incoming.Result != nil && !stored.Status.Settled() {
			merged.Result = new(*incoming.Result)
		}
	case newer:
		in := incoming.Clone()
		merged.Status, merged.Result, merged.Error, merged.Media = in.Status, in.Result, in.Error, in.Media
		merged.StartedAt, merged.CompletedAt, merged.Completion = in.StartedAt, in.CompletedAt, in.Completion
		merged.Handover, merged.Payload = in.Handover, in.Payload
	}
	for key, value := range incoming.Notes {
		if _, ok := merged.Notes[key]; ok && !newer {
			continue
		}
		if merged.Notes == nil {
			merged.Notes = map[string]string{}
		}
		merged.Notes[key] = value
	}
	return merged
}
