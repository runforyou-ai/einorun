package einorun

import (
	"context"
	"errors"
	"fmt"
)

// Step is what the runtime saves at a safe point: the process changes since
// the last successful save, and the run's current usage, plan, active
// completion and checkpoint, which replace the stored values.
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
//   - Every call the runtime writes, through SaveToolCall or the Calls of a
//     Step, is a complete snapshot of the record at its Rev: all fields,
//     including every note. A higher Rev never carries less than a lower one.
//   - SaveStep writes a step atomically. Changes are incremental; Usage, Plan,
//     Completion and State are the run's current values and replace the stored
//     ones (nil clears them). Steps are written one at a time; SaveToolCall may
//     be called concurrently with each other and with SaveStep.
//   - Calls are merged with MergeCall semantics. A Step may carry a call
//     snapshot older than a SaveToolCall that already happened; it must not
//     win. Blocks only link to their call; the call itself is written through
//     Calls or SaveToolCall.
//   - Writes the host makes outside the runtime (dispatching to an external
//     executor, settling an external call, deciding a submission) are
//     authoritative; OverlayExternal describes them.
//   - SaveToolCall that submits a call for a decision (Handover
//     HandoverSubmitted) is where the host creates the submission, in the same
//     transaction. It returns an error wrapping ErrCallRejected to refuse it.
//     A host that notifies reviewers of calls needing review does so when the
//     merged record newly reaches StatusNeedsReview, not because the incoming
//     snapshot says so.
//   - Any other error aborts the run and is returned from Run.
//
// The journaltest package checks these rules against an implementation.
type Journal interface {
	SaveStep(ctx context.Context, step Step) error
	SaveToolCall(ctx context.Context, call ToolCall) error
}

// Validate reports whether c can be written: it needs an ID and a status.
func (c ToolCall) Validate() error {
	switch {
	case c.ID == "":
		return errors.New("einorun: tool call without an ID")
	case c.Status == "":
		return fmt.Errorf("einorun: tool call %s without a status", c.ID)
	}
	return nil
}

// MergeCall returns the record that results from writing the runtime's
// snapshot incoming over stored, following the Journal contract. Journals
// that keep records as documents use it directly; SQL journals implement the
// same rules in their upsert.
//
//   - A snapshot with a Rev not above the stored one changes nothing.
//   - A newer snapshot replaces the descriptive fields (name, arguments, model
//     call, traits) and the notes.
//   - It replaces the outcome (status, result, error, media, times, completion,
//     handover, payload) only while the stored call is neither settled nor
//     handed over. Settled outcomes are final, whatever the newer status.
//   - When the host recorded a handover first, settled or not, a newer
//     snapshot that hands the call over too fills in the payload if there is
//     none and the receipt while the call has neither a result nor an error.
//     Status and times stay the host's.
func MergeCall(stored *ToolCall, incoming ToolCall) ToolCall {
	if stored == nil {
		return incoming.Clone()
	}
	merged := stored.Clone()
	if incoming.Rev <= stored.Rev {
		return merged
	}
	in := incoming.Clone()
	merged.Rev = in.Rev
	merged.ParentID, merged.ModelCallID, merged.CallID = in.ParentID, in.ModelCallID, in.CallID
	merged.Name, merged.Arguments = in.Name, in.Arguments
	merged.Replayable, merged.SideEffects = in.Replayable, in.SideEffects
	merged.Notes = in.Notes
	switch {
	case stored.Handover != HandoverNone:
		// The host handed the call over first, possibly settling it already;
		// a handover snapshot only fills in what is still missing.
		if in.Handover != HandoverNone {
			if len(merged.Payload) == 0 {
				merged.Payload = in.Payload
			}
			if merged.Result == nil && merged.Error == nil {
				merged.Result = in.Result
			}
		}
	case stored.Status.Settled():
	default:
		merged.Status, merged.Result, merged.Error, merged.Media = in.Status, in.Result, in.Error, in.Media
		merged.StartedAt, merged.CompletedAt, merged.Completion = in.StartedAt, in.CompletedAt, in.Completion
		merged.Handover, merged.Payload = in.Handover, in.Payload
	}
	return merged
}

// OverlayExternal returns stored with a write the host makes outside the
// runtime applied: Status, Result, Error, Media, CompletedAt, Handover and
// Payload come from update when set (a non-nil empty Media clears the media);
// Rev and every other field are kept. It describes the authoritative writes
// the Journal contract refers to.
func OverlayExternal(stored ToolCall, update ToolCall) ToolCall {
	merged := stored.Clone()
	u := update.Clone()
	if u.Status != "" {
		merged.Status = u.Status
	}
	if u.Result != nil {
		merged.Result = u.Result
	}
	if u.Error != nil {
		merged.Error = u.Error
	}
	if u.Media != nil {
		merged.Media = u.Media
	}
	if u.CompletedAt != nil {
		merged.CompletedAt = u.CompletedAt
	}
	if u.Handover != HandoverNone {
		merged.Handover = u.Handover
	}
	if len(u.Payload) > 0 {
		merged.Payload = u.Payload
	}
	return merged
}
