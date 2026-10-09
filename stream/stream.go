// Package stream defines the display projection of an agent run: snapshots,
// deltas and the operations that turn one into the next.
//
// A run publishes deltas while it executes. A relay or client keeps a
// Snapshot and applies each Delta in order; a gap or a mismatched stream means
// the receiver has to fetch a fresh snapshot. The package depends only on the
// standard library so that relays and clients can use it without pulling in
// the runtime.
package stream

import (
	"errors"
	"fmt"
	"slices"
	"time"
)

var (
	// ErrGap reports that a delta does not start where the snapshot ends.
	ErrGap = errors.New("stream: sequence gap")
	// ErrMismatch reports that a delta belongs to another stream.
	ErrMismatch = errors.New("stream: stream mismatch")
)

// BlockKind is the kind of a content block.
type BlockKind string

const (
	// KindThinking is model reasoning.
	KindThinking BlockKind = "thinking"
	// KindContent is model text.
	KindContent BlockKind = "content"
	// KindToolCall is a tool call.
	KindToolCall BlockKind = "tool_call"
)

// Block is a content block as shown to a viewer. Tool calls carry their name
// and status only, never arguments or results.
type Block struct {
	ID          string
	Position    int64
	ModelCallID string
	Kind        BlockKind
	Text        string
	Call        *CallView
}

// CallView is the visible state of a tool call.
type CallView struct {
	CallID      string
	Name        string
	Status      string
	StartedAt   *time.Time
	CompletedAt *time.Time
	// Description describes the delegated task of a delegation call once its
	// arguments are complete.
	Description string
	// Activity names the tool a delegated agent is calling right now.
	Activity string
}

// PlanStatus is the status of a plan task.
type PlanStatus string

const (
	// PlanPending is a task that has not started.
	PlanPending PlanStatus = "pending"
	// PlanInProgress is the task being worked on.
	PlanInProgress PlanStatus = "in_progress"
	// PlanCompleted is a finished task.
	PlanCompleted PlanStatus = "completed"
)

// PlanTask is one entry of the run's task list.
type PlanTask struct {
	ID         string     `json:"id"`
	Subject    string     `json:"subject"`
	ActiveForm string     `json:"activeForm,omitempty"`
	Status     PlanStatus `json:"status"`
}

// OpKind is the kind of an operation.
type OpKind string

const (
	// OpUpsertBlock inserts a block or replaces the block with the same ID.
	OpUpsertBlock OpKind = "upsert_block"
	// OpAppendText appends text to a block.
	OpAppendText OpKind = "append_text"
	// OpRemoveBlocks removes blocks.
	OpRemoveBlocks OpKind = "remove_blocks"
	// OpAppendCandidate appends text to the candidate reply.
	OpAppendCandidate OpKind = "append_candidate"
	// OpClearCandidate clears the candidate reply.
	OpClearCandidate OpKind = "clear_candidate"
	// OpSetPlan replaces the task list.
	OpSetPlan OpKind = "set_plan"
)

// Operation is one change to a snapshot.
type Operation struct {
	Kind     OpKind
	Block    *Block     // OpUpsertBlock
	BlockID  string     // OpAppendText
	BlockIDs []string   // OpRemoveBlocks
	Text     string     // OpAppendText, OpAppendCandidate
	Plan     []PlanTask // OpSetPlan
}

// Delta moves a snapshot of Stream from sequence Base to Sequence.
type Delta struct {
	// Stream identifies the stream. The host chooses it; each execution attempt
	// of a run uses a new identifier and its sequence starts at 1.
	Stream     string
	Base       int64
	Sequence   int64
	Operations []Operation
}

// Snapshot is the full display state of a stream at Sequence.
type Snapshot struct {
	Stream    string
	Sequence  int64
	Blocks    []Block
	Candidate string
	Plan      []PlanTask
}

// Apply applies d. It returns false without an error when d ends at or before
// the snapshot's sequence (a duplicate), ErrMismatch when d belongs to another
// stream and ErrGap when d does not start at the snapshot's sequence. When an
// operation cannot be applied the snapshot is left unchanged.
func (s *Snapshot) Apply(d Delta) (bool, error) {
	if d.Stream != s.Stream {
		return false, ErrMismatch
	}
	if d.Sequence <= s.Sequence {
		return false, nil
	}
	if d.Base != s.Sequence {
		return false, ErrGap
	}
	blocks, candidate, plan := slices.Clone(s.Blocks), s.Candidate, s.Plan
	for _, op := range d.Operations {
		switch op.Kind {
		case OpUpsertBlock:
			if op.Block == nil {
				return false, errors.New("stream: upsert without a block")
			}
			if i := indexOf(blocks, op.Block.ID); i >= 0 {
				blocks[i] = op.Block.clone()
			} else {
				blocks = append(blocks, op.Block.clone())
			}
		case OpAppendText:
			i := indexOf(blocks, op.BlockID)
			if i < 0 {
				return false, fmt.Errorf("stream: append text to unknown block %q", op.BlockID)
			}
			blocks[i].Text += op.Text
		case OpRemoveBlocks:
			for _, id := range op.BlockIDs {
				i := indexOf(blocks, id)
				if i < 0 {
					return false, fmt.Errorf("stream: remove unknown block %q", id)
				}
				blocks = slices.Delete(blocks, i, i+1)
			}
		case OpAppendCandidate:
			candidate += op.Text
		case OpClearCandidate:
			candidate = ""
		case OpSetPlan:
			plan = slices.Clone(op.Plan)
		default:
			return false, fmt.Errorf("stream: unsupported operation %q", op.Kind)
		}
	}
	s.Blocks, s.Candidate, s.Plan, s.Sequence = blocks, candidate, plan, d.Sequence
	return true, nil
}

// Clone returns a deep copy of s.
func (s Snapshot) Clone() Snapshot {
	s.Plan = slices.Clone(s.Plan)
	s.Blocks = slices.Clone(s.Blocks)
	for i, b := range s.Blocks {
		s.Blocks[i] = b.clone()
	}
	return s
}

// MergeDeltas joins two consecutive deltas of the same stream. It returns false
// when later does not start where earlier ends.
func MergeDeltas(earlier, later Delta) (Delta, bool) {
	if earlier.Stream != later.Stream || earlier.Sequence != later.Base {
		return Delta{}, false
	}
	merged := later
	merged.Base = earlier.Base
	merged.Operations = MergeOperations(append(slices.Clone(earlier.Operations), later.Operations...))
	return merged, true
}

// MergeOperations folds adjacent operations that can be combined: text
// appended to the same block, candidate appends, repeated upserts of the same
// block and repeated plan updates.
func MergeOperations(ops []Operation) []Operation {
	merged := make([]Operation, 0, len(ops))
	for _, op := range ops {
		if n := len(merged); n > 0 {
			last := &merged[n-1]
			switch {
			case op.Kind == OpAppendText && last.Kind == OpAppendText && last.BlockID == op.BlockID,
				op.Kind == OpAppendCandidate && last.Kind == OpAppendCandidate:
				last.Text += op.Text
				continue
			case op.Kind == OpAppendText && last.Kind == OpUpsertBlock && last.Block.ID == op.BlockID:
				block := *last.Block
				block.Text += op.Text
				last.Block = &block
				continue
			case op.Kind == OpUpsertBlock && last.Kind == OpUpsertBlock && last.Block.ID == op.Block.ID:
				last.Block = op.Block
				continue
			case op.Kind == OpSetPlan && last.Kind == OpSetPlan:
				last.Plan = op.Plan
				continue
			}
		}
		merged = append(merged, op)
	}
	return merged
}

// TextBytes returns the number of text bytes the delta carries, so relays can
// bound the size of merged deltas.
func (d Delta) TextBytes() int {
	total := 0
	for _, op := range d.Operations {
		total += len(op.Text)
		if op.Block != nil {
			total += len(op.Block.Text)
		}
		for _, task := range op.Plan {
			total += len(task.Subject) + len(task.ActiveForm)
		}
	}
	return total
}

// indexOf returns the index of the block with id, or -1.
func indexOf(blocks []Block, id string) int {
	return slices.IndexFunc(blocks, func(b Block) bool { return b.ID == id })
}

// clone copies the block and its call view.
func (b Block) clone() Block {
	if b.Call != nil {
		call := *b.Call
		if call.StartedAt != nil {
			call.StartedAt = new(*call.StartedAt)
		}
		if call.CompletedAt != nil {
			call.CompletedAt = new(*call.CompletedAt)
		}
		b.Call = &call
	}
	return b
}
